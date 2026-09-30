import json
from typing import Any, Callable

import requests

from src.config.settings import settings


class LLMError(RuntimeError):
    """Raised when the LLM server is unreachable or returns an unusable response."""


class LMStudioClient:
    """Client for LM Studio's OpenAI-compatible chat completions API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout: int | None = None,
        reasoning_effort: str | None = None,
    ):
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self.temperature = (
            settings.llm_temperature if temperature is None else temperature
        )
        self.timeout = timeout or settings.llm_timeout
        self.reasoning_effort = (
            settings.llm_reasoning_effort
            if reasoning_effort is None
            else reasoning_effort
        )

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        json_schema: dict[str, Any] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """Return the model's reply. Raises LLMError on any failure.

        If json_schema is given, the server is asked to constrain output to it.

        If on_token is given, the reply is streamed and each piece of
        ``content`` is passed to it as it arrives; the full reply is still
        returned. A failure after some pieces were delivered raises LLMError
        like any other, so the caller must be ready to retract what it showed.

        Only ``message.content`` is returned. Gemma 4 also emits
        ``reasoning_content``, which is not part of the answer and must never
        reach the user or the citation validator.
        """

        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": (
                self.temperature if temperature is None else temperature
            ),
            "stream": False,
        }

        # Gemma 4 reasons before answering, and those tokens dominate latency
        # (a RAG answer can spend thousands of them before the first visible
        # word). "default" leaves the model's own behaviour alone.
        if self.reasoning_effort and self.reasoning_effort != "default":
            payload["reasoning_effort"] = self.reasoning_effort

        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": json_schema,
                },
            }

        try:
            if on_token is not None:
                content = self._stream(payload, on_token)
            else:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
        except requests.RequestException as exc:
            raise LLMError(f"LM Studio request failed: {exc}") from exc
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                f"LM Studio returned a malformed response: {exc}"
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise LLMError("LM Studio returned an empty completion.")

        return content.strip()

    def _stream(
        self,
        payload: dict[str, Any],
        on_token: Callable[[str], None],
    ) -> str:
        """Server-sent events: ``data: {chunk}`` lines, then ``data: [DONE]``.

        Lines are decoded as UTF-8 here rather than by requests: LM Studio
        sends ``text/event-stream`` with no charset, which requests would read
        as ISO-8859-1 and turn the bulletin's curly quotes into mojibake.
        Only ``delta.content`` is kept -- ``reasoning_content`` never reaches
        the caller, streamed or not.
        """
        payload = dict(payload, stream=True)
        pieces: list[str] = []

        with requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=self.timeout,
            stream=True,
        ) as response:
            response.raise_for_status()

            for raw in response.iter_lines():
                line = raw.decode("utf-8").strip() if raw else ""

                if not line.startswith("data:"):
                    continue

                data = line[len("data:"):].strip()

                if data == "[DONE]":
                    break

                chunk = json.loads(data)
                piece = (chunk["choices"][0].get("delta") or {}).get("content")

                if piece:
                    pieces.append(piece)
                    on_token(piece)

        return "".join(pieces)

    def health_check(self) -> bool:
        """Check whether the LM Studio server is reachable."""

        try:
            response = requests.get(f"{self.base_url}/models", timeout=10)
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False
