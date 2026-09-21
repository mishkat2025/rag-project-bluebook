from typing import Any

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
    ):
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self.temperature = (
            settings.llm_temperature if temperature is None else temperature
        )
        self.timeout = timeout or settings.llm_timeout

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """Return the model's reply. Raises LLMError on any failure.

        If json_schema is given, the server is asked to constrain output to it.
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

    def health_check(self) -> bool:
        """Check whether the LM Studio server is reachable."""

        try:
            response = requests.get(f"{self.base_url}/models", timeout=10)
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False
