from typing import Any

import requests

from src.config.settings import settings


class OllamaClient:
    """Client for the local Ollama text-generation API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout: int | None = None,
    ):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model or "qwen2.5:3b"
        self.temperature = (
            settings.ollama_temperature
            if temperature is None
            else temperature
        )
        self.timeout = timeout or settings.ollama_timeout

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
    ) -> str:
        """Generate text using the local Ollama model."""

        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": (
                    self.temperature
                    if temperature is None
                    else temperature
                )
            },
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        generated_text = data.get("response")

        if generated_text is None:
            raise RuntimeError(
                "Ollama response did not contain a 'response' field."
            )

        return generated_text.strip()

    def health_check(self) -> bool:
        """Check whether the Ollama server is reachable."""

        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10,
            )
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False