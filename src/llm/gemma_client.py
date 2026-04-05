"""
Ollama API client for Gemma 4.

Ollama runs LLMs locally on your machine. It exposes an HTTP API that
mimics the OpenAI format — so the same code that talks to GPT-4 can
talk to Gemma 4 running on your laptop.

We use streaming so the UI can show the answer word-by-word as it generates,
rather than waiting for the full response.
"""

import json
import logging
import os
from typing import Generator

import httpx

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:27b")


class GemmaClient:
    """
    Client for Gemma 4 running locally via Ollama.

    Uses the OpenAI-compatible /v1/chat/completions endpoint.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_HOST,
        model: str = OLLAMA_MODEL,
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def health_check(self) -> bool:
        """Verify that Ollama is running and the model is available."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            # Accept partial match (e.g. "gemma4:27b" matches "gemma4")
            model_base = self.model.split(":")[0]
            return any(model_base in m for m in models)
        except Exception as e:
            logger.warning("Ollama health check failed: %s", e)
            return False

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.3,
        stream: bool = False,
    ) -> dict:
        """
        Send a chat completion request to Ollama.

        messages format: [{"role": "user", "content": "..."}, ...]
        Returns: {"content": "...", "tool_calls": [...] or None}
        """
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,  # we handle streaming separately
        }
        if tools:
            payload["tools"] = tools

        try:
            resp = httpx.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            body = resp.json()
            choice = body.get("choices", [{}])[0]
            message = choice.get("message", {})
            return {
                "content": message.get("content", ""),
                "tool_calls": message.get("tool_calls"),
                "finish_reason": choice.get("finish_reason", ""),
            }
        except httpx.ConnectError:
            raise ConnectionError(
                "Cannot connect to Ollama. Is it running? Start with: ollama serve"
            )
        except httpx.TimeoutException:
            raise TimeoutError(
                f"Ollama timed out after {self.timeout}s. "
                "Try a smaller model or increase timeout."
            )

    def stream_chat(
        self, messages: list[dict], temperature: float = 0.3
    ) -> Generator[str, None, None]:
        """
        Streaming version of chat. Yields text chunks as they arrive.
        Use this in the Streamlit UI for a "typing" effect.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            ) as resp:
                for line in resp.iter_lines():
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    try:
                        chunk = json.loads(line)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        text = delta.get("content", "")
                        if text:
                            yield text
                    except json.JSONDecodeError:
                        continue
        except httpx.ConnectError:
            yield "\n\n[Error: Ollama is not running. Start with: ollama serve]\n"

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Simple one-shot generation (no chat history)."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        result = self.chat(messages)
        return result.get("content", "")
