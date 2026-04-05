"""
Claude API client — drop-in replacement for GemmaClient when Ollama isn't available.

Uses claude-haiku (fast + cheap) by default. Set CLAUDE_MODEL in .env to override.
"""

import os
import logging
from typing import Generator

logger = logging.getLogger(__name__)

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")


class ClaudeClient:
    def __init__(self, model: str = CLAUDE_MODEL):
        import anthropic
        self.model = model
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    def health_check(self) -> bool:
        try:
            self._client.messages.create(
                model=self.model,
                max_tokens=5,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True
        except Exception as e:
            logger.warning("Claude health check failed: %s", e)
            return False

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.3,
        **kwargs,
    ) -> dict:
        """Same interface as GemmaClient.chat()"""
        # Separate system message from the rest
        system = ""
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_messages.append({"role": m["role"], "content": m["content"]})

        kwargs_extra = {}
        if tools:
            # Convert OpenAI-style tool defs to Anthropic format
            kwargs_extra["tools"] = _convert_tools(tools)

        response = self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=temperature,
            system=system or "You are a helpful assistant.",
            messages=chat_messages,
            **kwargs_extra,
        )

        # Extract text content and any tool use
        content_text = ""
        tool_calls = None

        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append({
                    "id": block.id,
                    "function": {
                        "name": block.name,
                        "arguments": block.input,
                    },
                })

        return {
            "content": content_text,
            "tool_calls": tool_calls,
            "finish_reason": response.stop_reason,
        }

    def generate(self, prompt: str, system: str | None = None) -> str:
        result = self.chat(
            messages=[{"role": "user", "content": prompt}],
            **({"system": system} if system else {}),
        )
        return result.get("content", "")

    def stream_chat(self, messages: list[dict], temperature: float = 0.3) -> Generator[str, None, None]:
        system = ""
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_messages.append({"role": m["role"], "content": m["content"]})

        with self._client.messages.stream(
            model=self.model,
            max_tokens=4096,
            temperature=temperature,
            system=system or "You are a helpful assistant.",
            messages=chat_messages,
        ) as stream:
            for text in stream.text_stream:
                yield text


def _convert_tools(openai_tools: list[dict]) -> list[dict]:
    """Convert OpenAI-style tool definitions to Anthropic format."""
    anthropic_tools = []
    for t in openai_tools:
        fn = t.get("function", {})
        anthropic_tools.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return anthropic_tools
