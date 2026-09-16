"""Ollama provider for local models."""

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import ollama

if TYPE_CHECKING:
    from tools.base import Tool

from config import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_HOST
from messages import Message, ToolCall
from providers.base import Provider, StreamEvent, Usage


class OllamaProvider(Provider):
    """Ollama provider for local LLM inference."""

    name = "ollama"

    def __init__(
        self,
        model_id: str = DEFAULT_OLLAMA_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
    ):
        self.model_id = model_id
        self.client = ollama.AsyncClient(host=host)

    def _tools_to_ollama(self, tools: list["Tool"]) -> list[dict]:
        """Convert tools to Ollama's format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def _messages_to_ollama(
        self, messages: list[Message], system: str
    ) -> list[dict]:
        """Convert messages to Ollama's format."""
        ollama_messages = []

        if system:
            ollama_messages.append({"role": "system", "content": system})

        for msg in messages:
            if msg.content:
                ollama_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

            for tc in msg.tool_calls:
                ollama_messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": tc.name,
                            "arguments": tc.args,
                        },
                    }],
                })

            for tr in msg.tool_results:
                ollama_messages.append({
                    "role": "tool",
                    "content": tr.content,
                })

        return ollama_messages

    async def stream(
        self,
        messages: list[Message],
        tools: list["Tool"],
        system: str = "",
    ) -> AsyncIterator[StreamEvent]:
        """Stream a response from Ollama."""
        ollama_messages = self._messages_to_ollama(messages, system)

        tool_calls = []
        usage = None

        async for chunk in await self.client.chat(
            model=self.model_id,
            messages=ollama_messages,
            tools=self._tools_to_ollama(tools) if tools else None,
            stream=True,
        ):
            message = chunk.get("message", {})

            # Handle text content
            if content := message.get("content"):
                yield StreamEvent(text=content)

            # Handle tool calls
            if tc_list := message.get("tool_calls"):
                if not tool_calls:
                    yield StreamEvent(tool_use_started=True)
                for tc in tc_list:
                    fn = tc.get("function", {})
                    # Ollama doesn't provide tool call IDs, using name as ID
                    tool_calls.append(ToolCall(
                        id=fn.get("name", "unknown"),
                        name=fn.get("name", "unknown"),
                        args=fn.get("arguments", {}),
                    ))

            # Check if done
            if chunk.get("done"):
                # Ollama reports tokens in final chunk
                usage = Usage(
                    input_tokens=chunk.get("prompt_eval_count", 0),
                    output_tokens=chunk.get("eval_count", 0),
                )
                stop_reason = "tool_use" if tool_calls else "end_turn"
                yield StreamEvent(
                    tool_calls=tool_calls if tool_calls else None,
                    stop_reason=stop_reason,
                    usage=usage,
                )
