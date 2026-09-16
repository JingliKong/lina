"""Base provider protocol for LLM backends."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools.base import Tool

from messages import Message, ToolCall


@dataclass
class Usage:
    """Token usage from a response."""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class StreamEvent:
    """An event from a streaming response."""
    text: str = ""
    tool_calls: list[ToolCall] | None = None
    stop_reason: str | None = None
    tool_use_started: bool = False  # Signal that tool use is beginning
    usage: Usage | None = None  # Token usage (typically at end of response)


class Provider(ABC):
    """Abstract base class for LLM providers."""

    name: str  # Provider identifier, e.g., "bedrock", "google", "ollama"

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        tools: list["Tool"],
        system: str = "",
    ) -> AsyncIterator[StreamEvent]:
        """Stream a response from the LLM.

        Args:
            messages: Conversation history
            tools: Available tools the LLM can call
            system: System prompt

        Yields:
            StreamEvent objects with text chunks, tool calls, and stop reason
        """
        pass
