"""Core message types for Henri."""

from dataclasses import dataclass, field
from typing import Literal

Role = Literal["user", "assistant", "tool"]


@dataclass
class ToolCall:
    """A request from the LLM to execute a tool."""
    id: str
    name: str
    args: dict


@dataclass
class ToolResult:
    """The result of executing a tool."""
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    """A message in the conversation."""
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: list[ToolCall] | None = None) -> "Message":
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool_result(cls, results: list[ToolResult]) -> "Message":
        return cls(role="tool", tool_results=results)
