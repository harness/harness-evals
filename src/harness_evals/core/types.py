"""Core types for structured agent trace data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A single tool/function call made by an agent.

    Maps to OpenAI function calls, Anthropic tool_use blocks, MCP tool
    invocations, and Langfuse/OTEL tool spans.
    """

    name: str
    input: dict | None = None
    output: str | dict | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict) -> ToolCall:
        return cls(
            name=data["name"],
            input=data.get("input"),
            output=data.get("output"),
        )


@dataclass
class Message:
    """A single message in a conversation.

    Maps to OpenAI chat messages, Langfuse generations, and OTEL LLM spans.
    """

    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = field(default=None)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls is not None:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Message:
        tool_calls = None
        if "tool_calls" in data and data["tool_calls"] is not None:
            tool_calls = [ToolCall.from_dict(tc) for tc in data["tool_calls"]]
        return cls(
            role=data["role"],
            content=data.get("content"),
            tool_calls=tool_calls,
        )
