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
        # Accept common field aliases so tool-call dicts from popular providers
        # work without pre-processing:
        #   OpenAI/LangChain: ``arguments`` (the tool input), ``result`` (output)
        #   Anthropic:        ``input`` (native match — no alias needed)
        # Explicit ``input``/``output`` keys always take precedence.
        input_val = data.get("input")
        if input_val is None:
            input_val = data.get("arguments")
        output_val = data.get("output")
        if output_val is None:
            output_val = data.get("result")
        return cls(
            name=data["name"],
            input=input_val,
            output=output_val,
        )


@dataclass
class Message:
    """A single message in a conversation.

    Maps to OpenAI chat messages, Langfuse generations, and OTEL LLM spans.
    """

    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = field(default=None)
    latency_ms: float | None = None
    token_count: int | None = None
    cost_usd: float | None = None

    # RAG turn data — the chunks retrieved to answer this turn and the
    # expected answer for it. Set on the assistant message of a turn so
    # turn-level RAG metrics can score each turn's retrieval/answer pair.
    retrieval_context: list[str] | None = field(default=None)
    expected: str | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls is not None:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.latency_ms is not None:
            d["latency_ms"] = self.latency_ms
        if self.token_count is not None:
            d["token_count"] = self.token_count
        if self.cost_usd is not None:
            d["cost_usd"] = self.cost_usd
        if self.retrieval_context is not None:
            d["retrieval_context"] = self.retrieval_context
        if self.expected is not None:
            d["expected"] = self.expected
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
            latency_ms=data.get("latency_ms"),
            token_count=data.get("token_count"),
            cost_usd=data.get("cost_usd"),
            retrieval_context=data.get("retrieval_context"),
            expected=data.get("expected"),
        )
