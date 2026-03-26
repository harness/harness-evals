from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from harness_evals.core.golden import Golden
from harness_evals.core.types import Message, ToolCall


@dataclass
class EvalCase:
    """A Golden enriched with the agent's output and runtime metadata.

    Created at evaluation time via ``from_golden()`` or loaded from a
    pre-captured file via ``from_dict()``. This is what metrics receive.
    """

    input: str | dict | list
    output: str | dict | list
    expected: str | dict | list | None = None
    context: list[str] | None = None

    latency_ms: float | None = None
    token_count: int | None = None
    cost_usd: float | None = None
    retry_count: int | None = None
    confidence: float | None = None

    messages: list[Message] | None = field(default=None)
    tool_calls: list[ToolCall] | None = field(default=None)
    expected_tools: list[str] | None = field(default=None)

    tags: dict[str, str] | None = field(default=None)
    metadata: dict[str, Any] | None = field(default=None)
    runs: list[EvalCase] | None = field(default=None)

    def meta(self, key: str, default: Any = None) -> Any:
        """Safely retrieve a metadata value without ``(self.metadata or {})`` boilerplate."""
        return (self.metadata or {}).get(key, default)

    def output_as_str(self) -> str:
        """Return output as a string, JSON-encoding dicts/lists."""
        if isinstance(self.output, str):
            return self.output
        return json.dumps(self.output, ensure_ascii=False)

    def output_as_dict(self) -> dict:
        """Return output as a dict, parsing JSON strings.

        Raises ``TypeError`` if output is a list or parses to a non-dict.
        Raises ``json.JSONDecodeError`` if output is an unparseable string.
        """
        if isinstance(self.output, dict):
            return self.output
        if isinstance(self.output, str):
            parsed = json.loads(self.output)
            if not isinstance(parsed, dict):
                raise TypeError(f"output parsed to {type(parsed).__name__}, expected dict")
            return parsed
        raise TypeError(f"output is {type(self.output).__name__}, expected str or dict")

    def expected_as_str(self) -> str:
        """Return expected as a string, JSON-encoding dicts/lists.

        Raises ``TypeError`` if expected is ``None``.
        """
        if self.expected is None:
            raise TypeError("expected is None")
        if isinstance(self.expected, str):
            return self.expected
        return json.dumps(self.expected, ensure_ascii=False)

    def expected_as_dict(self) -> dict:
        """Return expected as a dict, parsing JSON strings.

        Raises ``TypeError`` if expected is ``None``, a list, or parses to a non-dict.
        Raises ``json.JSONDecodeError`` if expected is an unparseable string.
        """
        if self.expected is None:
            raise TypeError("expected is None")
        if isinstance(self.expected, dict):
            return self.expected
        if isinstance(self.expected, str):
            parsed = json.loads(self.expected)
            if not isinstance(parsed, dict):
                raise TypeError(f"expected parsed to {type(parsed).__name__}, expected dict")
            return parsed
        raise TypeError(f"expected is {type(self.expected).__name__}, expected str or dict")

    def to_dict(self) -> dict:
        result = {}
        for k, v in asdict(self).items():
            if v is not None:
                result[k] = v
        return result

    @classmethod
    def from_dict(cls, data: dict) -> EvalCase:
        """Create an EvalCase from a dict with backward-compat aliases.

        Accepts ``actual_output`` for ``output``, ``expected_output`` for
        ``expected``, and ``token_usage`` for ``token_count``.
        Deserializes ``messages`` and ``tool_calls`` from plain dicts.
        """
        mapped = dict(data)
        if "actual_output" in mapped and "output" not in mapped:
            mapped["output"] = mapped.pop("actual_output")
        if "expected_output" in mapped and "expected" not in mapped:
            mapped["expected"] = mapped.pop("expected_output")
        if "token_usage" in mapped and "token_count" not in mapped:
            mapped["token_count"] = mapped.pop("token_usage")

        if "messages" in mapped and mapped["messages"] is not None:
            mapped["messages"] = [m if isinstance(m, Message) else Message.from_dict(m) for m in mapped["messages"]]
        if "tool_calls" in mapped and mapped["tool_calls"] is not None:
            mapped["tool_calls"] = [
                tc if isinstance(tc, ToolCall) else ToolCall.from_dict(tc) for tc in mapped["tool_calls"]
            ]

        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in mapped.items() if k in known})

    @classmethod
    def from_golden(
        cls,
        golden: Golden,
        output: str | dict | list,
        *,
        metadata_extra: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> EvalCase:
        """Create an EvalCase by combining a Golden with agent output.

        ``metadata_extra`` is merged on top of ``golden.metadata``, so
        runtime keys (model name, conversation ID, etc.) win on conflicts.
        """
        if golden.metadata or metadata_extra:
            merged_meta = {**(golden.metadata or {}), **(metadata_extra or {})}
        else:
            merged_meta = None
        return cls(
            input=golden.input,
            output=output,
            expected=golden.expected,
            context=golden.context,
            expected_tools=golden.expected_tools,
            tags=golden.tags,
            metadata=merged_meta,
            **kwargs,
        )
