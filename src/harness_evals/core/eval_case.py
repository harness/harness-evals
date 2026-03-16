from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from harness_evals.core.golden import Golden


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

    tags: dict[str, str] | None = field(default=None)
    metadata: dict[str, Any] | None = field(default=None)
    runs: list[EvalCase] | None = field(default=None)

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
        """
        mapped = dict(data)
        if "actual_output" in mapped and "output" not in mapped:
            mapped["output"] = mapped.pop("actual_output")
        if "expected_output" in mapped and "expected" not in mapped:
            mapped["expected"] = mapped.pop("expected_output")
        if "token_usage" in mapped and "token_count" not in mapped:
            mapped["token_count"] = mapped.pop("token_usage")

        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in mapped.items() if k in known})

    @classmethod
    def from_golden(cls, golden: Golden, output: str | dict | list, **kwargs: Any) -> EvalCase:
        """Create an EvalCase by combining a Golden with agent output."""
        return cls(
            input=golden.input,
            output=output,
            expected=golden.expected,
            context=golden.context,
            tags=golden.tags,
            metadata=golden.metadata,
            **kwargs,
        )
