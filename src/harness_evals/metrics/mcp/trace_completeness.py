"""MCPTraceCompleteness metric — checks all expected MCP operations were executed."""

from __future__ import annotations

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score


class MCPTraceCompletenessMetric(BaseMetric):
    """Fraction of expected trace entries found in the actual MCP trace.

    Reads ``metadata["mcp_trace"]`` and ``metadata["expected_trace"]``
    — both lists of ``{"tool": str, "input": dict, ...}``.

    Matching: exact tool name + input dict equality.
    Each actual entry can match at most one expected entry.
    Returns 0.0 if either field is missing.
    """

    def __init__(self, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(name="mcp_trace_completeness", threshold=threshold, **kwargs)

    def measure(self, eval_case: EvalCase) -> Score:
        mcp_trace: list[dict] | None = eval_case.meta("mcp_trace")
        expected_trace: list[dict] | None = eval_case.meta("expected_trace")

        if mcp_trace is None or expected_trace is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason='metadata must contain "mcp_trace" and "expected_trace"',
            )

        if not expected_trace:
            return Score(
                name=self.name,
                value=1.0,
                threshold=self.threshold,
                reason="No expected trace entries",
            )

        used = [False] * len(mcp_trace)
        matched_expected = [False] * len(expected_trace)

        for j, expected_entry in enumerate(expected_trace):
            exp_tool = expected_entry.get("tool")
            exp_input = expected_entry.get("input")
            for i, actual_entry in enumerate(mcp_trace):
                if used[i]:
                    continue
                if actual_entry.get("tool") == exp_tool and actual_entry.get("input") == exp_input:
                    used[i] = True
                    matched_expected[j] = True
                    break

        found = sum(matched_expected)
        value = found / len(expected_trace)
        missing = [e for e, m in zip(expected_trace, matched_expected, strict=True) if not m]

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=f"{found}/{len(expected_trace)} expected operations found",
            metadata={
                "found": found,
                "total_expected": len(expected_trace),
                "missing": missing,
            },
        )
