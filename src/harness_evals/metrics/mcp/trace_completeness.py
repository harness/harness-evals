"""MCPTraceCompleteness metric — checks all expected MCP operations were executed."""

from __future__ import annotations

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.core.types import ToolCall


class MCPTraceCompletenessMetric(BaseMetric):
    """Fraction of expected trace entries found in the actual tool calls.

    Reads ``eval_case.tool_calls`` for the actual calls. Takes
    ``expected_trace`` — a list of ``ToolCall`` — as a constructor parameter.

    Matching: exact tool name + input dict equality.
    Each actual entry can match at most one expected entry.
    Returns 0.0 if ``tool_calls`` is missing.
    """

    def __init__(
        self,
        expected_trace: list[ToolCall],
        threshold: float = 0.7,
        **kwargs: object,
    ) -> None:
        super().__init__(name="mcp_trace_completeness", dimension=Dimension.TRAJECTORY, threshold=threshold, **kwargs)
        self.expected_trace = expected_trace

    def measure(self, eval_case: EvalCase) -> Score:
        if eval_case.tool_calls is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="tool_calls not provided on EvalCase",
            )

        if not self.expected_trace:
            return Score(
                name=self.name,
                value=1.0,
                threshold=self.threshold,
                reason="No expected trace entries",
            )

        used = [False] * len(eval_case.tool_calls)
        matched_expected = [False] * len(self.expected_trace)

        for j, expected_entry in enumerate(self.expected_trace):
            for i, actual_tc in enumerate(eval_case.tool_calls):
                if used[i]:
                    continue
                if actual_tc.name == expected_entry.name and actual_tc.input == expected_entry.input:
                    used[i] = True
                    matched_expected[j] = True
                    break

        found = sum(matched_expected)
        value = found / len(self.expected_trace)
        missing = [e.to_dict() for e, m in zip(self.expected_trace, matched_expected, strict=True) if not m]

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=f"{found}/{len(self.expected_trace)} expected operations found",
            metadata={
                "found": found,
                "total_expected": len(self.expected_trace),
                "missing": missing,
            },
        )
