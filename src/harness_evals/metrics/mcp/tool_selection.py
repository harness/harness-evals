"""ToolSelectionAccuracy metric — Jaccard similarity of tool calls vs expected tools."""

from __future__ import annotations

from collections import Counter

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score


class ToolSelectionAccuracyMetric(BaseMetric):
    """Fraction of tool calls matching expected tools (Jaccard over multisets).

    Reads ``eval_case.tool_calls`` (list of ``ToolCall``) and
    ``eval_case.expected_tools`` (list of tool name strings).

    Score = ``|intersection| / |union|`` of tool name multisets.
    Returns 0.0 if either field is missing.
    """

    def __init__(self, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(name="tool_selection_accuracy", dimension=Dimension.TRAJECTORY, threshold=threshold, **kwargs)

    def measure(self, eval_case: EvalCase) -> Score:
        if eval_case.tool_calls is None or eval_case.expected_tools is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="tool_calls and expected_tools must both be provided on EvalCase",
            )

        if not eval_case.expected_tools and not eval_case.tool_calls:
            return Score(
                name=self.name,
                value=1.0,
                threshold=self.threshold,
                reason="No tools expected and none called",
            )

        actual_names = [tc.name for tc in eval_case.tool_calls]
        actual_counts = Counter(actual_names)
        expected_counts = Counter(eval_case.expected_tools)

        all_tools = set(actual_counts) | set(expected_counts)
        if not all_tools:
            return Score(
                name=self.name,
                value=1.0,
                threshold=self.threshold,
                reason="Both sets empty",
            )

        intersection = sum(min(actual_counts[t], expected_counts[t]) for t in all_tools)
        union = sum(max(actual_counts[t], expected_counts[t]) for t in all_tools)

        value = intersection / union if union > 0 else 0.0

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=f"Jaccard similarity: {intersection}/{union}",
            metadata={
                "actual_tools": actual_names,
                "expected_tools": eval_case.expected_tools,
                "intersection": intersection,
                "union": union,
            },
        )
