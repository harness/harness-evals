"""ToolCorrectness metric — compares called tools against expected tool sequence."""

from __future__ import annotations

from collections import Counter

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score


class ToolCorrectnessMetric(BaseMetric):
    """Compare ``eval_case.tool_calls`` names against ``eval_case.expected_tools``.

    Supports two modes:

    - **exact** (default): Order and count must match exactly.
      Score is the fraction of positions where the called tool matches
      the expected tool. Uses ``max(len(called), len(expected))`` as the
      denominator, so extra tools reduce the score proportionally while
      missing tools reduce it more heavily.
    - **subset**: All expected tools must appear in ``tool_calls``,
      order-independent but count-sensitive (duplicates in expected
      require matching duplicates in called). Score is the fraction of
      expected tool occurrences found.
    """

    def __init__(
        self,
        mode: str = "exact",
        threshold: float = 1.0,
        **kwargs: object,
    ) -> None:
        super().__init__(name="tool_correctness", dimension=Dimension.TRAJECTORY, threshold=threshold, **kwargs)
        if mode not in ("exact", "subset"):
            raise ValueError(f"mode must be 'exact' or 'subset', got '{mode}'")
        self.mode = mode

    def measure(self, eval_case: EvalCase) -> Score:
        expected_tools = eval_case.expected_tools

        if expected_tools is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="Cannot evaluate tool correctness — 'expected_tools' not provided on the eval case",
            )

        if eval_case.tool_calls is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="Cannot evaluate tool correctness — 'tool_calls' not provided on the eval case",
            )

        tools_called = [tc.name for tc in eval_case.tool_calls]

        if not expected_tools:
            value = 1.0 if not tools_called else 0.0
            reason = "Perfect match — no tools were expected and none were called" if value == 1.0 else "Agent called tools but none were expected"
            return Score(
                name=self.name,
                value=value,
                threshold=self.threshold,
                reason=reason,
            )

        if self.mode == "exact":
            return self._measure_exact(tools_called, expected_tools)
        return self._measure_subset(tools_called, expected_tools)

    def _measure_exact(self, called: list[str], expected: list[str]) -> Score:
        max_len = max(len(called), len(expected))
        matches = sum(1 for a, b in zip(called, expected, strict=False) if a == b)
        value = matches / max_len

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=f"{matches} of {max_len} tool calls match the expected sequence ({matches}/{max_len}, exact mode)",
            metadata={
                "mode": "exact",
                "tools_called": called,
                "expected_tools": expected,
                "matches": matches,
            },
        )

    def _measure_subset(self, called: list[str], expected: list[str]) -> Score:
        called_counts = Counter(called)
        expected_counts = Counter(expected)
        found = sum(min(called_counts[tool], count) for tool, count in expected_counts.items())
        total_expected = len(expected)
        value = found / total_expected

        missing: list[str] = []
        for tool, count in expected_counts.items():
            deficit = count - called_counts.get(tool, 0)
            if deficit > 0:
                missing.extend([tool] * deficit)

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=f"{found} of {total_expected} expected tools were called ({found}/{total_expected}, subset mode)",
            metadata={
                "mode": "subset",
                "tools_called": called,
                "expected_tools": expected,
                "found": found,
                "missing": missing,
            },
        )
