"""ToolCorrectness metric — compares called tools against expected tool sequence."""

from __future__ import annotations

from collections import Counter

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score


class ToolCorrectnessMetric(BaseMetric):
    """Compare ``metadata["tools_called"]`` against ``metadata["expected_tools"]``.

    Supports two modes:

    - **exact** (default): Order and count must match exactly.
      Score is the fraction of positions where the called tool matches
      the expected tool. Uses ``max(len(called), len(expected))`` as the
      denominator, so extra tools reduce the score proportionally while
      missing tools reduce it more heavily.
    - **subset**: All expected tools must appear in ``tools_called``,
      order-independent but count-sensitive (duplicates in expected
      require matching duplicates in called). Score is the fraction of
      expected tool occurrences found.

    Both lists should be ``list[str]`` of tool/function names.
    """

    def __init__(
        self,
        mode: str = "exact",
        threshold: float = 1.0,
        **kwargs: object,
    ) -> None:
        super().__init__(name="tool_correctness", threshold=threshold, **kwargs)
        if mode not in ("exact", "subset"):
            raise ValueError(f"mode must be 'exact' or 'subset', got '{mode}'")
        self.mode = mode

    def measure(self, eval_case: EvalCase) -> Score:
        metadata = eval_case.metadata or {}
        tools_called: list[str] | None = metadata.get("tools_called")
        expected_tools: list[str] | None = metadata.get("expected_tools")

        if expected_tools is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="metadata['expected_tools'] not provided",
            )

        if tools_called is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="metadata['tools_called'] not provided",
            )

        if not expected_tools:
            value = 1.0 if not tools_called else 0.0
            reason = "No tools expected" if value == 1.0 else "Tools called but none expected"
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
            reason=f"{matches}/{max_len} tools match (exact mode)",
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
            reason=f"{found}/{total_expected} expected tools found (subset mode)",
            metadata={
                "mode": "subset",
                "tools_called": called,
                "expected_tools": expected,
                "found": found,
                "missing": missing,
            },
        )
