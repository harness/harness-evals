"""ToolArgumentMatchMetric — deterministic comparison of tool-call arguments."""

from __future__ import annotations

from typing import Any

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.core.types import ToolCall


class ToolArgumentMatchMetric(BaseMetric):
    """Deterministic comparison of tool-call arguments against expectations.

    Reads ``eval_case.tool_calls`` (actual) and ``eval_case.expected_tool_calls``
    (authored). Names are checked as a precondition for arg credit; this metric
    is intended to be paired with :class:`ToolCorrectnessMetric` for full
    name + argument coverage.

    Pairing modes (mirror :class:`ToolCorrectnessMetric` for a consistent mental
    model):

    - **exact** (default): pair calls by index. Denominator is
      ``max(len(actual), len(expected))`` so extra/missing calls are penalised.
    - **subset**: order-independent multiset pairing on tool ``name``.
      Denominator is ``len(expected)``. Greedy FIFO assignment for v1.

    Argument matching strategies (``arg_match``):

    - **exact** (default): the dicts must be equal after applying ``ignore_keys``
      and treating any expected value equal to ``wildcard_value`` as
      "matches anything".
    - **subset**: every expected key must be present in actual with equal
      value (with the same ignore/wildcard rules); extra actual keys are
      ignored.

    Per-pair argument scoring is binary in v1 (a pair either fully matches
    or it doesn't). The final value is ``matches / denominator``.

    For the LLM-judged sibling, see :class:`ArgumentCorrectnessMetric`.
    """

    def __init__(
        self,
        pair: str = "exact",
        arg_match: str = "exact",
        ignore_keys: set[str] | None = None,
        wildcard_value: object = "*",
        threshold: float = 1.0,
        **kwargs: object,
    ) -> None:
        super().__init__(
            name="tool_argument_match",
            dimension=Dimension.TRAJECTORY,
            threshold=threshold,
            **kwargs,
        )
        if pair not in ("exact", "subset"):
            raise ValueError(f"pair must be 'exact' or 'subset', got '{pair}'")
        if arg_match not in ("exact", "subset"):
            raise ValueError(f"arg_match must be 'exact' or 'subset', got '{arg_match}'")
        self.pair = pair
        self.arg_match = arg_match
        self.ignore_keys: set[str] = set(ignore_keys) if ignore_keys else set()
        self.wildcard_value = wildcard_value

    def measure(self, eval_case: EvalCase) -> Score:
        expected_calls = eval_case.expected_tool_calls

        if expected_calls is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="expected_tool_calls not provided on EvalCase",
            )

        if eval_case.tool_calls is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="tool_calls not provided on EvalCase",
            )

        actual_calls = eval_case.tool_calls

        if not expected_calls:
            value = 1.0 if not actual_calls else 0.0
            reason = "No tool calls expected" if value == 1.0 else "Tool calls made but none expected"
            return Score(
                name=self.name,
                value=value,
                threshold=self.threshold,
                reason=reason,
                metadata={
                    "pair": self.pair,
                    "arg_match": self.arg_match,
                    "n_pairs": 0,
                    "matches": 0,
                    "details": [],
                },
            )

        if self.pair == "exact":
            return self._measure_exact(actual_calls, expected_calls)
        return self._measure_subset(actual_calls, expected_calls)

    # ------------------------------------------------------------------
    # Pairing strategies
    # ------------------------------------------------------------------

    def _measure_exact(self, actual: list[ToolCall], expected: list[ToolCall]) -> Score:
        denominator = max(len(actual), len(expected))
        details: list[dict[str, Any]] = []
        matches = 0
        for i in range(denominator):
            a = actual[i] if i < len(actual) else None
            e = expected[i] if i < len(expected) else None
            detail = self._compare_pair(a, e)
            details.append(detail)
            if detail["matched"]:
                matches += 1

        value = matches / denominator
        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=f"{matches}/{denominator} tool calls match args (exact pairing, {self.arg_match} args)",
            metadata={
                "pair": "exact",
                "arg_match": self.arg_match,
                "n_pairs": denominator,
                "matches": matches,
                "details": details,
            },
        )

    def _measure_subset(self, actual: list[ToolCall], expected: list[ToolCall]) -> Score:
        used: set[int] = set()
        details: list[dict[str, Any]] = []
        matches = 0
        for e in expected:
            chosen_idx: int | None = None
            chosen_detail: dict[str, Any] | None = None
            # Prefer a fully-matching candidate; fall back to any name match.
            fallback_idx: int | None = None
            fallback_detail: dict[str, Any] | None = None
            for j, a in enumerate(actual):
                if j in used or a.name != e.name:
                    continue
                d = self._compare_pair(a, e)
                if d["matched"]:
                    chosen_idx = j
                    chosen_detail = d
                    break
                if fallback_idx is None:
                    fallback_idx = j
                    fallback_detail = d

            if chosen_idx is not None and chosen_detail is not None:
                used.add(chosen_idx)
                details.append(chosen_detail)
                matches += 1
            elif fallback_idx is not None and fallback_detail is not None:
                used.add(fallback_idx)
                details.append(fallback_detail)
            else:
                details.append(self._compare_pair(None, e))

        denominator = len(expected)
        value = matches / denominator
        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=f"{matches}/{denominator} expected tool calls matched (subset pairing, {self.arg_match} args)",
            metadata={
                "pair": "subset",
                "arg_match": self.arg_match,
                "n_pairs": denominator,
                "matches": matches,
                "details": details,
            },
        )

    # ------------------------------------------------------------------
    # Per-pair argument comparison
    # ------------------------------------------------------------------

    def _compare_pair(self, actual: ToolCall | None, expected: ToolCall | None) -> dict[str, Any]:
        """Compare a single (actual, expected) pair. Returns a structured detail dict.

        Either side may be ``None`` (extra actual call or missing expected call).
        ``matched`` is True only when both names match and arg comparison passes.
        """
        if expected is None:
            return {
                "matched": False,
                "reason": "extra_actual_call",
                "actual": actual.to_dict() if actual is not None else None,
                "expected": None,
                "name_match": False,
            }
        if actual is None:
            return {
                "matched": False,
                "reason": "missing_actual_call",
                "actual": None,
                "expected": expected.to_dict(),
                "name_match": False,
            }

        name_match = actual.name == expected.name
        if not name_match:
            return {
                "matched": False,
                "reason": "name_mismatch",
                "actual": actual.to_dict(),
                "expected": expected.to_dict(),
                "name_match": False,
            }

        actual_args = self._strip(actual.input or {})
        expected_args = self._strip(expected.input or {})

        missing_keys: list[str] = []
        extra_keys: list[str] = []
        value_mismatches: list[dict[str, Any]] = []

        for k, v_exp in expected_args.items():
            if k not in actual_args:
                missing_keys.append(k)
                continue
            v_act = actual_args[k]
            if not self._values_equal(v_exp, v_act):
                value_mismatches.append({"key": k, "expected": v_exp, "actual": v_act})

        if self.arg_match == "exact":
            for k in actual_args:
                if k not in expected_args:
                    extra_keys.append(k)

        matched = not missing_keys and not value_mismatches and (self.arg_match == "subset" or not extra_keys)

        return {
            "matched": matched,
            "reason": "ok" if matched else "arg_mismatch",
            "actual": actual.to_dict(),
            "expected": expected.to_dict(),
            "name_match": True,
            "missing_keys": missing_keys,
            "extra_keys": extra_keys,
            "value_mismatches": value_mismatches,
        }

    def _strip(self, args: dict) -> dict:
        if not self.ignore_keys:
            return dict(args)
        return {k: v for k, v in args.items() if k not in self.ignore_keys}

    def _values_equal(self, expected: Any, actual: Any) -> bool:
        if expected == self.wildcard_value:
            return True
        return expected == actual
