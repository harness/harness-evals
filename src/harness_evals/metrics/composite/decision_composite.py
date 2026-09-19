"""DecisionCompositeMetric — batches decision-primitive sub-checks into one call per shared state."""

from __future__ import annotations

from typing import Any

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.decision.base import BaseDecisionProvider
from harness_evals.decision.types import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion
from harness_evals.metrics.composite._combine import fold_sub_scores
from harness_evals.metrics.decision._common import level_norm, parse_bool_expected
from harness_evals.utils.path import extract_path


def _resolve_sub_mode(explicit_mode: str | None, expected: Any) -> str:
    if explicit_mode == "correctness" and expected is None:
        raise ValueError("mode='correctness' requires an expected value (via expected_field)")
    if explicit_mode is not None:
        return explicit_mode
    return "correctness" if expected is not None else "confidence"


def _score_answer(
    question: Question, answer: Any, mode: str, expected: Any, invert: bool
) -> tuple[float, dict[str, Any]]:
    if isinstance(question, ChoiceQuestion):
        metadata: dict[str, Any] = {
            "choice": answer.choice,
            "confidence": answer.confidence,
            "probabilities": answer.probabilities,
            "mode": mode,
        }
        if mode == "correctness":
            value = 1.0 if answer.choice == expected else 0.0
        else:
            value = max(0.0, min(1.0, answer.confidence))
        return value, metadata

    if isinstance(question, ScoreQuestion):
        num_levels = len(question.criteria)
        metadata = {
            "score": answer.score,
            "confidence": answer.confidence,
            "legend": answer.legend,
            "probabilities": answer.probabilities,
            "mode": mode,
        }
        if mode == "correctness":
            if not isinstance(expected, int | float) or isinstance(expected, bool):
                raise ValueError(f"ScoreQuestion expected must be a numeric rubric level (0-indexed), got {expected!r}")
            if not (0 <= expected <= num_levels - 1):
                raise ValueError(
                    f"ScoreQuestion expected level {expected} is out of range for {num_levels} criteria "
                    f"(valid range: 0-{num_levels - 1})"
                )
            value = 1.0 - abs(level_norm(answer.score, num_levels) - level_norm(expected, num_levels))
        else:
            value = answer.confidence
        return max(0.0, min(1.0, value)), metadata

    if isinstance(question, NoulQuestion):
        noul = max(0.0, min(1.0, answer.noul))
        metadata = {"noul": noul, "mode": mode}
        if mode == "correctness":
            expected_bool = parse_bool_expected(expected)
            value = 1.0 if (noul >= 0.5) == expected_bool else 0.0
        else:
            value = noul if not invert else 1.0 - noul
        return value, metadata

    raise TypeError(f"Unknown question type: {type(question)!r}")


class DecisionCompositeMetric(BaseMetric):
    """Batch decision-primitive sub-checks sharing a ``state_field`` into one provider call per group.

    Each entry in ``sub_scores`` needs ``name``, ``weight``, ``state_field``,
    and a ``question`` (a ``ChoiceQuestion``/``ScoreQuestion``/``NoulQuestion``).
    Optional keys: ``expected_field`` (a path into the eval case used for
    correctness-mode comparison), ``mode`` (explicit override, else inferred
    from whether ``expected_field`` resolves to a value), ``invert`` (Noul
    confidence-mode only), ``skip_when_missing``.

    Sub-checks whose ``state_field`` resolves to ``None`` are skipped (if
    ``skip_when_missing``) or marked ``status="error"`` before any network
    call. Remaining sub-checks are grouped by ``state_field`` and answered
    with one ``provider.a_ask()`` call per group. Unlike the standalone
    ``ChoiceMetric``/``ScoreMetric``/``NoulMetric``, a provider failure marks
    only that group's sub-checks ``status="error"`` (excluded from the
    weighted sum) rather than raising — this metric is a multi-check
    aggregator, matching ``CompositeMetric``'s existing partial-failure
    semantics.
    """

    def __init__(
        self,
        provider: BaseDecisionProvider,
        sub_scores: list[dict[str, Any]],
        threshold: float = 0.85,
        dimension: Dimension = Dimension.CORRECTNESS,
        name: str = "decision_composite",
        **kwargs: object,
    ) -> None:
        super().__init__(name=name, dimension=dimension, threshold=threshold, **kwargs)
        for i, sub in enumerate(sub_scores):
            if "name" not in sub:
                raise ValueError(f"sub_scores[{i}] missing required 'name' key")
            if "question" not in sub:
                raise ValueError(f"sub_scores[{i}] ({sub['name']}) missing required 'question' key")
            if "state_field" not in sub:
                raise ValueError(f"sub_scores[{i}] ({sub['name']}) missing required 'state_field' key")
        self.provider = provider
        self.sub_scores = sub_scores

    def measure(self, eval_case: EvalCase) -> Score:
        return _run_async(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        eval_case_dict = eval_case.to_dict()
        results: dict[str, dict[str, Any]] = {}
        groups: dict[str, list[dict[str, Any]]] = {}

        for sub in self.sub_scores:
            sub_name = sub["name"]
            state_field = sub["state_field"]
            skip_when_missing = sub.get("skip_when_missing", False)
            state = extract_path(eval_case_dict, state_field)

            if state is None:
                if skip_when_missing:
                    results[sub_name] = {
                        "value": None,
                        "status": "skipped",
                        "reason": f"Missing state field '{state_field}'",
                    }
                else:
                    results[sub_name] = {
                        "value": 0.0,
                        "status": "error",
                        "reason": f"Missing state field '{state_field}'",
                    }
                continue

            groups.setdefault(state_field, []).append({**sub, "_state": state})

        for subs in groups.values():
            state = subs[0]["_state"]
            questions = {sub["name"]: sub["question"] for sub in subs}
            try:
                response = await self.provider.a_ask(state, questions)
            except Exception as e:
                for sub in subs:
                    results[sub["name"]] = {"value": 0.0, "status": "error", "reason": f"Provider error: {e}"}
                continue

            for sub in subs:
                sub_name = sub["name"]
                question = sub["question"]
                answer = response.answers[sub_name]
                expected_field = sub.get("expected_field")
                expected = extract_path(eval_case_dict, expected_field) if expected_field else None

                try:
                    mode = _resolve_sub_mode(sub.get("mode"), expected)
                    value, metadata = _score_answer(question, answer, mode, expected, sub.get("invert", False))
                except ValueError as e:
                    results[sub_name] = {"value": 0.0, "status": "error", "reason": str(e)}
                    continue

                metadata.update(
                    {
                        "model": response.model,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                    }
                )
                results[sub_name] = {"value": value, "status": "ok", "metadata": metadata}

        final_score, details = fold_sub_scores(self.sub_scores, results)
        active_weight_sum = sum(
            float(sub.get("weight", 0.0))
            for sub in self.sub_scores
            if results.get(sub["name"], {}).get("status") != "skipped"
        )

        return Score(
            name=self.name,
            value=final_score,
            threshold=self.threshold,
            reason=f"Decision composite aggregated {len(details['sub_scores'])} sub-scores with total active weight {active_weight_sum:g}",
            metadata=details,
        )
