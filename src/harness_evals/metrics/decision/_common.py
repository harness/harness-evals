"""Shared helpers for decision-primitive metrics (Choice/Score/Noul).

Not a public import surface — internal to ``metrics/decision/`` (and reused by
``metrics/composite/decision_composite.py``, which batches the same question
types across multiple sub-checks).
"""

from __future__ import annotations

from typing import Any, Literal

from harness_evals.core.eval_case import EvalCase
from harness_evals.decision.types import Answer
from harness_evals.utils.path import extract_path

_MISSING = object()
Mode = Literal["correctness", "confidence"]


def resolve_state(eval_case: EvalCase, state_field: str) -> Any:
    """Extract state from the eval case, or ``_MISSING`` if the path resolves to nothing."""
    value = extract_path(eval_case.to_dict(), state_field)
    return _MISSING if value is None else value


def resolve_mode(mode: Mode | None, expected: Any) -> Mode:
    """Resolve the effective mode: explicit ``mode`` wins, else infer from ``expected``.

    ``mode="correctness"`` with no ``expected`` is a hard error — an author who
    forces correctness mode intends every case in the run to carry ``expected``.
    """
    if mode == "correctness" and expected is None:
        raise ValueError("mode='correctness' requires an expected value to be set")
    if mode is not None:
        return mode
    return "correctness" if expected is not None else "confidence"


def _clamp(value: float) -> float:
    """Clamp a provider-reported value to [0, 1].

    Providers are expected to return calibrated probabilities in range, but a
    slightly-off response (e.g. floating-point overshoot) shouldn't blow up an
    eval run — clamp defensively rather than let ``Score.__post_init__`` raise.
    """
    return max(0.0, min(1.0, value))


def score_choice(answer: Answer, mode: Mode, expected: Any) -> tuple[float, dict[str, Any]]:
    """Score a ``ChoiceQuestion`` answer. Shared by ``ChoiceMetric`` and ``DecisionCompositeMetric``."""
    metadata: dict[str, Any] = {
        "choice": answer.choice,
        "confidence": answer.confidence,
        "probabilities": answer.probabilities,
        "mode": mode,
    }
    value = 1.0 if answer.choice == expected else 0.0 if mode == "correctness" else _clamp(answer.confidence)
    return value, metadata


def score_rubric(answer: Answer, mode: Mode, expected: Any, num_levels: int) -> tuple[float, dict[str, Any]]:
    """Score a ``ScoreQuestion`` answer. Shared by ``ScoreMetric`` and ``DecisionCompositeMetric``."""
    metadata: dict[str, Any] = {
        "score": answer.score,
        "confidence": answer.confidence,
        "legend": answer.legend,
        "probabilities": answer.probabilities,
        "mode": mode,
    }
    if mode == "correctness":
        if not isinstance(expected, int | float) or isinstance(expected, bool):
            raise ValueError(f"expected must be a numeric rubric level (0-indexed), got {expected!r}")
        if not (0 <= expected <= num_levels - 1):
            raise ValueError(
                f"expected level {expected} is out of range for {num_levels} criteria (valid range: 0-{num_levels - 1})"
            )
        value = 1.0 - abs(level_norm(answer.score, num_levels) - level_norm(expected, num_levels))
    else:
        value = answer.confidence
    return _clamp(value), metadata


def score_noul(answer: Answer, mode: Mode, expected: Any, invert: bool = False) -> tuple[float, dict[str, Any]]:
    """Score a ``NoulQuestion`` answer. Shared by ``NoulMetric`` and ``DecisionCompositeMetric``."""
    noul = _clamp(answer.noul)
    metadata: dict[str, Any] = {"noul": noul, "mode": mode}
    if mode == "correctness":
        expected_bool = parse_bool_expected(expected)
        value = 1.0 if (noul >= 0.5) == expected_bool else 0.0
    else:
        value = noul if not invert else 1.0 - noul
    return value, metadata


def parse_bool_expected(expected: Any) -> bool:
    """Parse a Noul ``expected`` value, rejecting ambiguous Python truthiness.

    Accepts ``True``/``False``, ``1``/``0``, and ``"true"``/``"false"``
    (case-insensitive). Anything else — including ``"yes"`` or ``2`` — is a
    validation error rather than a silent ``bool()`` cast.
    """
    if isinstance(expected, bool):
        return expected
    if isinstance(expected, int) and expected in (0, 1):
        return bool(expected)
    if isinstance(expected, str) and expected.strip().lower() in ("true", "false"):
        return expected.strip().lower() == "true"
    raise ValueError(f"Noul expected must be a bool, 0/1, or 'true'/'false' (case-insensitive); got {expected!r}")


def level_norm(level: float, num_levels: int) -> float:
    """Normalize a score level (0-indexed) to [0, 1] given the rubric's level count."""
    if num_levels <= 1:
        return 0.0
    return level / (num_levels - 1)
