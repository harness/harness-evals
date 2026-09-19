"""Shared helpers for decision-primitive metrics (Choice/Score/Noul).

Not a public import surface — internal to ``metrics/decision/``.
"""

from __future__ import annotations

from typing import Any, Literal

from harness_evals.core.eval_case import EvalCase
from harness_evals.utils.path import extract_path

_MISSING = object()
Mode = Literal["correctness", "confidence"]


def resolve_state(eval_case: EvalCase, state_field: str) -> Any:
    """Extract state from the eval case, or ``_MISSING`` if the path resolves to nothing."""
    value = extract_path(eval_case.to_dict(), state_field)
    return _MISSING if value is None else value


def resolve_mode(mode: Mode | None, eval_case: EvalCase) -> Mode:
    """Resolve the effective mode: explicit ``mode`` wins, else infer from ``expected``.

    ``mode="correctness"`` with no ``expected`` is a hard error — an author who
    forces correctness mode intends every case in the run to carry ``expected``.
    """
    if mode == "correctness" and eval_case.expected is None:
        raise ValueError("mode='correctness' requires eval_case.expected to be set")
    if mode is not None:
        return mode
    return "correctness" if eval_case.expected is not None else "confidence"


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
