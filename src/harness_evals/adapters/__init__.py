"""Input adapters — convert external trace/span shapes into EvalCase."""

from harness_evals.adapters.trace import (
    SpanType,
    classify_span,
    normalize_span,
    spans_to_eval_case,
    spans_to_eval_case_for_span,
)

__all__ = [
    "SpanType",
    "classify_span",
    "normalize_span",
    "spans_to_eval_case",
    "spans_to_eval_case_for_span",
]
