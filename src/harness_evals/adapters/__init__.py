"""Input adapters — convert external trace/span shapes into EvalCase."""

from harness_evals.adapters.trace import (
    SpanType,
    classify_span,
    extract_span_subtree,
    normalize_span,
    spans_to_eval_case,
    spans_to_eval_case_for_span,
)
from harness_evals.applicability import TRACE_INPUTS

__all__ = [
    "TRACE_INPUTS",
    "SpanType",
    "classify_span",
    "extract_span_subtree",
    "normalize_span",
    "spans_to_eval_case",
    "spans_to_eval_case_for_span",
]
