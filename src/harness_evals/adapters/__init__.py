"""Input adapters — convert external trace/span shapes into EvalCase."""

from harness_evals.adapters.group import (
    GroupAssembly,
    GroupAuthoredData,
    GroupDiagnostics,
    GroupErrorCode,
    GroupInputError,
    GroupIssue,
    GroupIssueCount,
    GroupTrace,
    GroupUsageCoverage,
    traces_to_group_eval_case,
)
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
    "GroupAssembly",
    "GroupAuthoredData",
    "GroupDiagnostics",
    "GroupErrorCode",
    "GroupInputError",
    "GroupIssue",
    "GroupIssueCount",
    "GroupTrace",
    "GroupUsageCoverage",
    "SpanType",
    "classify_span",
    "extract_span_subtree",
    "normalize_span",
    "spans_to_eval_case",
    "spans_to_eval_case_for_span",
    "traces_to_group_eval_case",
]
