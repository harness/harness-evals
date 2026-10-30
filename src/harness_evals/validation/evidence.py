"""Pure configured-metric eligibility and evidence admission; never scoring.

Group diagnostics must belong to the supplied case. The caller is responsible
for that association: counts and source assertions are not cryptographic proof.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import fields
from typing import Any

import jsonpath_ng.ext
import yaml

from harness_evals.adapters.group import GroupDiagnostics, GroupIssue, GroupIssueCount, GroupUsageCoverage
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import EvaluationMode, MetricInput, MetricInputs, OnlineScope
from harness_evals.core.types import Message, ToolCall
from harness_evals.metrics.decision._common import (
    parse_bool_expected,
    resolve_mode,
    validate_choice_expected,
    validate_rubric_expected,
)
from harness_evals.utils.path import extract_path
from harness_evals.validation.types import (
    ConfiguredMetricContract,
    EvidenceRequirement,
    ExecutionKind,
    ReasonCode,
    ValidationContext,
    ValidationReason,
    ValidationResult,
)


class _ConfigurationError(ValueError):
    pass


def _reason(code: ReasonCode, coordinate: str | None = None, *, warning: bool = False) -> ValidationReason:
    return ValidationReason(code, "warning" if warning else "error", coordinate)


def _errors(reasons: Sequence[ValidationReason]) -> bool:
    return any(reason.severity == "error" for reason in reasons)


def _integer(value: Any) -> bool:
    return type(value) is int and value >= 0


def _number(value: Any, *, integral: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value >= 0 and (not integral or value % 1 == 0)
    except (OverflowError, ValueError):
        return False


def _active_weights(rule: EvidenceRequirement) -> list[int | float] | None:
    weights = rule.parameters.get("active_weights")
    if weights is not None and (
        rule.op != "all"
        or type(weights) is not list
        or len(weights) != len(rule.children)
        or any(type(weight) not in (int, float) or weight < 0 or not math.isfinite(weight) for weight in weights)
    ):
        raise _ConfigurationError
    return weights


def _plain(value: Any, depth: int = 0) -> Any:
    """Snapshot only SDK data/JSON containers, without invoking case methods."""
    if depth > 64:
        raise ValueError
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, (Message, ToolCall)):
        model = Message if isinstance(value, Message) else ToolCall
        return {item.name: _plain(getattr(value, item.name), depth + 1) for item in fields(model)}
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _plain(item, depth + 1) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [_plain(item, depth + 1) for item in value]
    raise ValueError


def _path(data: Any, path: Any) -> Any:
    # extract_path intentionally swallows parser errors; validate syntax first so
    # malformed configuration cannot masquerade as an absent observation.
    if not isinstance(path, str) or len(path) > 1024 or path.count(".") > 64 or ".." in path:
        raise _ConfigurationError
    if path.startswith("$") or any(char in path for char in "[*?"):
        try:
            jsonpath_ng.ext.parse(path)
        except Exception:
            raise _ConfigurationError from None
    try:
        return extract_path(data, path)
    except Exception:
        raise _ConfigurationError from None


def _tool_calls(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(call, dict)
        and isinstance(call.get("name"), str)
        and bool(call["name"])
        and (call.get("input") is None or isinstance(call["input"], dict))
        and (call.get("output") is None or isinstance(call["output"], (str, dict)))
        for call in value
    )


def _messages(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(message, dict)
        and isinstance(message.get("role"), str)
        and bool(message["role"])
        and (message.get("content") is None or isinstance(message["content"], str))
        and (message.get("metadata") is None or isinstance(message["metadata"], dict))
        and (message.get("tool_calls") is None or _tool_calls(message["tool_calls"]))
        and (message.get("expected") is None or isinstance(message["expected"], str))
        and all(
            message.get(name) is None or _number(message[name], integral=name == "token_count")
            for name in ("latency_ms", "token_count", "cost_usd")
        )
        and (message.get("retrieval_context") is None or _predicate(message["retrieval_context"], "string_list"))
        for message in value
    )


def _predicate(value: Any, predicate: str) -> bool:
    if predicate == "present":
        return value is not None
    if predicate == "nonempty":
        return isinstance(value, (str, list, dict, tuple)) and len(value) > 0
    if predicate == "nonempty_text":
        return isinstance(value, str) and bool(value.strip())
    if predicate == "string":
        return isinstance(value, str)
    if predicate == "sequence":
        return isinstance(value, (list, tuple))
    if predicate == "string_list":
        return isinstance(value, list) and all(isinstance(item, str) for item in value)
    if predicate == "tool_calls":
        return _tool_calls(value)
    if predicate == "tool_calls_with_arguments":
        return _tool_calls(value) and all(isinstance(call.get("input"), dict) for call in value)
    if predicate == "messages":
        return _messages(value)
    if predicate == "nonnegative_number":
        return _number(value)
    if predicate == "nonnegative_integer":
        return _number(value, integral=True)
    if predicate == "probability":
        return _number(value) and value <= 1
    raise _ConfigurationError


def _field(rule: EvidenceRequirement, data: dict[str, Any], coordinate: str) -> list[ValidationReason]:
    parameters = rule.parameters
    policy = parameters.get("missing_policy", "reject")
    minimum = parameters.get("min_items", 0)
    if policy not in ("skip", "score_absence", "reject") or not _integer(minimum):
        raise _ConfigurationError
    # Check the predicate even for missing evidence to fail closed on unknown rules.
    _predicate(None, rule.predicate)
    value = _path(data, rule.path)
    if rule.op == "fallback" and value is None:
        fallback = _path(data, parameters.get("default_path"))
        if parameters.get("fallback_string_only", True) is not True:
            raise _ConfigurationError
        if isinstance(fallback, str):
            value = fallback
    if value is None:
        if policy == "score_absence":
            return []
        return [
            _reason(
                ReasonCode.PARTIAL_EVIDENCE if policy == "skip" else ReasonCode.MISSING_EVIDENCE,
                coordinate,
                warning=policy == "skip",
            )
        ]
    if not _predicate(value, rule.predicate):
        return [_reason(ReasonCode.INVALID_EVIDENCE, coordinate)]
    if minimum and (not isinstance(value, (str, list, tuple, dict)) or len(value) < minimum):
        return [_reason(ReasonCode.INSUFFICIENT_EVIDENCE, coordinate)]
    return []


def _turns(rule: EvidenceRequirement, data: dict[str, Any], coordinate: str) -> list[ValidationReason]:
    parameters = rule.parameters
    metric = parameters.get("metric")
    known = {
        "turn_latency",
        "turn_token_cost",
        "turn_faithfulness",
        "turn_contextual_precision",
        "turn_contextual_recall",
        "turn_contextual_relevancy",
    }
    if metric not in known or type(parameters.get("allow_skips", False)) is not bool:
        raise _ConfigurationError
    messages = _path(data, rule.path or "messages")
    if messages is None:
        return [_reason(ReasonCode.MISSING_EVIDENCE, coordinate)]
    if not isinstance(messages, list) or any(
        not isinstance(message, dict) or not isinstance(message.get("role"), str) or not message["role"]
        for message in messages
    ):
        return [_reason(ReasonCode.INVALID_EVIDENCE, coordinate)]
    usable = total = 0
    last_user = None
    for message in messages:
        if message["role"] == "user":
            last_user = message.get("content")
        if message["role"] != "assistant":
            continue
        total += 1
        well_formed = _messages([message])
        if metric in ("turn_latency", "turn_token_cost"):
            if not well_formed:
                return [_reason(ReasonCode.INVALID_EVIDENCE, coordinate)]
            name = "latency_ms" if metric == "turn_latency" else "token_count"
            value = message.get(name)
            if metric == "turn_latency" and value is None and isinstance(message.get("metadata"), dict):
                value = message["metadata"].get(name)
            valid = _number(value, integral=metric == "turn_token_cost")
            if value is not None and not valid:
                return [_reason(ReasonCode.INVALID_EVIDENCE, coordinate)]
        else:
            chunks = message.get("retrieval_context")
            valid = _predicate(chunks, "string_list") and bool(chunks) and any(chunk.strip() for chunk in chunks)
            if metric == "turn_faithfulness":
                valid = valid and _predicate(message.get("content"), "nonempty_text")
            if metric in ("turn_contextual_precision", "turn_contextual_relevancy"):
                valid = valid and _predicate(last_user, "nonempty_text")
            if metric == "turn_contextual_recall":
                valid = valid and _predicate(message.get("expected"), "nonempty_text")
        usable += bool(valid) and well_formed
    if not usable:
        return [_reason(ReasonCode.INSUFFICIENT_EVIDENCE, coordinate)]
    if usable < total:
        # The contract retains allow_skips; validation never changes the scorer's
        # skip-versus-localized-zero policy or invents replacement measurements.
        return [_reason(ReasonCode.PARTIAL_EVIDENCE, coordinate, warning=True)]
    return []


def _references(rule: EvidenceRequirement, data: dict[str, Any], coordinate: str) -> list[ValidationReason]:
    parameters = rule.parameters
    flags = (
        "include_messages_as_reference",
        "include_assistant_tool_inputs_as_reference",
        "include_assistant_tool_results_as_reference",
        "include_scenario_metadata_as_reference",
        "include_sse_events_as_reference",
    )
    if any(type(parameters.get(flag, False)) is not bool for flag in flags):
        raise _ConfigurationError
    names = parameters.get("sse_reference_events", [])
    if not _predicate(names, "string_list"):
        raise _ConfigurationError
    candidates: list[bool] = []
    chunks = data.get("context")
    candidates.append(_predicate(chunks, "string_list") and any(chunk.strip() for chunk in chunks))
    expected = data.get("expected")
    candidates.append(expected is not None and (not isinstance(expected, str) or bool(expected.strip())))
    messages = data.get("messages")
    if any(parameters.get(flag, False) for flag in flags[:3]) and _messages(messages):
        from harness_evals.metrics.safety.hallucination import (
            _assistant_tool_input_references,
            _assistant_tool_result_references,
        )

        typed = [Message.from_dict(message) for message in messages]
        if parameters.get(flags[0]):
            candidates.append(any(message.role != "assistant" and bool(message.content) for message in typed))
        if parameters.get(flags[1]):
            candidates.append(bool(_assistant_tool_input_references(typed)))
        if parameters.get(flags[2]):
            candidates.append(bool(_assistant_tool_result_references(typed)))
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        from harness_evals.metrics.safety.hallucination import _scenario_metadata_references, _sse_event_references

        if parameters.get(flags[3]):
            candidates.append(bool(_scenario_metadata_references(metadata)))
        if parameters.get(flags[4]):
            candidates.append(bool(_sse_event_references(metadata, tuple(names))))
    return [] if any(candidates) else [_reason(ReasonCode.INSUFFICIENT_EVIDENCE, coordinate)]


def _decision(rule: EvidenceRequirement, data: dict[str, Any], coordinate: str) -> list[ValidationReason]:
    parameters = rule.parameters
    question = parameters.get("question_type")
    criteria = parameters.get("criteria")
    if question not in ("choice", "score", "noul") or parameters.get("mode") not in (None, "correctness", "confidence"):
        raise _ConfigurationError
    if question == "choice" and (
        not isinstance(criteria, dict)
        or not criteria
        or any(
            not key or (value is not None and not isinstance(value, (str, dict, list)))
            for key, value in criteria.items()
        )
    ):
        raise _ConfigurationError
    if question == "score" and (
        not isinstance(criteria, list)
        or not 2 <= len(criteria) <= 10
        or any(not isinstance(value, (str, dict, list)) for value in criteria)
    ):
        raise _ConfigurationError
    if (
        question == "noul"
        and criteria is not None
        and (not isinstance(criteria, dict) or any(not isinstance(value, str) for value in criteria.values()))
    ):
        raise _ConfigurationError
    skip = parameters.get("skip_when_missing", False)
    if type(skip) is not bool:
        raise _ConfigurationError
    state = _path(data, parameters.get("state_field", rule.path or "output"))
    if state is None:
        return [_reason(ReasonCode.PARTIAL_EVIDENCE if skip else ReasonCode.MISSING_EVIDENCE, coordinate, warning=skip)]
    # An explicit None expected_field means no comparison for a composite child;
    # it must not silently read the top-level expected label.
    expected_path = parameters.get("expected_field", "expected")
    expected = _path(data, expected_path) if expected_path is not None else None
    try:
        mode = resolve_mode(parameters.get("mode"), expected)
        if question == "choice":
            validate_choice_expected(mode, expected, criteria)
        elif question == "score":
            validate_rubric_expected(mode, expected, len(criteria))
        elif mode == "correctness":
            parse_bool_expected(expected)
    except (ValueError, TypeError, OverflowError):
        return [_reason(ReasonCode.INVALID_EVIDENCE, coordinate)]
    return []


_OPERATORS = frozenset(
    {
        "deep_diff",
        "json_schema",
        "field_exists",
        "equals",
        "contains",
        "contains_all",
        "count_match",
        "list_field_match",
        "unique_ratio",
        "regex_match",
        "range",
        "length",
    }
)


class _NonFiniteValue(ValueError):
    pass


def _reject_constant(value: str) -> Any:
    raise _NonFiniteValue


def _finite(value: Any, depth: int = 0) -> None:
    if depth > 64 or (type(value) is float and not math.isfinite(value)):
        raise _NonFiniteValue
    if isinstance(value, dict):
        for key, item in value.items():
            _finite(key, depth + 1)
            _finite(item, depth + 1)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            _finite(item, depth + 1)


def _parse_structure(value: Any, output_format: str) -> Any:
    if isinstance(value, str):
        try:
            value = (
                json.loads(value, parse_constant=_reject_constant) if output_format == "json" else yaml.safe_load(value)
            )
        except _NonFiniteValue:
            raise
        except Exception:
            # Scorers retain failed parses. Only an actual operand remaining a
            # string is a parse failure; expected strings can be compared as-is.
            pass
    _finite(value)
    return value


def _structural(rule: EvidenceRequirement, data: dict[str, Any], coordinate: str) -> list[ValidationReason]:
    parameters = rule.parameters
    output_format = parameters.get("format", "json")
    if output_format not in ("json", "yaml"):
        raise _ConfigurationError
    for default in ("output", "expected"):
        path = parameters.get(f"{default}_field")
        if path is not None and not isinstance(path, str):
            raise _ConfigurationError
        raw = _path(data, path) if path else data.get(default)
        if raw is None and path and isinstance(data.get(default), str):
            raw = data[default]
        if raw is None:
            return [_reason(ReasonCode.MISSING_EVIDENCE, coordinate)]
        try:
            parsed = _parse_structure(raw, output_format)
        except _NonFiniteValue:
            return [_reason(ReasonCode.INVALID_EVIDENCE, coordinate)]
        if default == "output" and isinstance(raw, str) and isinstance(parsed, str):
            return [_reason(ReasonCode.INVALID_EVIDENCE, coordinate)]
    return []


def _composite(rule: EvidenceRequirement, data: dict[str, Any], coordinate: str) -> list[ValidationReason]:
    parameters = rule.parameters
    subs = parameters.get("sub_scores")
    output_format = parameters.get("output_format", "json")
    if not isinstance(subs, list) or not subs or output_format not in ("json", "yaml"):
        raise _ConfigurationError
    if data.get("output") is None:
        return [_reason(ReasonCode.MISSING_EVIDENCE, coordinate)]
    parsed = dict(data)
    for name in ("output", "expected"):
        raw = data.get(name)
        try:
            value = _parse_structure(raw, output_format)
        except _NonFiniteValue:
            return [_reason(ReasonCode.INVALID_EVIDENCE, coordinate)]
        if name == "output" and isinstance(raw, str) and isinstance(value, str):
            return [_reason(ReasonCode.INVALID_EVIDENCE, coordinate)]
        parsed[name] = value
    reasons: list[ValidationReason] = []
    active_weight = total_weight = 0.0
    for index, sub in enumerate(subs):
        child_id = f"{coordinate}.{index}"
        if not isinstance(sub, dict) or not isinstance(sub.get("check"), dict):
            raise _ConfigurationError
        weight = sub.get("weight", 0)
        if type(weight) not in (int, float) or not _number(weight):
            raise _ConfigurationError
        total_weight += float(weight)
        check = sub["check"]
        operator = check.get("type")
        if operator not in _OPERATORS or type(sub.get("skip_when_missing", False)) is not bool:
            raise _ConfigurationError
        # Literal equality ignores expected_field entirely, including its syntax.
        keys = ("output_field", "expected_field") if operator == "deep_diff" else ("field",)
        if operator in ("count_match", "list_field_match") or (operator == "equals" and "value" not in check):
            keys = ("field", "expected_field")
        if any(not isinstance(check.get(key), str) or not check[key] for key in keys):
            raise _ConfigurationError
        values = {key: _path(parsed, check[key]) for key in keys}
        actual = values.get("field")
        if check.get("field") and sub.get("skip_when_missing") and actual is None:
            reasons.append(_reason(ReasonCode.PARTIAL_EVIDENCE, child_id, warning=True))
            continue
        active_weight += float(weight)
        if operator == "deep_diff" and any(value is None for value in values.values()):
            reasons.append(_reason(ReasonCode.MISSING_EVIDENCE, child_id))
        elif operator == "json_schema":
            if not check.get("schema"):
                raise _ConfigurationError
            if actual is None:
                reasons.append(_reason(ReasonCode.MISSING_EVIDENCE, child_id))
        elif operator in ("list_field_match", "unique_ratio"):
            if not all(isinstance(value, list) for value in values.values()):
                reasons.append(_reason(ReasonCode.INVALID_EVIDENCE, child_id))
            elif operator == "unique_ratio":
                try:
                    set(actual)
                except TypeError:
                    reasons.append(_reason(ReasonCode.INVALID_EVIDENCE, child_id))
        elif operator == "count_match" and any(
            value is not None and not isinstance(value, (str, dict, list, tuple)) for value in values.values()
        ):
            reasons.append(_reason(ReasonCode.INVALID_EVIDENCE, child_id))
        # The other operators deliberately score absent/wrong-type observations
        # as failed quality assertions; presence is not their prerequisite.
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise _ConfigurationError
    if not math.isfinite(active_weight) or active_weight <= 0:
        reasons.append(_reason(ReasonCode.INSUFFICIENT_EVIDENCE, coordinate))
    return reasons


def _evaluate(
    rule: EvidenceRequirement, data: dict[str, Any], coordinate: str = "r", depth: int = 0
) -> list[ValidationReason]:
    if not isinstance(rule, EvidenceRequirement) or depth > 64:
        return [_reason(ReasonCode.INVALID_CONFIGURATION, coordinate)]
    try:
        if rule.op in ("all", "any"):
            decision_children = bool(rule.children) and all(child.op == "decision" for child in rule.children)
            minimum = rule.parameters.get("minimum_active", int(rule.op == "all" and decision_children))
            if not _integer(minimum):
                raise _ConfigurationError
            branches = [
                _evaluate(child, data, f"{coordinate}.{index}", depth + 1) for index, child in enumerate(rule.children)
            ]
            if rule.op == "any":
                if not branches:
                    raise _ConfigurationError
                if any(not _errors(branch) for branch in branches):
                    return [reason for branch in branches for reason in branch if reason.severity == "warning"]
            reasons = [reason for branch in branches for reason in branch]
            if rule.op == "all":
                usable = [
                    not _errors(branch) and not any(reason.code == ReasonCode.PARTIAL_EVIDENCE for reason in branch)
                    for branch in branches
                ]
                weights = _active_weights(rule)
                active_total = (
                    math.fsum(weight for weight, active in zip(weights, usable, strict=True) if active)
                    if weights is not None
                    else None
                )
                if sum(usable) < minimum or (active_total is not None and active_total <= 0):
                    reasons.append(_reason(ReasonCode.INSUFFICIENT_EVIDENCE, coordinate))
            return reasons
        if rule.op in ("field", "fallback"):
            return _field(rule, data, coordinate)
        if rule.op == "assistant_turns":
            return _turns(rule, data, coordinate)
        if rule.op == "hallucination_references":
            return _references(rule, data, coordinate)
        if rule.op == "decision":
            return _decision(rule, data, coordinate)
        if rule.op == "structural":
            return _structural(rule, data, coordinate)
        if rule.op == "composite":
            return _composite(rule, data, coordinate)
        raise _ConfigurationError
    except _ConfigurationError:
        return [_reason(ReasonCode.INVALID_CONFIGURATION, coordinate)]
    except Exception:
        return [_reason(ReasonCode.INVALID_EVIDENCE, coordinate)]


def _group(contract: ConfiguredMetricContract, diagnostics: Any) -> list[ValidationReason]:
    if not contract.group_compatible:
        return [_reason(ReasonCode.UNSUPPORTED_SCOPE)]
    if diagnostics is None:
        return [_reason(ReasonCode.UNKNOWN_CAPTURE)]
    if not isinstance(diagnostics, GroupDiagnostics):
        return [_reason(ReasonCode.INVALID_EVIDENCE)]
    count_names = (
        "member_count",
        "span_count",
        "duplicate_span_count",
        "message_count",
        "tool_execution_count",
        "complete_member_count",
        "incomplete_member_count",
        "unknown_member_count",
    )
    if any(not _integer(getattr(diagnostics, name)) for name in count_names):
        return [_reason(ReasonCode.INVALID_EVIDENCE)]
    if (
        not 2 <= diagnostics.member_count <= 64
        or sum(
            (diagnostics.complete_member_count, diagnostics.incomplete_member_count, diagnostics.unknown_member_count)
        )
        != diagnostics.member_count
    ):
        return [_reason(ReasonCode.INVALID_EVIDENCE)]
    if (
        not isinstance(diagnostics.issues, tuple)
        or any(
            not isinstance(item, GroupIssueCount) or not isinstance(item.issue, GroupIssue) or not _integer(item.count)
            for item in diagnostics.issues
        )
        or len({item.issue for item in diagnostics.issues}) != len(diagnostics.issues)
    ):
        return [_reason(ReasonCode.INVALID_EVIDENCE)]
    for name in ("input_tokens", "output_tokens", "cost_usd"):
        coverage = getattr(diagnostics, name)
        if (
            not isinstance(coverage, GroupUsageCoverage)
            or any(not _integer(getattr(coverage, item.name)) for item in fields(coverage))
            or coverage.observed != coverage.selected + coverage.overlapping + coverage.ambiguous
        ):
            return [_reason(ReasonCode.INVALID_EVIDENCE)]
    issues = {item.issue for item in diagnostics.issues if item.count}
    reasons = []
    if diagnostics.complete_member_count != diagnostics.member_count:
        code = ReasonCode.INCOMPLETE_CAPTURE if contract.group_requires_complete else ReasonCode.PARTIAL_EVIDENCE
        reasons.append(_reason(code, warning=not contract.group_requires_complete))
    order_issues = {
        GroupIssue.CYCLIC_ORDER,
        GroupIssue.UNKNOWN_TIMESTAMP,
        GroupIssue.MISSING_PARENT,
        GroupIssue.AMBIGUOUS_TOOL_ORDER,
    }
    if contract.group_requires_order and issues & order_issues:
        reasons.append(_reason(ReasonCode.AMBIGUOUS_ORDER))
    identity_issues = {
        GroupIssue.MISSING_TRACE_ID,
        GroupIssue.MISSING_SPAN_ID,
        GroupIssue.AMBIGUOUS_TOOL_LINK,
        GroupIssue.UNMATCHED_TOOL_RESULT,
        GroupIssue.AMBIGUOUS_MESSAGE_IDENTITY,
    }
    if contract.group_requires_complete:
        if issues & identity_issues:
            reasons.append(_reason(ReasonCode.AMBIGUOUS_TOOLS))
        if issues & {GroupIssue.MISSING_PARENT, GroupIssue.CYCLIC_ORDER, GroupIssue.UNSUPPORTED_EVIDENCE}:
            reasons.append(_reason(ReasonCode.INCOMPLETE_CAPTURE))
    elif issues & {GroupIssue.AMBIGUOUS_USAGE, GroupIssue.INVALID_TELEMETRY}:
        reasons.append(_reason(ReasonCode.AMBIGUOUS_USAGE, warning=True))
    return reasons


def _rule_roots(rule: EvidenceRequirement, depth: int = 0) -> set[str]:
    """Find only the roots read by the interpreter, validating rule containers."""
    if (
        not isinstance(rule, EvidenceRequirement)
        or depth > 64
        or type(rule.op) is not str
        or (rule.path is not None and type(rule.path) is not str)
        or type(rule.predicate) is not str
        or type(rule.children) is not tuple
        or type(rule.parameters) is not dict
    ):
        raise _ConfigurationError
    parameters = rule.parameters
    roots: set[str] = set()

    def add(path: Any) -> None:
        _path({}, path)
        root = MetricInput.from_path(path)
        roots.update({root.value} if root is not None else {item.value for item in MetricInput})

    if rule.op in ("all", "any"):
        _active_weights(rule)
        if not _integer(parameters.get("minimum_active", 0)) or (rule.op == "any" and not rule.children):
            raise _ConfigurationError
        for child in rule.children:
            roots.update(_rule_roots(child, depth + 1))
    elif rule.children:
        raise _ConfigurationError
    elif rule.op in ("field", "fallback"):
        add(rule.path)
        _predicate(None, rule.predicate)
        if rule.op == "fallback":
            add(parameters.get("default_path"))
    elif rule.op == "assistant_turns":
        add(rule.path or "messages")
    elif rule.op == "hallucination_references":
        roots.update({"context", "expected"})
        if any(
            parameters.get(flag, False)
            for flag in (
                "include_messages_as_reference",
                "include_assistant_tool_inputs_as_reference",
                "include_assistant_tool_results_as_reference",
            )
        ):
            roots.add("messages")
        if parameters.get("include_scenario_metadata_as_reference") or parameters.get(
            "include_sse_events_as_reference"
        ):
            roots.add("metadata")
    elif rule.op == "decision":
        add(parameters.get("state_field", rule.path or "output"))
        if parameters.get("expected_field", "expected") is not None:
            add(parameters.get("expected_field", "expected"))
    elif rule.op == "structural":
        if parameters.get("format", "json") not in ("json", "yaml"):
            raise _ConfigurationError
        for name in ("output", "expected"):
            roots.add(name)
            if parameters.get(f"{name}_field") not in (None, ""):
                add(parameters[f"{name}_field"])
    elif rule.op == "composite":
        roots.update({"output", "expected"})
        subs = parameters.get("sub_scores")
        if not isinstance(subs, list):
            raise _ConfigurationError
        for sub in subs:
            if not isinstance(sub, dict) or not isinstance(sub.get("check"), dict):
                raise _ConfigurationError
            check = sub["check"]
            for name in ("field", "output_field", "expected_field"):
                if name == "expected_field" and check.get("type") == "equals" and "value" in check:
                    continue
                if name in check:
                    add(check[name])
    else:
        raise _ConfigurationError
    return roots


def _projection(contract: ConfiguredMetricContract) -> set[str]:
    inputs = contract.inputs
    if (
        not isinstance(inputs, MetricInputs)
        or type(inputs.any_of) is not tuple
        or any(
            not isinstance(group, frozenset) or any(not isinstance(item, MetricInput) for item in group)
            for group in (inputs.required, inputs.optional, *inputs.any_of)
        )
    ):
        raise _ConfigurationError
    return {item.value for item in inputs.fields} | _rule_roots(contract.requirements)


def _typed_calls(value: Any) -> None:
    if value is not None and (not isinstance(value, list) or any(not isinstance(call, ToolCall) for call in value)):
        raise ValueError


def _snapshot(case: EvalCase, roots: set[str], depth: int = 0) -> dict[str, Any]:
    """Check scorer-facing runtime types before copying the requested roots."""
    if depth > 64 or not isinstance(case, EvalCase):
        raise ValueError
    data = {}
    for name in sorted(roots):
        value = getattr(case, name)
        if name in ("tool_calls", "expected_tool_calls"):
            _typed_calls(value)
        elif name == "messages" and value is not None:
            if not isinstance(value, list) or any(not isinstance(message, Message) for message in value):
                raise ValueError
            for message in value:
                _typed_calls(message.tool_calls)
        elif name == "runs" and value is not None:
            if not isinstance(value, list):
                raise ValueError
            nested_roots = roots | set(_RUN_PREDICATES) | {"output"}
            data[name] = [_snapshot(run, nested_roots, depth + 1) for run in value]
            continue
        data[name] = _plain(value, depth + 1)
    return data


def _eligibility(contract: ConfiguredMetricContract, context: ValidationContext) -> list[ValidationReason]:
    if not isinstance(contract, ConfiguredMetricContract) or not isinstance(context, ValidationContext):
        return [_reason(ReasonCode.INVALID_CONFIGURATION)]
    if not isinstance(contract.reasons, tuple) or any(
        not isinstance(reason, ValidationReason) for reason in contract.reasons
    ):
        return [_reason(ReasonCode.INVALID_CONFIGURATION)]
    reasons = list(contract.reasons)
    if contract.reviewed is not True:
        reasons.append(_reason(ReasonCode.UNREVIEWED_CONTRACT))
    if (
        type(contract.kind) is not str
        or not isinstance(contract.modes, frozenset)
        or not all(isinstance(mode, EvaluationMode) for mode in contract.modes)
        or not isinstance(contract.online_scopes, frozenset)
        or not all(isinstance(scope, OnlineScope) for scope in contract.online_scopes)
        or not isinstance(contract.execution, ExecutionKind)
        or not isinstance(contract.resources, tuple)
        or not all(isinstance(resource, str) for resource in contract.resources)
        or any(
            type(value) is not bool
            for value in (contract.group_compatible, contract.group_requires_complete, contract.group_requires_order)
        )
    ):
        return [*reasons, _reason(ReasonCode.INVALID_CONFIGURATION)]
    try:
        _projection(contract)
    except Exception:
        return [*reasons, _reason(ReasonCode.INVALID_CONFIGURATION)]
    try:
        mode = EvaluationMode(context.mode)
    except (ValueError, TypeError):
        reasons.append(_reason(ReasonCode.UNSUPPORTED_MODE))
        mode = None
    if mode not in contract.modes:
        reasons.append(_reason(ReasonCode.UNSUPPORTED_MODE))
    try:
        scope = OnlineScope(context.scope) if context.scope is not None else None
    except (ValueError, TypeError):
        scope = None
        reasons.append(_reason(ReasonCode.UNSUPPORTED_SCOPE))
    if mode == EvaluationMode.ONLINE:
        if scope is None or scope not in contract.online_scopes:
            reasons.append(_reason(ReasonCode.UNSUPPORTED_SCOPE))
    elif context.scope is not None:
        reasons.append(_reason(ReasonCode.UNSUPPORTED_SCOPE))
    try:
        execution = ExecutionKind(context.execution)
    except (ValueError, TypeError):
        execution = None
    if execution is None or execution != contract.execution:
        reasons.append(_reason(ReasonCode.UNSUPPORTED_EXECUTION))
    if not _integer(contract.min_runs) or not isinstance(contract.requirements, EvidenceRequirement):
        reasons.append(_reason(ReasonCode.INVALID_CONFIGURATION))
    resources = context.available_resources
    if resources is not None and (
        not isinstance(resources, frozenset) or not all(isinstance(item, str) for item in resources)
    ):
        reasons.append(_reason(ReasonCode.INVALID_CONFIGURATION))
    elif contract.resources:
        if resources is None:
            reasons.append(_reason(ReasonCode.PROVIDER_NOT_CHECKED, warning=True))
        elif not set(contract.resources).issubset(resources):
            reasons.append(_reason(ReasonCode.PROVIDER_UNAVAILABLE))
    if scope == OnlineScope.GROUP:
        reasons.extend(_group(contract, context.group_diagnostics))
    return reasons


def _reads_runs(rule: EvidenceRequirement) -> bool:
    path = rule.path
    root = isinstance(path, str) and (
        path in ("runs", "$.runs")
        or path.startswith(("runs.", "runs[", "$.runs.", "$.runs[", "$['runs']", '$["runs"]'))
    )
    return root or any(_reads_runs(child) for child in rule.children)


_RUN_PREDICATES = {
    "messages": "messages",
    "tool_calls": "tool_calls",
    "context": "string_list",
    "latency_ms": "nonnegative_number",
    "token_count": "nonnegative_integer",
    "input_tokens": "nonnegative_integer",
    "output_tokens": "nonnegative_integer",
    "cost_usd": "nonnegative_number",
    "retry_count": "nonnegative_integer",
    "confidence": "probability",
}


def _run_evidence(data: dict[str, Any], coordinate: str) -> list[ValidationReason]:
    """A repeated execution needs actual output, not a Group member placeholder."""
    if data.get("output") is None:
        return [_reason(ReasonCode.MISSING_EVIDENCE, coordinate)]
    if not isinstance(data["output"], (str, dict, list)):
        return [_reason(ReasonCode.INVALID_EVIDENCE, coordinate)]
    if any(
        data.get(name) is not None and not _predicate(data[name], predicate)
        for name, predicate in _RUN_PREDICATES.items()
    ):
        return [_reason(ReasonCode.INVALID_EVIDENCE, coordinate)]
    return []


def validate_metric(
    contract: ConfiguredMetricContract,
    case: EvalCase | Sequence[EvalCase] | None = None,
    *,
    context: ValidationContext = ValidationContext(),  # noqa: B008 - immutable public API default
) -> ValidationResult:
    """Admit a configured metric without I/O, provider construction or scoring.

    Context-only success establishes eligibility, never readiness. Evidence is
    not inspected when the context or selected contract is already ineligible.
    """
    try:
        reasons = _eligibility(contract, context)
    except Exception:
        reasons = [_reason(ReasonCode.INVALID_CONFIGURATION)]
    if _errors(reasons) or case is None:
        return ValidationResult(not _errors(reasons), False, tuple(dict.fromkeys(reasons)))
    try:
        roots = _projection(contract)
        if context.execution == ExecutionKind.DATASET:
            if (
                not isinstance(case, Sequence)
                or isinstance(case, (str, bytes))
                or not isinstance(context.outcomes, tuple)
                or len(context.outcomes) != len(case)
                or any(type(outcome) is not bool for outcome in context.outcomes)
            ):
                reasons.append(_reason(ReasonCode.INVALID_EVIDENCE))
            elif len(case) < 2:
                reasons.append(_reason(ReasonCode.INSUFFICIENT_EVIDENCE))
            else:
                for index, item in enumerate(case):
                    if not isinstance(item, EvalCase):
                        reasons.append(_reason(ReasonCode.INVALID_EVIDENCE, f"r.{index}"))
                        continue
                    data = _snapshot(item, roots | {"confidence"})
                    if not _predicate(data.get("confidence"), "probability"):
                        reasons.append(_reason(ReasonCode.INVALID_EVIDENCE, f"r.{index}"))
                    reasons.extend(_evaluate(contract.requirements, data, f"r.{index}"))
                if contract.kind == "discrimination" and len(set(context.outcomes)) < 2:
                    reasons.append(_reason(ReasonCode.INSUFFICIENT_EVIDENCE))
        elif not isinstance(case, EvalCase):
            reasons.append(_reason(ReasonCode.INVALID_EVIDENCE))
        elif context.execution == ExecutionKind.REPEATED_RUNS:
            runs = case.runs
            if not isinstance(runs, list) or any(not isinstance(run, EvalCase) for run in runs):
                reasons.append(_reason(ReasonCode.MISSING_EVIDENCE if runs is None else ReasonCode.INVALID_EVIDENCE))
            elif len(runs) < max(2, contract.min_runs):
                reasons.append(_reason(ReasonCode.INSUFFICIENT_EVIDENCE))
            else:
                reads_runs = _reads_runs(contract.requirements)
                if reads_runs:
                    reasons.extend(_evaluate(contract.requirements, _snapshot(case, roots)))
                for index, run in enumerate(runs):
                    data = _snapshot(run, roots | set(_RUN_PREDICATES) | {"output"})
                    reasons.extend(_run_evidence(data, f"r.{index}"))
                    if not reads_runs:
                        reasons.extend(_evaluate(contract.requirements, data, f"r.{index}"))
        else:
            data = _snapshot(case, roots)
            if context.scope == OnlineScope.GROUP:
                diagnostics = context.group_diagnostics
                for name, count in (
                    ("messages", diagnostics.message_count),
                    ("tool_calls", diagnostics.tool_execution_count),
                ):
                    if isinstance(data.get(name), list) and len(data[name]) != count:
                        reasons.append(_reason(ReasonCode.INVALID_EVIDENCE))
            reasons.extend(_evaluate(contract.requirements, data))
    except Exception:
        reasons.append(_reason(ReasonCode.INVALID_EVIDENCE))
    reasons = list(dict.fromkeys(reasons))
    return ValidationResult(not _errors(reasons), True, tuple(reasons))
