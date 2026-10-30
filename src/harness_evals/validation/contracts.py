"""Pure resolution of selected, explicitly reviewed built-in metric contracts.

Only reviewed constructors run, with inert resource sentinels. Plugin discovery,
provider construction, external code loading and scoring never run here.
"""

from __future__ import annotations

import inspect
import json
import math
import re
import types
from enum import Enum
from typing import Any, Union, get_args, get_origin, get_type_hints
from urllib.parse import unquote

from jsonschema.validators import validator_for

from harness_evals.applicability import applicability_for_inputs
from harness_evals.catalog import _build_registry
from harness_evals.core.metric import BaseMetric, EvaluationMode, MetricInputs, OnlineScope, ReliabilityMetric
from harness_evals.core.types import ToolCall
from harness_evals.decision.types import ChoiceQuestion, NoulQuestion, ScoreQuestion
from harness_evals.metrics.llm_judge.geval import GEvalMetric
from harness_evals.metrics.llm_judge.types import RubricLevel
from harness_evals.validation.types import (
    ConfiguredMetricContract,
    EvidenceRequirement,
    ExecutionKind,
    ReasonCode,
    ValidationReason,
)

# Review is an allowlist, not inference from names, inheritance or declarations.
_REVIEWED_KINDS = frozenset(
    {
        "exact_match",
        "contains",
        "list_contains",
        "regex",
        "json_diff",
        "schema_validation",
        "structural_similarity",
        "composite",
        "decision_choice",
        "decision_score",
        "decision_noul",
        "decision_composite",
        "levenshtein",
        "bleu",
        "rouge",
        "embedding_similarity",
        "latency",
        "token_cost",
        "cost_efficiency",
        "retry_count",
        "turn_latency",
        "turn_token_cost",
        "geval",
        "rubric_judge",
        "pairwise",
        "summarization",
        "prompt_alignment",
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "answer_similarity",
        "context_precision",
        "context_recall",
        "context_relevancy",
        "context_entity_recall",
        "turn_faithfulness",
        "turn_contextual_precision",
        "turn_contextual_recall",
        "turn_contextual_relevancy",
        "pii",
        "toxicity",
        "prompt_injection",
        "hallucination",
        "bias",
        "misuse_detection",
        "harmful_advice",
        "role_violation",
        "compliance",
        "harm_severity",
        "task_completion",
        "tool_correctness",
        "argument_correctness",
        "tool_argument_match",
        "tool_selection",
        "mcp_trace_completeness",
        "outcome_consistency",
        "brier_score",
        "calibration",
        "discrimination",
        "vulnerability_correctness",
        "security_completeness",
        "code_safety",
        "code_quality",
        "explanation_quality",
        "root_cause_analysis",
        "actionability",
    }
)
_GROUP_CANDIDATES = frozenset(
    {
        "tool_correctness",
        "tool_selection",
        "tool_argument_match",
        "mcp_trace_completeness",
        "turn_latency",
        "turn_token_cost",
    }
)
_TURN_KINDS = frozenset(
    {
        "turn_latency",
        "turn_token_cost",
        "turn_faithfulness",
        "turn_contextual_precision",
        "turn_contextual_recall",
        "turn_contextual_relevancy",
    }
)
_REFERENCE_FLAGS = (
    "include_messages_as_reference",
    "include_assistant_tool_inputs_as_reference",
    "include_assistant_tool_results_as_reference",
    "include_scenario_metadata_as_reference",
    "include_sse_events_as_reference",
)
_RESOURCES = {"llm": "llm", "embedding": "embedding", "provider": "decision_provider"}
_FORBIDDEN_OPTIONS = frozenset(
    {
        "llm",
        "embedding",
        "provider",
        "target",
        "api_key",
        "token",
        "password",
        "credentials",
        "headers",
        "metadata",
        "entry_point",
        "entrypoint",
        "code",
        "module",
        "class_path",
        "file",
        "path",
        "url",
        "base_url",
    }
)


class _InertDependency:
    """Not a provider: any attempted resource use fails closed."""

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError("Resources cannot be used during contract resolution")


_INERT = _InertDependency()


def _json_copy(value: Any) -> Any:
    """Reject Python coercions, non-finite numbers and arbitrary object hooks."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is list:
        return [_json_copy(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _json_copy(item) for key, item in value.items()}
    raise ValueError("Expected finite JSON configuration")


def _rejected(
    kind: str,
    code: ReasonCode,
    *,
    inputs: MetricInputs | None = None,
    execution: ExecutionKind = ExecutionKind.CASE,
    min_runs: int = 0,
) -> ConfiguredMetricContract:
    return ConfiguredMetricContract(
        kind=kind,
        inputs=inputs if inputs is not None else MetricInputs.unknown(),
        requirements=EvidenceRequirement("all"),
        execution=execution,
        min_runs=min_runs,
        reviewed=False,
        reasons=(ValidationReason(code),),
    )


def _catalog_contract_metadata(kind: str, metric_class: type[BaseMetric]) -> dict[str, Any]:
    """Discovery only: never resolve defaults or invoke a constructor."""
    supported = _build_registry().get(kind) is metric_class and kind in _REVIEWED_KINDS
    return {
        "configured_contract_supported": supported,
        "contract_review": "configuration_required" if supported else "unreviewed",
        "group_candidate": supported and kind in _GROUP_CANDIDATES,
    }


def _constructor_parameters(cls: type[BaseMetric]) -> dict[str, Any]:
    annotations: dict[str, Any] = {}
    for base in cls.__mro__:
        if base.__module__ == "harness_evals.core.metric":
            continue
        init = base.__dict__.get("__init__")
        if init is None:
            continue
        hints = get_type_hints(init)
        for name, parameter in inspect.signature(init).parameters.items():
            if name != "self" and parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                annotations.setdefault(name, hints.get(name, Any))
    return annotations


def _matches_type(value: Any, annotation: Any) -> bool:
    if annotation in (Any, object):
        return True
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, types.UnionType):
        return any(_matches_type(value, member) for member in args)
    if annotation is type(None):
        return value is None
    if annotation is float:
        return type(value) in (int, float)
    if annotation in (str, bool, int):
        return type(value) is annotation
    if annotation in (dict, list) or origin in (dict, list, set):
        container = origin or annotation
        if container is set:
            container = list  # Constructor set options use JSON arrays.
        if type(value) is not container:
            return False
        if not args:
            return True
        if container is list:
            return all(_matches_type(item, args[0]) for item in value)
        return all(_matches_type(key, args[0]) and _matches_type(item, args[1]) for key, item in value.items())
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return value in {member.value for member in annotation}
    return False  # Runtime dataclasses, DAGs, callables and providers are not JSON options.


def _tool_call(value: Any) -> ToolCall:
    if type(value) is not dict or set(value) - {"name", "input", "output"}:
        raise ValueError("Invalid tool configuration")
    if type(value.get("name")) is not str or not value["name"]:
        raise ValueError("Invalid tool configuration")
    if value.get("input") is not None and type(value["input"]) is not dict:
        raise ValueError("Invalid tool configuration")
    if value.get("output") is not None and type(value["output"]) not in (str, dict):
        raise ValueError("Invalid tool configuration")
    return ToolCall(**value)


def _criteria(question_type: str, value: Any) -> Any:
    value = _json_copy(value)
    annotations = {
        "choice": dict[str, str | dict | list | None],
        "score": list[str | dict | list],
        "noul": dict[str, str] | None,
    }
    if not _matches_type(value, annotations[question_type]):
        raise ValueError("Invalid question criteria")
    if question_type == "choice" and (not value or any(not key for key in value)):
        raise ValueError("Invalid choices")
    if question_type == "score" and not 2 <= len(value) <= 10:
        raise ValueError("Invalid rubric")
    return value


def _question(value: Any) -> ChoiceQuestion | ScoreQuestion | NoulQuestion:
    if type(value) is not dict or set(value) - {"type", "instructions", "criteria"}:
        raise ValueError("Invalid question configuration")
    classes = {"choice": ChoiceQuestion, "score": ScoreQuestion, "noul": NoulQuestion}
    cls = classes[value["type"]]
    instructions = _json_copy(value["instructions"])
    if not _matches_type(instructions, str | dict | list):
        raise ValueError("Invalid question instructions")
    return cls(instructions=instructions, criteria=_criteria(value["type"], value.get("criteria")))


def _question_options(question: Any) -> dict[str, Any]:
    classes = {ChoiceQuestion: "choice", ScoreQuestion: "score", NoulQuestion: "noul"}
    return {
        "type": classes[type(question)],
        "instructions": _json_copy(question.instructions),
        "criteria": _json_copy(question.criteria),
    }


def _schema(value: Any) -> None:
    """Check only local schema structure; never resolve references or instances."""
    value = _json_copy(value)

    validator = validator_for(value)
    validator.check_schema(value)

    def anchor(resource: Any, name: str, *, root: bool = True) -> Any:
        if type(resource) is dict:
            identifier = validator.ID_OF(resource)
            if not root and isinstance(identifier, str) and identifier and not identifier.startswith("#"):
                return None
            if name in (resource.get("$anchor"), resource.get("$dynamicAnchor")) or identifier == f"#{name}":
                return resource
            for child in resource.values():
                found = anchor(child, name, root=False)
                if found is not None:
                    return found
        elif type(resource) is list:
            for child in resource:
                found = anchor(child, name, root=False)
                if found is not None:
                    return found
        return None

    def target(resource: Any, reference: str) -> Any:
        fragment = unquote(reference.removeprefix("#"))
        if not fragment:
            return resource
        if not fragment.startswith("/"):
            return anchor(resource, fragment)
        current = resource
        for part in fragment[1:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if type(current) is dict and part in current:
                current = current[part]
            elif type(current) is list and part.isdecimal() and int(part) < len(current):
                current = current[int(part)]
            else:
                return None
        return current

    def local_references(node: Any, resource: Any) -> None:
        if type(node) is dict:
            identifier = validator.ID_OF(node)
            if isinstance(identifier, str) and identifier and not identifier.startswith("#"):
                resource = node
            for key, child in node.items():
                if key in {"$ref", "$dynamicRef", "$recursiveRef"}:
                    if type(child) is not str or (child and not child.startswith("#")):
                        raise ValueError("External schema references are unsupported")
                    resolved = target(resource, child)
                    if type(resolved) not in (dict, bool):
                        raise ValueError("Local schema reference has no schema target")
                    validator.check_schema(resolved)
                local_references(child, resource)
        elif type(node) is list:
            for child in node:
                local_references(child, resource)

    local_references(value, value)


def _construct_metric(kind: str, cls: type[BaseMetric], options: dict[str, Any] | None) -> BaseMetric:
    if options is not None and type(options) is not dict:
        raise ValueError("Expected constructor options")
    selected = _json_copy(options if options is not None else {})
    parameters = _constructor_parameters(cls)
    if set(selected) & _FORBIDDEN_OPTIONS or set(selected) - set(parameters):
        raise ValueError("Unsupported constructor options")
    for name, value in selected.items():
        if name == "expected_trace" and kind == "mcp_trace_completeness":
            if type(value) is not list:
                raise ValueError("Invalid expected trace")
            selected[name] = [_tool_call(item) for item in value]
        elif name == "sub_scores" and kind == "decision_composite":
            if type(value) is not list:
                raise ValueError("Invalid sub-checks")
            selected[name] = [{**sub, "question": _question(sub["question"])} for sub in value]
        elif name == "rubric" and kind == "rubric_judge" and value is not None:
            if type(value) is not dict or any(
                type(key) is not int and (type(key) is not str or not re.fullmatch(r"[+-]?[0-9]+", key))
                for key in value
            ):
                raise ValueError("Invalid judge rubric")
            rubric = {int(key): item for key, item in value.items()}
            if len(rubric) != len(value) or not _matches_type(rubric, dict[int, str]):
                raise ValueError("Invalid judge rubric")
            selected[name] = rubric
        elif name == "rubric" and issubclass(cls, GEvalMetric) and value is not None:
            if type(value) is not list:
                raise ValueError("Invalid judge rubric")
            bands = []
            for level in value:
                if (
                    type(level) is not dict
                    or set(level) != {"min_score", "max_score", "description"}
                    or type(level["min_score"]) is not int
                    or type(level["max_score"]) is not int
                    or type(level["description"]) is not str
                ):
                    raise ValueError("Invalid rubric band")
                bands.append(RubricLevel(**level))
            selected[name] = bands
        elif not _matches_type(value, parameters[name]):
            raise ValueError("Invalid constructor option type")
    for name in _RESOURCES:
        if name in parameters:
            selected[name] = _INERT
    return cls(**selected)


def _current_options(kind: str, metric: BaseMetric) -> dict[str, Any]:
    """Copy and validate only reviewed constructor fields, never resources/hooks."""
    selected = {}
    for name, annotation in _constructor_parameters(type(metric)).items():
        if name in _RESOURCES:
            selected[name] = _INERT
            continue
        # These subclasses hard-code inherited constructor keywords themselves.
        if kind in _TURN_KINDS and name in {"name", "dimension"}:
            continue
        if isinstance(metric, GEvalMetric) and kind != "geval" and name == "dimension":
            continue
        value = getattr(metric, name)
        if name == "expected_trace" and kind == "mcp_trace_completeness":
            if type(value) is not list or any(type(call) is not ToolCall for call in value):
                raise ValueError("Invalid expected trace")
            selected[name] = [
                _tool_call(_json_copy({"name": call.name, "input": call.input, "output": call.output}))
                for call in value
            ]
            continue
        if name == "sub_scores" and kind == "decision_composite":
            if type(value) is not list or any(type(sub) is not dict for sub in value):
                raise ValueError("Invalid sub-checks")
            selected[name] = [
                {
                    **_json_copy({key: item for key, item in sub.items() if key != "question"}),
                    "question": _question(_question_options(sub["question"])),
                }
                for sub in value
            ]
            continue
        if name == "rubric" and kind == "rubric_judge":
            if not _matches_type(value, dict[int, str]):
                raise ValueError("Invalid judge rubric")
            selected[name] = dict(value)
            continue
        if name == "rubric" and isinstance(metric, GEvalMetric):
            if type(value) is not list or any(type(level) is not RubricLevel for level in value):
                raise ValueError("Invalid judge rubric")
            levels = []
            for level in value:
                if (
                    type(level.min_score) is not int
                    or type(level.max_score) is not int
                    or type(level.description) is not str
                ):
                    raise ValueError("Invalid rubric band")
                levels.append(RubricLevel(level.min_score, level.max_score, level.description))
            selected[name] = levels
            continue
        if kind == "hallucination" and name == "sse_reference_events" and type(value) is tuple:
            value = list(value)
        elif kind == "tool_argument_match" and name == "ignore_keys" and type(value) is set:
            value = sorted(_json_copy(list(value)))
        elif isinstance(annotation, type) and issubclass(annotation, Enum) and type(value) is annotation:
            value = value.value
        elif kind == "schema_validation" and name == "schema" and type(value) is bool:
            # A JSON string schema may have been parsed to a boolean by __init__.
            value = json.dumps(value)
        value = _json_copy(value)
        if not _matches_type(value, annotation):
            raise ValueError("Invalid constructor option type")
        selected[name] = value
    return selected


def resolve_metric_contract(kind: str, options: dict[str, Any] | None = None) -> ConfiguredMetricContract:
    """Resolve constructor-style finite JSON options without providers or scoring.

    ``expected_trace`` uses ToolCall-shaped JSON objects. Decision-composite
    questions use JSON objects tagged with ``type`` (choice/score/noul).
    Factory envelopes and live runtime objects are not constructor options.
    """
    if type(kind) is not str:
        return _rejected("unknown", ReasonCode.UNKNOWN_METRIC)
    cls = _build_registry().get(kind)
    if cls is None:
        return _rejected(kind, ReasonCode.UNKNOWN_METRIC)
    if kind not in _REVIEWED_KINDS:
        execution = ExecutionKind.REPEATED_RUNS if issubclass(cls, ReliabilityMetric) else ExecutionKind.CASE
        return _rejected(
            kind,
            ReasonCode.UNREVIEWED_CONTRACT,
            inputs=cls.eval_inputs,
            execution=execution,
            min_runs=2 if execution == ExecutionKind.REPEATED_RUNS else 0,
        )
    try:
        return _describe(_construct_metric(kind, cls, options), frozenset(), rebuild=False)
    except Exception:
        # Never echo constructor/JSON/regex/path exception strings or raw options.
        return _rejected(kind, ReasonCode.INVALID_CONFIGURATION)


def _field(path: str, predicate: str = "present", **parameters: Any) -> EvidenceRequirement:
    return EvidenceRequirement("field", path=path, predicate=predicate, parameters=parameters)


def _all(*children: EvidenceRequirement) -> EvidenceRequirement:
    return EvidenceRequirement("all", children=children)


def _declaration_rule(inputs: MetricInputs) -> EvidenceRequirement:
    def field_rule(name: str) -> EvidenceRequirement:
        predicates = {
            "context": "string_list",
            "messages": "messages",
            "tool_calls": "tool_calls",
            "expected_tools": "string_list",
            "expected_tool_calls": "tool_calls",
            "latency_ms": "nonnegative_number",
            "cost_usd": "nonnegative_number",
            "token_count": "nonnegative_integer",
            "input_tokens": "nonnegative_integer",
            "output_tokens": "nonnegative_integer",
            "retry_count": "nonnegative_integer",
            "confidence": "probability",
            "runs": "sequence",
        }
        return _field(name, predicates.get(name, "present"))

    children = [field_rule(field.value) for field in sorted(inputs.required, key=lambda field: field.value)]
    if inputs.any_of:
        children.append(
            EvidenceRequirement(
                "any",
                children=tuple(
                    _all(*(field_rule(field.value) for field in sorted(group, key=lambda field: field.value)))
                    for group in inputs.any_of
                ),
            )
        )
    return _all(*children)


def _decision_rule(
    question_type: str,
    mode: Any,
    state_field: Any,
    expected_field: Any,
    criteria: Any,
    skip_when_missing: Any = False,
) -> EvidenceRequirement:
    if mode not in (None, "confidence", "correctness") or type(skip_when_missing) is not bool:
        raise ValueError("Invalid decision configuration")
    if type(state_field) is not str or not state_field:
        raise ValueError("Invalid decision state path")
    for path in (state_field, expected_field):
        if path is not None and (
            type(path) is not str or not path or MetricInputs.from_paths([path]) == MetricInputs.unknown()
        ):
            raise ValueError("Invalid decision path")
    if mode == "correctness" and expected_field is None:
        raise ValueError("Missing decision expected path")
    criteria = _criteria(question_type, criteria)
    return EvidenceRequirement(
        "decision",
        parameters={
            "question_type": question_type,
            "mode": mode,
            "state_field": state_field,
            "expected_field": expected_field,
            "criteria": criteria,
            "skip_when_missing": skip_when_missing,
        },
    )


def _composite_config(metric: BaseMetric) -> dict[str, Any]:
    sub_scores = _json_copy(metric.sub_scores)
    if metric.output_format not in ("json", "yaml") or not sub_scores:
        raise ValueError("Invalid composite configuration")
    # Static operator allowlist: never access or call the operator function registry.
    paths = {
        "deep_diff": ("output_field", "expected_field"),
        "json_schema": ("field",),
        "field_exists": ("field",),
        "equals": ("field",),
        "contains": ("field",),
        "contains_all": ("field",),
        "count_match": ("field", "expected_field"),
        "list_field_match": ("field", "expected_field"),
        "unique_ratio": ("field",),
        "regex_match": ("field",),
        "range": ("field",),
        "length": ("field",),
    }
    options = {
        "deep_diff": {"ignore_order", "ignore_keys"},
        "json_schema": {"schema"},
        "field_exists": set(),
        "equals": {"value", "expected_field"},
        "contains": {"value"},
        "contains_all": {"values"},
        "count_match": set(),
        "list_field_match": set(),
        "unique_ratio": {"min_ratio"},
        "regex_match": {"pattern"},
        "range": {"min", "max"},
        "length": {"min_length", "max_length"},
    }
    names = set()
    total_weight = 0.0
    for sub in sub_scores:
        if set(sub) - {"name", "weight", "check", "skip_when_missing"}:
            raise ValueError("Unsupported sub-check configuration")
        if type(sub.get("name")) is not str or not sub["name"] or sub["name"] in names:
            raise ValueError("Invalid sub-check name")
        names.add(sub["name"])
        weight = sub.get("weight", 0)
        if type(weight) not in (int, float) or weight < 0 or not math.isfinite(float(weight)):
            raise ValueError("Invalid sub-check weight")
        total_weight += float(weight)
        if type(sub.get("skip_when_missing", False)) is not bool:
            raise ValueError("Invalid missing-data policy")
        check = sub["check"]
        operator = check["type"]
        if set(check) - {"type", *paths[operator], *options[operator]}:
            raise ValueError("Unsupported operator configuration")
        for key in paths[operator]:
            if type(check.get(key)) is not str or not check[key]:
                raise ValueError("Missing operator path")
        if operator == "equals" and "value" in check:
            check.pop("expected_field", None)  # Literal precedence, including projection.
        elif operator == "equals" and not check.get("expected_field"):
            raise ValueError("Missing comparison operand")
        for key in ("field", "output_field", "expected_field"):
            if key in check and (
                type(check[key]) is not str or MetricInputs.from_paths([check[key]]) == MetricInputs.unknown()
            ):
                raise ValueError("Invalid operator path")
        if operator == "deep_diff" and (
            type(check.get("ignore_order", True)) is not bool
            or not _matches_type(check.get("ignore_keys", []), list[str])
        ):
            raise ValueError("Invalid structural comparison options")
        if operator == "json_schema":
            if type(check.get("schema")) is not dict or not check["schema"]:
                raise ValueError("Missing JSON schema")
            _schema(check["schema"])
        if operator == "contains_all" and type(check.get("values", [])) is not list:
            raise ValueError("Invalid containment options")
        if operator == "regex_match":
            if type(check.get("pattern")) is not str or not check["pattern"]:
                raise ValueError("Missing regex pattern")
            re.compile(check["pattern"])
        for key in ("min", "max", "min_ratio"):
            if key in check and type(check[key]) not in (int, float):
                raise ValueError("Invalid numeric comparison options")
        for key in ("min_length", "max_length"):
            if key in check and (type(check[key]) is not int or check[key] < 0):
                raise ValueError("Invalid length comparison options")
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("No usable composite weight")
    return {"sub_scores": sub_scores, "output_format": metric.output_format}


def _requirements(kind: str, metric: BaseMetric) -> EvidenceRequirement:
    if kind in _TURN_KINDS:
        allow_skips = getattr(metric, "allow_skips", kind in {"turn_latency", "turn_token_cost"})
        if type(allow_skips) is not bool:
            raise ValueError("Invalid turn policy")
        return EvidenceRequirement("assistant_turns", parameters={"metric": kind, "allow_skips": allow_skips})
    if kind == "hallucination":
        flags = {name: getattr(metric, name) for name in _REFERENCE_FLAGS}
        if any(type(value) is not bool for value in flags.values()):
            raise ValueError("Invalid reference policy")
        events = list(metric.sse_reference_events)
        if any(type(event) is not str for event in events):
            raise ValueError("Invalid reference events")
        return _all(
            _field("output"),
            EvidenceRequirement(
                "hallucination_references",
                parameters={
                    **flags,
                    "sse_reference_events": events,
                },
            ),
        )
    if kind == "schema_validation":
        _schema(metric.schema)
    if kind == "structural_similarity":
        if (
            metric.format not in ("json", "yaml")
            or metric.level not in ("raw", "structural", "schema_validated")
            or metric.extra_keys not in ("ignore", "penalize")
        ):
            raise ValueError("Invalid structural configuration")
        for path in (metric.output_field, metric.expected_field):
            if path is not None and (
                type(path) is not str or (path and MetricInputs.from_paths([path]) == MetricInputs.unknown())
            ):
                raise ValueError("Invalid structural path")
        schema_validator = _json_copy(metric.schema_validator)
        if schema_validator is not None:
            if schema_validator.get("type") != "json_schema" or set(schema_validator) - {"type", "schema"}:
                raise ValueError("External validators are not supported")
            if not schema_validator.get("schema"):
                raise ValueError("Missing JSON schema")
            _schema(schema_validator["schema"])
        return EvidenceRequirement(
            "structural",
            parameters={
                "format": metric.format,
                "output_field": metric.output_field,
                "expected_field": metric.expected_field,
            },
        )
    if kind == "composite":
        return EvidenceRequirement("composite", parameters=_composite_config(metric))
    if kind in ("decision_choice", "decision_score", "decision_noul"):
        return _decision_rule(
            kind.removeprefix("decision_"), metric.mode, metric.state_field, "expected", metric.criteria
        )
    if kind == "decision_composite":
        if type(metric.sub_scores) is not list or not metric.sub_scores:
            raise ValueError("No decision sub-checks")
        children = []
        names = set()
        active_weights = []
        total_weight = 0.0
        for sub in metric.sub_scores:
            if type(sub) is not dict or set(sub) - {
                "name",
                "weight",
                "state_field",
                "expected_field",
                "mode",
                "question",
                "invert",
                "skip_when_missing",
            }:
                raise ValueError("Unsupported decision sub-check configuration")
            if type(sub.get("name")) is not str or not sub["name"] or sub["name"] in names:
                raise ValueError("Invalid decision sub-check name")
            names.add(sub["name"])
            weight = sub.get("weight", 0.0)  # Same default and conversion as fold_sub_scores.
            if type(weight) not in (int, float) or weight < 0 or not math.isfinite(float(weight)):
                raise ValueError("Invalid decision sub-check weight")
            total_weight += float(weight)
            active_weights.append(float(weight))
            if type(sub.get("invert", False)) is not bool:
                raise ValueError("Invalid decision inversion policy")
            question = _question_options(sub["question"])
            _question(question)  # Validate the entire typed question before admission.
            children.append(
                _decision_rule(
                    question["type"],
                    sub.get("mode"),
                    sub["state_field"],
                    sub.get("expected_field"),
                    question["criteria"],
                    sub.get("skip_when_missing", False),
                )
            )
        if not math.isfinite(total_weight) or total_weight <= 0:
            raise ValueError("No usable decision weight")
        # Retain every child; at least one must supply non-skipped evidence.
        return EvidenceRequirement(
            "all", children=tuple(children), parameters={"minimum_active": 1, "active_weights": active_weights}
        )
    if kind == "outcome_consistency":
        return _field("runs", "sequence", min_items=2)
    if kind == "tool_argument_match":
        if type(metric.skip_when_missing) is not bool:
            raise ValueError("Invalid missing-data policy")
        policy = "skip" if metric.skip_when_missing else "reject"
        return _all(
            *(
                _field(path, "tool_calls_with_arguments", missing_policy=policy)
                for path in ("tool_calls", "expected_tool_calls")
            )
        )
    if kind == "argument_correctness":
        return _all(_field("input"), _field("tool_calls", "tool_calls", min_items=1))
    if kind in {"faithfulness", "context_precision", "context_recall", "context_relevancy", "context_entity_recall"}:
        return _all(_declaration_rule(metric.eval_inputs), _field("context", "string_list", min_items=1))
    return _declaration_rule(metric.eval_inputs)


def _factory_composite_class(cls: type) -> bool:
    """Recognize factory-owned classes without interpreter-specific code inspection."""
    from harness_evals.metrics.factory import _FACTORY_COMPOSITE_CLASSES

    return cls in _FACTORY_COMPOSITE_CLASSES


def _describe_composite(metric: BaseMetric, seen: frozenset[int]) -> ConfiguredMetricContract:
    if type(metric.sub_metrics) is not list or not metric.sub_metrics:
        return _rejected("composite", ReasonCode.INVALID_CONFIGURATION)
    weights = metric.weights
    if (
        type(weights) is not list
        or len(weights) != len(metric.sub_metrics)
        or any(type(weight) not in (int, float) or not math.isfinite(weight) for weight in weights)
        or metric.aggregation not in {"average", "weighted_average", "min", "max", "all_pass"}
    ):
        return _rejected("composite", ReasonCode.INVALID_CONFIGURATION)
    total_weight = math.fsum(weights)
    if metric.aggregation == "weighted_average" and (
        not math.isfinite(total_weight) or total_weight <= 0 or any(weight < 0 for weight in weights)
    ):
        return _rejected("composite", ReasonCode.INVALID_CONFIGURATION)
    children = tuple(_describe(child, seen) for child in metric.sub_metrics)
    if not all(child.reviewed for child in children):
        return _rejected("composite", ReasonCode.UNREVIEWED_CONTRACT)
    if len({child.execution for child in children}) != 1 or children[0].execution == ExecutionKind.DATASET:
        # Repeated-run children dispatch through a_measure(case.runs), but the
        # factory never dispatches child measure_dataset(cases, outcomes).
        return _rejected("composite", ReasonCode.UNSUPPORTED_EXECUTION)
    modes = frozenset.intersection(*(child.modes for child in children))
    scopes = frozenset.intersection(*(child.online_scopes for child in children)) - {OnlineScope.GROUP}
    if not scopes:
        modes -= {EvaluationMode.ONLINE}
    return ConfiguredMetricContract(
        kind="composite",
        inputs=MetricInputs.union(*(child.inputs for child in children)),
        requirements=_all(*(child.requirements for child in children)),
        modes=modes,
        online_scopes=scopes,
        execution=children[0].execution,
        min_runs=max(child.min_runs for child in children),
        resources=tuple(sorted({resource for child in children for resource in child.resources})),
    )


def _describe(metric: BaseMetric, seen: frozenset[int], *, rebuild: bool = True) -> ConfiguredMetricContract:
    if id(metric) in seen:
        return _rejected("composite", ReasonCode.INVALID_CONFIGURATION)
    seen = seen | {id(metric)}
    registry = _build_registry()
    kind = next((kind for kind, cls in registry.items() if type(metric) is cls), None)
    if kind is None:
        if _factory_composite_class(type(metric)):
            return _describe_composite(metric, seen)
        return _rejected("custom", ReasonCode.UNREVIEWED_CONTRACT)
    execution = (
        ExecutionKind.REPEATED_RUNS
        if isinstance(metric, ReliabilityMetric)
        else ExecutionKind.DATASET
        if kind in {"brier_score", "calibration", "discrimination"}
        else ExecutionKind.CASE
    )
    min_runs = 2 if execution == ExecutionKind.REPEATED_RUNS else 0
    if kind not in _REVIEWED_KINDS:
        return _rejected(
            kind,
            ReasonCode.UNREVIEWED_CONTRACT,
            inputs=type(metric).eval_inputs,
            execution=execution,
            min_runs=min_runs,
        )
    if rebuild:
        metric = type(metric)(**_current_options(kind, metric))
    if kind == "calibration" and (type(metric.n_bins) is not int or metric.n_bins < 1):
        raise ValueError("Invalid calibration bins")
    if kind == "bleu" and (type(metric.max_n) is not int or metric.max_n < 1):
        raise ValueError("Invalid n-gram order")
    requirements = _requirements(kind, metric)
    if metric.eval_inputs == MetricInputs.unknown():
        return _rejected(
            kind, ReasonCode.UNREVIEWED_CONTRACT, inputs=metric.eval_inputs, execution=execution, min_runs=min_runs
        )
    applicability = applicability_for_inputs(metric.eval_inputs)
    group = kind in _GROUP_CANDIDATES
    if kind == "tool_argument_match":
        group = metric.skip_when_missing is False
    complete = group and kind not in {"turn_latency", "turn_token_cost"}
    order = group and (
        (kind == "tool_correctness" and metric.mode in ("exact", "subsequence"))
        # Subset argument pairing also consumes order through its greedy fallback.
        or kind == "tool_argument_match"
    )
    scopes = (applicability.online_scopes - {OnlineScope.GROUP}) | ({OnlineScope.GROUP} if group else set())
    modes = applicability.modes | ({EvaluationMode.ONLINE} if group else set())
    if not scopes:
        modes -= {EvaluationMode.ONLINE}
    resources = tuple(
        sorted(resource for name, resource in _RESOURCES.items() if name in _constructor_parameters(type(metric)))
    )
    return ConfiguredMetricContract(
        kind=kind,
        inputs=metric.eval_inputs,
        requirements=requirements,
        modes=frozenset(modes),
        online_scopes=frozenset(scopes),
        execution=execution,
        min_runs=min_runs,
        resources=resources,
        group_compatible=group,
        group_requires_complete=complete,
        group_requires_order=order,
    )


def describe_configured_metric(metric: BaseMetric) -> ConfiguredMetricContract:
    """Describe exact built-ins or factory composites, never arbitrary instance state."""
    try:
        return _describe(metric, frozenset())
    except Exception:
        return _rejected("unknown", ReasonCode.INVALID_CONFIGURATION)
