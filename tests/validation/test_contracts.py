"""Contract resolution/discovery regressions; no provider, scorer or target calls."""

from __future__ import annotations

import json
import socket
import urllib.request
from unittest.mock import Mock

import pytest

from harness_evals.catalog import _build_registry, catalog
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension, EvaluationMode, MetricInput, MetricInputs, OnlineScope
from harness_evals.core.types import ToolCall
from harness_evals.decision.types import ChoiceQuestion, NoulQuestion, ScoreQuestion
from harness_evals.metrics import ExactMatchMetric, HallucinationMetric, MCPTraceCompletenessMetric
from harness_evals.validation.contracts import describe_configured_metric, resolve_metric_contract
from harness_evals.validation.evidence import validate_metric
from harness_evals.validation.types import ConfiguredMetricContract, ExecutionKind, ReasonCode

pytestmark = pytest.mark.unit


def _explode(*args, **kwargs):
    raise AssertionError("Provider, scorer, target or external code was invoked")


@pytest.fixture
def no_execution(monkeypatch):
    invocations = []

    def forbidden(*args, **kwargs):
        invocations.append(True)
        _explode()

    # Patch every shipped scorer, including dataset/repeated-run and async hooks.
    for cls in _build_registry().values():
        for name in ("measure", "a_measure", "measure_runs", "a_measure_runs", "measure_dataset"):
            if hasattr(cls, name):
                monkeypatch.setattr(cls, name, forbidden)
    from harness_evals.decision.typesafe import TypeSafeDecisionProvider
    from harness_evals.llm.anthropic import AnthropicLLM
    from harness_evals.llm.openai import OpenAILLM
    from harness_evals.llm.openai_embedding import OpenAIEmbedding
    from harness_evals.metrics import factory
    from harness_evals.metrics.composite.operators import OPERATORS
    from harness_evals.metrics.structural import structural_similarity
    from harness_evals.targets.http import HttpTarget

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    for cls in (OpenAILLM, AnthropicLLM, OpenAIEmbedding, TypeSafeDecisionProvider, HttpTarget):
        monkeypatch.setattr(cls, "__init__", forbidden)
    monkeypatch.setattr(factory, "build_llm_provider", forbidden)
    monkeypatch.setattr(factory, "_build_code_metric", forbidden)
    monkeypatch.setattr(structural_similarity, "_load_entry_point", forbidden)
    for operator in OPERATORS:
        monkeypatch.setitem(OPERATORS, operator, forbidden)
    yield
    # Also catch forbidden calls swallowed by fail-closed exception handling.
    assert not invocations


def _roundtrip(contract):
    snapshot = contract.to_dict()
    encoded = json.dumps(snapshot, allow_nan=False, sort_keys=True)
    assert json.loads(encoded) == snapshot
    assert ConfiguredMetricContract.from_dict(json.loads(encoded)).to_dict() == snapshot
    return snapshot


@pytest.mark.parametrize(
    "kind, options",
    [
        ("exact_match", {}),
        ("exact_match", {"negate": True, "forbidden": "restricted"}),
        ("contains", {}),
        ("list_contains", {}),
        ("regex", {}),
        ("json_diff", {}),
        ("schema_validation", {"schema": {"type": "object"}}),
        ("structural_similarity", {"output_field": "metadata.actual", "expected_field": "metadata.reference"}),
        ("levenshtein", {}),
        ("bleu", {}),
        ("rouge", {}),
        ("embedding_similarity", {}),
        ("latency", {}),
        ("token_cost", {}),
        ("cost_efficiency", {}),
        ("retry_count", {}),
        ("turn_latency", {}),
        ("turn_token_cost", {}),
        ("geval", {}),
        ("rubric_judge", {}),
        ("pairwise", {}),
        ("summarization", {}),
        ("prompt_alignment", {"prompt_instructions": ["Be accurate"]}),
        ("faithfulness", {}),
        ("answer_relevancy", {}),
        ("answer_correctness", {}),
        ("answer_similarity", {}),
        ("context_precision", {}),
        ("context_recall", {}),
        ("context_relevancy", {}),
        ("context_entity_recall", {}),
        ("turn_faithfulness", {}),
        ("turn_contextual_precision", {"allow_skips": True}),
        ("turn_contextual_recall", {}),
        ("turn_contextual_relevancy", {}),
        ("pii", {}),
        ("toxicity", {}),
        ("prompt_injection", {}),
        ("hallucination", {}),
        ("bias", {}),
        ("misuse_detection", {}),
        ("harmful_advice", {}),
        ("role_violation", {"role_description": "Support agent"}),
        ("compliance", {"policies": ["Protect user data"]}),
        ("harm_severity", {}),
        ("task_completion", {}),
        ("tool_correctness", {}),
        ("argument_correctness", {}),
        ("tool_argument_match", {"ignore_keys": ["timestamp"]}),
        ("tool_selection", {}),
        ("mcp_trace_completeness", {"expected_trace": [{"name": "lookup", "input": {}}]}),
        ("outcome_consistency", {"k": 7}),
        ("brier_score", {}),
        ("calibration", {}),
        ("discrimination", {}),
        ("decision_choice", {"instructions": "Select", "criteria": {"a": None, "b": None}}),
        ("decision_score", {"instructions": "Rate", "criteria": ["Bad", "Good"]}),
        ("decision_noul", {"instructions": "Accept?"}),
        (
            "composite",
            {"sub_scores": [{"name": "exists", "weight": 1, "check": {"type": "field_exists", "field": "output.ok"}}]},
        ),
        (
            "decision_composite",
            {
                "sub_scores": [
                    {
                        "name": "label",
                        "weight": 1,
                        "state_field": "output",
                        "question": {"type": "choice", "instructions": "Pick", "criteria": {"a": None, "b": None}},
                    }
                ]
            },
        ),
        ("vulnerability_correctness", {}),
        ("security_completeness", {}),
        ("code_safety", {}),
        ("code_quality", {}),
        ("explanation_quality", {}),
        ("root_cause_analysis", {}),
        ("actionability", {}),
    ],
)
def test_reviewed_contracts_are_pure_strict_json(kind, options, no_execution):
    original = json.dumps(options, sort_keys=True)
    contract = resolve_metric_contract(kind, options)
    assert contract.reviewed, contract.to_dict()
    assert not contract.reasons
    assert json.dumps(options, sort_keys=True) == original
    snapshot = _roundtrip(contract)
    assert snapshot["description_kind"] == "configured"
    assert "metric_class" not in snapshot
    from harness_evals.validation.contracts import _construct_metric

    described = describe_configured_metric(_construct_metric(kind, _build_registry()[kind], options))
    assert described.reviewed
    assert _roundtrip(described) == snapshot


@pytest.mark.parametrize(
    "options",
    [
        [],
        "options",
        {"threshold": float("nan")},
        {"threshold": float("inf")},
        {"threshold": True},
        {"threshold": "0.8"},
        {"case_sensitive": 1},
        {"case_sensitive": None},
        {"typo": True},
        {"case_sensitive": object()},
        {"case_sensitive": (True,)},
        {"case_sensitive": {True}},
        {1: "value"},
        {"llm": {}},
        {"provider": "openai"},
        {"target": {}},
        {"api_key": "not-a-credential"},
        {"metadata": {}},
        {"entry_point": "somewhere"},
        {"code": "pass"},
    ],
)
def test_invalid_non_json_runtime_and_factory_options_fail_closed(options, no_execution):
    contract = resolve_metric_contract("exact_match", options)
    assert not contract.reviewed
    assert [reason.code for reason in contract.reasons] == [ReasonCode.INVALID_CONFIGURATION]
    assert "not-a-credential" not in json.dumps(contract.to_dict())


@pytest.mark.parametrize(
    "kind, options",
    [
        ("regex", {"negate": True, "forbidden": "["}),
        ("tool_correctness", {"mode": "not-valid"}),
        ("tool_argument_match", {"pair": "not-valid"}),
        ("decision_choice", {"instructions": "Pick", "criteria": {}, "mode": "incorrect"}),
        ("decision_score", {"instructions": "Rate", "criteria": ["Only one"]}),
        ("decision_noul", {"instructions": "Accept?", "state_field": "$..output"}),
        ("structural_similarity", {"schema_validator": {"type": "entry_point", "name": "user-code"}}),
        ("mcp_trace_completeness", {"expected_trace": [{"name": "lookup", "input": []}]}),
        ("mcp_trace_completeness", {"expected_trace": [{"name": "lookup"}]}),
        ("mcp_trace_completeness", {"expected_trace": [{"name": "lookup", "input": None}]}),
        ("composite", {"sub_scores": [{"name": "x", "check": {"type": "custom_code", "field": "output"}}]}),
        (
            "decision_composite",
            {
                "sub_scores": [
                    {
                        "name": "x",
                        "state_field": "output",
                        "mode": "correctness",
                        "question": {"type": "noul", "instructions": "Accept?"},
                    }
                ]
            },
        ),
    ],
)
def test_constructor_and_nested_configuration_errors_are_static(kind, options, no_execution):
    contract = resolve_metric_contract(kind, options)
    assert not contract.reviewed
    assert contract.reasons[0].code == ReasonCode.INVALID_CONFIGURATION
    reasons = json.dumps([reason.to_dict() for reason in contract.reasons])
    assert "user-code" not in reasons and "not-valid" not in reasons


@pytest.mark.parametrize(
    "kind,options",
    [
        ("calibration", {"n_bins": 0}),
        ("bleu", {"max_n": 0}),
        (
            "composite",
            {
                "sub_scores": [
                    {
                        "name": "bad_regex",
                        "check": {
                            "type": "regex_match",
                            "field": "output.text",
                            "pattern": "[",
                        },
                    }
                ]
            },
        ),
        (
            "composite",
            {
                "sub_scores": [
                    {
                        "name": "bad_range",
                        "check": {
                            "type": "range",
                            "field": "output.value",
                            "min": "zero",
                        },
                    }
                ]
            },
        ),
        (
            "composite",
            {
                "sub_scores": [
                    {
                        "name": "missing_schema",
                        "check": {
                            "type": "json_schema",
                            "field": "output.value",
                        },
                    }
                ]
            },
        ),
    ],
)
def test_static_scorer_configuration_errors_fail_before_admission(kind, options, no_execution):
    contract = resolve_metric_contract(kind, options)
    assert not contract.reviewed
    assert contract.reasons[0].code == ReasonCode.INVALID_CONFIGURATION


def test_selected_configuration_not_class_or_catalog_possibilities(no_execution):
    entry = next(entry for entry in catalog() if entry.kind == "exact_match")
    positive = entry.configured_contract()
    negative = entry.configured_contract({"negate": True, "forbidden": "restricted"})
    assert positive.inputs.required == {MetricInput.OUTPUT, MetricInput.EXPECTED}
    assert negative.inputs.required == {MetricInput.OUTPUT}
    assert EvaluationMode.ONLINE not in positive.modes
    assert EvaluationMode.ONLINE in negative.modes
    assert EvaluationMode.ONLINE in entry.applicability.modes
    assert ExactMatchMetric.eval_inputs == positive.inputs
    assert entry.to_dict()["applicability_kind"] == "possibilities"


def test_hallucination_projection_and_reference_flags_are_selected(no_execution):
    default = resolve_metric_contract("hallucination")
    selected = resolve_metric_contract(
        "hallucination",
        {
            "include_messages_as_reference": True,
            "include_sse_events_as_reference": True,
            "sse_reference_events": ["reference"],
        },
    )
    assert default.inputs.any_of == (frozenset({MetricInput.CONTEXT}), frozenset({MetricInput.EXPECTED}))
    assert selected.inputs.any_of == tuple(
        frozenset({field})
        for field in (
            MetricInput.CONTEXT,
            MetricInput.EXPECTED,
            MetricInput.MESSAGES,
            MetricInput.METADATA,
        )
    )
    rule = selected.requirements.children[1]
    assert rule.op == "hallucination_references"
    assert rule.parameters == {
        "include_messages_as_reference": True,
        "include_assistant_tool_inputs_as_reference": False,
        "include_assistant_tool_results_as_reference": False,
        "include_scenario_metadata_as_reference": False,
        "include_sse_events_as_reference": True,
        "sse_reference_events": ["reference"],
    }
    assert EvaluationMode.ONLINE not in default.modes
    assert EvaluationMode.ONLINE in selected.modes
    assert not selected.group_compatible


def test_structural_nested_paths_and_fallback_precedence(no_execution):
    contract = resolve_metric_contract(
        "structural_similarity",
        {
            "output_field": "metadata.actual",
            "expected_field": "metadata.reference",
        },
    )
    assert contract.inputs.any_of
    assert contract.inputs.fields == {MetricInput.OUTPUT, MetricInput.EXPECTED, MetricInput.METADATA}
    assert contract.requirements.op == "structural"
    assert contract.requirements.parameters == {
        "format": "yaml",
        "output_field": "metadata.actual",
        "expected_field": "metadata.reference",
    }


@pytest.mark.parametrize(
    "kind,options,complete,order",
    [
        ("tool_correctness", {}, True, True),
        ("tool_correctness", {"mode": "subsequence"}, True, True),
        ("tool_correctness", {"mode": "subset"}, True, False),
        ("tool_selection", {}, True, False),
        ("tool_argument_match", {}, True, True),
        ("tool_argument_match", {"pair": "subset", "arg_match": "subset"}, True, True),
        ("tool_argument_match", {"pair": "subsequence"}, True, True),
        ("mcp_trace_completeness", {"expected_trace": []}, True, False),
        ("turn_latency", {}, False, False),
        ("turn_token_cost", {}, False, False),
    ],
)
def test_exact_group_policy_positive(kind, options, complete, order, no_execution):
    contract = resolve_metric_contract(kind, options)
    assert contract.reviewed and contract.group_compatible
    assert OnlineScope.GROUP in contract.online_scopes
    assert EvaluationMode.ONLINE in contract.modes
    assert contract.group_requires_complete is complete
    assert contract.group_requires_order is order
    entry = next(entry for entry in catalog() if entry.kind == kind)
    assert OnlineScope.GROUP not in entry.applicability.online_scopes
    assert entry.to_dict()["group_candidate"]


@pytest.mark.parametrize(
    "kind, options",
    [
        ("tool_argument_match", {"skip_when_missing": True}),
        ("turn_faithfulness", {}),
        ("turn_contextual_precision", {}),
        ("hallucination", {"include_messages_as_reference": True}),
        ("latency", {}),
        ("token_cost", {}),
        ("exact_match", {"negate": True, "forbidden": "restricted"}),
        ("outcome_consistency", {}),
        ("calibration", {}),
        ("decision_noul", {"instructions": "Accept?"}),
    ],
)
def test_group_never_inferred_from_fields(kind, options, no_execution):
    contract = resolve_metric_contract(kind, options)
    assert contract.reviewed
    assert not contract.group_compatible
    assert OnlineScope.GROUP not in contract.online_scopes


def test_group_policy_does_not_inherit_a_global_field_capability(monkeypatch, no_execution):
    from harness_evals import applicability

    monkeypatch.setattr(applicability, "ONLINE_SCOPE_INPUTS", {OnlineScope.GROUP: frozenset(MetricInput)})
    contract = resolve_metric_contract("exact_match")
    assert contract.reviewed
    assert not contract.group_compatible
    assert contract.online_scopes == frozenset()
    assert contract.modes == {EvaluationMode.OFFLINE}


def test_turn_and_rag_nested_requirements(no_execution):
    for kind in ("turn_latency", "turn_token_cost", "turn_faithfulness", "turn_contextual_recall"):
        contract = resolve_metric_contract(kind)
        assert contract.inputs.required == {MetricInput.MESSAGES}
        assert contract.requirements.op == "assistant_turns"
        assert contract.requirements.parameters["metric"] == kind
    assert resolve_metric_contract("turn_faithfulness", {"allow_skips": True}).requirements.parameters["allow_skips"]
    context = resolve_metric_contract("faithfulness").requirements.children[1]
    assert (context.path, context.predicate, context.parameters) == ("context", "string_list", {"min_items": 1})


def test_decision_expected_and_mode_protocol(no_execution):
    options = {"instructions": "Choose", "criteria": {"a": None, "b": None}, "state_field": "metadata.state"}
    automatic = resolve_metric_contract("decision_choice", options)
    assert automatic.inputs.required == {MetricInput.METADATA}
    assert automatic.inputs.optional == {MetricInput.EXPECTED}
    assert automatic.requirements.parameters == {
        "question_type": "choice",
        "mode": None,
        "state_field": "metadata.state",
        "expected_field": "expected",
        "criteria": {"a": None, "b": None},
        "skip_when_missing": False,
    }
    correctness = resolve_metric_contract("decision_choice", {**options, "mode": "correctness"})
    assert MetricInput.EXPECTED in correctness.inputs.required
    assert correctness.resources == ("decision_provider",)


def test_describe_decision_composite_preserves_explicit_expected(no_execution):
    from harness_evals.metrics import DecisionCompositeMetric

    metric = DecisionCompositeMetric(
        provider=object(),
        sub_scores=[
            {
                "name": "label",
                "state_field": "metadata.state",
                "expected_field": "metadata.label",
                "mode": "correctness",
                "question": ChoiceQuestion("Choose", {"a": None, "b": None}),
                "weight": 1,
            },
            {
                "name": "optional",
                "state_field": "output.optional",
                "skip_when_missing": True,
                "question": ChoiceQuestion("Choose", {"a": None, "b": None}),
                "weight": 1,
            },
        ],
    )
    contract = describe_configured_metric(metric)
    assert contract.reviewed
    first, second = contract.requirements.children
    assert first.parameters["expected_field"] == "metadata.label"
    assert second.parameters["expected_field"] is None
    assert second.parameters["skip_when_missing"] is True
    _roundtrip(contract)


def test_operator_rules_preserve_literals_absence_and_skips(no_execution):
    options = {
        "output_format": "json",
        "sub_scores": [
            {"name": "absence", "weight": 1, "check": {"type": "field_exists", "field": "output.optional"}},
            {
                "name": "literal",
                "weight": 1,
                "check": {"type": "equals", "field": "output.value", "value": 0, "expected_field": "metadata.ignored"},
            },
            {
                "name": "nested",
                "weight": 1,
                "skip_when_missing": True,
                "check": {"type": "list_field_match", "field": "output.items", "expected_field": "expected.items"},
            },
        ],
    }
    contract = resolve_metric_contract("composite", options)
    assert contract.reviewed
    assert contract.requirements.op == "composite"
    assert "expected_field" not in contract.requirements.parameters["sub_scores"][1]["check"]
    assert MetricInput.METADATA not in contract.inputs.fields
    assert contract.requirements.parameters["sub_scores"][0]["check"]["field"] == "output.optional"
    assert contract.requirements.parameters["sub_scores"][2]["skip_when_missing"]
    assert "expected_field" in options["sub_scores"][1]["check"]  # No caller mutation.


def test_dataset_and_repeated_execution_are_not_case_or_group(no_execution):
    for kind in ("brier_score", "calibration", "discrimination"):
        contract = resolve_metric_contract(kind)
        assert contract.execution == ExecutionKind.DATASET
        assert contract.inputs.required == {MetricInput.CONFIDENCE}
        assert contract.requirements.children[0].predicate == "probability"
    repeated = resolve_metric_contract("outcome_consistency", {"k": 10})
    assert repeated.execution == ExecutionKind.REPEATED_RUNS
    assert repeated.min_runs == 2  # Actual scorer minimum, not the expected k.
    assert repeated.inputs.required == {MetricInput.RUNS}
    assert repeated.requirements.parameters == {"min_items": 2}
    for kind in ("resource_consistency", "trajectory_consistency"):
        contract = resolve_metric_contract(kind)
        assert contract.execution == ExecutionKind.REPEATED_RUNS
        assert contract.min_runs == 2
        assert not contract.reviewed  # Per-run nested contracts need further review.


def test_unknown_custom_subclasses_and_arbitrary_state_fail_closed(no_execution):
    class Custom(ExactMatchMetric):
        pass

    class Undeclared(BaseMetric):
        def measure(self, case):
            _explode()

    custom = Custom()
    custom.private_state = object()
    for metric in (custom, Undeclared("custom", Dimension.CORRECTNESS)):
        contract = describe_configured_metric(metric)
        assert not contract.reviewed
        assert contract.reasons[0].code == ReasonCode.UNREVIEWED_CONTRACT
        assert "private_state" not in json.dumps(_roundtrip(contract))
    unknown = resolve_metric_contract("not_shipped")
    assert unknown.reasons[0].code == ReasonCode.UNKNOWN_METRIC
    assert not describe_configured_metric(object()).reviewed


def test_describe_instance_does_not_serialize_or_touch_provider(no_execution):
    provider = Mock()
    provider.generate_json.side_effect = _explode
    metric = HallucinationMetric(provider, include_messages_as_reference=True)
    metric.private_state = object()
    contract = describe_configured_metric(metric)
    assert contract.reviewed and contract.resources == ("llm",)
    assert "private_state" not in json.dumps(_roundtrip(contract))
    provider.assert_not_called()
    provider.generate_json.assert_not_called()
    mcp = describe_configured_metric(MCPTraceCompletenessMetric([ToolCall("lookup", {"key": "value"})]))
    assert mcp.reviewed and mcp.group_compatible
    assert "lookup" not in json.dumps(mcp.to_dict())


def test_invalid_config_objects_never_invoke_copy_or_serialization_hooks(no_execution):
    class Hook:
        calls = 0

        def __deepcopy__(self, memo):
            Hook.calls += 1
            _explode()

        def __str__(self):
            Hook.calls += 1
            _explode()

    metric = MCPTraceCompletenessMetric([ToolCall("lookup", {"key": Hook()})])
    contract = describe_configured_metric(metric)
    assert not contract.reviewed
    assert contract.reasons[0].code == ReasonCode.INVALID_CONFIGURATION
    assert Hook.calls == 0
    assert not resolve_metric_contract("exact_match", {"forbidden": Hook()}).reviewed
    assert Hook.calls == 0


def test_factory_composite_children_remain_nested_and_require_all(no_execution):
    from harness_evals.metrics.factory import build_metric

    metric = build_metric(
        "composite",
        {
            "metrics": [
                {
                    "ref": "text",
                    "type": "heuristic",
                    "config": {"kind": "exact_match", "options": {"negate": True, "forbidden": "x"}},
                },
                {"ref": "time", "type": "heuristic", "config": {"kind": "latency"}},
            ]
        },
        allow_code_loading=False,
    )
    contract = describe_configured_metric(metric)
    assert contract.reviewed
    assert contract.inputs.required == {MetricInput.OUTPUT, MetricInput.LATENCY_MS}
    assert contract.requirements.op == "all" and len(contract.requirements.children) == 2
    assert all(child.op == "all" for child in contract.requirements.children)
    assert not contract.group_compatible
    _roundtrip(contract)
    metric.sub_metrics[0] = metric
    assert not describe_configured_metric(metric).reviewed


def test_factory_composite_keeps_compatible_repeated_execution(no_execution):
    from harness_evals.metrics.factory import build_metric

    metric = build_metric(
        "composite",
        {
            "metrics": [
                {"ref": "first", "type": "heuristic", "config": {"kind": "outcome_consistency"}},
                {"ref": "second", "type": "heuristic", "config": {"kind": "outcome_consistency", "options": {"k": 7}}},
            ]
        },
        allow_code_loading=False,
    )
    contract = describe_configured_metric(metric)
    assert contract.reviewed and contract.execution == ExecutionKind.REPEATED_RUNS
    assert contract.min_runs == 2
    assert contract.inputs.required == {MetricInput.RUNS}
    assert len(contract.requirements.children) == 2
    _roundtrip(contract)


def test_factory_composite_rejects_incompatible_execution(no_execution):
    from harness_evals.metrics.factory import build_metric

    metric = build_metric(
        "composite",
        {
            "metrics": [
                {"ref": "case", "type": "heuristic", "config": {"kind": "latency"}},
                {"ref": "dataset", "type": "heuristic", "config": {"kind": "calibration"}},
            ]
        },
        allow_code_loading=False,
    )
    contract = describe_configured_metric(metric)
    assert not contract.reviewed
    assert contract.reasons[0].code == ReasonCode.UNSUPPORTED_EXECUTION


def test_catalog_json_additions_are_deterministic_and_not_selected(no_execution):
    for entry in catalog():
        snapshot = entry.to_dict()
        assert snapshot == entry.to_dict()
        assert json.loads(json.dumps(snapshot, allow_nan=False)) == snapshot
        assert snapshot["description_kind"] == "catalog"
        assert "metric_class" not in snapshot
        assert "group" not in snapshot["applicability"]["online_scopes"]
        assert snapshot["projection_kind"] == "class_declaration"
        assert snapshot["applicability_kind"] == "possibilities"
        assert snapshot["eval_inputs"]["required"] == sorted(field.value for field in entry.eval_inputs.required)
        assert snapshot["eval_inputs"]["any_of"] == [
            sorted(field.value for field in group) for group in entry.eval_inputs.any_of
        ]


def test_catalog_plugin_collision_is_discoverable_but_never_admitted(monkeypatch, no_execution):
    from harness_evals import plugins

    class Plugin(ExactMatchMetric):
        pass

    monkeypatch.setattr(plugins, "registered_metrics", lambda: {"exact_match": Plugin, "custom_plugin": Plugin})
    entries = {entry.kind: entry for entry in catalog()}
    for kind in ("exact_match", "custom_plugin"):
        entry = entries[kind]
        assert entry.metric_class is Plugin
        assert not entry.to_dict()["configured_contract_supported"]
        assert not entry.configured_contract().reviewed
    assert resolve_metric_contract("exact_match").reviewed
    assert not resolve_metric_contract("custom_plugin").reviewed


def test_resolution_does_not_recurse_into_catalog_or_discover_plugins(monkeypatch, no_execution):
    from harness_evals import plugins
    from harness_evals.catalog import CatalogEntry

    monkeypatch.setattr(CatalogEntry, "to_dict", _explode)
    monkeypatch.setattr(plugins, "registered_metrics", _explode)
    assert resolve_metric_contract("hallucination", {"include_messages_as_reference": True}).reviewed
    assert resolve_metric_contract("exact_match").reviewed


def test_cached_instance_declaration_is_not_the_current_builtin_configuration(no_execution):
    metric = ExactMatchMetric()
    metric.eval_inputs = MetricInputs.unknown()
    assert describe_configured_metric(metric).to_dict() == resolve_metric_contract("exact_match").to_dict()


@pytest.mark.parametrize(
    "kind,initial,changes",
    [
        ("exact_match", {"negate": True, "forbidden": "x"}, {"negate": False, "forbidden": None}),
        ("contains", {"negate": True, "forbidden": "x"}, {"negate": False, "forbidden": None}),
        ("regex", {"negate": True, "forbidden": "x"}, {"negate": False, "forbidden": None}),
        ("exact_match", {}, {"negate": True, "forbidden": "x"}),
        ("hallucination", {}, {"include_messages_as_reference": True}),
        ("decision_noul", {"instructions": "Accept?"}, {"state_field": "metadata.state", "mode": "correctness"}),
        ("structural_similarity", {}, {"output_field": "metadata.actual", "expected_field": "metadata.reference"}),
        ("tool_argument_match", {}, {"ignore_keys": {"timestamp"}}),
    ],
)
def test_describe_rederives_current_configuration_and_projection(kind, initial, changes, no_execution):
    from harness_evals.validation.contracts import _construct_metric

    metric = _construct_metric(kind, _build_registry()[kind], initial)
    for name, value in changes.items():
        setattr(metric, name, value)
    original_inputs = metric.eval_inputs
    selected = {**initial, **changes}
    if kind == "tool_argument_match":
        selected["ignore_keys"] = ["timestamp"]
    described = describe_configured_metric(metric)
    resolved = resolve_metric_contract(kind, selected)
    assert described.reviewed and resolved.reviewed
    assert _roundtrip(described) == _roundtrip(resolved)
    assert metric.eval_inputs == original_inputs  # Description never repairs/mutates the caller's metric.


@pytest.mark.parametrize(
    "kind,initial,name,value",
    [
        ("exact_match", {}, "negate", 1),
        ("exact_match", {}, "forbidden", object()),
        ("exact_match", {}, "threshold", float("nan")),
        ("hallucination", {}, "include_messages_as_reference", "yes"),
        ("structural_similarity", {}, "output_field", float("nan")),
        ("structural_similarity", {}, "expected_field", float("inf")),
        ("structural_similarity", {}, "output_field", "$..output"),
        ("structural_similarity", {}, "level", "unsupported"),
        ("structural_similarity", {}, "format", "unsupported"),
        ("structural_similarity", {}, "extra_keys_penalty", float("nan")),
        ("structural_similarity", {}, "required_fields", {"value": float("nan")}),
        ("decision_noul", {"instructions": "Accept?"}, "instructions", 3),
        ("tool_correctness", {}, "mode", "unsupported"),
        ("tool_correctness", {}, "mode", []),
        ("mcp_trace_completeness", {"expected_trace": []}, "expected_trace", None),
        ("mcp_trace_completeness", {"expected_trace": []}, "expected_trace", [{"name": "lookup"}]),
        ("mcp_trace_completeness", {"expected_trace": []}, "expected_trace", [ToolCall("")]),
        ("mcp_trace_completeness", {"expected_trace": []}, "expected_trace", [ToolCall("lookup")]),
        ("mcp_trace_completeness", {"expected_trace": []}, "expected_trace", [ToolCall("lookup", [])]),
        ("mcp_trace_completeness", {"expected_trace": []}, "expected_trace", [ToolCall("lookup", {}, [])]),
        ("mcp_trace_completeness", {"expected_trace": []}, "expected_trace", [ToolCall("lookup", {"x": float("nan")})]),
    ],
)
def test_mutated_builtin_configuration_fails_closed_and_stays_strict_json(kind, initial, name, value, no_execution):
    from harness_evals.validation.contracts import _construct_metric

    metric = _construct_metric(kind, _build_registry()[kind], initial)
    setattr(metric, name, value)
    contract = describe_configured_metric(metric)
    assert not contract.reviewed
    assert contract.reasons[0].code == ReasonCode.INVALID_CONFIGURATION
    _roundtrip(contract)


def _decision_sub(**updates):
    return {
        "name": "label",
        "weight": 1,
        "state_field": "metadata.state",
        "question": {"type": "choice", "instructions": "Choose", "criteria": {"a": None, "b": None}},
        **updates,
    }


@pytest.mark.parametrize(
    "updates",
    [
        {"weight": "broken"},
        {"weight": True},
        {"weight": None},
        {"weight": float("nan")},
        {"weight": float("inf")},
        {"weight": 0},
        {"weight": -1},
        {"name": ""},
        {"name": 2},
        {"unsupported": "private-config"},
        {"invert": 1},
        {"skip_when_missing": "true"},
        {"question": {"type": "choice", "instructions": 42, "criteria": {"a": None}}},
        {"question": {"type": "choice", "instructions": None, "criteria": {"a": None}}},
        {"question": {"type": "choice", "instructions": "Choose", "criteria": {"a": 42}}},
        {"question": {"type": "choice", "instructions": "Choose", "criteria": {"": None}}},
        {"question": {"type": "score", "instructions": "Rate", "criteria": [0, 1]}},
        {"question": {"type": "score", "instructions": "Rate", "criteria": ["Only one"]}},
        {"question": {"type": "noul", "instructions": "Accept?", "criteria": {"yes": 1}}},
        {"question": {"type": "noul", "instructions": "Accept?", "extra": True}},
    ],
)
def test_decision_composite_validates_entire_children_before_admission(updates, no_execution):
    contract = resolve_metric_contract("decision_composite", {"sub_scores": [_decision_sub(**updates)]})
    assert not contract.reviewed
    assert contract.reasons[0].code == ReasonCode.INVALID_CONFIGURATION
    reasons = json.dumps([reason.to_dict() for reason in contract.reasons])
    assert "private-config" not in reasons and "metadata.state" not in reasons and "label" not in reasons
    _roundtrip(contract)


@pytest.mark.parametrize(
    "subs",
    [
        [_decision_sub(), _decision_sub()],
        [_decision_sub(weight=1e308), _decision_sub(name="second", weight=1e308)],
        [_decision_sub(weight=1), _decision_sub(name="second", weight=-1)],
        [_decision_sub(weight=-1), _decision_sub(name="second", weight=2)],
        [{key: value for key, value in _decision_sub().items() if key != "weight"}],
    ],
)
def test_decision_composite_names_and_fold_weight_totals(subs, no_execution):
    contract = resolve_metric_contract("decision_composite", {"sub_scores": subs})
    assert not contract.reviewed
    assert contract.reasons[0].code == ReasonCode.INVALID_CONFIGURATION


@pytest.mark.parametrize(
    "question",
    [
        ChoiceQuestion(42, {"a": None}),
        ChoiceQuestion("Choose", {"a": True}),
        ScoreQuestion("Rate", [1, 2]),
        NoulQuestion("Accept?", {"yes": False}),
    ],
)
def test_describe_decision_composite_rejects_invalid_typed_questions(question, no_execution):
    from harness_evals.metrics import DecisionCompositeMetric

    metric = DecisionCompositeMetric(object(), [_decision_sub(question=question)])
    contract = describe_configured_metric(metric)
    assert not contract.reviewed
    assert contract.reasons[0].code == ReasonCode.INVALID_CONFIGURATION
    _roundtrip(contract)


def test_decision_composite_preserves_all_skippable_children_and_fold_defaults(no_execution):
    contract = resolve_metric_contract(
        "decision_composite",
        {"sub_scores": [_decision_sub(skip_when_missing=True), _decision_sub(name="second", weight=0)]},
    )
    assert contract.reviewed
    assert contract.requirements.op == "all"
    assert contract.requirements.parameters == {"minimum_active": 1, "active_weights": [1.0, 0.0]}
    assert len(contract.requirements.children) == 2
    assert contract.requirements.children[0].parameters["skip_when_missing"] is True
    assert contract.requirements.children[1].parameters["skip_when_missing"] is False
    _roundtrip(contract)


def _schema_configuration(kind, schema):
    if kind == "schema_validation":
        return {"schema": schema}
    if kind == "structural_similarity":
        return {"level": "schema_validated", "schema_validator": {"type": "json_schema", "schema": schema}}
    return {
        "sub_scores": [
            {"name": "schema", "weight": 1, "check": {"type": "json_schema", "field": "output", "schema": schema}}
        ]
    }


@pytest.mark.parametrize("kind", ["schema_validation", "composite", "structural_similarity"])
@pytest.mark.parametrize(
    "schema",
    [
        {"type": "nonsense"},
        {"type": "object", "properties": {"nested": {"type": "nonsense"}}},
        {"type": "object", "required": "not-an-array"},
        {"$ref": "https://invalid.example/schema"},
        {"$dynamicRef": "https://invalid.example/schema#value"},
        {"$ref": "file:///private/schema.json"},
        {"$defs": {"nested": {"$ref": "other.json"}}},
    ],
)
def test_invalid_schemas_and_external_references_rejected_without_io(kind, schema, no_execution):
    options = _schema_configuration(kind, schema)
    contract = resolve_metric_contract(kind, options)
    assert not contract.reviewed
    assert contract.reasons[0].code == ReasonCode.INVALID_CONFIGURATION
    reasons = json.dumps([reason.to_dict() for reason in contract.reasons])
    assert "invalid.example" not in reasons and "private/schema" not in reasons
    from harness_evals.validation.contracts import _construct_metric

    metric = _construct_metric(kind, _build_registry()[kind], options)
    assert describe_configured_metric(metric).reasons[0].code == ReasonCode.INVALID_CONFIGURATION
    _roundtrip(contract)


@pytest.mark.parametrize("kind", ["schema_validation", "composite", "structural_similarity"])
@pytest.mark.parametrize(
    "schema",
    [
        {"$defs": {"local": {"type": "object"}}, "$ref": "#/$defs/local"},
        {"$dynamicAnchor": "local", "$dynamicRef": "#local", "type": "object"},
        {"type": "object", "properties": {"value": {"type": "string"}}},
    ],
)
def test_local_schemas_and_internal_references_are_pure(kind, schema, no_execution):
    from harness_evals.validation.contracts import _construct_metric

    options = _schema_configuration(kind, schema)
    resolved = resolve_metric_contract(kind, options)
    described = describe_configured_metric(_construct_metric(kind, _build_registry()[kind], options))
    assert resolved.reviewed and described.reviewed
    assert _roundtrip(resolved) == _roundtrip(described)


@pytest.mark.parametrize("kind", ["schema_validation", "composite", "structural_similarity"])
@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": {"$ref": {"type": "string"}}},
        {"properties": {"$dynamicRef": {}, "$recursiveRef": {}, "$id": {}, "$anchor": {}}},
        {"$defs": {"$ref": {"type": "string"}}, "$ref": "#/$defs/$ref"},
        {"const": {"$ref": "https://invalid.example/literal"}},
        {"enum": [{"$ref": "file:///private/literal", "$anchor": "literal"}]},
        {"default": {"$ref": "missing.json"}, "examples": [{"$ref": 42}]},
        {"extension": {"$ref": "missing.json", "$id": "ignored", "$schema": "ignored"}},
        {"$defs": {"named": {"$anchor": "local", "const": {"$ref": "literal"}}}, "$ref": "#local"},
        {
            "$schema": "http://json-schema.org/draft-04/schema#",
            "definitions": {"node": {"id": "#node"}},
            "$ref": "#node",
        },
        {"$schema": "https://json-schema.org/draft/2019-09/schema", "$recursiveAnchor": True, "$recursiveRef": "#"},
        {"$defs": {"node": {"$id": "nested", "$defs": {"s": {"type": "string"}}, "$ref": "#/$defs/s"}}},
        {"$defs": {"node": {"$anchor": "local", "properties": {"next": {"$ref": "#local"}}}}},
    ],
)
def test_schema_positions_distinguish_property_names_anchors_and_literals(kind, schema, no_execution):
    from harness_evals.validation.contracts import _construct_metric

    options = _schema_configuration(kind, schema)
    configured = resolve_metric_contract(kind, options)
    described = describe_configured_metric(_construct_metric(kind, _build_registry()[kind], options))
    assert configured.reviewed and described.reviewed
    assert _roundtrip(configured) == _roundtrip(described)
    assert validate_metric(configured, EvalCase(input="", output={}, expected={})).ready


@pytest.mark.parametrize("kind", ["schema_validation", "composite", "structural_similarity"])
@pytest.mark.parametrize(
    "schema",
    [
        {"properties": {"$ref": {"$ref": "https://invalid.example/schema"}}},
        {"patternProperties": {"$ref": {"$dynamicRef": "file:///private/schema"}}},
        {"additionalProperties": {"$ref": "missing.json"}},
        {"items": {"$ref": "missing.json"}},
        {"prefixItems": [{"$ref": "missing.json"}]},
        {"allOf": [{"$ref": "missing.json"}]},
        {"anyOf": [{"$ref": "missing.json"}]},
        {"oneOf": [{"$ref": "missing.json"}]},
        {"not": {"$ref": "missing.json"}},
        {"if": {"$ref": "missing.json"}},
        {"then": {"$ref": "missing.json"}},
        {"else": {"$ref": "missing.json"}},
        {"contains": {"$ref": "missing.json"}},
        {"propertyNames": {"$ref": "missing.json"}},
        {"unevaluatedProperties": {"$ref": "missing.json"}},
        {"unevaluatedItems": {"$ref": "missing.json"}},
        {"dependentSchemas": {"$ref": {"$ref": "missing.json"}}},
        {"contentSchema": {"$ref": "missing.json"}},
        {"$schema": "http://json-schema.org/draft-07/schema#", "items": [{"$ref": "missing.json"}]},
        {"$schema": "http://json-schema.org/draft-07/schema#", "additionalItems": {"$ref": "missing.json"}},
        {
            "$schema": "http://json-schema.org/draft-04/schema#",
            "dependencies": {"a": ["b"], "b": {"$ref": "missing.json"}},
        },
        {"$schema": "http://json-schema.org/draft-03/schema#", "extends": {"$ref": "missing.json"}},
        {"$schema": "http://json-schema.org/draft-03/schema#", "type": [{"$ref": "missing.json"}]},
        {"$schema": "http://json-schema.org/draft-03/schema#", "disallow": [{"$ref": "missing.json"}]},
        {"$schema": "https://json-schema.org/draft/2019-09/schema", "items": {"$recursiveRef": "missing.json"}},
        {
            "$defs": {
                "old": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "dependencies": {"a": {"$ref": "missing.json"}},
                }
            }
        },
        {"extension": {"$ref": "https://invalid.example/schema"}, "$ref": "#/extension"},
        {"const": {"$ref": "file:///private/schema"}, "$ref": "#/const"},
        {"enum": [{"$ref": "missing.json"}], "$ref": "#/enum/0"},
        {"const": {"$anchor": "literal"}, "$ref": "#literal"},
        {"properties": {"$anchor": {"type": "string"}, "local": {}}, "$ref": "#local"},
        {"$defs": {"named": {"$id": "nested", "$anchor": "local"}}, "$ref": "#local"},
    ],
)
def test_nested_schema_references_cannot_hide_in_applicable_positions(kind, schema, no_execution):
    configured = resolve_metric_contract(kind, _schema_configuration(kind, schema))
    assert not configured.reviewed
    assert configured.reasons[0].code == ReasonCode.INVALID_CONFIGURATION
    _roundtrip(configured)


def test_schema_json_string_is_checked_after_parsing(no_execution):
    assert not resolve_metric_contract("schema_validation", {"schema": '{"type":"nonsense"}'}).reviewed
    assert not resolve_metric_contract("schema_validation", {"schema": '{"maximum":NaN}'}).reviewed
    assert resolve_metric_contract("schema_validation", {"schema": '{"type":"object"}'}).reviewed


def test_factory_composite_rederives_mutated_child_configuration(no_execution):
    from harness_evals.metrics.factory import build_metric

    metric = build_metric(
        "composite",
        {
            "metrics": [
                {
                    "ref": "child",
                    "type": "heuristic",
                    "config": {"kind": "exact_match", "options": {"negate": True, "forbidden": "x"}},
                }
            ]
        },
        allow_code_loading=False,
    )
    metric.sub_metrics[0].negate = False
    metric.sub_metrics[0].forbidden = None
    contract = describe_configured_metric(metric)
    assert contract.reviewed
    assert contract.inputs.required == {MetricInput.OUTPUT, MetricInput.EXPECTED}
    _roundtrip(contract)


@pytest.mark.parametrize(
    "output,expected,metadata,ready",
    [
        ("{}", "{}", None, True),
        ("{", {}, None, False),
        ({}, {}, None, False),  # A missing selected path falls back only to a string.
        ("{}", "{}", {"actual": "{", "reference": {}}, False),
        ("{", "{}", {"actual": {}, "reference": {}}, True),  # Selected operands win over defaults.
        ("{}", "{}", {"actual": {"value": float("nan")}, "reference": {}}, False),
        ("{}", None, None, False),
    ],
)
def test_resolved_structural_rules_select_parse_and_validate_both_operands(
    output, expected, metadata, ready, no_execution
):
    contract = resolve_metric_contract(
        "structural_similarity",
        {"format": "json", "output_field": "metadata.actual", "expected_field": "metadata.reference"},
    )
    restored = ConfiguredMetricContract.from_dict(_roundtrip(contract))
    result = validate_metric(restored, EvalCase(input="", output=output, expected=expected, metadata=metadata))
    assert result.ready is ready


def test_schema_validated_structural_rule_still_requires_expected(no_execution):
    contract = resolve_metric_contract(
        "structural_similarity", _schema_configuration("structural_similarity", {"type": "object"})
    )
    result = validate_metric(contract, EvalCase(input="", output={}))
    assert not result.ready
    assert ReasonCode.MISSING_EVIDENCE in {reason.code for reason in result.reasons}


def test_resolved_decision_composite_rejects_all_skipped_evidence(no_execution):
    contract = resolve_metric_contract("decision_composite", {"sub_scores": [_decision_sub(skip_when_missing=True)]})
    missing = validate_metric(contract, EvalCase(input="", output=""))
    assert not missing.ready
    assert ReasonCode.INSUFFICIENT_EVIDENCE in {reason.code for reason in missing.reasons}
    active = validate_metric(contract, EvalCase(input="", output="", metadata={"state": "answer"}))
    assert active.ready


def test_snapshots_are_call_owned_and_versioned(no_execution):
    contract = resolve_metric_contract(
        "hallucination", {"include_sse_events_as_reference": True, "sse_reference_events": ["ref"]}
    )
    snapshot = contract.to_dict()
    snapshot["requirements"]["children"][1]["parameters"]["sse_reference_events"].append("mutated")
    assert contract.requirements.children[1].parameters["sse_reference_events"] == ["ref"]
    assert resolve_metric_contract("hallucination").requirements.children[1].parameters["sse_reference_events"] == []
    snapshot["schema_version"] = 999
    with pytest.raises(ValueError, match="version"):
        ConfiguredMetricContract.from_dict(snapshot)


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("weights", [float("nan")]),
        ("weights", [float("inf")]),
        ("weights", []),
        ("weights", [0]),
        ("weights", [-1]),
        ("weights", [True]),
        ("aggregation", "not_an_aggregation"),
    ],
)
def test_factory_composite_checks_current_weights_and_aggregation(attribute, value, no_execution):
    from harness_evals.metrics.factory import build_metric

    metric = build_metric(
        "composite",
        {
            "aggregation": "weighted_average",
            "metrics": [{"ref": "x", "type": "heuristic", "config": {"kind": "exact_match"}}],
        },
        allow_code_loading=False,
    )
    assert describe_configured_metric(metric).reviewed
    setattr(metric, attribute, value)
    rejected = describe_configured_metric(metric)
    assert not rejected.reviewed
    assert rejected.reasons[0].code == ReasonCode.INVALID_CONFIGURATION


def test_decision_composite_requires_positive_usable_weight_after_skips(no_execution):
    contract = resolve_metric_contract(
        "decision_composite",
        {
            "sub_scores": [
                _decision_sub(skip_when_missing=True),
                _decision_sub(name="zero", weight=0, state_field="metadata.other"),
            ]
        },
    )
    restored = ConfiguredMetricContract.from_dict(_roundtrip(contract))
    assert contract.reviewed
    result = validate_metric(restored, EvalCase(input="", output="", metadata={"other": "observed"}))
    assert not result.ready
    assert ReasonCode.INSUFFICIENT_EVIDENCE in {reason.code for reason in result.reasons}
    assert validate_metric(
        restored, EvalCase(input="", output="", metadata={"state": "observed", "other": "observed"})
    ).ready


@pytest.mark.parametrize("kind", ["schema_validation", "composite", "structural_similarity"])
@pytest.mark.parametrize("schema", [{"$ref": "#/missing"}, {"$ref": "#missing"}, {"x": {"type": 42}, "$ref": "#/x"}])
def test_dangling_or_invalid_local_schema_targets_are_configuration_errors(kind, schema, no_execution):
    configured = resolve_metric_contract(kind, _schema_configuration(kind, schema))
    assert not configured.reviewed
    assert configured.reasons[0].code == ReasonCode.INVALID_CONFIGURATION
    _roundtrip(configured)


@pytest.mark.parametrize("kind", ["schema_validation", "composite", "structural_similarity"])
@pytest.mark.parametrize(
    "schema",
    [
        {"$defs": {"a/b": {"type": "object"}}, "$ref": "#/$defs/a~1b"},
        {"$defs": {"named": {"$anchor": "known", "type": "object"}}, "$ref": "#known"},
        {"$defs": {"false": False}, "$ref": "#/$defs/false"},
    ],
)
def test_valid_local_schema_targets_do_not_require_external_resolution(kind, schema, no_execution):
    configured = resolve_metric_contract(kind, _schema_configuration(kind, schema))
    assert configured.reviewed
    assert validate_metric(configured, EvalCase(input="", output={}, expected={})).ready


@pytest.mark.parametrize(
    "kind,options",
    [
        ("rubric_judge", {"rubric": {"-1": "Poor", "1": "Good"}}),
        ("rubric_judge", {"rubric": None}),
        (
            "geval",
            {
                "rubric": [
                    {"min_score": 0, "max_score": 2, "description": "Limited"},
                    {"min_score": 3, "max_score": 5, "description": "Complete"},
                ]
            },
        ),
    ],
)
def test_json_judge_rubrics_resolve_without_provider_construction(kind, options, no_execution):
    configured = resolve_metric_contract(kind, options)
    assert configured.reviewed
    assert validate_metric(configured, EvalCase(input="task", output="answer")).ready
    _roundtrip(configured)


def test_existing_rubric_judge_rebuilds_integer_keys_without_serializing_provider(no_execution):
    from harness_evals.metrics import RubricJudgeMetric

    configured = describe_configured_metric(RubricJudgeMetric(object(), rubric={-1: "Poor", 1: "Good"}))
    assert configured.reviewed
    _roundtrip(configured)
