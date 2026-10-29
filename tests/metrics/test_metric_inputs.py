"""Metric input declarations, their derived applicability, and their accuracy."""

from __future__ import annotations

import abc
import inspect

import pytest

import harness_evals.metrics as builtin_metrics
from harness_evals.applicability import applicability_for_inputs
from harness_evals.catalog import CatalogEntry, _build_registry, catalog
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import (
    BaseMetric,
    Dimension,
    EvaluationMode,
    MetricApplicability,
    MetricInput,
    MetricInputs,
    OnlineScope,
)
from harness_evals.core.score import Score
from harness_evals.core.types import Message, ToolCall
from harness_evals.decision.base import BaseDecisionProvider
from harness_evals.decision.types import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)
from harness_evals.llm.embedding import BaseEmbedding
from harness_evals.metrics import (
    ChoiceMetric,
    CompositeMetric,
    ContainsMetric,
    DecisionCompositeMetric,
    ExactMatchMetric,
    GEvalMetric,
    HallucinationMetric,
    NoulMetric,
    RegexMetric,
    RoleAdherenceMetric,
    ScoreMetric,
    StructuralSimilarityMetric,
    WebhookMetric,
)
from harness_evals.metrics.llm_judge.dag import BinaryJudgementNode, DeepAcyclicGraph, VerdictNode
from harness_evals.plugins import register_metric
from tests.conftest import MockLLM

INPUT, OUTPUT, EXPECTED, CONTEXT = MetricInput.INPUT, MetricInput.OUTPUT, MetricInput.EXPECTED, MetricInput.CONTEXT
ONLINE_EVERYWHERE = MetricApplicability(
    modes=frozenset({EvaluationMode.OFFLINE, EvaluationMode.ONLINE}),
    online_scopes=frozenset({OnlineScope.TRACE, OnlineScope.SPAN}),
)
OFFLINE_ONLY = MetricApplicability()


def _custom_metric(**namespace: object) -> type[BaseMetric]:
    def __init__(self):
        BaseMetric.__init__(self, name="custom", dimension=Dimension.CORRECTNESS)

    def measure(self, eval_case):
        return Score(name="custom", value=1.0, threshold=1.0)

    return type("CustomMetric", (BaseMetric,), {"__init__": __init__, "measure": measure, **namespace})


class _Provider(BaseDecisionProvider):
    async def a_ask(self, state, questions):
        answers = {}
        for name, question in questions.items():
            if isinstance(question, ChoiceQuestion):
                answers[name] = ChoiceAnswer(choice="Paris", confidence=0.9, probabilities={"Paris": 0.9})
            elif isinstance(question, ScoreQuestion):
                answers[name] = ScoreAnswer(score=1, confidence=0.9, legend={}, probabilities={1: 0.9})
            else:
                answers[name] = NoulAnswer(noul=0.9)
        return DecisionResponse(model="fake", answers=answers, input_tokens=1, output_tokens=1)


class _Embedding(BaseEmbedding):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


@pytest.mark.unit
class TestMetricInputs:
    def test_coerces_values_and_derives_fields(self):
        inputs = MetricInputs(required=["output"], optional=[MetricInput.EXPECTED])

        assert inputs.required == frozenset({OUTPUT})
        assert inputs.optional == frozenset({EXPECTED})
        assert inputs.fields == frozenset({OUTPUT, EXPECTED})

    def test_rejects_a_field_that_is_both_required_and_optional(self):
        with pytest.raises(ValueError, match="both required and optional: output"):
            MetricInputs(required={OUTPUT}, optional={OUTPUT})

    def test_rejects_a_bare_string(self):
        with pytest.raises(TypeError, match="not a string"):
            MetricInputs(required="output")

    def test_rejects_an_unknown_field(self):
        with pytest.raises(ValueError):
            MetricInputs(required=["outputs"])

    def test_unknown_requires_every_field(self):
        assert MetricInputs.unknown() == MetricInputs(required=MetricInput)

    def test_union_keeps_a_field_required_when_any_declaration_requires_it(self):
        combined = MetricInputs.union(
            MetricInputs(required={OUTPUT}, optional={EXPECTED}), MetricInputs(required={EXPECTED, CONTEXT})
        )

        assert combined == MetricInputs(required={OUTPUT, EXPECTED, CONTEXT})

    @pytest.mark.parametrize(
        ("path", "field"),
        [
            ("output", OUTPUT),
            ("metadata.user.id", MetricInput.METADATA),
            ("messages[0].content", MetricInput.MESSAGES),
            ("$.tool_calls[*].name", MetricInput.TOOL_CALLS),
            ("$.tool_calls[?(@.name == 'search')].input", MetricInput.TOOL_CALLS),
            ("messages.-1.content", MetricInput.MESSAGES),
            ("$..id", None),
            ("metadata..id", None),
            ("$.output.a | $.expected", None),
            ("$.*", None),
            ("$", None),
            ("", None),
            ("unknown_field", None),
        ],
    )
    def test_from_path_resolves_the_root_field(self, path, field):
        assert MetricInput.from_path(path) is field

    def test_from_paths_falls_back_to_unknown_for_an_unresolvable_path(self):
        assert MetricInputs.from_paths(required=["output", "$..id"]) == MetricInputs.unknown()

    def test_from_paths_drops_optional_paths_that_are_also_required(self):
        assert MetricInputs.from_paths(required=["output.a"], optional=["output.b", "expected"]) == MetricInputs(
            required={OUTPUT}, optional={EXPECTED}
        )


@pytest.mark.unit
class TestDeclarationContract:
    def test_undeclared_metric_requires_every_field(self):
        assert _custom_metric().eval_inputs == MetricInputs.unknown()

    def test_declared_metric_exposes_its_declaration(self):
        inputs = MetricInputs(required={OUTPUT})

        assert _custom_metric(eval_inputs=inputs).eval_inputs is inputs

    def test_invalid_declaration_fails_at_class_creation(self):
        with pytest.raises(
            TypeError, match=r"CustomMetric\.eval_inputs must be a MetricInputs instance, got frozenset"
        ):
            _custom_metric(eval_inputs=frozenset({OUTPUT}))

    def test_subclass_that_keeps_scoring_inherits_the_declaration(self):
        class SubGEval(GEvalMetric):
            criteria = "Is the answer correct?"

        assert SubGEval.eval_inputs is GEvalMetric.eval_inputs

    def test_subclass_that_overrides_scoring_reverts_to_unknown(self):
        class SubGEval(GEvalMetric):
            def measure(self, eval_case):
                return Score(name=self.name, value=float(bool(eval_case.context)), threshold=self.threshold)

        assert SubGEval.eval_inputs == MetricInputs.unknown()

    def test_implementing_an_abstract_hook_inherits_the_declaration(self):
        declared = MetricInputs(required={MetricInput.RUNS})

        class Base(BaseMetric):
            eval_inputs = declared

            @abc.abstractmethod
            def measure_runs(self, runs): ...

        class Concrete(Base):
            def measure_runs(self, runs):
                return Score(name="runs", value=1.0, threshold=1.0)

        assert Concrete.eval_inputs is declared

    def test_configuration_does_not_override_a_subclass_declaration(self):
        declared = MetricInputs(required={MetricInput.METADATA})

        class SubChoice(ChoiceMetric):
            eval_inputs = declared

        metric = SubChoice(provider=_Provider(), instructions="?", criteria={"a": None}, state_field="tags")

        assert metric.eval_inputs is declared

    @pytest.mark.parametrize("indirect", [False, True])
    def test_configuration_preserves_an_explicitly_reused_parent_declaration(self, indirect):
        class SubChoice(ChoiceMetric):
            eval_inputs = ChoiceMetric.eval_inputs

            async def a_measure(self, eval_case):
                return Score(name=self.name, value=float(bool(eval_case.output)), threshold=self.threshold)

        class Child(SubChoice):
            pass

        metric_class = Child if indirect else SubChoice
        metric = metric_class(
            provider=_Provider(), instructions="?", criteria={"a": None}, state_field="metadata.answer"
        )

        assert metric.eval_inputs is SubChoice.eval_inputs
        assert OUTPUT in metric.eval_inputs.required
        assert MetricInput.METADATA not in metric.eval_inputs.fields
        assert metric.measure(EvalCase(input="q", output="answer")).value == 1.0

    def test_configuration_still_narrows_a_transparent_subclass(self):
        class SubChoice(ChoiceMetric):
            pass

        metric = SubChoice(provider=_Provider(), instructions="?", criteria={"a": None}, state_field="tags.answer")

        assert metric.eval_inputs == MetricInputs(required={MetricInput.TAGS}, optional={EXPECTED})

    def test_configuration_does_not_narrow_a_reverted_subclass(self):
        class SubChoice(ChoiceMetric):
            async def a_measure(self, eval_case):
                return Score(name=self.name, value=1.0, threshold=self.threshold)

        metric = SubChoice(provider=_Provider(), instructions="?", criteria={"a": None})

        assert metric.eval_inputs == MetricInputs.unknown()

    def test_every_public_builtin_declares_its_inputs(self):
        configuration_defined = {CompositeMetric, DecisionCompositeMetric}
        undeclared = [
            metric.__name__
            for name in builtin_metrics.__all__
            if isinstance(metric := getattr(builtin_metrics, name), type)
            and issubclass(metric, BaseMetric)
            and metric not in configuration_defined
            and metric.eval_inputs == MetricInputs.unknown()
        ]

        assert undeclared == []


@pytest.mark.unit
class TestConfiguredInputs:
    def test_decision_primitives_read_their_state_field(self):
        for metric_class in (ChoiceMetric, ScoreMetric, NoulMetric):
            metric = metric_class(
                provider=_Provider(), instructions="?", criteria=["a", "b"], state_field="metadata.answer"
            )

            assert metric.eval_inputs == MetricInputs(required={MetricInput.METADATA}, optional={EXPECTED})

    def test_correctness_mode_requires_expected(self):
        metric = ChoiceMetric(provider=_Provider(), instructions="?", criteria={"a": None}, mode="correctness")

        assert metric.eval_inputs == MetricInputs(required={OUTPUT, EXPECTED})

    def test_unresolvable_state_field_reads_every_field(self):
        metric = NoulMetric(provider=_Provider(), instructions="?", state_field="$..answer")

        assert metric.eval_inputs == MetricInputs.unknown()

    def test_composite_reads_its_sub_check_paths(self):
        metric = CompositeMetric(
            sub_scores=[
                {"name": "tools", "weight": 1.0, "check": {"type": "count_match", "field": "tool_calls"}},
                {
                    "name": "tagged",
                    "weight": 1.0,
                    "skip_when_missing": True,
                    "check": {"type": "field_exists", "field": "tags.suite"},
                },
            ]
        )

        assert metric.eval_inputs == MetricInputs(
            required={OUTPUT, MetricInput.TOOL_CALLS}, optional={EXPECTED, MetricInput.TAGS}
        )

    def test_decision_composite_reads_its_sub_check_paths(self):
        question = NoulQuestion(instructions="?")
        metric = DecisionCompositeMetric(
            provider=_Provider(),
            sub_scores=[
                {
                    "name": "a",
                    "weight": 1.0,
                    "state_field": "output",
                    "question": question,
                    "expected_field": "expected",
                },
                {
                    "name": "b",
                    "weight": 1.0,
                    "state_field": "metadata.plan",
                    "question": question,
                    "skip_when_missing": True,
                },
            ],
        )

        assert metric.eval_inputs == MetricInputs(required={OUTPUT}, optional={EXPECTED, MetricInput.METADATA})

    def test_structural_similarity_reads_its_field_paths(self):
        metric = StructuralSimilarityMetric(output_field="metadata.manifest")

        assert metric.eval_inputs == MetricInputs(required={MetricInput.METADATA, EXPECTED}, optional={OUTPUT})

    def test_negated_checks_need_only_output(self):
        assert ContainsMetric(negate=True, forbidden="secret").eval_inputs == MetricInputs(required={OUTPUT})
        assert ExactMatchMetric().eval_inputs == MetricInputs(required={OUTPUT, EXPECTED})

    def test_custom_webhook_payload_reads_every_field(self):
        metric = WebhookMetric("https://example.invalid", payload_fn=lambda case: {}, use_status_code=True)

        assert metric.eval_inputs == MetricInputs.unknown()

    def test_custom_webhook_payload_can_declare_its_inputs(self):
        declared = MetricInputs(required={OUTPUT})
        metric = WebhookMetric(
            "https://example.invalid", payload_fn=lambda case: {}, eval_inputs=declared, use_status_code=True
        )

        assert metric.eval_inputs is declared

    def test_hallucination_references_follow_configuration(self):
        assert HallucinationMetric(llm=MockLLM()).eval_inputs == MetricInputs(
            required={OUTPUT, CONTEXT}, optional={EXPECTED}
        )
        assert HallucinationMetric(llm=MockLLM(), include_messages_as_reference=True).eval_inputs == MetricInputs(
            required={OUTPUT, MetricInput.MESSAGES}, optional={CONTEXT, EXPECTED}
        )


@pytest.mark.unit
class TestApplicability:
    def test_applicability_follows_required_inputs_only(self):
        assert (
            applicability_for_inputs(MetricInputs(required={INPUT, OUTPUT}, optional={EXPECTED})) == ONLINE_EVERYWHERE
        )
        assert applicability_for_inputs(MetricInputs(required={OUTPUT, EXPECTED})) == OFFLINE_ONLY

    def test_authored_metadata_keeps_a_metric_offline(self):
        assert RoleAdherenceMetric(llm=MockLLM()).applicability == OFFLINE_ONLY

    def test_custom_metric_applicability_follows_its_declaration(self):
        assert _custom_metric()().applicability == OFFLINE_ONLY
        assert _custom_metric(eval_inputs=MetricInputs(required={OUTPUT}))().applicability == ONLINE_EVERYWHERE

    def test_configuration_changes_applicability(self):
        assert ContainsMetric().applicability == OFFLINE_ONLY
        assert ContainsMetric(negate=True, forbidden="secret").applicability == ONLINE_EVERYWHERE
        assert HallucinationMetric(llm=MockLLM()).applicability == OFFLINE_ONLY
        assert HallucinationMetric(llm=MockLLM(), include_messages_as_reference=True).applicability == ONLINE_EVERYWHERE

    @pytest.mark.parametrize("scope", [OnlineScope.TRACE, OnlineScope.SPAN])
    @pytest.mark.parametrize(
        "offline, online",
        [
            pytest.param(ContainsMetric(), ContainsMetric(negate=True, forbidden="secret"), id="contains"),
            pytest.param(ExactMatchMetric(), ExactMatchMetric(negate=True, forbidden="secret"), id="exact-match"),
            pytest.param(RegexMetric(), RegexMetric(negate=True, forbidden="secret"), id="regex"),
            pytest.param(
                HallucinationMetric(llm=MockLLM()),
                HallucinationMetric(llm=MockLLM(), include_messages_as_reference=True),
                id="hallucination",
            ),
            pytest.param(
                StructuralSimilarityMetric(),
                StructuralSimilarityMetric(expected_field="input"),
                id="structural-similarity",
            ),
            pytest.param(
                CompositeMetric(
                    sub_scores=[{"name": "check", "check": {"type": "field_exists", "field": "metadata.x"}}]
                ),
                CompositeMetric(sub_scores=[{"name": "check", "check": {"type": "field_exists", "field": "output.x"}}]),
                id="composite",
            ),
            pytest.param(
                DecisionCompositeMetric(
                    provider=_Provider(),
                    sub_scores=[
                        {"name": "check", "state_field": "metadata.x", "question": NoulQuestion(instructions="?")}
                    ],
                ),
                DecisionCompositeMetric(
                    provider=_Provider(),
                    sub_scores=[{"name": "check", "state_field": "output", "question": NoulQuestion(instructions="?")}],
                ),
                id="decision-composite",
            ),
        ],
    )
    def test_catalog_filters_keep_metrics_with_online_configurations(self, scope, offline, online):
        entries = catalog()
        entry = next(entry for entry in entries if entry.metric_class is type(online))
        visible = {
            entry.kind
            for entry in entries
            if EvaluationMode.ONLINE in entry.applicability.modes and scope in entry.applicability.online_scopes
        }

        assert entry.kind in visible
        assert entry.applicability == ONLINE_EVERYWHERE
        assert offline.applicability == OFFLINE_ONLY
        assert online.applicability == ONLINE_EVERYWHERE
        assert entry.eval_inputs is type(online).eval_inputs

    def test_catalog_override_is_inherited_only_when_scoring_and_inputs_are_preserved(self):
        @register_metric("transparent_contains")
        class Transparent(ContainsMetric):
            pass

        @register_metric("metadata_contains")
        class Metadata(ContainsMetric):
            eval_inputs = MetricInputs(required={MetricInput.METADATA})

        @register_metric("custom_contains")
        class Custom(ContainsMetric):
            def measure(self, eval_case):
                return Score(name=self.name, value=float(bool(eval_case.context)), threshold=self.threshold)

        entries = {entry.kind: entry for entry in catalog()}

        assert entries["transparent_contains"].applicability == ONLINE_EVERYWHERE
        assert entries["metadata_contains"].applicability == OFFLINE_ONLY
        assert entries["custom_contains"].applicability == OFFLINE_ONLY
        assert Metadata(negate=True, forbidden="secret").applicability == OFFLINE_ONLY
        assert Custom(negate=True, forbidden="secret").eval_inputs == MetricInputs.unknown()

    def test_plugin_can_explicitly_declare_catalog_support_without_changing_runtime_inputs(self):
        metric_class = register_metric("configured_plugin")(_custom_metric(catalog_applicability=ONLINE_EVERYWHERE))

        entry = next(entry for entry in catalog() if entry.kind == "configured_plugin")

        assert entry.applicability == ONLINE_EVERYWHERE
        assert entry.eval_inputs == MetricInputs.unknown()
        assert metric_class().applicability == OFFLINE_ONLY

    def test_catalog_reports_class_declarations(self):
        entries = {entry.kind: entry for entry in catalog()}

        assert all(entry.eval_inputs is entry.metric_class.eval_inputs for entry in entries.values())
        assert entries["geval"].applicability == ONLINE_EVERYWHERE
        assert entries["role_adherence"].applicability == OFFLINE_ONLY
        assert all(EvaluationMode.OFFLINE in entry.applicability.modes for entry in entries.values())
        assert not any(OnlineScope.GROUP in entry.applicability.online_scopes for entry in entries.values())

    def test_catalog_entry_accepts_its_original_fields_positionally(self):
        entry = CatalogEntry(
            "custom", "Custom", BaseMetric, Dimension.CORRECTNESS, "custom", 1.0, False, False, False, "A metric."
        )

        assert entry.description == "A metric."
        assert entry.eval_inputs == MetricInputs.unknown()
        assert entry.applicability == OFFLINE_ONLY


_FIELD_NAMES = frozenset(field.value for field in MetricInput)
_reads: set[str] = set()


class _RecordingDict(dict):
    """``EvalCase.to_dict()`` result that records top-level key lookups."""

    def __getitem__(self, key):
        _reads.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        _reads.add(key)
        return super().get(key, default)

    def __contains__(self, key):
        _reads.add(key)
        return super().__contains__(key)


class _RecordingEvalCase(EvalCase):
    """Records each top-level field a metric reads, directly or through ``to_dict``."""

    def __getattribute__(self, name: str):
        if name in _FIELD_NAMES:
            _reads.add(name)
        return super().__getattribute__(name)

    def to_dict(self) -> dict:
        values = {name: object.__getattribute__(self, name) for name in _FIELD_NAMES}
        return _RecordingDict(EvalCase(**values).to_dict())


def _complete_case(**overrides: object) -> _RecordingEvalCase:
    search = ToolCall(name="search", input={"q": "capital of France"}, output="Paris")
    turn = {"retrieval_context": ["Paris is the capital of France."], "expected": "Paris", "latency_ms": 60.0}
    fields: dict[str, object] = dict(
        input="What is the capital of France?",
        output="The capital of France is Paris.",
        expected="Paris",
        context=["Paris is the capital of France."],
        latency_ms=120.0,
        token_count=30,
        input_tokens=10,
        output_tokens=20,
        cost_usd=0.01,
        retry_count=0,
        confidence=0.9,
        messages=[
            Message(role="user", content="What is the capital of France?"),
            Message(role="assistant", content="Let me search.", tool_calls=[search], token_count=12, **turn),
            Message(role="tool", content="Paris"),
            Message(role="user", content="Thanks."),
            Message(role="assistant", content="The capital of France is Paris.", token_count=18, **turn),
        ],
        tool_calls=[search],
        expected_tools=["search"],
        expected_tool_calls=[ToolCall(name="search", input={"q": "capital of France"})],
        tags={"suite": "geography"},
        metadata={
            "chatbot_role": "a geography tutor",
            "expected_turns": 2,
            "expected_outcome": "The user learns the capital.",
            "nominal_passed": True,
            "perturbed_results": [True, False],
        },
        runs=[
            EvalCase(input="q", output="Paris", token_count=30, tool_calls=[search]),
            EvalCase(input="q", output="Paris", token_count=32, tool_calls=[search]),
        ],
    )
    return _RecordingEvalCase(**{**fields, **overrides})


def _assert_reads_match_declaration(metric: BaseMetric, **case_overrides: object) -> None:
    _reads.clear()
    if type(metric).measure_dataset is not BaseMetric.measure_dataset:
        metric.measure_dataset([_complete_case(**case_overrides), _complete_case(**case_overrides)], [True, False])
    else:
        metric.measure(_complete_case(**case_overrides))

    declared = metric.eval_inputs
    assert {MetricInput(name) for name in _reads} - declared.fields == set(), "reads undeclared fields"
    assert {field for field in declared.required if field.value not in _reads} == set(), "never reads a required field"


_CASE_OVERRIDES: dict[str, dict[str, object]] = {
    "numeric_diff": {"output": "42", "expected": "42"},
    "decision_score": {"expected": 1},
    "decision_noul": {"expected": True},
}


_JUDGE_RESPONSE = {
    "score": 1,
    "reason": "ok",
    "reasoning": "ok",
    "verdict": True,
    "verdicts": [],
    "claims": [],
    "statements": [],
    "questions": [],
    "entities": [],
    "requirements": [],
}
_CONSTRUCTOR_OPTIONS: dict[str, dict[str, object]] = {
    "schema_validation": {"schema": {"type": "string"}},
    "composite": {
        "sub_scores": [{"name": "present", "weight": 1.0, "check": {"type": "field_exists", "field": "output"}}]
    },
    "decision_choice": {"instructions": "Pick the city.", "criteria": {"Paris": None, "Lyon": None}},
    "decision_score": {"instructions": "Rate it.", "criteria": ["wrong", "right"]},
    "decision_noul": {"instructions": "Is it correct?"},
    "decision_composite": {
        "sub_scores": [
            {"name": "correct", "weight": 1.0, "state_field": "output", "question": NoulQuestion(instructions="?")}
        ]
    },
    "dag": {
        "dag": DeepAcyclicGraph(
            root_nodes=[
                BinaryJudgementNode(
                    criteria="Is it correct?",
                    children=[VerdictNode(verdict=True, score=1.0), VerdictNode(verdict=False, score=0.0)],
                )
            ]
        )
    },
    "prompt_alignment": {"prompt_instructions": ["Answer concisely."]},
    "role_violation": {"role_description": "A geography tutor."},
    "compliance": {"policies": ["Do not share personal data."]},
    "conversational_geval": {"criteria": "Is the conversation helpful?"},
    "topic_adherence": {"allowed_topics": ["geography"]},
    "mcp_trace_completeness": {"expected_trace": [ToolCall(name="search")]},
    "spec_grounding": {"requirements": ["The answer names the capital."]},
}


def _build(kind: str, metric_class: type[BaseMetric]) -> BaseMetric:
    parameters = inspect.signature(metric_class.__init__).parameters
    options = dict(_CONSTRUCTOR_OPTIONS.get(kind, {}))
    if "llm" in parameters:
        options["llm"] = MockLLM(default=_JUDGE_RESPONSE)
    if "embedding" in parameters:
        options["embedding"] = _Embedding()
    if "provider" in parameters:
        options["provider"] = _Provider()
    return metric_class(**options)


@pytest.mark.unit
@pytest.mark.parametrize(("kind", "metric_class"), _build_registry().items(), ids=str)
def test_builtin_metric_reads_match_its_declaration(kind, metric_class):
    _assert_reads_match_declaration(_build(kind, metric_class), **_CASE_OVERRIDES.get(kind, {}))


_CONFIGURED: dict[str, tuple[object, dict[str, object]]] = {
    "hallucination_messages": (
        lambda: HallucinationMetric(llm=MockLLM(default=_JUDGE_RESPONSE), include_messages_as_reference=True),
        {},
    ),
    "hallucination_tool_results": (
        lambda: HallucinationMetric(
            llm=MockLLM(default=_JUDGE_RESPONSE), include_assistant_tool_results_as_reference=True
        ),
        {},
    ),
    "hallucination_scenario": (
        lambda: HallucinationMetric(llm=MockLLM(default=_JUDGE_RESPONSE), include_scenario_metadata_as_reference=True),
        {},
    ),
    "structural_output_field": (
        lambda: StructuralSimilarityMetric(output_field="metadata.manifest"),
        {"metadata": {"manifest": {"a": 1}}, "expected": {"a": 1}},
    ),
    "choice_metadata_state": (
        lambda: ChoiceMetric(
            provider=_Provider(), instructions="?", criteria={"Paris": None}, state_field="metadata.answer"
        ),
        {"metadata": {"answer": "Paris"}},
    ),
    "choice_correctness": (
        lambda: ChoiceMetric(provider=_Provider(), instructions="?", criteria={"Paris": None}, mode="correctness"),
        {},
    ),
    "composite_paths": (
        lambda: CompositeMetric(
            sub_scores=[
                {"name": "tools", "weight": 1.0, "check": {"type": "count_match", "field": "tool_calls"}},
                {
                    "name": "tagged",
                    "weight": 1.0,
                    "skip_when_missing": True,
                    "check": {"type": "field_exists", "field": "tags.suite"},
                },
            ]
        ),
        {"output": '{"answer": "Paris"}'},
    ),
    "decision_composite_paths": (
        lambda: DecisionCompositeMetric(
            provider=_Provider(),
            sub_scores=[
                {
                    "name": "plan",
                    "weight": 1.0,
                    "state_field": "metadata.plan",
                    "question": NoulQuestion(instructions="?"),
                }
            ],
        ),
        {"metadata": {"plan": "search, then answer"}},
    ),
    "contains_negate": (lambda: ContainsMetric(negate=True, forbidden="secret"), {}),
}


@pytest.mark.unit
@pytest.mark.parametrize(("build", "case_overrides"), _CONFIGURED.values(), ids=_CONFIGURED.keys())
def test_configured_metric_reads_match_its_declaration(build, case_overrides):
    _assert_reads_match_declaration(build(), **case_overrides)
