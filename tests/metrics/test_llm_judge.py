"""Tests for LLM-judged metrics with mocked LLM."""

import pytest

from harness_evals import EvalCase
from harness_evals.core.metric import Dimension
from harness_evals.llm.base import BaseLLM
from harness_evals.metrics.llm_judge.geval import GEvalMetric
from harness_evals.metrics.llm_judge.rubric_judge import RubricJudgeMetric
from harness_evals.metrics.llm_judge.types import RubricLevel


class MockLLM(BaseLLM):
    def __init__(self, json_response: dict):
        self._json_response = json_response
        self.last_prompt: str | None = None
        self.last_schema: dict | None = None

    async def generate(self, prompt: str, **kwargs) -> str:
        return ""

    async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
        self.last_prompt = prompt
        self.last_schema = schema
        return self._json_response


@pytest.mark.unit
class TestGEvalMetric:
    async def test_high_score(self):
        llm = MockLLM({"reasoning": "Accurate and complete", "score": 0.9})
        metric = GEvalMetric(llm=llm, criteria="accuracy", threshold=0.7)
        ec = EvalCase(input="What is 2+2?", output="4", expected="4")
        score = await metric.a_measure(ec)
        assert score.value == 0.9
        assert score.passed
        assert "Accurate" in score.reason

    async def test_low_score(self):
        llm = MockLLM({"reasoning": "Wrong answer", "score": 0.2})
        metric = GEvalMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="What is 2+2?", output="5", expected="4")
        score = await metric.a_measure(ec)
        assert score.value == 0.2
        assert not score.passed

    async def test_clamps_score(self):
        llm = MockLLM({"reasoning": "test", "score": 1.5})
        metric = GEvalMetric(llm=llm)
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 1.0

    async def test_missing_score_defaults_zero(self):
        llm = MockLLM({"reasoning": "test"})
        metric = GEvalMetric(llm=llm)
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 0.0

    def test_sync_measure(self):
        llm = MockLLM({"reasoning": "ok", "score": 0.8})
        metric = GEvalMetric(llm=llm, threshold=0.5)
        ec = EvalCase(input="q", output="a")
        score = metric.measure(ec)
        assert score.value == 0.8
        assert score.passed


@pytest.mark.unit
class TestRubricJudgeMetric:
    async def test_top_level(self):
        llm = MockLLM({"reasoning": "Excellent work", "level": 5})
        metric = RubricJudgeMetric(llm=llm, threshold=0.5)
        ec = EvalCase(input="q", output="a", expected="a")
        score = await metric.a_measure(ec)
        assert score.value == 1.0  # 5 out of 5 -> 1.0
        assert score.passed
        assert score.metadata["level"] == 5

    async def test_mid_level(self):
        llm = MockLLM({"reasoning": "Acceptable", "level": 3})
        metric = RubricJudgeMetric(llm=llm, threshold=0.5)
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 0.5  # (3-1)/(5-1) = 0.5

    async def test_lowest_level(self):
        llm = MockLLM({"reasoning": "Very poor", "level": 1})
        metric = RubricJudgeMetric(llm=llm, threshold=0.5)
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 0.0  # (1-1)/(5-1) = 0.0

    async def test_clamps_level(self):
        llm = MockLLM({"reasoning": "test", "level": 10})
        metric = RubricJudgeMetric(llm=llm)
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 1.0  # clamped to max

    async def test_custom_rubric(self):
        custom = {1: "Bad", 2: "OK", 3: "Great"}
        llm = MockLLM({"reasoning": "OK", "level": 2})
        metric = RubricJudgeMetric(llm=llm, rubric=custom, threshold=0.3)
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 0.5  # (2-1)/(3-1) = 0.5
        assert score.passed

    def test_empty_rubric_raises(self):
        llm = MockLLM({"reasoning": "test", "level": 1})
        with pytest.raises(ValueError, match="non-empty"):
            RubricJudgeMetric(llm=llm, rubric={})


# ---------------------------------------------------------------------------
# RubricLevel dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRubricLevel:
    def test_valid_level(self):
        r = RubricLevel(0, 5, "description")
        assert r.min_score == 0
        assert r.max_score == 5
        assert r.description == "description"

    def test_rejects_inverted_range(self):
        with pytest.raises(ValueError, match="min_score"):
            RubricLevel(5, 2, "bad")

    def test_rejects_negative_min(self):
        with pytest.raises(ValueError, match="non-negative"):
            RubricLevel(-1, 3, "bad")

    def test_rejects_empty_description(self):
        with pytest.raises(ValueError, match="description"):
            RubricLevel(0, 5, "")

    def test_frozen(self):
        r = RubricLevel(0, 5, "x")
        with pytest.raises((AttributeError, Exception)):
            r.min_score = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GEvalMetric enrichments: evaluation_steps, rubric, subclassing, name override
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGEvalMetricEvaluationSteps:
    async def test_steps_appear_numbered_in_prompt(self):
        llm = MockLLM({"reasoning": "ok", "score": 0.8})
        metric = GEvalMetric(
            llm=llm,
            criteria="is it good",
            evaluation_steps=["first step", "second step", "third step"],
        )
        await metric.a_measure(EvalCase(input="q", output="a"))
        assert "Evaluation steps" in llm.last_prompt
        assert "  1. first step" in llm.last_prompt
        assert "  2. second step" in llm.last_prompt
        assert "  3. third step" in llm.last_prompt

    async def test_no_steps_section_when_empty(self):
        llm = MockLLM({"reasoning": "ok", "score": 0.8})
        metric = GEvalMetric(llm=llm, criteria="is it good")
        await metric.a_measure(EvalCase(input="q", output="a"))
        assert "Evaluation steps" not in llm.last_prompt


@pytest.mark.unit
class TestGEvalMetricWithRubric:
    def _rubric(self) -> list[RubricLevel]:
        return [
            RubricLevel(0, 2, "Poor"),
            RubricLevel(3, 5, "OK"),
            RubricLevel(6, 8, "Good"),
            RubricLevel(9, 10, "Excellent"),
        ]

    async def test_integer_scoring_normalized_to_01(self):
        llm = MockLLM({"reasoning": "good fix", "score": 8})
        metric = GEvalMetric(llm=llm, criteria="quality", rubric=self._rubric(), threshold=0.5)
        score = await metric.a_measure(EvalCase(input="q", output="a"))
        assert score.value == 0.8
        assert score.passed
        assert score.metadata["raw_score"] == 8
        assert score.metadata["min_score"] == 0
        assert score.metadata["max_score"] == 10

    async def test_rubric_prompt_contains_ranges_and_int_instruction(self):
        llm = MockLLM({"reasoning": "ok", "score": 5})
        metric = GEvalMetric(llm=llm, criteria="quality", rubric=self._rubric())
        await metric.a_measure(EvalCase(input="q", output="a"))
        assert "Rubric" in llm.last_prompt
        assert "0-2: Poor" in llm.last_prompt
        assert "9-10: Excellent" in llm.last_prompt
        assert "integer 0-10" in llm.last_prompt
        assert llm.last_schema["properties"]["score"]["type"] == "integer"

    async def test_rubric_clamps_above(self):
        llm = MockLLM({"reasoning": "overshoot", "score": 15})
        metric = GEvalMetric(llm=llm, criteria="q", rubric=self._rubric())
        score = await metric.a_measure(EvalCase(input="q", output="a"))
        assert score.value == 1.0
        assert score.metadata["raw_score"] == 10

    async def test_rubric_clamps_below(self):
        llm = MockLLM({"reasoning": "undershoot", "score": -3})
        metric = GEvalMetric(llm=llm, criteria="q", rubric=self._rubric())
        score = await metric.a_measure(EvalCase(input="q", output="a"))
        assert score.value == 0.0
        assert score.metadata["raw_score"] == 0

    async def test_rubric_missing_score_defaults_to_min(self):
        llm = MockLLM({"reasoning": "forgot"})
        metric = GEvalMetric(llm=llm, criteria="q", rubric=self._rubric())
        score = await metric.a_measure(EvalCase(input="q", output="a"))
        assert score.value == 0.0

    async def test_rubric_non_integer_score_falls_back(self):
        llm = MockLLM({"reasoning": "weird", "score": "abc"})
        metric = GEvalMetric(llm=llm, criteria="q", rubric=self._rubric())
        score = await metric.a_measure(EvalCase(input="q", output="a"))
        assert score.value == 0.0

    def test_rubric_must_contain_rubric_level_instances(self):
        llm = MockLLM({"reasoning": "x", "score": 0})
        with pytest.raises(TypeError, match="RubricLevel"):
            GEvalMetric(llm=llm, criteria="q", rubric=[{"min": 0, "max": 10, "desc": "x"}])  # type: ignore[list-item]


@pytest.mark.unit
class TestGEvalMetricSubclassing:
    async def test_class_level_attributes_used_by_default(self):
        class MyMetric(GEvalMetric):
            criteria = "Is the answer precise?"
            evaluation_steps = ["check precision", "check clarity"]
            rubric = [
                RubricLevel(0, 4, "Imprecise"),
                RubricLevel(5, 10, "Precise"),
            ]

            def __init__(self, llm: BaseLLM, threshold: float = 0.5, **kwargs: object) -> None:
                super().__init__(llm=llm, threshold=threshold, **kwargs)

        llm = MockLLM({"reasoning": "precise", "score": 9})
        metric = MyMetric(llm=llm)
        assert metric.name == "MyMetric"
        score = await metric.a_measure(EvalCase(input="q", output="a"))
        assert score.value == 0.9
        assert "Is the answer precise?" in llm.last_prompt
        assert "  1. check precision" in llm.last_prompt
        assert "0-4: Imprecise" in llm.last_prompt

    async def test_constructor_args_override_class_attrs(self):
        class MyMetric(GEvalMetric):
            criteria = "default criteria"
            evaluation_steps = ["default step"]

        llm = MockLLM({"reasoning": "ok", "score": 0.7})
        metric = MyMetric(
            llm=llm,
            criteria="overridden criteria",
            evaluation_steps=["overridden step"],
        )
        await metric.a_measure(EvalCase(input="q", output="a"))
        assert "overridden criteria" in llm.last_prompt
        assert "overridden step" in llm.last_prompt
        assert "default criteria" not in llm.last_prompt

    def test_base_class_keeps_geval_name(self):
        llm = MockLLM({"reasoning": "ok", "score": 0.8})
        metric = GEvalMetric(llm=llm, criteria="x")
        assert metric.name == "geval"

    def test_explicit_name_override(self):
        llm = MockLLM({"reasoning": "ok", "score": 0.8})
        metric = GEvalMetric(llm=llm, criteria="x", name="MyCustomJudge")
        assert metric.name == "MyCustomJudge"

    def test_dimension_override(self):
        llm = MockLLM({"reasoning": "ok", "score": 0.8})
        metric = GEvalMetric(llm=llm, criteria="x", dimension=Dimension.SAFETY)
        assert metric.dimension == Dimension.SAFETY


@pytest.mark.unit
class TestGEvalMetricGrounding:
    async def test_grounding_instruction_in_prompt(self):
        llm = MockLLM({"reasoning": "ok", "score": 0.8})
        metric = GEvalMetric(llm=llm, criteria="x")
        await metric.a_measure(EvalCase(input="q", output="a"))
        assert "Evaluate ONLY" in llm.last_prompt
        assert "Do not infer or assume" in llm.last_prompt


# ---------------------------------------------------------------------------
# RubricJudgeMetric enrichments: evaluation_steps
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRubricJudgeMetricEvaluationSteps:
    async def test_steps_appear_numbered_in_prompt(self):
        llm = MockLLM({"reasoning": "ok", "level": 4})
        metric = RubricJudgeMetric(
            llm=llm,
            evaluation_steps=["step one", "step two"],
        )
        await metric.a_measure(EvalCase(input="q", output="a"))
        assert "Evaluation steps" in llm.last_prompt
        assert "  1. step one" in llm.last_prompt
        assert "  2. step two" in llm.last_prompt

    async def test_no_steps_section_when_empty(self):
        llm = MockLLM({"reasoning": "ok", "level": 4})
        metric = RubricJudgeMetric(llm=llm)
        await metric.a_measure(EvalCase(input="q", output="a"))
        assert "Evaluation steps" not in llm.last_prompt

    async def test_grounding_instruction_in_prompt(self):
        llm = MockLLM({"reasoning": "ok", "level": 4})
        metric = RubricJudgeMetric(llm=llm)
        await metric.a_measure(EvalCase(input="q", output="a"))
        assert "Evaluate ONLY" in llm.last_prompt
        assert "Do not infer or assume" in llm.last_prompt


# ---------------------------------------------------------------------------
# Mutable-state isolation: no aliasing across instances / caller / class attrs
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMutableStateIsolation:
    def test_geval_sibling_instances_have_independent_lists(self):
        llm = MockLLM({"reasoning": "x", "score": 0})
        a = GEvalMetric(llm=llm, criteria="x", evaluation_steps=["s"], rubric=[RubricLevel(0, 5, "d")])
        b = GEvalMetric(llm=llm, criteria="x", evaluation_steps=["s"], rubric=[RubricLevel(0, 5, "d")])
        assert a.evaluation_steps is not b.evaluation_steps
        assert a.rubric is not b.rubric
        a.evaluation_steps.append("leaked")
        a.rubric.append(RubricLevel(0, 0, "leaked"))
        assert "leaked" not in b.evaluation_steps
        assert not any(r.description == "leaked" for r in b.rubric)

    def test_geval_caller_list_not_aliased(self):
        llm = MockLLM({"reasoning": "x", "score": 0})
        caller_steps = ["a"]
        caller_rubric = [RubricLevel(0, 5, "d")]
        metric = GEvalMetric(llm=llm, criteria="x", evaluation_steps=caller_steps, rubric=caller_rubric)
        assert metric.evaluation_steps is not caller_steps
        assert metric.rubric is not caller_rubric
        caller_steps.append("mutated")
        caller_rubric.append(RubricLevel(0, 0, "mutated"))
        assert "mutated" not in metric.evaluation_steps
        assert not any(r.description == "mutated" for r in metric.rubric)

    def test_geval_class_default_not_aliased(self):
        class WithDefaults(GEvalMetric):
            criteria = "x"
            evaluation_steps = ["default"]
            rubric = [RubricLevel(0, 5, "default")]

        llm = MockLLM({"reasoning": "x", "score": 0})
        m1 = WithDefaults(llm=llm)
        m2 = WithDefaults(llm=llm)
        assert m1.evaluation_steps is not WithDefaults.evaluation_steps
        assert m1.rubric is not WithDefaults.rubric
        assert m1.evaluation_steps is not m2.evaluation_steps
        m1.evaluation_steps.append("added")
        m1.rubric.append(RubricLevel(0, 0, "added"))
        assert "added" not in m2.evaluation_steps
        assert "added" not in WithDefaults.evaluation_steps
        assert not any(r.description == "added" for r in m2.rubric)
        assert not any(r.description == "added" for r in WithDefaults.rubric)

    def test_rubric_judge_dict_not_aliased(self):
        llm = MockLLM({"reasoning": "x", "level": 1})
        caller_dict = {1: "Poor", 2: "OK", 3: "Great"}
        metric = RubricJudgeMetric(llm=llm, rubric=caller_dict)
        assert metric.rubric is not caller_dict
        caller_dict[4] = "Excellent"
        assert 4 not in metric.rubric

    def test_rubric_judge_default_rubric_not_aliased(self):
        llm = MockLLM({"reasoning": "x", "level": 1})
        m1 = RubricJudgeMetric(llm=llm)
        m2 = RubricJudgeMetric(llm=llm)
        assert m1.rubric is not m2.rubric
        m1.rubric[99] = "injected"
        assert 99 not in m2.rubric


# ---------------------------------------------------------------------------
# Edge cases: degenerate rubrics, empty lists, format-unsafe inputs
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestGEvalMetricEdgeCases:
    async def test_single_band_rubric_span_zero(self):
        """Rubric with one level where min==max: normalized value is 1.0."""
        llm = MockLLM({"reasoning": "x", "score": 5})
        metric = GEvalMetric(llm=llm, criteria="x", rubric=[RubricLevel(5, 5, "only")])
        s = await metric.a_measure(EvalCase(input="q", output="a"))
        assert s.value == 1.0
        assert s.metadata["raw_score"] == 5
        assert s.metadata["min_score"] == 5
        assert s.metadata["max_score"] == 5

    async def test_rubric_not_starting_at_zero(self):
        """1-5 rubric normalizes (3-1)/(5-1) = 0.5."""
        llm = MockLLM({"reasoning": "x", "score": 3})
        metric = GEvalMetric(
            llm=llm,
            criteria="x",
            rubric=[RubricLevel(1, 2, "low"), RubricLevel(3, 4, "mid"), RubricLevel(5, 5, "high")],
        )
        s = await metric.a_measure(EvalCase(input="q", output="a"))
        assert abs(s.value - 0.5) < 1e-9
        assert "from 1 to 5" in llm.last_prompt
        assert "integer 1-5" in llm.last_prompt

    async def test_non_contiguous_rubric_with_gaps(self):
        """Gaps between bands are allowed; LLM can still score in the full range."""
        llm = MockLLM({"reasoning": "x", "score": 6})
        metric = GEvalMetric(
            llm=llm,
            criteria="x",
            rubric=[RubricLevel(0, 3, "low"), RubricLevel(8, 10, "high")],
        )
        s = await metric.a_measure(EvalCase(input="q", output="a"))
        assert abs(s.value - 0.6) < 1e-9
        assert "0-3: low" in llm.last_prompt
        assert "8-10: high" in llm.last_prompt

    async def test_empty_evaluation_steps_omits_section(self):
        llm = MockLLM({"reasoning": "x", "score": 0.5})
        metric = GEvalMetric(llm=llm, criteria="x", evaluation_steps=[])
        await metric.a_measure(EvalCase(input="q", output="a"))
        assert "Evaluation steps" not in llm.last_prompt

    async def test_empty_rubric_list_uses_float_path(self):
        llm = MockLLM({"reasoning": "x", "score": 0.7})
        metric = GEvalMetric(llm=llm, criteria="x", rubric=[])
        s = await metric.a_measure(EvalCase(input="q", output="a"))
        assert s.value == 0.7
        assert s.metadata is None

    async def test_float_score_with_rubric_truncated_to_int(self):
        """LLM disobeys 'integer' -> int() truncates for metadata and normalization."""
        llm = MockLLM({"reasoning": "x", "score": 5.7})
        metric = GEvalMetric(llm=llm, criteria="x", rubric=[RubricLevel(0, 10, "d")])
        s = await metric.a_measure(EvalCase(input="q", output="a"))
        assert abs(s.value - 0.5) < 1e-9
        assert s.metadata["raw_score"] == 5

    async def test_none_score_returns_zero(self):
        llm = MockLLM({"reasoning": "x", "score": None})
        metric = GEvalMetric(llm=llm, criteria="x")
        s = await metric.a_measure(EvalCase(input="q", output="a"))
        assert s.value == 0.0

    async def test_none_score_with_rubric_returns_zero(self):
        llm = MockLLM({"reasoning": "x", "score": None})
        metric = GEvalMetric(llm=llm, criteria="x", rubric=[RubricLevel(0, 10, "d")])
        s = await metric.a_measure(EvalCase(input="q", output="a"))
        assert s.value == 0.0

    async def test_braces_in_input_output_dont_break_format(self):
        """Curly braces in user-supplied text must not trigger .format() errors."""
        ec = EvalCase(
            input="Test {foo} {bar} {{nested}}",
            output='```json\n{"key": {"nested": "value"}}\n```',
            expected="{a: 1}",
        )
        llm = MockLLM({"reasoning": "x", "score": 0.9})
        metric = GEvalMetric(llm=llm, criteria="x")
        s = await metric.a_measure(ec)
        assert s.value == 0.9
        assert '{"key"' in llm.last_prompt
        assert "{foo}" in llm.last_prompt

    async def test_schema_selection_integer_vs_number(self):
        llm = MockLLM({"reasoning": "x", "score": 5})
        metric = GEvalMetric(llm=llm, criteria="x", rubric=[RubricLevel(0, 10, "d")])
        await metric.a_measure(EvalCase(input="q", output="a"))
        assert llm.last_schema["properties"]["score"]["type"] == "integer"

        llm = MockLLM({"reasoning": "x", "score": 0.5})
        metric = GEvalMetric(llm=llm, criteria="x")
        await metric.a_measure(EvalCase(input="q", output="a"))
        assert llm.last_schema["properties"]["score"]["type"] == "number"


# ---------------------------------------------------------------------------
# Prompt formatting invariants: no triple newlines, no unsubstituted fields
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPromptFormattingInvariants:
    """Guard against regressions in prompt assembly."""

    def _cases(self):
        return [
            EvalCase(input="q", output="a"),
            EvalCase(input="q", output="a", expected="ref"),
            EvalCase(input="q with {braces}", output='{"json": 1}'),
        ]

    def test_geval_no_triple_newlines(self):
        llm = MockLLM({"reasoning": "x", "score": 0.5})
        for ec in self._cases():
            for metric in [
                GEvalMetric(llm=llm, criteria="x"),
                GEvalMetric(llm=llm, criteria="x", evaluation_steps=["s"]),
                GEvalMetric(llm=llm, criteria="x", rubric=[RubricLevel(0, 10, "d")]),
                GEvalMetric(
                    llm=llm,
                    criteria="x",
                    evaluation_steps=["s1", "s2"],
                    rubric=[RubricLevel(0, 10, "d")],
                ),
            ]:
                p = metric._build_prompt(ec)
                assert "\n\n\n" not in p, f"triple newline in {type(metric).__name__} prompt"
                assert "{{" not in p and "}}" not in p, "unresolved brace escape"

    def test_rubric_judge_no_triple_newlines(self):
        llm = MockLLM({"reasoning": "x", "level": 3})
        for ec in self._cases():
            for metric in [
                RubricJudgeMetric(llm=llm),
                RubricJudgeMetric(llm=llm, evaluation_steps=["a", "b"]),
                RubricJudgeMetric(llm=llm, rubric={0: "bad", 1: "ok", 2: "good"}),
            ]:
                p = metric._build_prompt(ec)
                assert "\n\n\n" not in p, "triple newline in rubric_judge prompt"
                assert "{{" not in p and "}}" not in p
