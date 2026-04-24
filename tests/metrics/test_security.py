"""Tests for security remediation metrics."""

from __future__ import annotations

import inspect
import os

import pytest

from harness_evals import EvalCase, a_evaluate, evaluate
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.metrics.llm_judge.geval import GEvalMetric
from harness_evals.metrics.security.actionability import ActionabilityMetric
from harness_evals.metrics.security.code_quality import CodeQualityMetric
from harness_evals.metrics.security.code_safety import CodeSafetyMetric
from harness_evals.metrics.security.composite import (
    REMEDIATION_WEIGHTS,
    remediation_quality_index,
)
from harness_evals.metrics.security.explanation_quality import ExplanationQualityMetric
from harness_evals.metrics.security.root_cause_analysis import RootCauseAnalysisMetric
from harness_evals.metrics.security.security_completeness import (
    SecurityCompletenessMetric,
)
from harness_evals.metrics.security.vulnerability_correctness import (
    VulnerabilityCorrectnessMetric,
)
from tests.conftest import MockLLM

ALL_METRIC_CLASSES = [
    VulnerabilityCorrectnessMetric,
    SecurityCompletenessMetric,
    CodeSafetyMetric,
    CodeQualityMetric,
    ExplanationQualityMetric,
    RootCauseAnalysisMetric,
    ActionabilityMetric,
]

XSS_CASE = EvalCase(
    input=(
        "CWE-79: Reflected XSS in user_profile.py line 42. "
        "User input from request.args['name'] rendered in HTML without sanitization."
    ),
    output=(
        "## CWE-79 Reflected XSS\n"
        "The `request.args['name']` parameter is passed to the template without escaping.\n\n"
        "```python\nfrom markupsafe import escape\nname = escape(request.args.get('name', ''))\n```\n\n"
        "Also add Content-Security-Policy headers."
    ),
)

SQLI_CASE = EvalCase(
    input="CWE-89: SQL injection in login.py line 18. User input concatenated into query.",
    output="Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))",
)


# ---------------------------------------------------------------------------
# VulnerabilityCorrectnessMetric
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVulnerabilityCorrectness:
    async def test_correct_root_cause(self):
        llm = MockLLM(default={"reasoning": "Fix addresses XSS root cause", "score": 9})
        metric = VulnerabilityCorrectnessMetric(llm=llm, threshold=0.5)
        score = await metric.a_measure(XSS_CASE)
        assert score.value == 0.9
        assert score.passed
        assert score.metadata["raw_score"] == 9

    async def test_wrong_vuln_type(self):
        llm = MockLLM(default={"reasoning": "Fix targets wrong vulnerability", "score": 1})
        metric = VulnerabilityCorrectnessMetric(llm=llm, threshold=0.5)
        score = await metric.a_measure(XSS_CASE)
        assert score.value == 0.1
        assert not score.passed

    async def test_missing_score_defaults_zero(self):
        llm = MockLLM(default={"reasoning": "confused"})
        metric = VulnerabilityCorrectnessMetric(llm=llm)
        score = await metric.a_measure(XSS_CASE)
        assert score.value == 0.0

    async def test_score_clamped_above(self):
        llm = MockLLM(default={"reasoning": "overenthusiastic", "score": 15})
        metric = VulnerabilityCorrectnessMetric(llm=llm)
        score = await metric.a_measure(XSS_CASE)
        assert score.value == 1.0

    async def test_score_clamped_below(self):
        llm = MockLLM(default={"reasoning": "negative", "score": -3})
        metric = VulnerabilityCorrectnessMetric(llm=llm)
        score = await metric.a_measure(XSS_CASE)
        assert score.value == 0.0

    def test_sync_measure(self):
        llm = MockLLM(default={"reasoning": "correct", "score": 8})
        metric = VulnerabilityCorrectnessMetric(llm=llm, threshold=0.5)
        score = metric.measure(XSS_CASE)
        assert score.value == 0.8
        assert score.passed

    def test_dimension(self):
        llm = MockLLM()
        metric = VulnerabilityCorrectnessMetric(llm=llm)
        assert metric.dimension == Dimension.CORRECTNESS
        assert isinstance(metric, BaseMetric)


# ---------------------------------------------------------------------------
# SecurityCompletenessMetric
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSecurityCompleteness:
    async def test_comprehensive_fix(self):
        llm = MockLLM(default={"reasoning": "Multiple layers of defense", "score": 9})
        metric = SecurityCompletenessMetric(llm=llm, threshold=0.5)
        score = await metric.a_measure(XSS_CASE)
        assert score.value == 0.9
        assert score.passed

    async def test_minimal_fix(self):
        llm = MockLLM(default={"reasoning": "Single-point fix only", "score": 4})
        metric = SecurityCompletenessMetric(llm=llm, threshold=0.5)
        score = await metric.a_measure(XSS_CASE)
        assert score.value == 0.4
        assert not score.passed

    def test_sync_measure(self):
        llm = MockLLM(default={"reasoning": "ok", "score": 7})
        metric = SecurityCompletenessMetric(llm=llm, threshold=0.5)
        score = metric.measure(XSS_CASE)
        assert score.passed

    def test_dimension(self):
        llm = MockLLM()
        assert SecurityCompletenessMetric(llm=llm).dimension == Dimension.CORRECTNESS


# ---------------------------------------------------------------------------
# CodeSafetyMetric
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCodeSafety:
    async def test_safe_fix(self):
        llm = MockLLM(default={"reasoning": "No new vulnerabilities introduced", "score": 9})
        metric = CodeSafetyMetric(llm=llm, threshold=0.5)
        score = await metric.a_measure(XSS_CASE)
        assert score.value == 0.9
        assert score.passed

    async def test_introduces_new_issue(self):
        llm = MockLLM(default={"reasoning": "Introduces null dereference", "score": 2})
        metric = CodeSafetyMetric(llm=llm, threshold=0.5)
        score = await metric.a_measure(XSS_CASE)
        assert score.value == 0.2
        assert not score.passed

    def test_sync_measure(self):
        llm = MockLLM(default={"reasoning": "safe", "score": 8})
        metric = CodeSafetyMetric(llm=llm, threshold=0.5)
        score = metric.measure(XSS_CASE)
        assert score.passed

    def test_dimension(self):
        llm = MockLLM()
        assert CodeSafetyMetric(llm=llm).dimension == Dimension.SAFETY


# ---------------------------------------------------------------------------
# CodeQualityMetric
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCodeQuality:
    async def test_idiomatic_code(self):
        llm = MockLLM(default={"reasoning": "Correct language, idiomatic", "score": 9})
        metric = CodeQualityMetric(llm=llm, threshold=0.5)
        score = await metric.a_measure(XSS_CASE)
        assert score.value == 0.9
        assert score.passed

    async def test_wrong_language(self):
        llm = MockLLM(default={"reasoning": "Python in a .tf file", "score": 1})
        metric = CodeQualityMetric(llm=llm, threshold=0.5)
        score = await metric.a_measure(XSS_CASE)
        assert score.value == 0.1
        assert not score.passed

    def test_sync_measure(self):
        llm = MockLLM(default={"reasoning": "clean", "score": 8})
        metric = CodeQualityMetric(llm=llm, threshold=0.5)
        score = metric.measure(XSS_CASE)
        assert score.passed

    def test_dimension(self):
        llm = MockLLM()
        assert CodeQualityMetric(llm=llm).dimension == Dimension.CORRECTNESS


# ---------------------------------------------------------------------------
# ExplanationQualityMetric
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExplanationQuality:
    async def test_specific_explanation(self):
        llm = MockLLM(default={"reasoning": "Code-specific with CWE reference", "score": 9})
        metric = ExplanationQualityMetric(llm=llm, threshold=0.5)
        score = await metric.a_measure(XSS_CASE)
        assert score.value == 0.9
        assert score.passed

    async def test_generic_textbook_advice(self):
        llm = MockLLM(default={"reasoning": "Generic filler steps", "score": 2})
        metric = ExplanationQualityMetric(llm=llm, threshold=0.5)
        score = await metric.a_measure(XSS_CASE)
        assert score.value == 0.2
        assert not score.passed

    def test_sync_measure(self):
        llm = MockLLM(default={"reasoning": "actionable", "score": 7})
        metric = ExplanationQualityMetric(llm=llm, threshold=0.5)
        score = metric.measure(XSS_CASE)
        assert score.passed

    def test_dimension(self):
        llm = MockLLM()
        assert ExplanationQualityMetric(llm=llm).dimension == Dimension.CORRECTNESS


# ---------------------------------------------------------------------------
# RootCauseAnalysisMetric
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRootCauseAnalysis:
    async def test_correct_taint_path(self):
        llm = MockLLM(default={"reasoning": "Source->propagation->sink traced", "score": 9})
        metric = RootCauseAnalysisMetric(llm=llm, threshold=0.5)
        score = await metric.a_measure(XSS_CASE)
        assert score.value == 0.9
        assert score.passed

    async def test_vague_analysis(self):
        llm = MockLLM(default={"reasoning": "Root cause partially correct", "score": 4})
        metric = RootCauseAnalysisMetric(llm=llm, threshold=0.5)
        score = await metric.a_measure(XSS_CASE)
        assert score.value == 0.4
        assert not score.passed

    def test_sync_measure(self):
        llm = MockLLM(default={"reasoning": "precise", "score": 8})
        metric = RootCauseAnalysisMetric(llm=llm, threshold=0.5)
        score = metric.measure(XSS_CASE)
        assert score.passed

    def test_dimension(self):
        llm = MockLLM()
        assert RootCauseAnalysisMetric(llm=llm).dimension == Dimension.GROUNDEDNESS


# ---------------------------------------------------------------------------
# ActionabilityMetric
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestActionability:
    async def test_copy_paste_ready(self):
        llm = MockLLM(default={"reasoning": "Complete with commands", "score": 9})
        metric = ActionabilityMetric(llm=llm, threshold=0.5)
        score = await metric.a_measure(XSS_CASE)
        assert score.value == 0.9
        assert score.passed

    async def test_truncated_output(self):
        llm = MockLLM(default={"reasoning": "Incomplete, truncated mid-line", "score": 2})
        metric = ActionabilityMetric(llm=llm, threshold=0.5)
        score = await metric.a_measure(XSS_CASE)
        assert score.value == 0.2
        assert not score.passed

    def test_sync_measure(self):
        llm = MockLLM(default={"reasoning": "actionable", "score": 8})
        metric = ActionabilityMetric(llm=llm, threshold=0.5)
        score = metric.measure(XSS_CASE)
        assert score.passed

    def test_dimension(self):
        llm = MockLLM()
        assert ActionabilityMetric(llm=llm).dimension == Dimension.CORRECTNESS


# ---------------------------------------------------------------------------
# Composite: Remediation Quality Index
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRemediationQualityIndex:
    def test_all_metrics_present(self):
        scores = [
            Score(name="VulnerabilityCorrectnessMetric", value=0.9, threshold=0.5),
            Score(name="SecurityCompletenessMetric", value=0.8, threshold=0.5),
            Score(name="CodeSafetyMetric", value=0.7, threshold=0.5),
            Score(name="CodeQualityMetric", value=0.6, threshold=0.5),
            Score(name="ExplanationQualityMetric", value=0.8, threshold=0.5),
            Score(name="RootCauseAnalysisMetric", value=0.7, threshold=0.5),
            Score(name="ActionabilityMetric", value=0.9, threshold=0.5),
        ]
        rqi = remediation_quality_index(scores)
        assert rqi.name == "RemediationQualityIndex"
        expected = 0.9 * 0.25 + 0.8 * 0.20 + 0.7 * 0.15 + 0.6 * 0.10 + 0.8 * 0.15 + 0.7 * 0.10 + 0.9 * 0.05
        assert abs(rqi.value - expected) < 1e-6
        assert rqi.passed

    def test_partial_metrics(self):
        scores = [
            Score(name="VulnerabilityCorrectnessMetric", value=0.8, threshold=0.5),
            Score(name="CodeSafetyMetric", value=0.6, threshold=0.5),
        ]
        rqi = remediation_quality_index(scores)
        expected = (0.8 * 0.25 + 0.6 * 0.15) / (0.25 + 0.15)
        assert abs(rqi.value - expected) < 1e-6
        assert len(rqi.metadata["matched_metrics"]) == 2

    def test_no_metrics(self):
        rqi = remediation_quality_index([])
        assert rqi.value == 0.0

    def test_weights_sum_to_one(self):
        assert abs(sum(REMEDIATION_WEIGHTS.values()) - 1.0) < 1e-6

    def test_custom_weights(self):
        scores = [
            Score(name="VulnerabilityCorrectnessMetric", value=1.0, threshold=0.5),
            Score(name="CodeSafetyMetric", value=0.0, threshold=0.5),
        ]
        rqi = remediation_quality_index(
            scores, weights={"VulnerabilityCorrectnessMetric": 0.5, "CodeSafetyMetric": 0.5}
        )
        assert abs(rqi.value - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Catalog Registration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCatalogRegistration:
    def test_security_metrics_in_catalog(self):
        from harness_evals.catalog import catalog

        entries = catalog()
        security_kinds = {
            "vulnerability_correctness",
            "security_completeness",
            "code_safety",
            "code_quality",
            "explanation_quality",
            "root_cause_analysis",
            "actionability",
        }
        found = {e.kind for e in entries}
        assert security_kinds.issubset(found)

    def test_security_metrics_require_llm(self):
        from harness_evals.catalog import catalog

        security_kinds = {
            "vulnerability_correctness",
            "security_completeness",
            "code_safety",
            "code_quality",
            "explanation_quality",
            "root_cause_analysis",
            "actionability",
        }
        for entry in catalog():
            if entry.kind in security_kinds:
                assert entry.requires_llm is True
                assert entry.category == "security"


# ---------------------------------------------------------------------------
# Pattern Consistency — structural invariants across all 7 metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPatternConsistency:
    def test_all_extend_base_metric(self):
        for cls in ALL_METRIC_CLASSES:
            assert issubclass(cls, BaseMetric)
            assert issubclass(cls, GEvalMetric)

    def test_all_require_llm_param(self):
        for cls in ALL_METRIC_CLASSES:
            sig = inspect.signature(cls.__init__)
            assert "llm" in sig.parameters, f"{cls.__name__} missing llm param"

    def test_all_have_criteria(self):
        for cls in ALL_METRIC_CLASSES:
            assert cls.criteria, f"{cls.__name__} has empty criteria"
            assert len(cls.criteria) > 20, f"{cls.__name__} criteria too short"

    def test_all_have_evaluation_steps(self):
        for cls in ALL_METRIC_CLASSES:
            assert len(cls.evaluation_steps) >= 3, f"{cls.__name__} needs >= 3 evaluation steps"

    def test_all_have_rubric_levels(self):
        for cls in ALL_METRIC_CLASSES:
            assert len(cls.rubric) >= 3, f"{cls.__name__} needs >= 3 rubric levels"
            mins = [r.min_score for r in cls.rubric]
            maxs = [r.max_score for r in cls.rubric]
            assert min(mins) == 0, f"{cls.__name__} rubric should start at 0"
            assert max(maxs) == 10, f"{cls.__name__} rubric should end at 10"

    def test_metric_names_are_class_names(self):
        llm = MockLLM()
        for cls in ALL_METRIC_CLASSES:
            metric = cls(llm=llm)
            assert metric.name == cls.__name__

    def test_all_metrics_have_weights(self):
        for cls in ALL_METRIC_CLASSES:
            assert cls.__name__ in REMEDIATION_WEIGHTS, f"{cls.__name__} missing from REMEDIATION_WEIGHTS"


# ---------------------------------------------------------------------------
# Prompt Template — verify rendering with real vulnerability data
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPromptTemplate:
    def test_prompt_contains_all_sections(self):
        llm = MockLLM()
        metric = VulnerabilityCorrectnessMetric(llm=llm)
        prompt = metric._build_prompt(XSS_CASE)

        assert "Criteria" in prompt
        assert "Evaluation steps" in prompt
        assert "Rubric" in prompt
        assert "Input" in prompt
        assert "Output" in prompt
        assert '{"reasoning":' in prompt
        assert '"score":' in prompt

    def test_prompt_contains_eval_case_data(self):
        llm = MockLLM()
        metric = VulnerabilityCorrectnessMetric(llm=llm)
        prompt = metric._build_prompt(XSS_CASE)

        assert "CWE-79" in prompt
        assert "request.args" in prompt
        assert "markupsafe" in prompt

    def test_prompt_steps_are_numbered(self):
        llm = MockLLM()
        metric = VulnerabilityCorrectnessMetric(llm=llm)
        prompt = metric._build_prompt(XSS_CASE)

        for i in range(1, len(metric.evaluation_steps) + 1):
            assert f"  {i}." in prompt

    def test_prompt_rubric_ranges_present(self):
        llm = MockLLM()
        metric = CodeSafetyMetric(llm=llm)
        prompt = metric._build_prompt(XSS_CASE)

        for level in metric.rubric:
            assert f"{level.min_score}-{level.max_score}" in prompt

    def test_all_metrics_render_without_error(self):
        llm = MockLLM()
        for cls in ALL_METRIC_CLASSES:
            metric = cls(llm=llm)
            prompt = metric._build_prompt(XSS_CASE)
            assert len(prompt) > 200, f"{cls.__name__} prompt suspiciously short"

    def test_prompt_requests_json_format(self):
        llm = MockLLM()
        metric = VulnerabilityCorrectnessMetric(llm=llm)
        prompt = metric._build_prompt(XSS_CASE)
        assert "Respond with JSON" in prompt
        assert '"reasoning"' in prompt
        assert '"score"' in prompt

    def test_grounding_instruction_present(self):
        llm = MockLLM()
        metric = VulnerabilityCorrectnessMetric(llm=llm)
        prompt = metric._build_prompt(XSS_CASE)
        assert "Evaluate ONLY" in prompt
        assert "Do not infer or assume" in prompt


# ---------------------------------------------------------------------------
# Full Pipeline — evaluate() and a_evaluate() integration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFullPipeline:
    def _make_metrics(self) -> list[BaseMetric]:
        llm = MockLLM(default={"reasoning": "Good fix.", "score": 8})
        return [cls(llm=llm, threshold=0.5) for cls in ALL_METRIC_CLASSES]

    def test_evaluate_all_7_metrics(self):
        metrics = self._make_metrics()
        scores = evaluate(XSS_CASE, metrics=metrics)
        assert len(scores) == 7
        for score in scores:
            assert 0.0 <= score.value <= 1.0
            assert score.passed

    async def test_a_evaluate_all_7_metrics(self):
        metrics = self._make_metrics()
        scores = await a_evaluate(XSS_CASE, metrics=metrics)
        assert len(scores) == 7

    def test_evaluate_then_composite(self):
        metrics = self._make_metrics()
        scores = evaluate(XSS_CASE, metrics=metrics)
        rqi = remediation_quality_index(scores)
        assert rqi.name == "RemediationQualityIndex"
        assert 0.0 <= rqi.value <= 1.0
        assert rqi.passed
        assert len(rqi.metadata["matched_metrics"]) == 7


# ---------------------------------------------------------------------------
# Real LLM integration (only when OPENAI_API_KEY is set)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestRealLLMIntegration:
    def _make_llm(self):
        from harness_evals.llm.openai import OpenAILLM

        return OpenAILLM(model="gpt-4o-mini", temperature=0.0)

    async def test_single_metric_real_llm(self):
        llm = self._make_llm()
        metric = VulnerabilityCorrectnessMetric(llm=llm, threshold=0.3)
        score = await metric.a_measure(XSS_CASE)

        assert isinstance(score, Score)
        assert 0.0 <= score.value <= 1.0
        assert score.reason is not None
        assert len(score.reason) > 10
        assert "raw_score" in score.metadata
        assert 0 <= score.metadata["raw_score"] <= 10

    async def test_all_metrics_real_llm(self):
        llm = self._make_llm()
        metrics = [cls(llm=llm, threshold=0.3) for cls in ALL_METRIC_CLASSES]
        scores = await a_evaluate(XSS_CASE, metrics=metrics)

        assert len(scores) == 7
        for score in scores:
            assert 0.0 <= score.value <= 1.0
            assert score.reason is not None

        rqi = remediation_quality_index(scores)
        assert rqi.value > 0.3, f"RQI unexpectedly low: {rqi.value}"

    async def test_xss_fix_scores_well(self):
        llm = self._make_llm()
        metric = VulnerabilityCorrectnessMetric(llm=llm, threshold=0.5)
        score = await metric.a_measure(XSS_CASE)
        assert score.value >= 0.5, f"Good XSS fix should score >= 0.5, got {score.value}"

    async def test_multiple_cases_real_llm(self):
        llm = self._make_llm()
        sqli = EvalCase(
            input="CWE-89: SQL Injection in login.py. User input concatenated into query.",
            output="Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))",
        )
        metric = VulnerabilityCorrectnessMetric(llm=llm, threshold=0.3)
        for case in [XSS_CASE, sqli]:
            score = await metric.a_measure(case)
            assert 0.0 <= score.value <= 1.0
