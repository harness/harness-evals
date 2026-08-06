"""Metric factory tests: catalog dispatch, code-loading guard, judge metadata."""

from unittest.mock import AsyncMock

import pytest

from harness_evals import EvalCase
from harness_evals.metrics.factory import build_metric


class TestBuildMetric:
    def test_heuristic_regex_from_catalog(self):
        m = build_metric("heuristic", {"kind": "regex"}, score_name="regex_match", threshold=1.0)
        assert m.name == "regex_match"

    def test_heuristic_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match="Unknown heuristic kind"):
            build_metric("heuristic", {"kind": "nope"}, score_name="x")

    def test_unknown_type_rejected(self):
        with pytest.raises(ValueError, match="Unknown metric type"):
            build_metric("wasm", {}, score_name="x")

    def test_code_metrics_rejected_when_disallowed(self):
        with pytest.raises(ValueError, match="not allowed"):
            build_metric("code", {"path": "mod:Cls"}, score_name="x", allow_code_loading=False)

    def test_nested_code_metrics_rejected_when_disallowed(self):
        with pytest.raises(ValueError, match="not allowed"):
            build_metric(
                "composite",
                {
                    "aggregation": "average",
                    "metrics": [{"ref": "unsafe", "type": "code", "config": {"path": "mod:Cls"}}],
                },
                score_name="composite",
                allow_code_loading=False,
            )

    def test_invalid_heuristic_options_fail_loudly(self):
        with pytest.raises(TypeError):
            build_metric("heuristic", {"kind": "regex", "options": {"not_an_option": True}})

    def test_factory_arguments_cannot_be_overridden_by_options(self):
        with pytest.raises(TypeError, match="threshold"):
            build_metric("heuristic", {"kind": "regex", "options": {"threshold": 0.8}})
        with pytest.raises(TypeError, match="conflict with factory-supplied arguments: name"):
            build_metric("heuristic", {"kind": "regex", "options": {"name": "custom"}})

    def test_inherited_metric_options_are_accepted(self):
        metric = build_metric(
            "llm",
            {"kind": "turn_faithfulness", "options": {"allow_skips": True}, "metadata": {"llm": object()}},
        )
        assert metric.name == "turn_faithfulness"

    def test_geval_dimension_option_accepted(self):
        metric = build_metric(
            "llm",
            {
                "kind": "geval",
                "criteria": "Is it correct?",
                "options": {"dimension": "groundedness"},
                "metadata": {"llm": object()},
            },
            score_name="geval_groundedness",
        )
        assert metric.name == "geval_groundedness"

    def test_injected_llm_options_cannot_be_overridden(self):
        with pytest.raises(TypeError, match="criteria"):
            build_metric(
                "llm",
                {
                    "kind": "conversational_geval",
                    "criteria": "a",
                    "options": {"criteria": "b"},
                    "metadata": {"llm": object()},
                },
            )

    def test_null_options_and_config_handled_gracefully(self):
        m = build_metric(
            "heuristic",
            {"kind": "regex", "options": None},
            score_name="regex_match",
        )
        assert m.name == "regex_match"

    def test_weighted_average_with_zero_weights_fails_loudly(self):
        with pytest.raises(ValueError, match="zero total weight"):
            build_metric(
                "composite",
                {
                    "aggregation": "weighted_average",
                    "metrics": [{"ref": "regex", "type": "heuristic", "config": {"kind": "regex"}, "weight": 0}],
                },
                score_name="weighted",
            )

    def test_weighted_average_keeps_weights_aligned_after_skip(self):
        metric = build_metric(
            "composite",
            {
                "aggregation": "weighted_average",
                "metrics": [
                    {"ref": "skipped", "type": "heuristic", "config": {"kind": "regex"}, "weight": 0},
                    {"ref": "scored", "type": "heuristic", "config": {"kind": "regex"}, "weight": 1},
                ],
            },
            score_name="weighted",
        )
        metric.sub_metrics[0].a_measure = AsyncMock(return_value=None)
        score = metric.measure(EvalCase(input="x", output="x", expected="x"))
        assert score.value == 1.0

    def test_llm_build_injects_judge_metadata(self):
        m = build_metric(
            "llm",
            {
                "kind": "geval",
                "criteria": "Is it correct?",
                "metadata": {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "k",
                    "base_url": "https://api.openai.com/v1",
                },
            },
            score_name="geval_correctness",
            threshold=0.8,
        )
        assert m.name == "geval_correctness"

    def test_llm_unknown_kind_falls_back_to_geval(self):
        m = build_metric(
            "llm",
            {"criteria": "rate it", "metadata": {"provider": "openai", "model": "gpt-4o", "api_key": "k"}},
            score_name="custom_judge",
        )
        assert m.name == "custom_judge"
    
    def test_llm_hardcoded_dimension_conflict_caught_at_runtime(self):
        with pytest.raises(TypeError, match="conflict with factory-supplied arguments: dimension"):
            build_metric(
                "llm",
                {
                    "kind": "code_safety",
                    "options": {"dimension": "safety"},
                    "metadata": {"llm": object()},
                },
            )
