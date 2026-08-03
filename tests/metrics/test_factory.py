"""Metric factory tests: catalog dispatch, code-loading guard, judge metadata."""

import pytest

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

    def test_llm_build_injects_judge_metadata(self):
        m = build_metric(
            "llm",
            {"kind": "geval", "criteria": "Is it correct?",
             "metadata": {"provider": "openai", "model": "gpt-4o",
                          "api_key": "k", "base_url": "https://api.openai.com/v1"}},
            score_name="geval_correctness", threshold=0.8,
        )
        assert m.name == "geval_correctness"

    def test_llm_unknown_kind_falls_back_to_geval(self):
        m = build_metric(
            "llm",
            {"criteria": "rate it",
             "metadata": {"provider": "openai", "model": "gpt-4o", "api_key": "k"}},
            score_name="custom_judge",
        )
        assert m.name == "custom_judge"
