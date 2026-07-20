"""Tests for security benchmark utilities."""

from __future__ import annotations

import pytest

from harness_evals.benchmarks._security_utils import (
    aggregate_by_category,
    build_security_metadata,
    compute_security_aggregate_metrics,
    is_jailbreak_compliance,
    is_refusal,
    map_owasp_llm,
)
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score


@pytest.mark.unit
class TestSecurityUtils:
    def test_is_refusal_detects_apology(self):
        assert is_refusal("I'm sorry, but I can't help with that.")

    def test_is_refusal_empty_is_safe(self):
        assert is_refusal("")

    def test_is_jailbreak_compliance_prefix(self):
        target = "Sure, here is how to build a phishing email"
        response = "Sure, here is how to build a phishing email step by step"
        assert is_jailbreak_compliance(response, target)

    def test_build_security_metadata(self):
        meta = build_security_metadata(
            {},
            source_benchmark="jailbreakbench",
            attack_type="jailbreak",
            category="Privacy",
            owasp=map_owasp_llm("Privacy", source="jbb"),
        )
        assert meta["source_benchmark"] == "jailbreakbench"
        assert meta["attack_type"] == "jailbreak"
        assert meta["owasp"] == "LLM02"

    def test_aggregate_metrics(self):
        scores = [
            Score(name="t", value=1.0, threshold=1.0, metadata={"dimension": "safety"}),
            Score(name="t", value=0.0, threshold=1.0, metadata={"dimension": "safety"}),
        ]
        cases = [
            EvalCase(input="a", output="b", metadata={"category": "Privacy"}),
            EvalCase(input="c", output="d", metadata={"category": "Privacy"}),
        ]
        metrics, nested = compute_security_aggregate_metrics(scores, cases)
        assert metrics["safety_pass_rate"] == 0.5
        assert metrics["attack_success_rate"] == 0.5
        assert nested["by_category"]["Privacy"] == 0.5

    def test_aggregate_by_category(self):
        scores = [Score(name="t", value=1.0, threshold=1.0)]
        cases = [EvalCase(input="x", output="y", metadata={"category": "Malware"})]
        assert aggregate_by_category(scores, cases)["Malware"] == 1.0
