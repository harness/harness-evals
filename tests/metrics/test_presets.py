"""Tests for GEval presets — ensure all presets are non-empty strings."""

import pytest

from harness_evals.metrics.llm_judge import presets


@pytest.mark.unit
class TestPresets:
    def test_all_presets_are_strings(self):
        for name in presets.__all__:
            value = getattr(presets, name)
            assert isinstance(value, str), f"{name} should be a string"
            assert len(value) > 10, f"{name} should be a non-trivial criteria string"

    def test_preset_count(self):
        assert len(presets.__all__) >= 8

    def test_factuality(self):
        assert "factual" in presets.FACTUALITY.lower()

    def test_closed_qa(self):
        assert "answer" in presets.CLOSED_QA.lower()

    def test_sql_equivalence(self):
        assert "SQL" in presets.SQL_EQUIVALENCE

    def test_summarization(self):
        assert "summary" in presets.SUMMARIZATION.lower()

    def test_translation(self):
        assert "translation" in presets.TRANSLATION.lower()

    def test_humor(self):
        assert "funny" in presets.HUMOR.lower() or "humor" in presets.HUMOR.lower()

    def test_feasibility(self):
        assert "feasib" in presets.FEASIBILITY.lower()

    def test_security(self):
        assert "vulnerabilit" in presets.SECURITY.lower()
