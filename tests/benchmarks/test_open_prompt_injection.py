"""Tests for OpenPromptInjection adapter."""

from __future__ import annotations

import pytest

from harness_evals.benchmarks.open_prompt_injection import OpenPromptInjection


@pytest.mark.unit
class TestOpenPromptInjection:
    def test_invalid_attack_mode(self):
        with pytest.raises(ValueError, match="attack_mode"):
            OpenPromptInjection(attack_mode="invalid")

    async def test_load_bundled_dataset(self):
        opi = OpenPromptInjection(attack_mode="direct")
        items = await opi.load_dataset(offline=True)
        assert len(items) >= 1
        assert items[0]["attack_mode"] == "direct"

    async def test_score_injection_success(self):
        opi = OpenPromptInjection()
        item = {"target_label": "positive", "injected_label": "spam"}
        value, reason = opi.score_response(item, "This is spam")
        assert value == 0.0
        assert "Injection succeeded" in (reason or "")

    async def test_score_target_preserved(self):
        opi = OpenPromptInjection()
        item = {"target_label": "positive", "injected_label": "spam"}
        value, reason = opi.score_response(item, "Sentiment is positive")
        assert value == 1.0
        assert reason is None

    async def test_run_with_mock(self, mock_llm_factory):
        opi = OpenPromptInjection(attack_mode="direct")
        llm = mock_llm_factory(responses=["positive"])
        result = await opi.run(llm, limit=1)

        assert result.num_total == 1
        assert result.eval_cases[0].metadata["attack_type"] == "prompt_injection"
