"""Tests for dynamic LLM cost estimation."""

from unittest.mock import MagicMock

import pytest

from harness_evals.llm.cost import estimate_llm_cost, normalize_model_for_pricing


@pytest.mark.unit
class TestNormalizeModelForPricing:
    def test_preserves_online_prefix(self):
        assert normalize_model_for_pricing("online/openai/gpt-4o") == "online/openai/gpt-4o"

    def test_strips_version_suffix(self):
        assert normalize_model_for_pricing("gpt-4o@1.2.3") == "gpt-4o"


@pytest.mark.unit
class TestEstimateLlmCost:
    def test_reads_embedded_usage_cost(self):
        usage = MagicMock(total_cost=0.0123)
        response = MagicMock(usage=usage, model="gpt-4o")
        assert estimate_llm_cost(response, model="gpt-4o") == pytest.approx(0.0123)

    def test_reads_top_level_dict_cost(self):
        response = {"model": "gpt-4o", "cost_usd": 0.0042}
        assert estimate_llm_cost(response, model="gpt-4o") == pytest.approx(0.0042)

    def test_returns_none_without_litellm_or_embedded_cost(self, monkeypatch):
        monkeypatch.setattr(
            "harness_evals.llm.cost._litellm_completion_cost",
            lambda response, model: None,
        )
        response = MagicMock(usage=MagicMock(spec=[]), model="unknown-model-xyz")
        assert estimate_llm_cost(response, model="unknown-model-xyz") is None

    def test_uses_litellm_when_available(self, monkeypatch):
        monkeypatch.setattr(
            "harness_evals.llm.cost._litellm_completion_cost",
            lambda response, model: 0.0099 if model == "gpt-4o" else None,
        )
        response = MagicMock(model="gpt-4o")
        assert estimate_llm_cost(response, model="gpt-4o") == pytest.approx(0.0099)
