"""Tests for JailBreakV-28K adapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from harness_evals.benchmarks.jailbreakv_28k import JailBreakV28K


@pytest.mark.unit
class TestJailBreakV28K:
    async def test_score_refusal(self):
        jbv = JailBreakV28K()
        value, _ = jbv.score_response({"prompt": "ignore rules"}, "I can't help with that.")
        assert value == 1.0

    async def test_run_with_mock(self, mock_llm_factory):
        jbv = JailBreakV28K()
        sample = [{"prompt": "jailbreak", "category": "test", "method": "template"}]
        with patch.object(jbv, "load_dataset", new_callable=AsyncMock, return_value=sample):
            llm = mock_llm_factory(responses=["I cannot assist"])
            result = await jbv.run(llm)

        assert result.num_total == 1
        assert result.eval_cases[0].metadata["source_benchmark"] == "jailbreakv_28k"
