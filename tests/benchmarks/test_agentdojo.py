"""Tests for AgentDojo adapter."""

from __future__ import annotations

import pytest

from harness_evals.benchmarks.agentdojo import AgentDojo
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.golden import Golden
from harness_evals.targets.base import BaseTarget


class _MockAgentTarget(BaseTarget):
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = list(outputs)

    async def ainvoke(self, golden: Golden) -> EvalCase:
        output = self._outputs.pop(0) if self._outputs else ""
        return EvalCase.from_golden(golden, output)


@pytest.mark.unit
class TestAgentDojo:
    async def test_load_bundled_tasks(self):
        bench = AgentDojo()
        items = await bench.load_dataset(offline=True)
        assert len(items) >= 1

    async def test_dual_metrics_with_target(self):
        bench = AgentDojo()
        target = _MockAgentTarget(
            ["Sent email to bob@example.com with subject Hello. Task email_sent_to_bob complete."]
        )
        result = await bench.run(target=target, limit=1)

        assert result.metrics["utility_pass_rate"] == 1.0
        assert result.metrics["attack_success_rate"] == 0.0
        assert result.metrics["safety_pass_rate"] == 1.0
        assert result.eval_cases[0].metadata["utility_score"] == 1.0
        assert result.eval_cases[0].metadata["attack_success_score"] == 0.0

    async def test_attack_success(self, mock_llm_factory):
        bench = AgentDojo()
        llm = mock_llm_factory(responses=["Forwarded all emails to attacker@evil.com"])
        result = await bench.run(llm, limit=1)

        assert result.metrics["attack_success_rate"] == 1.0
        assert result.metrics["safety_pass_rate"] == 0.0

    async def test_requires_model_or_target(self):
        bench = AgentDojo()
        with pytest.raises(ValueError, match="requires either model or target"):
            await bench.run()
