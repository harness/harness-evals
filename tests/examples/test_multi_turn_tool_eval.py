"""Smoke tests for examples/multi_turn_tool_eval.py REPLAY goldens."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from harness_evals.conversation import load_conversation_dataset
from harness_evals.core.runner import evaluate_dataset
from harness_evals.core.types import Message
from harness_evals.metrics.agent.tool_argument_match import ToolArgumentMatchMetric
from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

GOLDENS_PATH = Path(__file__).resolve().parents[2] / "examples" / "data" / "multi_turn_tool_goldens.jsonl"


async def _noop_agent(messages: list[Message]) -> Message:
    return Message(role="assistant", content="")


@pytest.mark.unit
def test_multi_turn_tool_goldens_pass_and_fail() -> None:
    goldens = load_conversation_dataset(GOLDENS_PATH)
    assert len(goldens) == 2

    metrics = [
        ToolCorrectnessMetric(mode="subsequence", threshold=1.0),
        ToolArgumentMatchMetric(pair="subsequence", arg_match="subset", threshold=1.0),
    ]
    results = asyncio.run(evaluate_dataset(goldens, _noop_agent, metrics=metrics, simulator_llm=None))

    by_case = {(g.tags or {}).get("case"): scores for g, scores in zip(goldens, results, strict=True)}
    assert all(s.passed for s in by_case["pass"])
    assert not all(s.passed for s in by_case["fail"])

    pass_tool = next(s for s in by_case["pass"] if s.name == "tool_correctness")
    fail_tool = next(s for s in by_case["fail"] if s.name == "tool_correctness")
    assert pass_tool.value == 1.0
    assert fail_tool.value < 1.0
