"""Convenience runners that chain conversation simulation with evaluation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from harness_evals.conversation.golden import ConversationGolden
from harness_evals.conversation.simulator import ConversationSimulator
from harness_evals.core.metric import BaseMetric
from harness_evals.core.runner import a_evaluate
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink
from harness_evals.core.types import Message
from harness_evals.llm.base import BaseLLM


async def evaluate_conversation(
    golden: ConversationGolden,
    agent_fn: Callable[[list[Message]], Awaitable[Message]],
    metrics: list[BaseMetric],
    *,
    simulator_llm: BaseLLM,
    sinks: list[BaseSink] | None = None,
) -> list[Score]:
    """Simulate a conversation from a golden, then evaluate with metrics."""
    simulator = ConversationSimulator(simulator_llm)
    eval_case = await simulator.simulate(golden, agent_fn)
    return await a_evaluate(eval_case, metrics, sinks)


async def evaluate_conversations(
    goldens: list[ConversationGolden],
    agent_fn: Callable[[list[Message]], Awaitable[Message]],
    metrics: list[BaseMetric],
    *,
    simulator_llm: BaseLLM,
    sinks: list[BaseSink] | None = None,
    concurrency: int = 5,
) -> list[list[Score]]:
    """Simulate and evaluate multiple conversation goldens concurrently."""
    simulator = ConversationSimulator(simulator_llm, max_concurrent=concurrency)
    eval_cases = await simulator.simulate_batch(goldens, agent_fn)

    scored = await asyncio.gather(*[a_evaluate(ec, metrics) for ec in eval_cases])

    if sinks:
        for eval_case, scores in zip(eval_cases, scored, strict=True):
            for sink in sinks:
                sink.write(scores, eval_case)
        for sink in sinks:
            sink.finalize()
            sink.shutdown()

    return list(scored)
