"""Evaluate an OpenAI Agents SDK agent with harness-evals.

The OpenAI Agents SDK (pip install openai-agents) provides a higher-level
agent framework. This example wraps an agent run as a PromptTarget-compatible
async function for evaluation.

Run:  python examples/integrations/openai_agents/example.py
Requires: OPENAI_API_KEY env var (or set USE_MOCK=1 for local testing)
"""

import asyncio
import os
from time import perf_counter

from harness_evals import EvalCase, Golden, evaluate_dataset
from harness_evals.metrics import ContainsMetric, LatencyMetric
from harness_evals.sinks import StdoutSink

MOCK_RESPONSES = {
    "What is the capital of France?": "Paris",
    "What is 2 + 2?": "4",
    "List three primary colors": "Red, blue, yellow",
}


async def mock_agent_invoke(golden: Golden) -> EvalCase:
    """Stub that simulates an agent run without real API calls."""
    input_str = golden.input if isinstance(golden.input, str) else str(golden.input)
    t0 = perf_counter()
    output = MOCK_RESPONSES.get(input_str, "I don't know")
    latency_ms = (perf_counter() - t0) * 1000
    return EvalCase.from_golden(golden, output=output, latency_ms=latency_ms)


async def real_agent_invoke(golden: Golden) -> EvalCase:
    """Wraps an OpenAI Agents SDK agent run as an eval target.

    Package: pip install openai-agents
    Import:  from agents import Agent, Runner
    """
    from agents import Agent, Runner

    agent = Agent(
        name="eval-agent",
        instructions="Answer questions concisely and accurately.",
    )

    input_str = golden.input if isinstance(golden.input, str) else str(golden.input)
    t0 = perf_counter()
    result = await Runner.run(agent, input_str)
    latency_ms = (perf_counter() - t0) * 1000

    return EvalCase.from_golden(golden, output=result.final_output, latency_ms=latency_ms)


goldens = [
    Golden(input="What is the capital of France?", expected="Paris"),
    Golden(input="What is 2 + 2?", expected="4"),
    Golden(input="List three primary colors", expected="Red, blue, yellow"),
]


async def main() -> None:
    use_mock = os.environ.get("USE_MOCK", "1") == "1"
    agent_fn = mock_agent_invoke if use_mock else real_agent_invoke

    results = await evaluate_dataset(
        goldens,
        agent_fn,
        metrics=[
            ContainsMetric(),
            LatencyMetric(max_ms=10000, threshold=0.5),
        ],
        sinks=[StdoutSink()],
    )
    passed = sum(all(s.passed for s in r) for r in results)
    print(f"\nPass rate: {passed}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
