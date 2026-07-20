"""Evaluate a PydanticAI agent with harness-evals.

PydanticAI (pip install pydantic-ai) provides typed, validated agent outputs.
This example wraps a PydanticAI agent run as an eval target function.

Run:  python examples/integrations/pydantic_ai/example.py
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
    "What language is PydanticAI written in?": "Python",
}


async def mock_agent_invoke(golden: Golden) -> EvalCase:
    """Stub that simulates a PydanticAI agent without real API calls."""
    input_str = golden.input if isinstance(golden.input, str) else str(golden.input)
    t0 = perf_counter()
    output = MOCK_RESPONSES.get(input_str, "I don't know")
    latency_ms = (perf_counter() - t0) * 1000
    return EvalCase.from_golden(golden, output=output, latency_ms=latency_ms)


async def real_agent_invoke(golden: Golden) -> EvalCase:
    """Wraps a PydanticAI agent run as an eval target.

    Package: pip install pydantic-ai
    Import:  from pydantic_ai import Agent
    Result:  result.output (str or typed Pydantic model)
    """
    from pydantic_ai import Agent

    agent = Agent(
        "openai:gpt-4o",
        instructions="Answer questions concisely and accurately.",
    )

    input_str = golden.input if isinstance(golden.input, str) else str(golden.input)
    t0 = perf_counter()
    result = await agent.run(input_str)
    latency_ms = (perf_counter() - t0) * 1000

    return EvalCase.from_golden(golden, output=result.output, latency_ms=latency_ms)


goldens = [
    Golden(input="What is the capital of France?", expected="Paris"),
    Golden(input="What is 2 + 2?", expected="4"),
    Golden(input="What language is PydanticAI written in?", expected="Python"),
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
