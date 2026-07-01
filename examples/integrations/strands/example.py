"""Evaluate a Strands agent with harness-evals.

Strands (pip install strands-agents) is an AWS open-source agent framework.
This example wraps a Strands agent call as an eval target function.

Run:  python examples/integrations/strands/example.py
Requires: AWS credentials for Bedrock (or set USE_MOCK=1 for local testing)
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
    "What AWS service runs serverless functions?": "AWS Lambda",
}


async def mock_agent_invoke(golden: Golden) -> EvalCase:
    """Stub that simulates a Strands agent without real API calls."""
    input_str = golden.input if isinstance(golden.input, str) else str(golden.input)
    t0 = perf_counter()
    output = MOCK_RESPONSES.get(input_str, "I don't know")
    latency_ms = (perf_counter() - t0) * 1000
    return EvalCase.from_golden(golden, output=output, latency_ms=latency_ms)


async def real_agent_invoke(golden: Golden) -> EvalCase:
    """Wraps a Strands agent as an eval target.

    Package: pip install strands-agents
    Import:  from strands import Agent
    Result:  str(agent(prompt)) gives the final text output
    """
    from strands import Agent

    agent = Agent(system_prompt="Answer questions concisely and accurately.")

    input_str = golden.input if isinstance(golden.input, str) else str(golden.input)
    t0 = perf_counter()
    result = await asyncio.to_thread(agent, input_str)
    latency_ms = (perf_counter() - t0) * 1000

    return EvalCase.from_golden(golden, output=str(result), latency_ms=latency_ms)


goldens = [
    Golden(input="What is the capital of France?", expected="Paris"),
    Golden(input="What is 2 + 2?", expected="4"),
    Golden(input="What AWS service runs serverless functions?", expected="AWS Lambda"),
]


async def main() -> None:
    use_mock = os.environ.get("USE_MOCK", "1") == "1"
    agent_fn = mock_agent_invoke if use_mock else real_agent_invoke

    results = await evaluate_dataset(
        goldens,
        agent_fn,
        metrics=[
            ContainsMetric(),
            LatencyMetric(max_ms=30000, threshold=0.5),
        ],
        sinks=[StdoutSink()],
    )
    passed = sum(all(s.passed for s in r) for r in results)
    print(f"\nPass rate: {passed}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
