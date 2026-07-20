"""Evaluate a deployed LangChain agent with harness-evals using HttpTarget.

This example shows how to evaluate a LangChain agent deployed as an HTTP
endpoint (e.g. via LangServe, LangGraph Platform, or a custom FastAPI app).

Run:  python examples/integrations/langchain/example.py
Requires: A running LangChain endpoint (or set USE_MOCK=1 for local testing)
"""

import asyncio
import os
from time import perf_counter

from harness_evals import EvalCase, Golden, evaluate_dataset
from harness_evals.metrics import ContainsMetric, LatencyMetric
from harness_evals.sinks import StdoutSink
from harness_evals.targets import BearerAuth, HttpTarget, NoAuth

MOCK_RESPONSES = {
    "What is the capital of France?": {"output": "Paris"},
    "What is 2 + 2?": {"output": "4"},
    "Summarize the latest AI news": {"output": "AI continues to advance rapidly in 2025."},
}


async def mock_agent_invoke(golden: Golden) -> EvalCase:
    """Stub that simulates an HTTP call to a LangChain endpoint."""
    input_str = golden.input if isinstance(golden.input, str) else str(golden.input)
    t0 = perf_counter()
    resp = MOCK_RESPONSES.get(input_str, {"output": "I don't know"})
    latency_ms = (perf_counter() - t0) * 1000
    return EvalCase.from_golden(golden, output=resp["output"], latency_ms=latency_ms)


def build_target() -> HttpTarget:
    """Build HttpTarget for a deployed LangChain endpoint.

    Typical LangServe endpoint shape:
      POST /invoke  {"input": "..."} -> {"output": "..."}
    """
    return HttpTarget(
        url=os.environ.get("LANGCHAIN_ENDPOINT", "http://localhost:8000/invoke"),
        method="POST",
        auth=BearerAuth(os.environ["LANGCHAIN_API_KEY"]) if os.environ.get("LANGCHAIN_API_KEY") else NoAuth(),
        body_template={"input": "{{input}}"},
        output_path="$.output",
    )


goldens = [
    Golden(input="What is the capital of France?", expected="Paris"),
    Golden(input="What is 2 + 2?", expected="4"),
    Golden(input="Summarize the latest AI news", expected="AI continues to advance rapidly in 2025."),
]


async def main() -> None:
    use_mock = os.environ.get("USE_MOCK", "1") == "1"

    if use_mock:
        agent_fn = mock_agent_invoke
    else:
        target = build_target()
        agent_fn = target.ainvoke

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
