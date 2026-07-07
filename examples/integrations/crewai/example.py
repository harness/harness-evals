"""Evaluate a CrewAI crew with harness-evals using HttpTarget.

CrewAI (pip install crewai) is a multi-agent orchestration framework.
This example shows how to evaluate a crew deployed as an HTTP endpoint
(e.g. via CrewAI AMP or a custom FastAPI wrapper).

Run:  python examples/integrations/crewai/example.py
Requires: A running CrewAI endpoint (or set USE_MOCK=1 for local testing)
"""

import asyncio
import os
from time import perf_counter

from harness_evals import EvalCase, Golden, evaluate_dataset
from harness_evals.metrics import ContainsMetric, LatencyMetric
from harness_evals.sinks import StdoutSink
from harness_evals.targets import BearerAuth, HttpTarget, NoAuth

MOCK_RESPONSES = {
    "Research the latest trends in AI agents": "AI agents are increasingly autonomous, with frameworks like CrewAI enabling multi-agent collaboration.",
    "Write a haiku about Python": "Code flows like water\nIndentation guides the way\nPython speaks in verse",
    "Summarize the benefits of microservices": "Microservices enable independent deployment, scaling, and technology diversity at the cost of distributed system complexity.",
}


async def mock_agent_invoke(golden: Golden) -> EvalCase:
    """Stub that simulates an HTTP call to a CrewAI endpoint."""
    input_str = golden.input if isinstance(golden.input, str) else str(golden.input)
    t0 = perf_counter()
    output = MOCK_RESPONSES.get(input_str, "I don't know")
    latency_ms = (perf_counter() - t0) * 1000
    return EvalCase.from_golden(golden, output=output, latency_ms=latency_ms)


def build_target() -> HttpTarget:
    """Build HttpTarget for a deployed CrewAI endpoint.

    Typical CrewAI AMP endpoint shape:
      POST /kickoff  {"topic": "..."} -> {"result": "..."}

    Or custom FastAPI wrapper:
      POST /run  {"input": "..."} -> {"output": "..."}
    """
    return HttpTarget(
        url=os.environ.get("CREWAI_ENDPOINT", "http://localhost:8000/run"),
        method="POST",
        auth=BearerAuth(os.environ["CREWAI_API_KEY"]) if os.environ.get("CREWAI_API_KEY") else NoAuth(),
        body_template={"input": "{{input}}"},
        output_path="$.output",
    )


goldens = [
    Golden(
        input="Research the latest trends in AI agents",
        expected="AI agents are increasingly autonomous, with frameworks like CrewAI enabling multi-agent collaboration.",
    ),
    Golden(
        input="Write a haiku about Python",
        expected="Code flows like water\nIndentation guides the way\nPython speaks in verse",
    ),
    Golden(
        input="Summarize the benefits of microservices",
        expected="Microservices enable independent deployment, scaling, and technology diversity at the cost of distributed system complexity.",
    ),
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
            LatencyMetric(max_ms=60000, threshold=0.5),
        ],
        sinks=[StdoutSink()],
        # For agent-specific metrics with a real endpoint:
        #   from harness_evals.metrics.agent import TaskCompletionMetric, ToolCorrectnessMetric
        #   metrics=[TaskCompletionMetric(llm=judge), ToolCorrectnessMetric()]
    )
    passed = sum(all(s.passed for s in r) for r in results)
    print(f"\nPass rate: {passed}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
