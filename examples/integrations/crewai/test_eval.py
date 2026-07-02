"""pytest integration test for CrewAI evaluation.

Run: pytest examples/integrations/crewai/test_eval.py
"""

import asyncio
from time import perf_counter

from harness_evals import EvalCase, Golden, assert_test
from harness_evals.metrics import ExactMatchMetric, LatencyMetric

MOCK_RESPONSES = {
    "Research the latest trends in AI agents": "AI agents are increasingly autonomous, with frameworks like CrewAI enabling multi-agent collaboration.",
    "Summarize the benefits of microservices": "Microservices enable independent deployment, scaling, and technology diversity at the cost of distributed system complexity.",
}


async def mock_agent_invoke(golden: Golden) -> EvalCase:
    input_str = golden.input if isinstance(golden.input, str) else str(golden.input)
    t0 = perf_counter()
    output = MOCK_RESPONSES.get(input_str, "I don't know")
    latency_ms = (perf_counter() - t0) * 1000
    return EvalCase.from_golden(golden, output=output, latency_ms=latency_ms)


def test_crewai_research():
    golden = Golden(
        input="Research the latest trends in AI agents",
        expected="AI agents are increasingly autonomous, with frameworks like CrewAI enabling multi-agent collaboration.",
    )
    ec = asyncio.run(mock_agent_invoke(golden))
    assert_test(ec, metrics=[ExactMatchMetric(), LatencyMetric(max_ms=5000)])


def test_crewai_summarize():
    golden = Golden(
        input="Summarize the benefits of microservices",
        expected="Microservices enable independent deployment, scaling, and technology diversity at the cost of distributed system complexity.",
    )
    ec = asyncio.run(mock_agent_invoke(golden))
    assert_test(ec, metrics=[ExactMatchMetric()])
