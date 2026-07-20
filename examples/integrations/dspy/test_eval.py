"""pytest integration test for DSPy module evaluation.

Run: pytest examples/integrations/dspy/test_eval.py
"""

import asyncio
from time import perf_counter

from harness_evals import EvalCase, Golden, assert_test
from harness_evals.metrics import ExactMatchMetric, LatencyMetric

MOCK_RESPONSES = {
    "What is the capital of France?": "Paris",
    "Who invented the telephone?": "Alexander Graham Bell",
}


async def mock_agent_invoke(golden: Golden) -> EvalCase:
    input_str = golden.input if isinstance(golden.input, str) else str(golden.input)
    t0 = perf_counter()
    output = MOCK_RESPONSES.get(input_str, "I don't know")
    latency_ms = (perf_counter() - t0) * 1000
    return EvalCase.from_golden(golden, output=output, latency_ms=latency_ms)


def test_dspy_capital():
    golden = Golden(input="What is the capital of France?", expected="Paris")
    ec = asyncio.run(mock_agent_invoke(golden))
    assert_test(ec, metrics=[ExactMatchMetric(), LatencyMetric(max_ms=5000)])


def test_dspy_history():
    golden = Golden(input="Who invented the telephone?", expected="Alexander Graham Bell")
    ec = asyncio.run(mock_agent_invoke(golden))
    assert_test(ec, metrics=[ExactMatchMetric()])
