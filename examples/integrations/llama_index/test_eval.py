"""pytest integration test for LlamaIndex RAG evaluation.

Run: pytest examples/integrations/llama_index/test_eval.py
"""

import asyncio
from time import perf_counter

from harness_evals import EvalCase, Golden, assert_test
from harness_evals.metrics import ExactMatchMetric, LatencyMetric

MOCK_RESPONSES = {
    "What is the main topic of the document?": "The document discusses machine learning fundamentals.",
    "Who is the author?": "The author is Dr. Smith.",
}


async def mock_agent_invoke(golden: Golden) -> EvalCase:
    input_str = golden.input if isinstance(golden.input, str) else str(golden.input)
    t0 = perf_counter()
    output = MOCK_RESPONSES.get(input_str, "I don't know")
    latency_ms = (perf_counter() - t0) * 1000
    return EvalCase.from_golden(golden, output=output, latency_ms=latency_ms)


def test_llamaindex_topic():
    golden = Golden(
        input="What is the main topic of the document?",
        expected="The document discusses machine learning fundamentals.",
    )
    ec = asyncio.run(mock_agent_invoke(golden))
    assert_test(ec, metrics=[ExactMatchMetric(), LatencyMetric(max_ms=5000)])


def test_llamaindex_author():
    golden = Golden(input="Who is the author?", expected="The author is Dr. Smith.")
    ec = asyncio.run(mock_agent_invoke(golden))
    assert_test(ec, metrics=[ExactMatchMetric()])
