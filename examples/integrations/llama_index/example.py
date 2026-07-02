"""Evaluate a LlamaIndex RAG pipeline with harness-evals using HttpTarget.

This example shows how to evaluate a LlamaIndex query engine deployed as an
HTTP endpoint (e.g. via FastAPI). Uses RAG-specific metrics to evaluate
faithfulness, context precision, and answer relevancy.

Run:  python examples/integrations/llama_index/example.py
Requires: A running LlamaIndex endpoint (or set USE_MOCK=1 for local testing)
"""

import asyncio
import os
from time import perf_counter

from harness_evals import EvalCase, Golden, evaluate_dataset
from harness_evals.metrics import ContainsMetric, LatencyMetric
from harness_evals.sinks import StdoutSink
from harness_evals.targets import HttpTarget

MOCK_RESPONSES = {
    "What is the main topic of the document?": {
        "response": "The document discusses machine learning fundamentals.",
        "context": ["Machine learning is a subset of AI that enables systems to learn from data."],
    },
    "Who is the author?": {
        "response": "The author is Dr. Smith.",
        "context": ["Written by Dr. Smith, Professor of Computer Science."],
    },
    "What year was it published?": {
        "response": "It was published in 2024.",
        "context": ["Publication date: 2024, Journal of AI Research."],
    },
}


async def mock_agent_invoke(golden: Golden) -> EvalCase:
    """Stub that simulates an HTTP call to a LlamaIndex RAG endpoint."""
    input_str = golden.input if isinstance(golden.input, str) else str(golden.input)
    t0 = perf_counter()
    resp = MOCK_RESPONSES.get(input_str, {"response": "I don't know", "context": []})
    latency_ms = (perf_counter() - t0) * 1000
    ec = EvalCase.from_golden(golden, output=resp["response"], latency_ms=latency_ms)
    ec.context = resp["context"]
    return ec


def build_target() -> HttpTarget:
    """Build HttpTarget for a deployed LlamaIndex query engine.

    Typical endpoint shape:
      POST /query  {"query": "..."} -> {"response": "...", "context": [...]}
    """
    return HttpTarget(
        url=os.environ.get("LLAMAINDEX_ENDPOINT", "http://localhost:8000/query"),
        method="POST",
        body_template={"query": ""},
        input_path="$.query",
        output_path="$.response",
        context_path="$.context",
    )


goldens = [
    Golden(
        input="What is the main topic of the document?",
        expected="The document discusses machine learning fundamentals.",
        context=["Machine learning is a subset of AI that enables systems to learn from data."],
    ),
    Golden(
        input="Who is the author?",
        expected="The author is Dr. Smith.",
        context=["Written by Dr. Smith, Professor of Computer Science."],
    ),
    Golden(
        input="What year was it published?",
        expected="It was published in 2024.",
        context=["Publication date: 2024, Journal of AI Research."],
    ),
]


async def main() -> None:
    use_mock = os.environ.get("USE_MOCK", "1") == "1"

    if use_mock:
        agent_fn = mock_agent_invoke
    else:
        target = build_target()
        agent_fn = target.ainvoke

    # RAG metrics (FaithfulnessMetric, ContextPrecisionMetric) require an LLM judge.
    # For this example we use deterministic metrics; swap in RAG metrics with a judge LLM:
    #   from harness_evals.metrics.rag import FaithfulnessMetric, ContextPrecisionMetric
    #   metrics = [FaithfulnessMetric(llm=judge), ContextPrecisionMetric(llm=judge)]
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
