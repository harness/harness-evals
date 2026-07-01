"""Evaluate an Anthropic Claude model with harness-evals using PromptTarget.

Run:  python examples/integrations/anthropic/example.py
Requires: ANTHROPIC_API_KEY env var (or set USE_MOCK=1 for local testing)
"""

import asyncio
import os

from harness_evals import Golden, evaluate_dataset
from harness_evals.llm.base import BaseLLM
from harness_evals.metrics import ContainsMetric, LatencyMetric
from harness_evals.prompts.template import PromptTemplate
from harness_evals.sinks import StdoutSink
from harness_evals.targets import PromptTarget


class MockAnthropicLLM(BaseLLM):
    """Deterministic stub — no API key needed."""

    RESPONSES = {
        "What is the capital of France?": "Paris",
        "What is 2 + 2?": "4",
        "Explain quantum computing in one sentence": (
            "Quantum computing uses quantum bits that can exist in superposition "
            "to solve certain problems exponentially faster than classical computers."
        ),
    }

    async def generate(self, prompt: str, **kwargs: object) -> str:
        for question, answer in self.RESPONSES.items():
            if question in prompt:
                return answer
        return "I don't know"

    async def generate_json(self, prompt: str, schema: dict, **kwargs: object) -> dict:
        return {"answer": await self.generate(prompt)}


def build_target() -> PromptTarget:
    prompt = PromptTemplate(template="Answer concisely: {{input}}")

    if os.environ.get("USE_MOCK", "1") == "1":
        model = MockAnthropicLLM()
    else:
        from harness_evals.llm.anthropic import AnthropicLLM

        model = AnthropicLLM(model="claude-sonnet-4-20250514")

    return PromptTarget(prompt=prompt, model=model)


goldens = [
    Golden(input="What is the capital of France?", expected="Paris"),
    Golden(input="What is 2 + 2?", expected="4"),
    Golden(
        input="Explain quantum computing in one sentence",
        expected="Quantum computing uses quantum bits that can exist in superposition "
        "to solve certain problems exponentially faster than classical computers.",
    ),
]


async def main() -> None:
    target = build_target()
    results = await evaluate_dataset(
        goldens,
        target.ainvoke,
        metrics=[
            ContainsMetric(),
            LatencyMetric(max_ms=5000, threshold=0.5),
        ],
        sinks=[StdoutSink()],
    )
    passed = sum(all(s.passed for s in r) for r in results)
    print(f"\nPass rate: {passed}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
