"""Evaluate any LLM via LiteLLM with harness-evals using PromptTarget.

LiteLLM supports 100+ models with a unified interface. Change the model
string to switch providers: "openai/gpt-4o", "anthropic/claude-sonnet-4-20250514",
"vertex_ai/gemini-2.5-flash", "bedrock/anthropic.claude-haiku-4-5-20251001:0", etc.

Run:  python examples/integrations/litellm/example.py
Requires: Provider-specific API key env var (or set USE_MOCK=1 for local testing)
"""

import asyncio
import os

from harness_evals import Golden, evaluate_dataset
from harness_evals.llm.base import BaseLLM
from harness_evals.metrics import ContainsMetric, LatencyMetric
from harness_evals.prompts.template import PromptTemplate
from harness_evals.sinks import StdoutSink
from harness_evals.targets import PromptTarget


class LiteLLMAdapter(BaseLLM):
    """BaseLLM wrapper around LiteLLM's acompletion — covers 100+ models."""

    def __init__(self, model: str = "openai/gpt-4o", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    async def generate(self, prompt: str, **kwargs: object) -> str:
        from litellm import acompletion

        response = await acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    async def generate_json(self, prompt: str, schema: dict, **kwargs: object) -> dict:
        import json

        from litellm import acompletion

        response = await acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content or "{}")


class MockLiteLLM(BaseLLM):
    """Deterministic stub — no API key or litellm install needed."""

    RESPONSES = {
        "What is the capital of France?": "Paris",
        "What is 2 + 2?": "4",
        "Who painted the Mona Lisa?": "Leonardo da Vinci",
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
        model: BaseLLM = MockLiteLLM()
    else:
        model = LiteLLMAdapter(model=os.environ.get("LITELLM_MODEL", "openai/gpt-4o"))

    return PromptTarget(prompt=prompt, model=model)


goldens = [
    Golden(input="What is the capital of France?", expected="Paris"),
    Golden(input="What is 2 + 2?", expected="4"),
    Golden(input="Who painted the Mona Lisa?", expected="Leonardo da Vinci"),
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
