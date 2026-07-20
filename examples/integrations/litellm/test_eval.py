"""pytest integration test for LiteLLM PromptTarget evaluation.

Run: pytest examples/integrations/litellm/test_eval.py
"""

import asyncio

from harness_evals import EvalCase, Golden, assert_test
from harness_evals.llm.base import BaseLLM
from harness_evals.metrics import ExactMatchMetric, LatencyMetric
from harness_evals.prompts.template import PromptTemplate
from harness_evals.targets import PromptTarget


class MockLiteLLM(BaseLLM):
    RESPONSES = {
        "What is the capital of France?": "Paris",
        "Who painted the Mona Lisa?": "Leonardo da Vinci",
    }

    async def generate(self, prompt: str, **kwargs: object) -> str:
        for question, answer in self.RESPONSES.items():
            if question in prompt:
                return answer
        return "I don't know"

    async def generate_json(self, prompt: str, schema: dict, **kwargs: object) -> dict:
        return {"answer": await self.generate(prompt)}


def _make_target() -> PromptTarget:
    return PromptTarget(
        prompt=PromptTemplate(template="Answer concisely: {{input}}"),
        model=MockLiteLLM(),
    )


async def _invoke(golden: Golden) -> EvalCase:
    target = _make_target()
    return await target.ainvoke(golden)


def test_litellm_capital():
    golden = Golden(input="What is the capital of France?", expected="Paris")
    ec = asyncio.run(_invoke(golden))
    assert_test(ec, metrics=[ExactMatchMetric(), LatencyMetric(max_ms=5000)])


def test_litellm_art():
    golden = Golden(input="Who painted the Mona Lisa?", expected="Leonardo da Vinci")
    ec = asyncio.run(_invoke(golden))
    assert_test(ec, metrics=[ExactMatchMetric()])
