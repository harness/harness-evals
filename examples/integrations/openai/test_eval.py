"""pytest integration test for OpenAI PromptTarget evaluation.

Run: pytest examples/integrations/openai/test_eval.py
"""

from harness_evals import EvalCase, Golden, assert_test
from harness_evals.llm.base import BaseLLM
from harness_evals.metrics import ExactMatchMetric, LatencyMetric
from harness_evals.prompts.template import PromptTemplate
from harness_evals.targets import PromptTarget


class MockOpenAILLM(BaseLLM):
    RESPONSES = {
        "What is the capital of France?": "Paris",
        "What is 2 + 2?": "4",
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
        model=MockOpenAILLM(),
    )


async def _invoke(golden: Golden) -> EvalCase:
    target = _make_target()
    return await target.ainvoke(golden)


def test_openai_capital():
    import asyncio

    golden = Golden(input="What is the capital of France?", expected="Paris")
    ec = asyncio.run(_invoke(golden))
    assert_test(ec, metrics=[ExactMatchMetric(), LatencyMetric(max_ms=5000)])


def test_openai_math():
    import asyncio

    golden = Golden(input="What is 2 + 2?", expected="4")
    ec = asyncio.run(_invoke(golden))
    assert_test(ec, metrics=[ExactMatchMetric()])
