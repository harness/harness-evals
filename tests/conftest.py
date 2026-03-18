"""Shared test fixtures for harness-evals."""

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.llm.base import BaseLLM


class MockLLM(BaseLLM):
    """In-memory LLM mock for testing LLM-judged metrics.

    Provide a list of ``responses`` to return in order, or a ``default``
    dict returned once the list is exhausted.
    """

    def __init__(self, responses: list[dict] | None = None, default: dict | None = None):
        self._responses = list(responses) if responses else []
        self._default = default or {}
        self._call_idx = 0

    async def generate(self, prompt: str, **kwargs) -> str:
        return ""

    async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
        if self._call_idx < len(self._responses):
            result = self._responses[self._call_idx]
            self._call_idx += 1
            return result
        return self._default


@pytest.fixture
def mock_llm():
    """Return a factory for MockLLM instances."""
    return MockLLM


@pytest.fixture
def simple_eval_case() -> EvalCase:
    return EvalCase(
        input="What is 2+2?",
        output="4",
        expected="4",
    )


@pytest.fixture
def json_eval_case() -> EvalCase:
    return EvalCase(
        input="Generate a K8s deployment",
        output={"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": "web"}},
        expected={"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": "web"}},
    )


@pytest.fixture
def operational_eval_case() -> EvalCase:
    return EvalCase(
        input="List pods",
        output="kubectl get pods",
        expected="kubectl get pods",
        latency_ms=1200,
        token_count=350,
        cost_usd=0.003,
        retry_count=0,
    )


@pytest.fixture
def multi_run_eval_case() -> EvalCase:
    runs = [
        EvalCase(input="task", output="result_a", token_count=100),
        EvalCase(input="task", output="result_a", token_count=110),
        EvalCase(input="task", output="result_a", token_count=105),
        EvalCase(input="task", output="result_b", token_count=95),
        EvalCase(input="task", output="result_a", token_count=102),
    ]
    return EvalCase(input="task", output="result_a", runs=runs)
