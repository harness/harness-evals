"""Shared fixtures for benchmark tests."""

from __future__ import annotations

import pytest

from harness_evals.llm.base import BaseLLM


class MockLLM(BaseLLM):
    """Mock LLM that returns pre-configured responses."""

    def __init__(self, responses: list[str] | None = None, default: str = "A") -> None:
        self.responses = list(responses) if responses else []
        self.default = default
        self.call_count = 0
        self.prompts: list[str] = []

    async def generate(self, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        self.call_count += 1
        if self.responses:
            resp = self.responses.pop(0)
            if isinstance(resp, BaseException):
                raise resp
            return resp
        return self.default

    async def generate_json(self, prompt: str, schema: dict, **kwargs: object) -> dict:
        return {}


@pytest.fixture
def mock_llm():
    """Create a mock LLM instance."""
    return MockLLM()


@pytest.fixture
def mock_llm_factory():
    """Create a mock LLM with specific responses."""

    def _factory(responses: list[str] | None = None, default: str = "A") -> MockLLM:
        return MockLLM(responses=responses, default=default)

    return _factory
