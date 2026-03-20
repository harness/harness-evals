"""Tests for the PromptRephrase LLM-based perturbation generator."""

import pytest

from harness_evals.perturbations.rephrase import PromptRephrase
from tests.conftest import MockLLM


@pytest.mark.unit
class TestPromptRephrase:
    async def test_produces_n_rephrasings(self):
        llm = MockLLM(
            responses=[
                {"rephrasings": ["r1", "r2", "r3"]},
            ]
        )
        gen = PromptRephrase(llm)
        results = await gen.perturb("What is the capital of France?", n=3)
        assert len(results) == 3
        assert results == ["r1", "r2", "r3"]

    async def test_pads_when_llm_returns_fewer(self):
        llm = MockLLM(
            responses=[
                {"rephrasings": ["only one"]},
            ]
        )
        gen = PromptRephrase(llm)
        results = await gen.perturb("Hello", n=3)
        assert len(results) == 3
        assert results[0] == "only one"
        assert results[1] == "Hello"
        assert results[2] == "Hello"

    async def test_truncates_when_llm_returns_more(self):
        llm = MockLLM(
            responses=[
                {"rephrasings": ["a", "b", "c", "d", "e"]},
            ]
        )
        gen = PromptRephrase(llm)
        results = await gen.perturb("Hello", n=2)
        assert len(results) == 2
        assert results == ["a", "b"]

    async def test_empty_string_returns_copies(self):
        llm = MockLLM()
        gen = PromptRephrase(llm)
        results = await gen.perturb("", n=4)
        assert len(results) == 4
        assert all(r == "" for r in results)

    async def test_whitespace_only_returns_copies(self):
        llm = MockLLM()
        gen = PromptRephrase(llm)
        results = await gen.perturb("   ", n=2)
        assert len(results) == 2
        assert all(r == "   " for r in results)

    async def test_handles_missing_rephrasings_key(self):
        llm = MockLLM(responses=[{"unexpected_key": "value"}])
        gen = PromptRephrase(llm)
        results = await gen.perturb("Test input", n=3)
        assert len(results) == 3
        assert all(r == "Test input" for r in results)

    async def test_handles_empty_rephrasings_list(self):
        llm = MockLLM(responses=[{"rephrasings": []}])
        gen = PromptRephrase(llm)
        results = await gen.perturb("Test input", n=2)
        assert len(results) == 2
        assert all(r == "Test input" for r in results)

    async def test_default_n_is_five(self):
        llm = MockLLM(
            responses=[
                {"rephrasings": ["a", "b", "c", "d", "e"]},
            ]
        )
        gen = PromptRephrase(llm)
        results = await gen.perturb("Test")
        assert len(results) == 5

    async def test_extends_base_perturbation(self):
        from harness_evals.perturbations.base import BasePerturbation

        llm = MockLLM()
        gen = PromptRephrase(llm)
        assert isinstance(gen, BasePerturbation)

    async def test_curly_braces_in_input(self):
        llm = MockLLM(responses=[{"rephrasings": ["rephrased"]}])
        gen = PromptRephrase(llm)
        results = await gen.perturb("Parse {name: 'test'}", n=1)
        assert len(results) == 1
        assert results[0] == "rephrased"

    def test_sync_perturb_not_needed(self):
        """PromptRephrase only exposes the async ``perturb()`` method."""
        llm = MockLLM()
        gen = PromptRephrase(llm)
        assert hasattr(gen, "perturb")
        assert not hasattr(gen, "perturb_sync")
