"""Tests for deterministic perturbation generators."""

import json

import pytest

from harness_evals.perturbations.json_reorder import JsonFieldReorder
from harness_evals.perturbations.schema_variation import SchemaVariation
from harness_evals.perturbations.typo import TypoInjection


@pytest.mark.unit
class TestJsonFieldReorder:
    @pytest.mark.asyncio
    async def test_produces_n_results(self):
        gen = JsonFieldReorder(seed=42)
        obj = json.dumps({"a": 1, "b": 2, "c": 3, "d": 4})
        results = await gen.perturb(obj, n=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_valid_json(self):
        gen = JsonFieldReorder(seed=42)
        obj = json.dumps({"name": "test", "value": 42, "active": True})
        results = await gen.perturb(obj, n=5)
        for r in results:
            parsed = json.loads(r)
            assert parsed["name"] == "test"
            assert parsed["value"] == 42

    @pytest.mark.asyncio
    async def test_different_orderings(self):
        gen = JsonFieldReorder(seed=42)
        obj = json.dumps({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})
        results = await gen.perturb(obj, n=5)
        # Should have some unique orderings
        unique = set(results)
        assert len(unique) > 1

    @pytest.mark.asyncio
    async def test_non_json_returns_original(self):
        gen = JsonFieldReorder()
        results = await gen.perturb("not json", n=3)
        assert all(r == "not json" for r in results)

    @pytest.mark.asyncio
    async def test_non_dict_returns_original(self):
        gen = JsonFieldReorder()
        results = await gen.perturb("[1, 2, 3]", n=3)
        assert all(r == "[1, 2, 3]" for r in results)

    @pytest.mark.asyncio
    async def test_recursive_mode(self):
        gen = JsonFieldReorder(seed=42, recursive=True)
        obj = json.dumps({"a": {"x": 1, "y": 2}, "b": {"m": 3, "n": 4}})
        results = await gen.perturb(obj, n=5)
        for r in results:
            parsed = json.loads(r)
            assert parsed["a"]["x"] == 1
            assert parsed["b"]["m"] == 3

    @pytest.mark.asyncio
    async def test_deterministic_with_seed(self):
        results1 = await JsonFieldReorder(seed=123).perturb('{"a": 1, "b": 2, "c": 3}', n=3)
        results2 = await JsonFieldReorder(seed=123).perturb('{"a": 1, "b": 2, "c": 3}', n=3)
        assert results1 == results2


@pytest.mark.unit
class TestSchemaVariation:
    @pytest.mark.asyncio
    async def test_produces_n_results(self):
        gen = SchemaVariation(seed=42)
        obj = json.dumps({"name": "test", "value": 42, "active": True})
        results = await gen.perturb(obj, n=4)
        assert len(results) == 4

    @pytest.mark.asyncio
    async def test_valid_json(self):
        gen = SchemaVariation(seed=42)
        obj = json.dumps({"name": "test", "value": 42})
        results = await gen.perturb(obj, n=4)
        for r in results:
            json.loads(r)  # should not raise

    @pytest.mark.asyncio
    async def test_variations_differ(self):
        gen = SchemaVariation(seed=42)
        obj = json.dumps({"first_name": "John", "age": 30, "is_active": True})
        results = await gen.perturb(obj, n=4)
        unique = set(results)
        assert len(unique) > 1

    @pytest.mark.asyncio
    async def test_non_json_returns_original(self):
        gen = SchemaVariation()
        results = await gen.perturb("plain text", n=3)
        assert all(r == "plain text" for r in results)

    @pytest.mark.asyncio
    async def test_casing_change(self):
        gen = SchemaVariation(seed=42)
        obj = json.dumps({"user_name": "test"})
        results = await gen.perturb(obj, n=4)
        # The casing transform (index 2) should produce camelCase
        parsed = json.loads(results[2])
        assert "userName" in parsed or "user_name" in parsed


@pytest.mark.unit
class TestTypoInjection:
    @pytest.mark.asyncio
    async def test_produces_n_results(self):
        gen = TypoInjection(seed=42)
        results = await gen.perturb("Hello world, how are you?", n=5)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_results_differ_from_original(self):
        gen = TypoInjection(rate=0.2, seed=42)
        text = "The quick brown fox jumps over the lazy dog"
        results = await gen.perturb(text, n=5)
        # At least some should differ
        assert any(r != text for r in results)

    @pytest.mark.asyncio
    async def test_deterministic_with_seed(self):
        results1 = await TypoInjection(seed=99).perturb("Hello world", n=3)
        results2 = await TypoInjection(seed=99).perturb("Hello world", n=3)
        assert results1 == results2

    @pytest.mark.asyncio
    async def test_empty_string(self):
        gen = TypoInjection()
        results = await gen.perturb("", n=3)
        assert all(r == "" for r in results)

    @pytest.mark.asyncio
    async def test_low_rate(self):
        gen = TypoInjection(rate=0.01, seed=42)
        text = "Short"
        results = await gen.perturb(text, n=3)
        # With very low rate on short text, changes should be minimal
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_high_rate(self):
        gen = TypoInjection(rate=0.5, seed=42)
        text = "Hello world test string"
        results = await gen.perturb(text, n=3)
        # With high rate, all should differ from original
        assert all(r != text for r in results)
