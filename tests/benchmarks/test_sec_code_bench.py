"""Tests for SecCodeBench adapter."""

from __future__ import annotations

from unittest.mock import PropertyMock, patch

import pytest

from harness_evals.benchmarks.sec_code_bench import SecCodeBench


@pytest.mark.unit
class TestSecCodeBench:
    async def test_load_bundled_manifest(self):
        bench = SecCodeBench()
        items = await bench.load_dataset(offline=True)
        assert len(items) >= 1
        assert items[0]["cwe"]

    async def test_heuristic_unsafe(self):
        bench = SecCodeBench()
        item = {"prompt": "fix code", "category": "Injection", "cwe": "CWE-89"}
        with patch.object(SecCodeBench, "docker_available", new_callable=PropertyMock, return_value=False):
            value, reason = bench.score_response(item, "os.system(user_input)")
        assert value == 0.0
        assert "unsafe" in (reason or "").lower()

    async def test_heuristic_safe(self):
        bench = SecCodeBench()
        item = {"prompt": "fix code", "category": "Injection", "cwe": "CWE-89"}
        with patch.object(SecCodeBench, "docker_available", new_callable=PropertyMock, return_value=False):
            value, _ = bench.score_response(item, "Use parameterized queries with placeholders.")
        assert value == 1.0

    async def test_run_with_mock(self, mock_llm_factory):
        bench = SecCodeBench()
        llm = mock_llm_factory(responses=["Use parameterized SQL with bound parameters."])
        result = await bench.run(llm, limit=1)
        assert result.num_total == 1
        assert "secure_pass_rate" in result.metrics
