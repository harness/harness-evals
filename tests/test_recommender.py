"""Tests for the eval recommender module."""

from __future__ import annotations
import json
import pytest
from harness_evals.recommender.scenarios import load_scenario, ScenarioType
from harness_evals.recommender.output import write_outputs


MOCK_RECOMMENDATION = {
    "dimensions_covered": [
        {"dimension": "correctness", "applies": True, "rationale": "Agent must produce accurate output."},
        {"dimension": "performance", "applies": True, "rationale": "Latency matters in production."},
    ],
    "recommended_metrics": [
        {"name": "exact_match", "dimension": "correctness", "rationale": "Checks exact output.", "threshold": 0.8},
        {"name": "latency", "dimension": "performance", "rationale": "Measures response time.", "threshold": 0.7},
    ],
    "recommended_dataset": [
        {
            "input": "What is 2+2?",
            "expected": "4",
            "context": None,
            "expected_tools": None,
            "expected_tool_calls": None,
            "metadata": {},
            "tags": {},
            "metric_tested": "exact_match",
        }
    ],
    "recommended_actions": "Run harness-evals run recommended.eval.yaml --fail-under 0.8",
}


def test_load_scenario_prompt():
    s = load_scenario(prompt="You are a helpful assistant.")
    assert s.type == ScenarioType.PROMPT
    assert "helpful assistant" in s.content


def test_load_scenario_endpoint():
    s = load_scenario(endpoint="http://localhost:8080/run")
    assert s.type == ScenarioType.HTTP_ENDPOINT
    assert s.content == "http://localhost:8080/run"


def test_load_scenario_traces(tmp_path):
    traces_file = tmp_path / "traces.jsonl"
    traces_file.write_text('{"input": "hi", "output": "hello"}\n')
    s = load_scenario(traces=str(traces_file))
    assert s.type == ScenarioType.TRACES
    assert "hi" in s.content


def test_load_scenario_requires_exactly_one():
    with pytest.raises(ValueError):
        load_scenario(prompt="a", endpoint="b")
    with pytest.raises(ValueError):
        load_scenario()


def test_write_outputs(tmp_path):
    config_path, goldens_path = write_outputs(MOCK_RECOMMENDATION, output_dir=str(tmp_path))
    assert config_path.exists()
    assert goldens_path.exists()
    with goldens_path.open() as f:
        lines = f.readlines()
    assert len(lines) == 1
    golden = json.loads(lines[0])
    assert golden["input"] == "What is 2+2?"
    assert golden["expected"] == "4"
    assert "context" in golden
    assert "expected_tools" in golden
    assert "expected_tool_calls" in golden
    assert "metadata" in golden
    assert "tags" in golden


def test_recommended_metrics_use_catalog_names():
    from harness_evals.catalog import catalog
    catalog_kinds = {e.kind for e in catalog()}
    for m in MOCK_RECOMMENDATION["recommended_metrics"]:
        assert m["name"] in catalog_kinds, f"{m['name']} not in catalog"
