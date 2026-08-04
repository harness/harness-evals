"""Tests for the eval recommender module."""

from __future__ import annotations

import json

import pytest
import yaml

from harness_evals.core.golden import Golden
from harness_evals.errors import HarnessEvalsError
from harness_evals.llm.base import BaseLLM
from harness_evals.recommender.engine import RECOMMENDATION_SCHEMA, recommend
from harness_evals.recommender.output import (
    build_eval_config,
    build_goldens,
    default_model,
    write_outputs,
)
from harness_evals.recommender.scenarios import ScenarioInput, ScenarioType, load_scenario

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
            "metric_tested": "exact_match",
        }
    ],
    "recommended_actions": "Run harness-evals run recommended.eval.yaml --fail-under 0.8",
}


class RecordingLLM(BaseLLM):
    """A BaseLLM that records generate_json calls and returns a canned dict."""

    def __init__(self, response: dict) -> None:
        self._response = response
        self.calls: list[dict] = []

    async def generate(self, prompt: str, **kwargs: object) -> str:
        return ""

    async def generate_json(self, prompt: str, schema: dict, **kwargs: object) -> dict:
        self.calls.append({"prompt": prompt, "schema": schema, "kwargs": kwargs})
        return self._response


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_scenario_prompt():
    s = load_scenario(prompt="You are a helpful assistant.")
    assert s.type == ScenarioType.PROMPT
    assert "helpful assistant" in s.content


@pytest.mark.unit
def test_load_scenario_endpoint():
    s = load_scenario(endpoint="http://localhost:8080/run")
    assert s.type == ScenarioType.HTTP_ENDPOINT
    assert s.content == "http://localhost:8080/run"


@pytest.mark.unit
def test_load_scenario_traces(tmp_path):
    traces_file = tmp_path / "traces.jsonl"
    traces_file.write_text('{"input": "hi", "output": "hello"}\n')
    s = load_scenario(traces=str(traces_file))
    assert s.type == ScenarioType.TRACES
    assert "hi" in s.content


@pytest.mark.unit
def test_load_scenario_requires_exactly_one():
    with pytest.raises(HarnessEvalsError):
        load_scenario(prompt="a", endpoint="b")
    with pytest.raises(HarnessEvalsError):
        load_scenario()


# ---------------------------------------------------------------------------
# engine — recommend() over a BaseLLM
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_recommend_uses_base_llm_generate_json():
    scenario = ScenarioInput(type=ScenarioType.PROMPT, content="You are a helpful assistant.")
    llm = RecordingLLM(MOCK_RECOMMENDATION)

    result = await recommend(scenario=scenario, llm=llm)

    assert result == MOCK_RECOMMENDATION
    assert len(llm.calls) == 1
    call = llm.calls[0]
    # Schema is passed through to generate_json.
    assert call["schema"] is RECOMMENDATION_SCHEMA
    # The system prompt is passed via kwargs (not the user prompt).
    assert "system_prompt" in call["kwargs"]
    assert "evaluation expert" in str(call["kwargs"]["system_prompt"])
    # The user prompt contains the scenario content and the catalog.
    assert "helpful assistant" in call["prompt"]
    assert "exact_match" in call["prompt"]


@pytest.mark.unit
def test_recommend_schema_is_valid_json_schema():
    assert RECOMMENDATION_SCHEMA["type"] == "object"
    props = RECOMMENDATION_SCHEMA["properties"]
    assert set(props) == {
        "dimensions_covered",
        "recommended_metrics",
        "recommended_dataset",
        "recommended_actions",
    }


@pytest.mark.unit
def test_recommended_metrics_use_catalog_names():
    from harness_evals.catalog import catalog

    catalog_kinds = {e.kind for e in catalog()}
    for m in MOCK_RECOMMENDATION["recommended_metrics"]:
        assert m["name"] in catalog_kinds, f"{m['name']} not in catalog"


# ---------------------------------------------------------------------------
# output — Golden + dataset + EvalConfig
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_goldens_returns_golden_objects():
    goldens = build_goldens(MOCK_RECOMMENDATION)
    assert len(goldens) == 1
    g = goldens[0]
    assert isinstance(g, Golden)
    assert g.input == "What is 2+2?"
    assert g.expected == "4"
    # metric_tested is folded into metadata.
    assert g.metadata is not None
    assert g.metadata["metric_tested"] == "exact_match"


@pytest.mark.unit
def test_build_eval_config_uses_specs():
    from harness_evals.config.schema import EvalConfig

    cfg = build_eval_config(MOCK_RECOMMENDATION, provider="anthropic")
    assert isinstance(cfg, EvalConfig)
    assert cfg.name == "recommended-eval"
    # judge_llm present with the anthropic default model.
    assert cfg.judge_llm is not None
    assert cfg.judge_llm.provider == "anthropic"
    assert cfg.judge_llm.name == default_model("anthropic")
    # metrics carried across with thresholds.
    kinds = {m.kind: m.threshold for m in cfg.metrics}
    assert kinds == {"exact_match": 0.8, "latency": 0.7}
    # relative dataset path.
    assert cfg.dataset.source == "local"
    assert cfg.dataset.id == "recommended.goldens.jsonl"
    # baseline present.
    assert cfg.baseline is not None


@pytest.mark.unit
def test_write_outputs(tmp_path):
    config_path, goldens_path = write_outputs(MOCK_RECOMMENDATION, output_dir=str(tmp_path), provider="anthropic")
    assert config_path.exists()
    assert goldens_path.exists()

    # Dataset is JSONL written via save_dataset — one Golden per line.
    with goldens_path.open() as f:
        lines = [line for line in f.read().splitlines() if line.strip()]
    assert len(lines) == 1
    golden = json.loads(lines[0])
    assert golden["input"] == "What is 2+2?"
    assert golden["expected"] == "4"

    # Config is valid YAML with judge_llm + anthropic provider + relative dataset.
    config_data = yaml.safe_load(config_path.read_text())
    assert config_data["dataset"] == "recommended.goldens.jsonl"
    assert config_data["judge_llm"]["provider"] == "anthropic"
    assert config_data["judge_llm"]["name"] == "claude-sonnet-4-20250514"
    assert "baseline" in config_data


@pytest.mark.unit
def test_write_outputs_config_reloads_as_eval_config(tmp_path):
    from harness_evals.config.schema import load_config

    config_path, _ = write_outputs(MOCK_RECOMMENDATION, output_dir=str(tmp_path), provider="openai")
    cfg = load_config(str(config_path))
    assert cfg.name == "recommended-eval"
    assert cfg.judge_llm is not None
    assert cfg.judge_llm.provider == "openai"
    assert {m.kind for m in cfg.metrics} == {"exact_match", "latency"}


@pytest.mark.unit
def test_write_outputs_openai_provider(tmp_path):
    config_path, _ = write_outputs(MOCK_RECOMMENDATION, output_dir=str(tmp_path), provider="openai")
    config_data = yaml.safe_load(config_path.read_text())
    assert config_data["judge_llm"]["provider"] == "openai"
    assert config_data["judge_llm"]["name"] == "gpt-4o"


@pytest.mark.unit
def test_default_model():
    assert default_model("anthropic") == "claude-sonnet-4-20250514"
    assert default_model("openai") == "gpt-4o"
    # Unknown providers fall back to the anthropic default.
    assert default_model("unknown") == "claude-sonnet-4-20250514"
