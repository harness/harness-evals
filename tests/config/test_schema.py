"""Tests for config schema parsing and validation."""

from __future__ import annotations

import pytest

from harness_evals.config.schema import (
    BaselineSpec,
    ConversationSpec,
    MetricSpec,
    ModelSpec,
    SinkSpec,
    load_config,
    loads_config,
)
from harness_evals.errors import HarnessEvalsError
from harness_evals.refs import ResourceRef

_MINIMAL_YAML = """\
name: test-eval
dataset: ./goldens.jsonl
target:
  type: prompt
  prompt: ./prompt.txt
  model: {provider: openai, name: gpt-4o}
metrics:
  - exact_match
"""


@pytest.mark.unit
class TestLoadsConfig:
    def test_minimal_config(self) -> None:
        cfg = loads_config(_MINIMAL_YAML)

        assert cfg.name == "test-eval"
        assert cfg.dataset == ResourceRef(source="local", id="./goldens.jsonl")
        assert cfg.target.type == "prompt"
        assert cfg.target.params["prompt"] == "./prompt.txt"
        assert len(cfg.metrics) == 1
        assert cfg.metrics[0].kind == "exact_match"
        assert cfg.sinks == [SinkSpec("stdout")]
        assert cfg.baseline is None
        assert cfg.plugins == []
        assert cfg.judge_llm is None

    def test_conversation_config(self) -> None:
        cfg = loads_config("""\
name: conversation-eval
conversation:
  mode: simulate
  max_turns: 1
  max_elicitation_rounds: 6
  simulator_llm: {provider: openai, name: gpt-4o-mini}
dataset: ./harness-agent.goldens.jsonl
target:
  type: conversational_streaming_http
  url: http://localhost:8080/stream
metrics:
  - {kind: goal_accuracy, threshold: 0.7}
judge_llm: {provider: openai, name: gpt-4o}
""")

        assert cfg.conversation == ConversationSpec(
            mode="simulate",
            max_turns=1,
            max_elicitation_rounds=6,
            simulator_llm=ModelSpec(provider="openai", name="gpt-4o-mini"),
        )
        assert cfg.target.type == "conversational_streaming_http"

    def test_full_config(self) -> None:
        cfg = loads_config("""\
name: nightly-regression
dataset: langfuse://datasets/support-goldens@3
target:
  type: http
  url: http://localhost:8080/run
  output_path: $.answer
metrics:
  - exact_match
  - {kind: geval, threshold: 0.7, params: {criteria: "Correct and helpful?"}}
judge_llm: {provider: openai, name: gpt-4o}
sinks:
  - stdout
  - {type: json, path: results.jsonl}
baseline: {store: json, path: .evals/baseline.json, tolerance: 0.03}
plugins: [acme_evals.adapters]
""")

        assert cfg.name == "nightly-regression"
        assert cfg.dataset.source == "langfuse"
        assert cfg.dataset.id == "datasets/support-goldens"
        assert cfg.dataset.version == "3"
        assert cfg.target.type == "http"
        assert cfg.target.params["url"] == "http://localhost:8080/run"
        assert len(cfg.metrics) == 2
        assert cfg.metrics[0] == MetricSpec(kind="exact_match")
        assert cfg.metrics[1].kind == "geval"
        assert cfg.metrics[1].threshold == 0.7
        assert cfg.metrics[1].params == {"criteria": "Correct and helpful?"}
        assert cfg.judge_llm == ModelSpec(provider="openai", name="gpt-4o")
        assert len(cfg.sinks) == 2
        assert cfg.sinks[0] == SinkSpec(type="stdout")
        assert cfg.sinks[1] == SinkSpec(type="json", params={"path": "results.jsonl"})
        assert cfg.baseline == BaselineSpec(store="json", path=".evals/baseline.json", tolerance=0.03)
        assert cfg.plugins == ["acme_evals.adapters"]

    def test_dataset_dict_form(self) -> None:
        cfg = loads_config("""\
name: test
dataset: {source: langfuse, id: support-goldens, version: 3}
target: {type: http, url: http://localhost:8080}
metrics: [exact_match]
""")
        assert cfg.dataset.source == "langfuse"
        assert cfg.dataset.id == "support-goldens"
        assert cfg.dataset.version == "3"


@pytest.mark.unit
class TestValidation:
    def test_missing_name(self) -> None:
        with pytest.raises(HarnessEvalsError, match="name"):
            loads_config("dataset: ./g.jsonl\ntarget: {type: http}\nmetrics: [x]")

    def test_missing_dataset(self) -> None:
        with pytest.raises(HarnessEvalsError, match="dataset"):
            loads_config("name: x\ntarget: {type: http}\nmetrics: [x]")

    def test_missing_target(self) -> None:
        with pytest.raises(HarnessEvalsError, match="target"):
            loads_config("name: x\ndataset: ./g.jsonl\nmetrics: [x]")

    def test_empty_metrics(self) -> None:
        with pytest.raises(HarnessEvalsError, match="metrics"):
            loads_config("name: x\ndataset: ./g.jsonl\ntarget: {type: http}\nmetrics: []")

    def test_missing_metrics(self) -> None:
        with pytest.raises(HarnessEvalsError, match="metrics"):
            loads_config("name: x\ndataset: ./g.jsonl\ntarget: {type: http}")

    def test_unknown_top_level_key(self) -> None:
        with pytest.raises(HarnessEvalsError, match="unknown_key"):
            loads_config(_MINIMAL_YAML + "unknown_key: true\n")

    def test_not_a_mapping(self) -> None:
        with pytest.raises(HarnessEvalsError, match="mapping"):
            loads_config("- item1\n- item2\n")

    def test_metric_dict_missing_kind(self) -> None:
        with pytest.raises(HarnessEvalsError, match="kind"):
            loads_config("name: x\ndataset: ./g.jsonl\ntarget: {type: http}\nmetrics: [{threshold: 0.5}]")

    def test_target_missing_type(self) -> None:
        with pytest.raises(HarnessEvalsError, match="type"):
            loads_config("name: x\ndataset: ./g.jsonl\ntarget: {url: http://x}\nmetrics: [x]")

    def test_model_spec_missing_fields(self) -> None:
        with pytest.raises(HarnessEvalsError, match="provider"):
            loads_config("name: x\ndataset: ./g.jsonl\ntarget: {type: http}\nmetrics: [x]\njudge_llm: {name: gpt-4o}")

    def test_conversation_requires_llm_for_simulate(self) -> None:
        with pytest.raises(HarnessEvalsError, match="simulator_llm"):
            loads_config("""\
name: x
conversation: {mode: simulate}
dataset: ./g.jsonl
target: {type: conversational_streaming_http, url: http://localhost:8080/stream}
metrics: [exact_match]
""")

    def test_conversation_rejects_invalid_mode(self) -> None:
        with pytest.raises(HarnessEvalsError, match="conversation.mode"):
            loads_config("""\
name: x
conversation: {mode: invalid}
dataset: ./g.jsonl
target: {type: conversational_streaming_http, url: http://localhost:8080/stream}
metrics: [exact_match]
judge_llm: {provider: openai, name: gpt-4o}
""")


@pytest.mark.unit
class TestLoadConfigFile:
    def test_load_from_file(self, tmp_path) -> None:
        config_path = tmp_path / "eval.yaml"
        config_path.write_text(_MINIMAL_YAML)
        cfg = load_config(str(config_path))
        assert cfg.name == "test-eval"

    def test_file_not_found(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "nonexistent.yaml"))


@pytest.mark.unit
class TestMetricSpecParsing:
    def test_bare_string(self) -> None:
        cfg = loads_config(_MINIMAL_YAML)
        assert cfg.metrics[0] == MetricSpec(kind="exact_match")

    def test_dict_with_params(self) -> None:
        cfg = loads_config("""\
name: x
dataset: ./g.jsonl
target: {type: http, url: http://x}
metrics:
  - {kind: latency, threshold: 0.5, params: {max_ms: 3000}}
""")
        m = cfg.metrics[0]
        assert m.kind == "latency"
        assert m.threshold == 0.5
        assert m.params == {"max_ms": 3000}


@pytest.mark.unit
class TestSinkSpecParsing:
    def test_default_stdout(self) -> None:
        cfg = loads_config(_MINIMAL_YAML)
        assert cfg.sinks == [SinkSpec(type="stdout")]

    def test_mixed_sinks(self) -> None:
        cfg = loads_config(_MINIMAL_YAML.rstrip() + "\nsinks: [stdout, {type: json, path: out.jsonl}]\n")
        assert len(cfg.sinks) == 2
        assert cfg.sinks[1].type == "json"
        assert cfg.sinks[1].params == {"path": "out.jsonl"}


@pytest.mark.unit
class TestBaselineSpecParsing:
    def test_defaults(self) -> None:
        cfg = loads_config(_MINIMAL_YAML.rstrip() + "\nbaseline: {store: json}\n")
        assert cfg.baseline == BaselineSpec()

    def test_custom(self) -> None:
        cfg = loads_config(
            _MINIMAL_YAML.rstrip() + "\nbaseline: {store: json, path: /tmp/b, tolerance: 0.1, run_id: abc}\n"
        )
        assert cfg.baseline == BaselineSpec(store="json", path="/tmp/b", tolerance=0.1, run_id="abc")
