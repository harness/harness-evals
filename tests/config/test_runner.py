"""Tests for config runner — builder functions and run_config."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from harness_evals.config.runner import (
    build_baseline_store,
    build_metric,
    build_sink,
    build_target,
    gate_against_baseline,
    run_config,
    scores_to_baseline_dict,
)
from harness_evals.config.schema import (
    BaselineSpec,
    EvalConfig,
    MetricSpec,
    SinkSpec,
    TargetSpec,
)
from harness_evals.core.score import Score
from harness_evals.errors import BaselineRegressionError, HarnessEvalsError, UnknownMetricError
from harness_evals.refs import ResourceRef
from harness_evals.sinks.json_sink import JsonSink
from harness_evals.sinks.stdout import StdoutSink

# ---------------------------------------------------------------------------
# build_metric
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildMetric:
    def test_deterministic_metric(self) -> None:
        metric = build_metric(MetricSpec(kind="exact_match"))
        assert metric.name == "exact_match"
        assert metric.threshold == 1.0

    def test_threshold_override(self) -> None:
        metric = build_metric(MetricSpec(kind="exact_match", threshold=0.8))
        assert metric.threshold == 0.8

    def test_params_forwarded(self) -> None:
        metric = build_metric(MetricSpec(kind="latency", params={"max_ms": 2000}))
        assert metric.max_ms == 2000

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(UnknownMetricError, match="no_such_metric"):
            build_metric(MetricSpec(kind="no_such_metric"))

    def test_llm_metric_without_llm_raises(self) -> None:
        with pytest.raises(HarnessEvalsError, match="requires an LLM"):
            build_metric(MetricSpec(kind="geval", params={"criteria": "test"}))

    def test_llm_metric_with_llm(self, mock_llm) -> None:
        llm = mock_llm()
        metric = build_metric(MetricSpec(kind="geval", params={"criteria": "Is it correct?"}), llm=llm)
        assert metric.name == "geval"

    @pytest.mark.parametrize(
        "kind",
        [
            "turn_faithfulness",
            "turn_contextual_precision",
            "turn_contextual_recall",
            "turn_contextual_relevancy",
        ],
    )
    def test_turn_level_rag_metrics_registered(self, kind: str, mock_llm) -> None:
        # Regression: these were exported but not in the registry, so config
        # users specifying them hit UnknownMetricError (PR #38 medium finding).
        metric = build_metric(MetricSpec(kind=kind), llm=mock_llm())
        assert metric.name == kind

    def test_turn_level_rag_metric_without_llm_raises(self) -> None:
        with pytest.raises(HarnessEvalsError, match="requires an LLM"):
            build_metric(MetricSpec(kind="turn_faithfulness"))


# ---------------------------------------------------------------------------
# env var resolution in model params
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnvVarResolution:
    def test_resolve_env_in_build_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_TEST_KEY", "sk-test-123")
        from harness_evals.config.runner import _resolve_env_in_params

        result = _resolve_env_in_params({"api_key": "${MY_TEST_KEY}", "temperature": 0.5})
        assert result["api_key"] == "sk-test-123"
        assert result["temperature"] == 0.5

    def test_missing_env_var_raises(self) -> None:
        from harness_evals.config.runner import _resolve_env_in_params

        with pytest.raises(ValueError, match="NONEXISTENT_VAR_XYZ"):
            _resolve_env_in_params({"api_key": "${NONEXISTENT_VAR_XYZ}"})

    def test_literal_strings_untouched(self) -> None:
        from harness_evals.config.runner import _resolve_env_in_params

        result = _resolve_env_in_params({"api_key": "sk-literal", "model": "gpt-4o"})
        assert result["api_key"] == "sk-literal"
        assert result["model"] == "gpt-4o"

    def test_resolves_nested_env_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("API_HOST", "http://localhost:8000")
        monkeypatch.setenv("ORG_ID", "default")
        monkeypatch.setenv("PROJECT_ID", "demo")
        from harness_evals.config.runner import _resolve_env_in_params

        result = _resolve_env_in_params(
            {
                "url": "${API_HOST}/chat?org=${ORG_ID}&project=${PROJECT_ID}",
                "headers": {"X-Org": "${ORG_ID}"},
                "body_template": {
                    "harness_context": {
                        "org_id": "${ORG_ID}",
                        "project_id": "${PROJECT_ID}",
                    },
                    "events": ["${ORG_ID}", "literal"],
                },
            }
        )

        assert result["url"] == "http://localhost:8000/chat?org=default&project=demo"
        assert result["headers"]["X-Org"] == "default"
        assert result["body_template"]["harness_context"]["org_id"] == "default"
        assert result["body_template"]["harness_context"]["project_id"] == "demo"
        assert result["body_template"]["events"] == ["default", "literal"]


# ---------------------------------------------------------------------------
# build_sink
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildSink:
    def test_stdout(self) -> None:
        sink = build_sink(SinkSpec(type="stdout"))
        assert isinstance(sink, StdoutSink)

    def test_json_with_path(self, tmp_path) -> None:
        sink = build_sink(SinkSpec(type="json", params={"path": str(tmp_path / "out.jsonl")}))
        assert isinstance(sink, JsonSink)

    def test_csv(self, tmp_path) -> None:
        from harness_evals.sinks.csv_sink import CsvSink

        sink = build_sink(SinkSpec(type="csv", params={"path": str(tmp_path / "out.csv")}))
        assert isinstance(sink, CsvSink)


# ---------------------------------------------------------------------------
# build_target
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildTarget:
    async def test_http_target(self) -> None:
        spec = TargetSpec(type="http", params={"url": "http://localhost:8080/run"})
        target = await build_target(spec)
        from harness_evals.targets.http import HttpTarget

        assert isinstance(target, HttpTarget)
        assert target.url == "http://localhost:8080/run"

    async def test_http_target_with_bearer_auth(self) -> None:
        spec = TargetSpec(
            type="http",
            params={
                "url": "http://localhost:8080/run",
                "auth": {"type": "bearer", "token": "tok123"},
            },
        )
        target = await build_target(spec)
        from harness_evals.targets.auth import BearerAuth

        assert isinstance(target.auth, BearerAuth)

    async def test_streaming_http_target_resolves_env_in_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("API_HOST", "http://localhost:8000")
        monkeypatch.setenv("ACCOUNT_ID", "acct")
        monkeypatch.setenv("ORG_ID", "default")
        monkeypatch.setenv("PROJECT_ID", "demo")
        spec = TargetSpec(
            type="streaming_http",
            params={
                "url": "${API_HOST}/chat/unified?orgIdentifier=${ORG_ID}&projectIdentifier=${PROJECT_ID}",
                "body_template": {
                    "prompt": None,
                    "stream": True,
                    "harness_context": {
                        "account_id": "${ACCOUNT_ID}",
                        "org_id": "${ORG_ID}",
                        "project_id": "${PROJECT_ID}",
                    },
                },
            },
        )

        target = await build_target(spec)
        from harness_evals.targets.streaming_http import StreamingHttpTarget

        assert isinstance(target, StreamingHttpTarget)
        assert target.url == "http://localhost:8000/chat/unified?orgIdentifier=default&projectIdentifier=demo"
        assert target.body_template is not None
        assert target.body_template["harness_context"] == {
            "account_id": "acct",
            "org_id": "default",
            "project_id": "demo",
        }

    async def test_prompt_target(self, tmp_path) -> None:
        prompt_path = tmp_path / "prompt.txt"
        prompt_path.write_text("Answer {{input}}", encoding="utf-8")

        spec = TargetSpec(
            type="prompt",
            params={
                "prompt": str(prompt_path),
                "model": {"provider": "openai", "name": "gpt-4o"},
            },
        )

        with patch("harness_evals.config.runner.build_llm") as mock_build_llm:
            mock_build_llm.return_value = AsyncMock()
            target = await build_target(spec)

        from harness_evals.targets.prompt import PromptTarget

        assert isinstance(target, PromptTarget)

    async def test_prompt_target_missing_prompt_raises(self) -> None:
        spec = TargetSpec(type="prompt", params={"model": {"provider": "openai", "name": "gpt-4o"}})
        with pytest.raises(HarnessEvalsError, match="prompt"):
            await build_target(spec)

    async def test_prompt_target_missing_model_raises(self, tmp_path) -> None:
        prompt_path = tmp_path / "prompt.txt"
        prompt_path.write_text("Answer {{input}}", encoding="utf-8")
        spec = TargetSpec(type="prompt", params={"prompt": str(prompt_path)})
        with pytest.raises(HarnessEvalsError, match="model"):
            await build_target(spec)


# ---------------------------------------------------------------------------
# build_baseline_store
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildBaselineStore:
    def test_json_store(self, tmp_path) -> None:
        spec = BaselineSpec(store="json", path=str(tmp_path / "baselines"))
        store = build_baseline_store(spec)
        from harness_evals.baseline.json_store import JsonBaselineStore

        assert isinstance(store, JsonBaselineStore)


# ---------------------------------------------------------------------------
# scores_to_baseline_dict
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScoresToBaselineDict:
    def test_pivots_correctly(self) -> None:
        scores = [
            [Score(name="exact_match", value=1.0, threshold=0.8), Score(name="latency", value=0.9, threshold=0.5)],
            [Score(name="exact_match", value=0.5, threshold=0.8), Score(name="latency", value=0.7, threshold=0.5)],
        ]
        result = scores_to_baseline_dict(scores)
        assert len(result["exact_match"]) == 2
        assert len(result["latency"]) == 2
        assert result["exact_match"][0].value == 1.0
        assert result["exact_match"][1].value == 0.5


# ---------------------------------------------------------------------------
# gate_against_baseline
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGateAgainstBaseline:
    def test_raises_on_regression(self, tmp_path) -> None:
        from harness_evals.baseline.json_store import JsonBaselineStore

        baseline_dir = str(tmp_path / "baselines")
        store = JsonBaselineStore(baseline_dir=baseline_dir)
        store.save("baseline-run", {"m": [Score(name="m", value=0.9, threshold=0.5)]})

        spec = BaselineSpec(store="json", path=baseline_dir, tolerance=0.05)
        current_scores = [[Score(name="m", value=0.5, threshold=0.5)]]

        with pytest.raises(BaselineRegressionError):
            gate_against_baseline(current_scores, spec)

    def test_passes_when_no_regression(self, tmp_path) -> None:
        from harness_evals.baseline.json_store import JsonBaselineStore

        baseline_dir = str(tmp_path / "baselines")
        store = JsonBaselineStore(baseline_dir=baseline_dir)
        store.save("baseline-run", {"m": [Score(name="m", value=0.9, threshold=0.5)]})

        spec = BaselineSpec(store="json", path=baseline_dir, tolerance=0.05)
        current_scores = [[Score(name="m", value=0.9, threshold=0.5)]]

        gate_against_baseline(current_scores, spec)


# ---------------------------------------------------------------------------
# run_config end-to-end (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunConfig:
    def test_end_to_end_with_local_dataset(self, tmp_path) -> None:
        dataset_path = tmp_path / "goldens.jsonl"
        dataset_path.write_text('{"input": "2+2?", "expected": "4"}\n')

        prompt_path = tmp_path / "prompt.txt"
        prompt_path.write_text("Answer: {{input}}", encoding="utf-8")

        from tests.conftest import MockLLM

        mock = MockLLM()

        cfg = EvalConfig(
            name="test",
            dataset=ResourceRef(source="local", id=str(dataset_path)),
            target=TargetSpec(
                type="prompt",
                params={
                    "prompt": str(prompt_path),
                    "model": mock,
                },
            ),
            metrics=[MetricSpec(kind="exact_match")],
            sinks=[],
        )

        with patch("harness_evals.config.runner.build_llm") as mock_build_llm:
            mock_build_llm.return_value = mock
            scores = run_config(cfg)

        assert len(scores) == 1
        assert len(scores[0]) == 1
        assert scores[0][0].name == "exact_match"
