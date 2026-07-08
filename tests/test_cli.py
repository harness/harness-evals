"""Tests for the harness-evals CLI."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness_evals.cli import main

_MINIMAL_CONFIG = """\
name: cli-test
dataset: {dataset_path}
target:
  type: prompt
  prompt: {prompt_path}
  model: {{provider: openai, name: gpt-4o}}
metrics:
  - exact_match
sinks: []
"""


def _write_config(tmp_path, goldens_content='{"input": "hi", "expected": "hello"}\n', prompt_content="{{input}}"):
    dataset_path = tmp_path / "goldens.jsonl"
    dataset_path.write_text(goldens_content)
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text(prompt_content, encoding="utf-8")
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(_MINIMAL_CONFIG.format(dataset_path=str(dataset_path), prompt_path=str(prompt_path)))
    return str(config_path)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCmdRun:
    def test_run_success(self, tmp_path) -> None:
        config_path = _write_config(tmp_path, goldens_content='{"input": "4", "expected": "4"}\n')

        from tests.conftest import MockLLM

        mock = MockLLM()
        mock._fixed_output = "4"

        async def fake_generate(prompt, **kwargs):
            return "4"

        mock.generate = fake_generate

        with patch("harness_evals.config.runner.build_llm", return_value=mock):
            exit_code = main(["run", config_path])

        assert exit_code == 0

    def test_run_file_not_found(self) -> None:
        exit_code = main(["run", "/nonexistent/config.yaml"])
        assert exit_code == 2

    def test_run_validate_only(self, tmp_path, capsys) -> None:
        config_path = _write_config(tmp_path)
        exit_code = main(["run", config_path, "--validate"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Config valid" in captured.err

    def test_run_fail_under(self, tmp_path) -> None:
        config_path = _write_config(tmp_path)

        from tests.conftest import MockLLM

        mock = MockLLM()

        async def fake_generate(prompt, **kwargs):
            return "wrong answer"

        mock.generate = fake_generate

        with patch("harness_evals.config.runner.build_llm", return_value=mock):
            exit_code = main(["run", config_path, "--fail-under", "0.99"])

        assert exit_code == 1

    def test_run_baseline_regression_exit_code(self, tmp_path, capsys) -> None:
        _write_config(tmp_path, goldens_content='{"input": "4", "expected": "4"}\n')

        from tests.conftest import MockLLM

        mock = MockLLM()

        async def fake_generate(prompt, **kwargs):
            return "4"

        mock.generate = fake_generate

        result_mock = MagicMock()
        result_mock.has_regressions = True
        result_mock.summary.return_value = "exact_match regressed by 0.10"

        with (
            patch("harness_evals.config.runner.build_llm", return_value=mock),
            patch("harness_evals.config.runner.compare_to_baseline", return_value=result_mock),
            patch("harness_evals.baseline.json_store.JsonBaselineStore.load", return_value={}),
        ):
            cfg_text = (
                "name: baseline-test\n"
                f"dataset: {tmp_path / 'goldens.jsonl'}\n"
                "target:\n"
                "  type: prompt\n"
                f"  prompt: {tmp_path / 'prompt.txt'}\n"
                "  model: {provider: openai, name: gpt-4o}\n"
                "metrics:\n"
                "  - exact_match\n"
                "baseline:\n"
                "  store: json\n"
                f"  path: {tmp_path / 'baselines'}\n"
            )
            cfg_path = tmp_path / "baseline-eval.yaml"
            cfg_path.write_text(cfg_text)
            exit_code = main(["run", str(cfg_path), "--baseline"])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Baseline regression" in captured.err

    def test_run_baseline_and_fail_under_both_reported(self, tmp_path, capsys) -> None:
        _write_config(tmp_path)

        from tests.conftest import MockLLM

        mock = MockLLM()

        async def fake_generate(prompt, **kwargs):
            return "wrong answer"

        mock.generate = fake_generate

        result_mock = MagicMock()
        result_mock.has_regressions = True
        result_mock.summary.return_value = "exact_match regressed"

        with (
            patch("harness_evals.config.runner.build_llm", return_value=mock),
            patch("harness_evals.config.runner.compare_to_baseline", return_value=result_mock),
            patch("harness_evals.baseline.json_store.JsonBaselineStore.load", return_value={}),
        ):
            cfg_text = (
                "name: both-test\n"
                f"dataset: {tmp_path / 'goldens.jsonl'}\n"
                "target:\n"
                "  type: prompt\n"
                f"  prompt: {tmp_path / 'prompt.txt'}\n"
                "  model: {provider: openai, name: gpt-4o}\n"
                "metrics:\n"
                "  - exact_match\n"
                "baseline:\n"
                "  store: json\n"
                f"  path: {tmp_path / 'baselines'}\n"
            )
            cfg_path = tmp_path / "both-eval.yaml"
            cfg_path.write_text(cfg_text)
            exit_code = main(["run", str(cfg_path), "--baseline", "--fail-under", "0.99"])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Baseline regression" in captured.err
        assert "--fail-under" in captured.err


# ---------------------------------------------------------------------------
# list-metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCmdListMetrics:
    def test_list_metrics_outputs_table(self, capsys) -> None:
        exit_code = main(["list-metrics"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "exact_match" in captured.out
        assert "metrics available" in captured.out


# ---------------------------------------------------------------------------
# discover
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCmdDiscover:
    def test_discovers_yaml_files(self, tmp_path) -> None:
        config_path = tmp_path / "my-eval.eval.yaml"
        config_path.write_text("""\
name: discovered-eval
dataset: ./goldens.jsonl
target: {type: http, url: http://localhost:8080}
metrics: [exact_match]
""")
        exit_code = main(["discover", str(tmp_path)])
        assert exit_code == 0

    def test_discovers_nothing_in_empty_dir(self, tmp_path, capsys) -> None:
        exit_code = main(["discover", str(tmp_path)])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "0 eval(s) discovered" in captured.err

    def test_skips_hidden_directories(self, tmp_path, capsys) -> None:
        hidden = tmp_path / ".git" / "hooks"
        hidden.mkdir(parents=True)
        (hidden / "pre-commit.eval.yaml").write_text(
            "name: shouldskip\ndataset: x\ntarget: {type: http, url: http://x}\nmetrics: [exact_match]\n"
        )
        visible = tmp_path / "evals"
        visible.mkdir()
        (visible / "real.eval.yaml").write_text(
            "name: real\ndataset: x\ntarget: {type: http, url: http://x}\nmetrics: [exact_match]\n"
        )
        exit_code = main(["discover", str(tmp_path)])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "name=real" in captured.out
        assert "shouldskip" not in captured.out
        assert "1 eval(s) discovered" in captured.err

    def test_nonexistent_path(self) -> None:
        exit_code = main(["discover", "/nonexistent/path"])
        assert exit_code == 2


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCmdImport:
    def test_import_writes_yaml(self, tmp_path, capsys) -> None:
        from harness_evals.config.schema import EvalConfig, MetricSpec, SinkSpec, TargetSpec
        from harness_evals.refs import ResourceRef

        fake_cfg = EvalConfig(
            name="imported-eval",
            dataset=ResourceRef(source="file", id="data.jsonl"),
            target=TargetSpec(type="http", params={"url": "http://localhost"}),
            metrics=[MetricSpec(kind="exact_match")],
            sinks=[SinkSpec(type="stdout")],
        )

        mock_source = AsyncMock()
        mock_source.fetch = AsyncMock(return_value=fake_cfg)
        mock_source_cls = MagicMock(return_value=mock_source)

        out_path = tmp_path / "out.yaml"
        with (
            patch("harness_evals.plugins.eval_config_source", return_value=mock_source_cls),
            patch("harness_evals.refs.resolve", return_value=MagicMock(source="harness")),
        ):
            exit_code = main(["import", "harness://evals/test@1", "-o", str(out_path)])

        assert exit_code == 0
        assert out_path.exists()
        content = out_path.read_text()
        assert "imported-eval" in content


# ---------------------------------------------------------------------------
# no command / help
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNoCommand:
    def test_no_args_prints_help(self, capsys) -> None:
        exit_code = main([])
        assert exit_code == 0
