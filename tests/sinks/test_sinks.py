"""Tests for StdoutSink and JsonSink."""

import json

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score
from harness_evals.sinks.json_sink import JsonSink
from harness_evals.sinks.stdout import StdoutSink


@pytest.fixture
def eval_case() -> EvalCase:
    return EvalCase(input="What is 2+2?", output="4", expected="4")


@pytest.fixture
def scores() -> list[Score]:
    return [
        Score(name="exact_match", value=1.0, threshold=1.0),
        Score(name="latency", value=0.75, threshold=0.5, reason="1250ms"),
    ]


@pytest.mark.unit
class TestStdoutSink:
    def test_write_prints_scores(self, capsys, eval_case, scores):
        sink = StdoutSink()
        sink.write(scores, eval_case)
        captured = capsys.readouterr().out

        assert "PASS" in captured
        assert "exact_match" in captured
        assert "latency" in captured

    def test_write_shows_fail(self, capsys):
        ec = EvalCase(input="q", output="wrong", expected="right")
        failing_scores = [Score(name="test", value=0.2, threshold=0.8, reason="bad")]
        sink = StdoutSink()
        sink.write(failing_scores, ec)
        captured = capsys.readouterr().out

        assert "FAIL" in captured
        assert "bad" in captured


@pytest.mark.unit
class TestJsonSink:
    def test_write_creates_file(self, tmp_path, eval_case, scores):
        path = tmp_path / "results.jsonl"
        sink = JsonSink(str(path))
        sink.write(scores, eval_case)

        assert path.exists()
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["input"] == "What is 2+2?"
        assert len(record["scores"]) == 2
        assert record["scores"][0]["name"] == "exact_match"

    def test_write_appends(self, tmp_path, eval_case, scores):
        path = tmp_path / "results.jsonl"
        sink = JsonSink(str(path))
        sink.write(scores, eval_case)
        sink.write(scores, eval_case)

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_creates_parent_dirs(self, tmp_path, eval_case, scores):
        path = tmp_path / "nested" / "deep" / "results.jsonl"
        sink = JsonSink(str(path))
        sink.write(scores, eval_case)

        assert path.exists()

    def test_scores_contain_passed(self, tmp_path, eval_case, scores):
        path = tmp_path / "results.jsonl"
        sink = JsonSink(str(path))
        sink.write(scores, eval_case)

        record = json.loads(path.read_text().strip())
        for s in record["scores"]:
            assert "passed" in s
