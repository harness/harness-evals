"""Tests for all sinks: StdoutSink, JsonSink, CsvSink, JUnitSink, OtlpSink."""

import csv
import json
import warnings
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import pytest

from harness_evals import EvalCase, Score
from harness_evals.sinks.csv_sink import CsvSink
from harness_evals.sinks.json_sink import JsonSink
from harness_evals.sinks.junit_sink import JUnitSink
from harness_evals.sinks.stdout import StdoutSink

# ---------------------------------------------------------------------------
# StdoutSink
# ---------------------------------------------------------------------------


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

    def test_finalize_prints_summary(self, capsys, eval_case, scores):
        sink = StdoutSink(summary=True)
        sink.write(scores, eval_case)
        sink.write(scores, eval_case)
        capsys.readouterr()  # clear per-case output

        sink.finalize()
        captured = capsys.readouterr().out

        assert "Summary" in captured
        assert "exact_match" in captured
        assert "latency" in captured
        assert "pass_rate" in captured
        assert "Overall pass rate" in captured

    def test_finalize_prints_dimension_block_and_safety(self, capsys):
        ec = EvalCase(input="q", output="a")
        dim_scores = [
            Score(name="exact_match", value=1.0, threshold=0.8, metadata={"dimension": "correctness"}),
            Score(name="pii_leak", value=0.0, threshold=1.0, metadata={"dimension": "safety"}),
        ]
        sink = StdoutSink(summary=True)
        sink.write(dim_scores, ec)
        capsys.readouterr()  # clear per-case output

        sink.finalize()
        captured = capsys.readouterr().out

        assert "Dimensions:" in captured
        assert "correctness" in captured
        # Safety surfaced separately as a hard constraint.
        assert "Safety: 1 violation(s)" in captured
        assert "violation(s)" in captured

    def test_finalize_no_safety_line_without_safety_scores(self, capsys):
        ec = EvalCase(input="q", output="a")
        sink = StdoutSink(summary=True)
        sink.write([Score(name="exact_match", value=1.0, threshold=0.8, metadata={"dimension": "correctness"})], ec)
        capsys.readouterr()

        sink.finalize()
        captured = capsys.readouterr().out
        assert "Safety:" not in captured
        assert "Dimensions:" in captured

    def test_finalize_no_summary_when_disabled(self, capsys, eval_case, scores):
        sink = StdoutSink(summary=False)
        sink.write(scores, eval_case)
        capsys.readouterr()

        sink.finalize()
        captured = capsys.readouterr().out
        assert captured == ""

    def test_finalize_no_output_when_no_writes(self, capsys):
        sink = StdoutSink(summary=True)
        sink.finalize()
        captured = capsys.readouterr().out
        assert captured == ""


# ---------------------------------------------------------------------------
# JsonSink
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CsvSink
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCsvSink:
    def test_creates_file_with_header(self, tmp_path):
        path = tmp_path / "scores.csv"
        sink = CsvSink(str(path))
        ec = EvalCase(input="What is 2+2?", output="4", expected="4")
        scores = [Score(name="exact_match", value=1.0, threshold=1.0)]
        sink.write(scores, ec)

        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["metric"] == "exact_match"
        assert rows[0]["passed"] == "True"

    def test_appends_without_repeating_header(self, tmp_path):
        path = tmp_path / "scores.csv"
        sink = CsvSink(str(path))
        ec = EvalCase(input="q", output="a")

        sink.write([Score(name="m1", value=1.0, threshold=0.5)], ec)
        sink.write([Score(name="m2", value=0.3, threshold=0.5)], ec)

        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["metric"] == "m1"
        assert rows[1]["metric"] == "m2"

    def test_multiple_scores_per_write(self, tmp_path):
        path = tmp_path / "scores.csv"
        sink = CsvSink(str(path))
        ec = EvalCase(input="q", output="a")
        scores = [
            Score(name="m1", value=1.0, threshold=0.5),
            Score(name="m2", value=0.8, threshold=0.5),
        ]
        sink.write(scores, ec)

        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "scores.csv"
        sink = CsvSink(str(path))
        ec = EvalCase(input="q", output="a")
        sink.write([Score(name="m", value=1.0, threshold=0.5)], ec)
        assert path.exists()

    def test_value_formatting(self, tmp_path):
        path = tmp_path / "scores.csv"
        sink = CsvSink(str(path))
        ec = EvalCase(input="q", output="a")
        sink.write([Score(name="m", value=0.123456, threshold=0.5)], ec)

        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["value"] == "0.1235"

    def test_failed_score_recorded(self, tmp_path):
        path = tmp_path / "scores.csv"
        sink = CsvSink(str(path))
        ec = EvalCase(input="q", output="a")
        sink.write([Score(name="m", value=0.3, threshold=0.8, reason="too low")], ec)

        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["passed"] == "False"
        assert rows[0]["reason"] == "too low"


# ---------------------------------------------------------------------------
# JUnitSink
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJUnitSink:
    def test_produces_valid_xml(self, tmp_path):
        path = tmp_path / "results.xml"
        sink = JUnitSink(str(path))
        ec = EvalCase(input="q", output="a")
        sink.write([Score(name="m1", value=1.0, threshold=0.5)], ec)
        sink.finalize()

        tree = ET.parse(path)
        root = tree.getroot()
        assert root.tag == "testsuite"
        assert root.attrib["tests"] == "1"
        assert root.attrib["failures"] == "0"

    def test_failure_elements(self, tmp_path):
        path = tmp_path / "results.xml"
        sink = JUnitSink(str(path))
        ec = EvalCase(input="q", output="a")
        sink.write(
            [Score(name="m1", value=0.3, threshold=0.8, reason="below threshold")],
            ec,
        )
        sink.finalize()

        tree = ET.parse(path)
        root = tree.getroot()
        assert root.attrib["failures"] == "1"
        tc = root.find("testcase")
        failure = tc.find("failure")
        assert failure is not None
        assert "below threshold" in failure.text

    def test_multiple_writes_accumulate(self, tmp_path):
        path = tmp_path / "results.xml"
        sink = JUnitSink(str(path))
        ec1 = EvalCase(input="q1", output="a1")
        ec2 = EvalCase(input="q2", output="a2")
        sink.write([Score(name="m1", value=1.0, threshold=0.5)], ec1)
        sink.write([Score(name="m2", value=0.2, threshold=0.5)], ec2)
        sink.finalize()

        tree = ET.parse(path)
        root = tree.getroot()
        assert root.attrib["tests"] == "2"
        assert root.attrib["failures"] == "1"

    def test_custom_suite_name(self, tmp_path):
        path = tmp_path / "results.xml"
        sink = JUnitSink(str(path), suite_name="my-suite")
        ec = EvalCase(input="q", output="a")
        sink.write([Score(name="m", value=1.0, threshold=0.5)], ec)
        sink.finalize()

        tree = ET.parse(path)
        assert tree.getroot().attrib["name"] == "my-suite"

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "results.xml"
        sink = JUnitSink(str(path))
        ec = EvalCase(input="q", output="a")
        sink.write([Score(name="m", value=1.0, threshold=0.5)], ec)
        sink.finalize()
        assert path.exists()

    def test_xml_declaration_present(self, tmp_path):
        path = tmp_path / "results.xml"
        sink = JUnitSink(str(path))
        ec = EvalCase(input="q", output="a")
        sink.write([Score(name="m", value=1.0, threshold=0.5)], ec)
        sink.finalize()

        content = path.read_text()
        assert content.startswith("<?xml")

    def test_mixed_pass_fail(self, tmp_path):
        path = tmp_path / "results.xml"
        sink = JUnitSink(str(path))
        ec = EvalCase(input="q", output="a")
        sink.write(
            [
                Score(name="pass_metric", value=1.0, threshold=0.5),
                Score(name="fail_metric", value=0.1, threshold=0.5),
            ],
            ec,
        )
        sink.finalize()

        tree = ET.parse(path)
        root = tree.getroot()
        assert root.attrib["tests"] == "2"
        assert root.attrib["failures"] == "1"
        testcases = root.findall("testcase")
        names = {tc.attrib["name"] for tc in testcases}
        assert names == {"pass_metric", "fail_metric"}

    def test_strips_illegal_control_chars(self, tmp_path):
        path = tmp_path / "results.xml"
        sink = JUnitSink(str(path))
        ec = EvalCase(input="bad\x01input", output="a")
        sink.write(
            [Score(name="m1", value=0.3, threshold=0.8, reason="bad\x0breason")],
            ec,
        )
        sink.finalize()

        # Must parse without a ParseError.
        ET.parse(path)

        raw = path.read_bytes()
        assert b"\x0b" not in raw
        assert b"\x01" not in raw

    def test_no_time_attribute_when_duration_not_set(self, tmp_path):
        path = tmp_path / "results.xml"
        sink = JUnitSink(str(path))
        ec = EvalCase(input="q", output="a")
        sink.write([Score(name="m", value=0.9, threshold=0.5)], ec)
        sink.finalize()

        tree = ET.parse(path)
        tc = tree.getroot().find("testcase")
        assert "time" not in tc.attrib

    def test_time_attribute_from_scoring_duration_ms(self, tmp_path):
        path = tmp_path / "results.xml"
        sink = JUnitSink(str(path))
        ec = EvalCase(input="q", output="a")
        sink.write([Score(name="m", value=0.9, threshold=0.5, scoring_duration_ms=1234.0)], ec)
        sink.finalize()

        tree = ET.parse(path)
        tc = tree.getroot().find("testcase")
        assert tc.attrib["time"] == "1.234"

    def test_suite_time_sums_testcases(self, tmp_path):
        path = tmp_path / "results.xml"
        sink = JUnitSink(str(path))
        ec = EvalCase(input="q", output="a")
        sink.write([Score(name="m1", value=1.0, threshold=0.5, scoring_duration_ms=1500.0)], ec)
        sink.write([Score(name="m2", value=0.8, threshold=0.5, scoring_duration_ms=2000.0)], ec)
        sink.finalize()

        tree = ET.parse(path)
        root = tree.getroot()
        assert root.attrib["time"] == "3.500"

    def test_failure_type_is_metric_failure(self, tmp_path):
        path = tmp_path / "results.xml"
        sink = JUnitSink(str(path))
        ec = EvalCase(input="q", output="a")
        sink.write([Score(name="m", value=0.1, threshold=0.5)], ec)
        sink.finalize()

        tree = ET.parse(path)
        failure = tree.getroot().find("testcase/failure")
        assert failure.attrib["type"] == "MetricFailure"


# ---------------------------------------------------------------------------
# OtlpSink
# ---------------------------------------------------------------------------

_OTLP_PATCH_TARGETS = {
    "tp": "harness_evals.sinks.otlp_sink.TracerProvider",
    "bsp": "harness_evals.sinks.otlp_sink.BatchSpanProcessor",
    "reader": "harness_evals.sinks.otlp_sink.PeriodicExportingMetricReader",
    "mp": "harness_evals.sinks.otlp_sink.MeterProvider",
    "resource": "harness_evals.sinks.otlp_sink.Resource",
    "load_grpc": "harness_evals.sinks.otlp_sink._load_grpc_exporters",
}


@pytest.fixture
def otlp_mocks():
    """Fixture that patches all OTel classes and yields (OtlpSink class, mocks dict)."""
    mock_gauge = MagicMock()
    mock_latency_hist = MagicMock()
    mock_token_hist = MagicMock()
    mock_cost_hist = MagicMock()
    mock_meter = MagicMock()

    def _create_gauge(**kwargs):
        return mock_gauge

    def _create_histogram(name="", **kwargs):
        if "latency" in name:
            return mock_latency_hist
        if "tokens" in name:
            return mock_token_hist
        if "cost" in name:
            return mock_cost_hist
        return MagicMock()

    mock_meter.create_gauge.side_effect = _create_gauge
    mock_meter.create_histogram.side_effect = _create_histogram

    mock_meter_provider = MagicMock()
    mock_meter_provider.get_meter.return_value = mock_meter

    mock_tracer = MagicMock()
    mock_root_span = MagicMock()
    mock_item_span = MagicMock()
    mock_tracer.start_span.side_effect = lambda name, **kw: mock_root_span if name == "eval-run" else mock_item_span

    mock_trace_provider = MagicMock()
    mock_trace_provider.get_tracer.return_value = mock_tracer

    mocks = {
        "gauge": mock_gauge,
        "latency_hist": mock_latency_hist,
        "token_hist": mock_token_hist,
        "cost_hist": mock_cost_hist,
        "meter": mock_meter,
        "meter_provider": mock_meter_provider,
        "tracer": mock_tracer,
        "root_span": mock_root_span,
        "item_span": mock_item_span,
        "trace_provider": mock_trace_provider,
    }

    patches = {k: patch(v) for k, v in _OTLP_PATCH_TARGETS.items()}
    started = {k: p.start() for k, p in patches.items()}
    started["mp"].return_value = mock_meter_provider
    started["tp"].return_value = mock_trace_provider
    started["load_grpc"].return_value = (MagicMock(), MagicMock())

    from harness_evals.sinks.otlp_sink import OtlpSink

    yield OtlpSink, mocks

    for p in patches.values():
        p.stop()


@pytest.mark.unit
class TestOtlpSink:
    """Tests for OtlpSink metrics emission."""

    def test_write_sets_gauge(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink(endpoint="http://localhost:4317")
        ec = EvalCase(input="q", output="a")
        sink.write([Score(name="exact_match", value=0.85, threshold=0.5)], ec)

        mocks["gauge"].set.assert_called_once()
        call_args = mocks["gauge"].set.call_args
        assert call_args[0][0] == 0.85
        attrs = call_args[1]["attributes"]
        assert attrs["eval.metric_name"] == "exact_match"
        assert attrs["eval.passed"] == "True"

    def test_write_includes_tags(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink()
        ec = EvalCase(input="q", output="a", tags={"env": "prod", "model": "gpt-4"})
        sink.write([Score(name="m", value=1.0, threshold=0.5)], ec)

        attrs = mocks["gauge"].set.call_args[1]["attributes"]
        assert attrs["tag.env"] == "prod"
        assert attrs["tag.model"] == "gpt-4"

    def test_shutdown_calls_both_providers(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink()
        sink.shutdown()

        mocks["meter_provider"].shutdown.assert_called_once()
        mocks["trace_provider"].shutdown.assert_called_once()

    def test_shutdown_calls_finalize(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink(run_id="run-1")
        ec = EvalCase(input="q", output="a")
        sink.write([Score(name="m", value=1.0, threshold=0.5)], ec)
        sink.shutdown()

        # shutdown should have called finalize, ending the root span
        mocks["root_span"].end.assert_called_once()
        mocks["meter_provider"].shutdown.assert_called_once()

    def test_resource_attributes_passed(self, otlp_mocks):
        OtlpSink, _mocks = otlp_mocks
        OtlpSink(resource_attributes={"my.platform.account": "acc-123"})
        from harness_evals.sinks.otlp_sink import Resource

        res_call = Resource.create.call_args[0][0]
        assert res_call["service.name"] == "harness-evals"
        assert res_call["my.platform.account"] == "acc-123"

    def test_extra_attributes_in_metrics(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink(extra_attributes={"eval.suite_id": "suite-1"})
        ec = EvalCase(input="q", output="a")
        sink.write([Score(name="m", value=0.9, threshold=0.5)], ec)

        attrs = mocks["gauge"].set.call_args[1]["attributes"]
        assert attrs["eval.suite_id"] == "suite-1"

    def test_latency_histogram(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink()
        ec = EvalCase(input="q", output="a", latency_ms=342.5)
        sink.write([Score(name="m", value=1.0, threshold=0.5)], ec)

        mocks["latency_hist"].record.assert_called_once()
        assert mocks["latency_hist"].record.call_args[0][0] == 342.5

    def test_token_histogram(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink()
        ec = EvalCase(input="q", output="a", token_count=180)
        sink.write([Score(name="m", value=1.0, threshold=0.5)], ec)

        mocks["token_hist"].record.assert_called_once()
        assert mocks["token_hist"].record.call_args[0][0] == 180

    def test_cost_histogram(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink()
        ec = EvalCase(input="q", output="a", cost_usd=0.003)
        sink.write([Score(name="m", value=1.0, threshold=0.5)], ec)

        mocks["cost_hist"].record.assert_called_once()
        assert mocks["cost_hist"].record.call_args[0][0] == 0.003

    def test_none_runtime_fields_skip_histograms(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink()
        ec = EvalCase(input="q", output="a")
        sink.write([Score(name="m", value=1.0, threshold=0.5)], ec)

        mocks["latency_hist"].record.assert_not_called()
        mocks["token_hist"].record.assert_not_called()
        mocks["cost_hist"].record.assert_not_called()

    def test_empty_scores_skipped(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink()
        sink.write([], EvalCase(input="q", output="a"))

        mocks["gauge"].set.assert_not_called()
        mocks["tracer"].start_span.assert_not_called()

    def test_dimension_included_when_present(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink()
        ec = EvalCase(input="q", output="a")
        sink.write([Score(name="m", value=0.9, threshold=0.5, metadata={"dimension": "correctness"})], ec)

        attrs = mocks["gauge"].set.call_args[1]["attributes"]
        assert attrs["eval.dimension"] == "correctness"

    def test_dimension_omitted_when_absent(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink()
        ec = EvalCase(input="q", output="a")
        sink.write([Score(name="m", value=0.9, threshold=0.5)], ec)

        attrs = mocks["gauge"].set.call_args[1]["attributes"]
        assert "eval.dimension" not in attrs

    def test_invalid_protocol_raises(self, otlp_mocks):
        OtlpSink, _mocks = otlp_mocks
        with pytest.raises(ValueError, match="Unsupported protocol"):
            OtlpSink(protocol="websocket")

    def test_headers_passed_to_grpc_exporters(self, otlp_mocks):
        OtlpSink, _mocks = otlp_mocks
        from harness_evals.sinks.otlp_sink import _load_grpc_exporters

        OtlpSink(headers={"x-api-key": "secret"})
        call_args = _load_grpc_exporters.call_args
        assert call_args[0][0] == "http://localhost:4317"  # endpoint
        assert call_args[0][1] is True  # insecure
        assert call_args[1]["headers"] == {"x-api-key": "secret"}

    def test_finalize_no_writes_still_flushes(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink()
        sink.finalize()

        mocks["root_span"].end.assert_not_called()
        mocks["trace_provider"].force_flush.assert_called_once()
        mocks["meter_provider"].force_flush.assert_called_once()


@pytest.mark.unit
class TestOtlpSinkTraces:
    """Tests for OtlpSink trace emission."""

    def test_root_span_created_on_first_write(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink(run_id="run-abc")
        mocks["tracer"].start_span.assert_not_called()

        sink.write([Score(name="m", value=1.0, threshold=0.5)], EvalCase(input="q", output="a"))

        calls = mocks["tracer"].start_span.call_args_list
        assert calls[0][0][0] == "eval-run"
        assert calls[0][1]["attributes"]["eval.run_id"] == "run-abc"

    def test_child_span_per_write(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink(run_id="run-1")
        ec = EvalCase(input="q", output="a")
        sink.write([Score(name="m", value=1.0, threshold=0.5)], ec)
        sink.write([Score(name="m", value=0.8, threshold=0.5)], ec)

        assert mocks["tracer"].start_span.call_count == 3  # 1 root + 2 children
        assert mocks["item_span"].end.call_count == 2

    def test_child_span_has_eval_case_attrs(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink(run_id="run-1")
        ec = EvalCase(input="question", output="answer", expected="answer", latency_ms=200.0, token_count=150)
        sink.write([Score(name="m", value=1.0, threshold=0.5)], ec)

        child_call = [c for c in mocks["tracer"].start_span.call_args_list if c[0][0] == "eval-item"][0]
        attrs = child_call[1]["attributes"]
        assert attrs["eval.item.input"] == "question"
        assert attrs["eval.item.output"] == "answer"
        assert attrs["eval.item.expected"] == "answer"
        assert attrs["eval.item.latency_ms"] == 200.0
        assert attrs["eval.item.token_count"] == 150

    def test_item_span_attributes_override_root(self, otlp_mocks):
        """Verify that item_span_attributes override/extend extra_attributes for item spans."""
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink(
            run_id="run-x",
            extra_attributes={
                "span.type": "eval_run",
                "eval.suite_id": "suite-1",
                "env": "prod",
            },
            item_span_attributes={
                "span.type": "eval_item",  # override root span.type
                "custom.item_field": "item_only",  # add new attribute
            },
        )
        sink.write([Score(name="m", value=1.0, threshold=0.5)], EvalCase(input="q", output="a"))

        calls = mocks["tracer"].start_span.call_args_list
        # Root span: has all extra_attributes
        root_attrs = calls[0][1]["attributes"]
        assert root_attrs["span.type"] == "eval_run"
        assert root_attrs["eval.suite_id"] == "suite-1"
        assert root_attrs["env"] == "prod"
        assert "custom.item_field" not in root_attrs

        # Item span: inherits extra_attributes but with item_span_attributes overrides
        item_call = [c for c in calls if c[0][0] == "eval-item"][0]
        item_attrs = item_call[1]["attributes"]
        assert item_attrs["span.type"] == "eval_item"  # overridden
        assert item_attrs["eval.suite_id"] == "suite-1"  # inherited
        assert item_attrs["env"] == "prod"  # inherited
        assert item_attrs["custom.item_field"] == "item_only"  # from item_span_attributes

    def test_score_events_on_child_span(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink(run_id="run-1")
        sink.write(
            [
                Score(name="accuracy", value=0.9, threshold=0.7, reason="good"),
                Score(name="relevance", value=0.6, threshold=0.7),
            ],
            EvalCase(input="q", output="a"),
        )

        event_calls = mocks["item_span"].add_event.call_args_list
        assert len(event_calls) == 2
        assert event_calls[0][1]["attributes"]["eval.metric_name"] == "accuracy"
        assert event_calls[0][1]["attributes"]["eval.score.passed"] is True
        assert event_calls[0][1]["attributes"]["eval.score.reason"] == "good"
        assert event_calls[1][1]["attributes"]["eval.metric_name"] == "relevance"
        assert event_calls[1][1]["attributes"]["eval.score.passed"] is False
        assert "eval.score.reason" not in event_calls[1][1]["attributes"]

    def test_score_event_dimension_when_present(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink(run_id="run-1")
        sink.write(
            [Score(name="m", value=0.9, threshold=0.5, metadata={"dimension": "safety"})],
            EvalCase(input="q", output="a"),
        )

        event_attrs = mocks["item_span"].add_event.call_args[1]["attributes"]
        assert event_attrs["eval.dimension"] == "safety"

    def test_finalize_ends_root_span_with_summary(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink(run_id="run-1")
        ec = EvalCase(input="q", output="a")
        sink.write([Score(name="m", value=1.0, threshold=0.5)], ec)
        sink.write([Score(name="m", value=0.3, threshold=0.5)], ec)
        sink.finalize()

        root = mocks["root_span"]
        root.set_attribute.assert_any_call("eval.summary.items_total", 2)
        root.set_attribute.assert_any_call("eval.summary.items_passed", 1)
        root.set_attribute.assert_any_call("eval.summary.status", "completed")
        root.end.assert_called_once()
        mocks["trace_provider"].force_flush.assert_called_once()
        mocks["meter_provider"].force_flush.assert_called_once()

    def test_finalize_summary_event_has_averages(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink(run_id="run-1")
        ec = EvalCase(input="q", output="a")
        sink.write([Score(name="accuracy", value=1.0, threshold=0.5)], ec)
        sink.write([Score(name="accuracy", value=0.8, threshold=0.5)], ec)
        sink.finalize()

        summary_event = [c for c in mocks["root_span"].add_event.call_args_list if c[0][0] == "eval.summary"]
        assert len(summary_event) == 1
        assert summary_event[0][1]["attributes"]["eval.summary.accuracy.mean"] == 0.9

    def test_finalize_emits_dimension_summary_attributes(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink(run_id="run-1")
        ec = EvalCase(input="q", output="a")
        sink.write(
            [
                Score(name="exact_match", value=1.0, threshold=0.8, metadata={"dimension": "correctness"}),
                Score(name="pii_leak", value=0.0, threshold=1.0, metadata={"dimension": "safety"}),
            ],
            ec,
        )
        sink.finalize()

        root = mocks["root_span"]
        # Per-dimension mean set on the root span (ADR-009).
        root.set_attribute.assert_any_call("eval.summary.dimension.correctness.mean", 1.0)
        root.set_attribute.assert_any_call("eval.summary.dimension.safety.mean", 0.0)
        # Safety violation count as a hard-constraint signal (ADR-003).
        root.set_attribute.assert_any_call("eval.summary.dimension.safety.violations", 1)
        # Also folded into the summary event.
        summary_event = [c for c in root.add_event.call_args_list if c[0][0] == "eval.summary"]
        attrs = summary_event[0][1]["attributes"]
        assert attrs["eval.summary.dimension.correctness.mean"] == 1.0
        assert attrs["eval.summary.dimension.safety.violations"] == 1

    def test_finalize_idempotent(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink(run_id="run-1")
        sink.write([Score(name="m", value=1.0, threshold=0.5)], EvalCase(input="q", output="a"))
        sink.finalize()
        sink.finalize()  # second call should be no-op

        mocks["root_span"].end.assert_called_once()

    def test_run_id_auto_generated(self, otlp_mocks):
        OtlpSink, _mocks = otlp_mocks
        sink = OtlpSink()
        assert sink._run_id
        assert len(sink._run_id) == 36

    def test_http_protocol(self, otlp_mocks):
        OtlpSink, _mocks = otlp_mocks
        with patch("harness_evals.sinks.otlp_sink._load_http_exporters") as mock_http:
            mock_http.return_value = (MagicMock(), MagicMock())
            OtlpSink(protocol="http", endpoint="http://collector:4318")
            mock_http.assert_called_once_with(
                "http://collector:4318/v1/traces",
                "http://collector:4318/v1/metrics",
                headers=None,
            )

    def test_http_insecure_false_warns(self, otlp_mocks):
        OtlpSink, _mocks = otlp_mocks
        with patch("harness_evals.sinks.otlp_sink._load_http_exporters") as mock_http:
            mock_http.return_value = (MagicMock(), MagicMock())
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                OtlpSink(protocol="http", insecure=False)
                assert len(w) == 1
                assert "insecure=False has no effect" in str(w[0].message)


@pytest.mark.unit
class TestHttpOtlpEndpoints:
    """Tests for _http_otlp_endpoints URL normalization."""

    def test_base_without_trailing_slash(self):
        from harness_evals.sinks.otlp_sink import _http_otlp_endpoints

        t, m = _http_otlp_endpoints("http://collector:4318")
        assert t == "http://collector:4318/v1/traces"
        assert m == "http://collector:4318/v1/metrics"

    def test_base_with_trailing_slash(self):
        from harness_evals.sinks.otlp_sink import _http_otlp_endpoints

        t, m = _http_otlp_endpoints("https://example.com/agenttrace/otlp/")
        assert t == "https://example.com/agenttrace/otlp/v1/traces"
        assert m == "https://example.com/agenttrace/otlp/v1/metrics"

    def test_already_trace_suffix(self):
        from harness_evals.sinks.otlp_sink import _http_otlp_endpoints

        t, m = _http_otlp_endpoints("https://harness.example/otel/v1/traces")
        assert t == "https://harness.example/otel/v1/traces"
        assert m == "https://harness.example/otel/v1/metrics"

    def test_already_metrics_suffix(self):
        from harness_evals.sinks.otlp_sink import _http_otlp_endpoints

        t, m = _http_otlp_endpoints("https://harness.example/otel/v1/metrics")
        assert t == "https://harness.example/otel/v1/traces"
        assert m == "https://harness.example/otel/v1/metrics"


@pytest.mark.unit
class TestOtlpSinkTruncate:
    """Tests for the _truncate helper."""

    def test_short_string_unchanged(self):
        from harness_evals.sinks.otlp_sink import _truncate

        assert _truncate("hello") == "hello"

    def test_long_string_truncated(self):
        from harness_evals.sinks.otlp_sink import _truncate

        long_str = "x" * 1500
        result = _truncate(long_str)
        assert len(result) == 1000

    def test_dict_serialized(self):
        from harness_evals.sinks.otlp_sink import _truncate

        result = _truncate({"key": "value"})
        assert result == '{"key": "value"}'

    def test_custom_max_len(self):
        from harness_evals.sinks.otlp_sink import _truncate

        assert _truncate("abcdef", max_len=3) == "abc"

    def test_non_ascii_truncated_by_char_not_byte(self):
        from harness_evals.sinks.otlp_sink import _truncate

        # Each character is 1 char but multiple bytes in UTF-8
        s = "\u00e9" * 10  # é repeated 10 times
        result = _truncate(s, max_len=5)
        assert len(result) == 5
        assert result == "\u00e9" * 5


# ---------------------------------------------------------------------------
# OtlpSink — Context Propagation & External Providers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOtlpSinkContextPropagation:
    """Tests for parent_context, tracer_provider, and meter_provider injection."""

    def test_parent_context_passed_to_root_span(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        fake_ctx = MagicMock()
        sink = OtlpSink(run_id="run-ctx", parent_context=fake_ctx)
        sink.write([Score(name="m", value=1.0, threshold=0.5)], EvalCase(input="q", output="a"))

        root_call = [c for c in mocks["tracer"].start_span.call_args_list if c[0][0] == "eval-run"][0]
        assert root_call[1]["context"] is fake_ctx

    def test_parent_context_none_by_default(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink(run_id="run-default")
        sink.write([Score(name="m", value=1.0, threshold=0.5)], EvalCase(input="q", output="a"))

        root_call = [c for c in mocks["tracer"].start_span.call_args_list if c[0][0] == "eval-run"][0]
        assert root_call[1]["context"] is None

    def test_external_tracer_provider_used(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        ext_tp = MagicMock()
        ext_tracer = MagicMock()
        ext_tp.get_tracer.return_value = ext_tracer
        ext_root = MagicMock()
        ext_item = MagicMock()
        ext_tracer.start_span.side_effect = lambda name, **kw: ext_root if name == "eval-run" else ext_item

        sink = OtlpSink(tracer_provider=ext_tp, run_id="run-ext")
        sink.write([Score(name="m", value=1.0, threshold=0.5)], EvalCase(input="q", output="a"))

        ext_tp.get_tracer.assert_called_once_with("harness-evals")
        assert ext_tracer.start_span.call_count == 2  # root + item

    def test_external_meter_provider_used(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        ext_mp = MagicMock()
        ext_meter = MagicMock()
        ext_mp.get_meter.return_value = ext_meter
        ext_gauge = MagicMock()
        ext_meter.create_gauge.return_value = ext_gauge
        ext_meter.create_histogram.return_value = MagicMock()

        sink = OtlpSink(meter_provider=ext_mp, run_id="run-ext-meter")
        sink.write([Score(name="m", value=0.9, threshold=0.5)], EvalCase(input="q", output="a"))

        ext_mp.get_meter.assert_called_once_with("harness-evals")
        ext_gauge.set.assert_called_once()

    def test_shutdown_does_not_shutdown_external_providers(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        ext_tp = MagicMock()
        ext_tp.get_tracer.return_value = mocks["tracer"]
        ext_mp = MagicMock()
        ext_mp.get_meter.return_value = mocks["meter"]

        sink = OtlpSink(tracer_provider=ext_tp, meter_provider=ext_mp)
        sink.shutdown()

        ext_tp.shutdown.assert_not_called()
        ext_mp.shutdown.assert_not_called()

    def test_finalize_does_not_flush_external_providers(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        ext_tp = MagicMock()
        ext_tp.get_tracer.return_value = mocks["tracer"]
        ext_mp = MagicMock()
        ext_mp.get_meter.return_value = mocks["meter"]

        sink = OtlpSink(tracer_provider=ext_tp, meter_provider=ext_mp, run_id="run-1")
        sink.write([Score(name="m", value=1.0, threshold=0.5)], EvalCase(input="q", output="a"))
        sink.finalize()

        ext_tp.force_flush.assert_not_called()
        ext_mp.force_flush.assert_not_called()

    def test_shutdown_still_shuts_own_meter_when_only_tracer_external(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        ext_tp = MagicMock()
        ext_tp.get_tracer.return_value = mocks["tracer"]

        sink = OtlpSink(tracer_provider=ext_tp, run_id="run-partial")
        sink.shutdown()

        ext_tp.shutdown.assert_not_called()
        mocks["meter_provider"].shutdown.assert_called_once()

    def test_warning_when_endpoint_and_provider_both_set(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        ext_tp = MagicMock()
        ext_tp.get_tracer.return_value = mocks["tracer"]
        ext_mp = MagicMock()
        ext_mp.get_meter.return_value = mocks["meter"]

        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            OtlpSink(
                endpoint="http://custom-collector:4317",
                tracer_provider=ext_tp,
                meter_provider=ext_mp,
            )
            assert len(w) == 1
            assert "endpoint/protocol/insecure/headers are ignored" in str(w[0].message)

    def test_no_warning_when_default_endpoint_and_provider(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        ext_tp = MagicMock()
        ext_tp.get_tracer.return_value = mocks["tracer"]
        ext_mp = MagicMock()
        ext_mp.get_meter.return_value = mocks["meter"]

        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            OtlpSink(tracer_provider=ext_tp, meter_provider=ext_mp)
            assert len(w) == 0

    def test_parent_context_with_external_provider(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        ext_tp = MagicMock()
        ext_tracer = MagicMock()
        ext_tp.get_tracer.return_value = ext_tracer
        ext_root = MagicMock()
        ext_item = MagicMock()
        ext_tracer.start_span.side_effect = lambda name, **kw: ext_root if name == "eval-run" else ext_item
        ext_mp = MagicMock()
        ext_mp.get_meter.return_value = mocks["meter"]

        fake_ctx = MagicMock()
        sink = OtlpSink(
            tracer_provider=ext_tp,
            meter_provider=ext_mp,
            parent_context=fake_ctx,
            run_id="run-both",
        )
        sink.write([Score(name="m", value=1.0, threshold=0.5)], EvalCase(input="q", output="a"))

        root_call = [c for c in ext_tracer.start_span.call_args_list if c[0][0] == "eval-run"][0]
        assert root_call[1]["context"] is fake_ctx


# ---------------------------------------------------------------------------
# OtlpSink — item_context (engine-owned item spans)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOtlpSinkItemContext:
    """Tests for write() with item_context — decorating engine-owned spans."""

    def test_item_context_adds_events_to_existing_span(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink(run_id="run-1")

        engine_span = MagicMock()
        fake_item_ctx = MagicMock()

        with patch("harness_evals.sinks.otlp_sink.trace_api.get_current_span", return_value=engine_span):
            sink.write(
                [Score(name="accuracy", value=0.9, threshold=0.7, reason="good")],
                EvalCase(input="q", output="a"),
                item_context=fake_item_ctx,
            )

        # Score event added to engine's span
        engine_span.add_event.assert_called_once()
        event_call = engine_span.add_event.call_args
        assert event_call[0][0] == "eval.score"
        assert event_call[1]["attributes"]["eval.metric_name"] == "accuracy"
        assert event_call[1]["attributes"]["eval.score.value"] == 0.9
        assert event_call[1]["attributes"]["eval.score.reason"] == "good"

    def test_item_context_does_not_create_child_span(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink(run_id="run-1")

        engine_span = MagicMock()
        fake_item_ctx = MagicMock()

        with patch("harness_evals.sinks.otlp_sink.trace_api.get_current_span", return_value=engine_span):
            sink.write(
                [Score(name="m", value=1.0, threshold=0.5)],
                EvalCase(input="q", output="a"),
                item_context=fake_item_ctx,
            )

        # Only root span created, no eval-item child
        span_calls = [c for c in mocks["tracer"].start_span.call_args_list if c[0][0] == "eval-item"]
        assert len(span_calls) == 0

    def test_item_context_does_not_end_engine_span(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink(run_id="run-1")

        engine_span = MagicMock()
        fake_item_ctx = MagicMock()

        with patch("harness_evals.sinks.otlp_sink.trace_api.get_current_span", return_value=engine_span):
            sink.write(
                [Score(name="m", value=1.0, threshold=0.5)],
                EvalCase(input="q", output="a"),
                item_context=fake_item_ctx,
            )

        engine_span.end.assert_not_called()

    def test_item_context_sets_passed_attribute(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink(run_id="run-1")

        engine_span = MagicMock()
        fake_item_ctx = MagicMock()

        with patch("harness_evals.sinks.otlp_sink.trace_api.get_current_span", return_value=engine_span):
            sink.write(
                [
                    Score(name="a", value=1.0, threshold=0.5),
                    Score(name="b", value=0.3, threshold=0.5),
                ],
                EvalCase(input="q", output="a"),
                item_context=fake_item_ctx,
            )

        engine_span.set_attribute.assert_called_with("eval.item.passed", False)

    def test_item_context_still_accumulates_summary(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink(run_id="run-1")

        engine_span = MagicMock()
        fake_item_ctx = MagicMock()

        with patch("harness_evals.sinks.otlp_sink.trace_api.get_current_span", return_value=engine_span):
            sink.write(
                [Score(name="m", value=1.0, threshold=0.5)],
                EvalCase(input="q", output="a"),
                item_context=fake_item_ctx,
            )
            sink.write(
                [Score(name="m", value=0.6, threshold=0.5)],
                EvalCase(input="q2", output="a2"),
                item_context=fake_item_ctx,
            )

        sink.finalize()

        # Root span gets summary despite item_context usage
        mocks["root_span"].set_attribute.assert_any_call("eval.summary.items_total", 2)
        mocks["root_span"].set_attribute.assert_any_call("eval.summary.items_passed", 2)

    def test_item_context_still_emits_metrics(self, otlp_mocks):
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink(run_id="run-1")

        engine_span = MagicMock()
        fake_item_ctx = MagicMock()

        with patch("harness_evals.sinks.otlp_sink.trace_api.get_current_span", return_value=engine_span):
            sink.write(
                [Score(name="m", value=0.85, threshold=0.5)],
                EvalCase(input="q", output="a", latency_ms=200.0),
                item_context=fake_item_ctx,
            )

        # Metrics still recorded
        mocks["gauge"].set.assert_called_once()
        mocks["latency_hist"].record.assert_called_once_with(200.0, attributes=mocks["latency_hist"].record.call_args[1]["attributes"])

    def test_item_context_none_uses_default_behavior(self, otlp_mocks):
        """Explicit item_context=None should behave identically to not passing it."""
        OtlpSink, mocks = otlp_mocks
        sink = OtlpSink(run_id="run-1")
        sink.write(
            [Score(name="m", value=1.0, threshold=0.5)],
            EvalCase(input="q", output="a"),
            item_context=None,
        )

        # Child span created as usual
        span_calls = [c for c in mocks["tracer"].start_span.call_args_list if c[0][0] == "eval-item"]
        assert len(span_calls) == 1
        mocks["item_span"].end.assert_called_once()


# ---------------------------------------------------------------------------
# LangfuseSink
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLangfuseSink:
    @patch("harness_evals.sinks.langfuse_sink.Langfuse")
    def test_write_sends_scores(self, mock_langfuse_cls):
        mock_client = MagicMock()
        mock_langfuse_cls.return_value = mock_client

        from harness_evals.sinks.langfuse_sink import LangfuseSink

        sink = LangfuseSink(secret_key="sk", public_key="pk")
        ec = EvalCase(input="q", output="a")
        sink.write(
            [Score(name="exact_match", value=1.0, threshold=0.5, reason="perfect")],
            ec,
        )

        mock_client.score.assert_called_once()
        call_kwargs = mock_client.score.call_args[1]
        assert call_kwargs["name"] == "exact_match"
        assert call_kwargs["value"] == 1.0
        assert call_kwargs["comment"] == "perfect"

    @patch("harness_evals.sinks.langfuse_sink.Langfuse")
    def test_write_with_trace_id(self, mock_langfuse_cls):
        mock_client = MagicMock()
        mock_langfuse_cls.return_value = mock_client

        from harness_evals.sinks.langfuse_sink import LangfuseSink

        sink = LangfuseSink(secret_key="sk", public_key="pk")
        ec = EvalCase(
            input="q",
            output="a",
            metadata={"langfuse_trace_id": "trace-123", "langfuse_observation_id": "obs-456"},
        )
        sink.write([Score(name="m", value=0.9, threshold=0.5)], ec)

        call_kwargs = mock_client.score.call_args[1]
        assert call_kwargs["trace_id"] == "trace-123"
        assert call_kwargs["observation_id"] == "obs-456"

    @patch("harness_evals.sinks.langfuse_sink.Langfuse")
    def test_write_without_trace_id(self, mock_langfuse_cls):
        mock_client = MagicMock()
        mock_langfuse_cls.return_value = mock_client

        from harness_evals.sinks.langfuse_sink import LangfuseSink

        sink = LangfuseSink(secret_key="sk", public_key="pk")
        ec = EvalCase(input="q", output="a")
        sink.write([Score(name="m", value=0.8, threshold=0.5)], ec)

        call_kwargs = mock_client.score.call_args[1]
        assert "trace_id" not in call_kwargs
        assert "observation_id" not in call_kwargs

    @patch("harness_evals.sinks.langfuse_sink.Langfuse")
    def test_write_includes_tags_in_metadata(self, mock_langfuse_cls):
        mock_client = MagicMock()
        mock_langfuse_cls.return_value = mock_client

        from harness_evals.sinks.langfuse_sink import LangfuseSink

        sink = LangfuseSink(secret_key="sk", public_key="pk")
        ec = EvalCase(input="q", output="a", tags={"env": "prod", "model": "gpt-4"})
        sink.write([Score(name="m", value=1.0, threshold=0.5)], ec)

        call_kwargs = mock_client.score.call_args[1]
        assert call_kwargs["metadata"]["tag.env"] == "prod"
        assert call_kwargs["metadata"]["tag.model"] == "gpt-4"

    @patch("harness_evals.sinks.langfuse_sink.Langfuse")
    def test_write_merges_score_metadata(self, mock_langfuse_cls):
        mock_client = MagicMock()
        mock_langfuse_cls.return_value = mock_client

        from harness_evals.sinks.langfuse_sink import LangfuseSink

        sink = LangfuseSink(secret_key="sk", public_key="pk")
        ec = EvalCase(input="q", output="a", tags={"env": "staging"})
        sink.write(
            [Score(name="m", value=0.9, threshold=0.5, metadata={"edit_distance": 2})],
            ec,
        )

        call_kwargs = mock_client.score.call_args[1]
        assert call_kwargs["metadata"]["edit_distance"] == 2
        assert call_kwargs["metadata"]["tag.env"] == "staging"

    @patch("harness_evals.sinks.langfuse_sink.Langfuse")
    def test_multiple_scores_per_write(self, mock_langfuse_cls):
        mock_client = MagicMock()
        mock_langfuse_cls.return_value = mock_client

        from harness_evals.sinks.langfuse_sink import LangfuseSink

        sink = LangfuseSink(secret_key="sk", public_key="pk")
        ec = EvalCase(input="q", output="a")
        sink.write(
            [
                Score(name="m1", value=1.0, threshold=0.5),
                Score(name="m2", value=0.3, threshold=0.5),
            ],
            ec,
        )

        assert mock_client.score.call_count == 2

    @patch("harness_evals.sinks.langfuse_sink.Langfuse")
    def test_finalize_flushes(self, mock_langfuse_cls):
        mock_client = MagicMock()
        mock_langfuse_cls.return_value = mock_client

        from harness_evals.sinks.langfuse_sink import LangfuseSink

        sink = LangfuseSink(secret_key="sk", public_key="pk")
        sink.finalize()

        mock_client.flush.assert_called_once()

    @patch("harness_evals.sinks.langfuse_sink.Langfuse")
    def test_shutdown_flushes_and_shuts_down(self, mock_langfuse_cls):
        mock_client = MagicMock()
        mock_langfuse_cls.return_value = mock_client

        from harness_evals.sinks.langfuse_sink import LangfuseSink

        sink = LangfuseSink(secret_key="sk", public_key="pk")
        sink.shutdown()

        mock_client.flush.assert_called_once()
        mock_client.shutdown.assert_called_once()
