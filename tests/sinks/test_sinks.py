"""Tests for CsvSink, JUnitSink, and OtlpSink."""

import csv
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import pytest

from harness_evals import EvalCase, Score
from harness_evals.sinks.csv_sink import CsvSink
from harness_evals.sinks.junit_sink import JUnitSink

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

    def test_no_time_attribute_on_testcase(self, tmp_path):
        path = tmp_path / "results.xml"
        sink = JUnitSink(str(path))
        ec = EvalCase(input="q", output="a")
        sink.write([Score(name="m", value=0.9, threshold=0.5)], ec)
        sink.finalize()

        tree = ET.parse(path)
        tc = tree.getroot().find("testcase")
        assert "time" not in tc.attrib

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


@pytest.mark.unit
class TestOtlpSink:
    @patch("harness_evals.sinks.otlp_sink.PeriodicExportingMetricReader")
    @patch("harness_evals.sinks.otlp_sink.OTLPMetricExporter")
    @patch("harness_evals.sinks.otlp_sink.MeterProvider")
    def test_write_sets_gauge(self, mock_provider_cls, mock_exporter_cls, mock_reader_cls):
        mock_gauge = MagicMock()
        mock_meter = MagicMock()
        mock_meter.create_gauge.return_value = mock_gauge
        mock_provider = MagicMock()
        mock_provider.get_meter.return_value = mock_meter
        mock_provider_cls.return_value = mock_provider

        from harness_evals.sinks.otlp_sink import OtlpSink

        sink = OtlpSink(endpoint="http://localhost:4317")
        ec = EvalCase(input="q", output="a")
        sink.write([Score(name="exact_match", value=0.85, threshold=0.5)], ec)

        mock_gauge.set.assert_called_once()
        call_args = mock_gauge.set.call_args
        assert call_args[0][0] == 0.85
        attrs = call_args[1]["attributes"]
        assert attrs["metric_name"] == "exact_match"
        assert attrs["passed"] == "True"

    @patch("harness_evals.sinks.otlp_sink.PeriodicExportingMetricReader")
    @patch("harness_evals.sinks.otlp_sink.OTLPMetricExporter")
    @patch("harness_evals.sinks.otlp_sink.MeterProvider")
    def test_write_includes_tags(self, mock_provider_cls, mock_exporter_cls, mock_reader_cls):
        mock_gauge = MagicMock()
        mock_meter = MagicMock()
        mock_meter.create_gauge.return_value = mock_gauge
        mock_provider = MagicMock()
        mock_provider.get_meter.return_value = mock_meter
        mock_provider_cls.return_value = mock_provider

        from harness_evals.sinks.otlp_sink import OtlpSink

        sink = OtlpSink()
        ec = EvalCase(input="q", output="a", tags={"env": "prod", "model": "gpt-4"})
        sink.write([Score(name="m", value=1.0, threshold=0.5)], ec)

        attrs = mock_gauge.set.call_args[1]["attributes"]
        assert attrs["tag.env"] == "prod"
        assert attrs["tag.model"] == "gpt-4"

    @patch("harness_evals.sinks.otlp_sink.PeriodicExportingMetricReader")
    @patch("harness_evals.sinks.otlp_sink.OTLPMetricExporter")
    @patch("harness_evals.sinks.otlp_sink.MeterProvider")
    def test_shutdown_calls_provider(self, mock_provider_cls, mock_exporter_cls, mock_reader_cls):
        mock_provider = MagicMock()
        mock_meter = MagicMock()
        mock_meter.create_gauge.return_value = MagicMock()
        mock_provider.get_meter.return_value = mock_meter
        mock_provider_cls.return_value = mock_provider

        from harness_evals.sinks.otlp_sink import OtlpSink

        sink = OtlpSink()
        sink.shutdown()

        mock_provider.shutdown.assert_called_once()

    @patch("harness_evals.sinks.otlp_sink.PeriodicExportingMetricReader")
    @patch("harness_evals.sinks.otlp_sink.OTLPMetricExporter")
    @patch("harness_evals.sinks.otlp_sink.MeterProvider")
    def test_no_global_state_mutation(self, mock_provider_cls, mock_exporter_cls, mock_reader_cls):
        mock_provider = MagicMock()
        mock_meter = MagicMock()
        mock_meter.create_gauge.return_value = MagicMock()
        mock_provider.get_meter.return_value = mock_meter
        mock_provider_cls.return_value = mock_provider

        from harness_evals.sinks.otlp_sink import OtlpSink

        sink = OtlpSink()
        assert sink._provider is mock_provider
        mock_provider.get_meter.assert_called_once_with("harness-evals")
