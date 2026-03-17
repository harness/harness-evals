"""Tests for the HTML reporting module."""

import tempfile
from pathlib import Path

import pytest

from harness_evals import EvalCase, Score
from harness_evals.reporting import HtmlReporter, HtmlSink


@pytest.mark.unit
class TestHtmlReporter:
    def _make_scores(self, overall: float, **kwargs: float) -> list[Score]:
        scores = [Score(name="overall", value=overall, threshold=0.7)]
        for k, v in kwargs.items():
            scores.append(Score(name=k, value=v, threshold=0.5))
        return scores

    def test_generate_returns_html_string(self):
        reporter = HtmlReporter(title="Test")
        ec = EvalCase(input="test prompt", output="response", expected="expected")
        reporter.add(ec, self._make_scores(0.9), group="g1", variant="good")
        html = reporter.generate()
        assert "<!DOCTYPE html>" in html
        assert "Test" in html

    def test_generate_writes_file(self):
        reporter = HtmlReporter(title="File Test")
        ec = EvalCase(input="prompt", output="out", expected="exp")
        reporter.add(ec, self._make_scores(0.8), group="g1", variant="good")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "report.html"
            reporter.generate(path)
            assert path.exists()
            content = path.read_text()
            assert "File Test" in content

    def test_groups_and_variants(self):
        reporter = HtmlReporter()
        for variant, score in [("good", 0.95), ("mediocre", 0.6), ("bad", 0.1)]:
            ec = EvalCase(input="question", output="answer")
            reporter.add(ec, self._make_scores(score), group="test_001", variant=variant)
        html = reporter.generate()
        assert "test_001" in html
        assert "95%" in html
        assert "60%" in html
        assert "10%" in html

    def test_metric_categories(self):
        reporter = HtmlReporter()
        ec = EvalCase(input="q", output="a")
        scores = self._make_scores(0.9, exact_match=1.0, relevance=0.8)
        reporter.add(ec, scores, group="g1", variant="good")
        reporter.set_metric_categories(
            {
                "Deterministic": ["exact_match"],
                "LLM Judge": ["relevance"],
            }
        )
        html = reporter.generate()
        assert "Deterministic" in html
        assert "LLM Judge" in html

    def test_overall_key_customizable(self):
        reporter = HtmlReporter()
        reporter.set_overall_key("quality_score")
        ec = EvalCase(input="q", output="a")
        scores = [Score(name="quality_score", value=0.85, threshold=0.7)]
        reporter.add(ec, scores, group="g1", variant="good")
        html = reporter.generate()
        assert "85%" in html

    def test_auto_label_from_input(self):
        reporter = HtmlReporter()
        ec = EvalCase(input="What is the meaning of life?", output="42")
        reporter.add(ec, self._make_scores(0.9))
        html = reporter.generate()
        assert "What is the meaning of life?" in html

    def test_narrative_generated(self):
        reporter = HtmlReporter()
        for variant, score in [("good", 0.95), ("bad", 0.1)]:
            ec = EvalCase(input="q", output="a")
            reporter.add(ec, self._make_scores(score), group="g1", variant=variant)
        html = reporter.generate()
        assert "Key Findings" in html
        assert "discrimination" in html.lower()

    def test_metadata_scores_included(self):
        reporter = HtmlReporter()
        ec = EvalCase(input="q", output="a")
        score = Score(
            name="composite",
            value=0.8,
            threshold=0.7,
            metadata={"sub_a": 0.9, "sub_b": 0.7, "non_numeric": "text"},
        )
        reporter.add(ec, [score], group="g1", variant="good")
        html = reporter.generate()
        assert "Sub A" in html
        assert "Sub B" in html

    def test_empty_report(self):
        reporter = HtmlReporter(title="Empty")
        html = reporter.generate()
        assert "<!DOCTYPE html>" in html
        assert "0 evaluations" in html

    def test_pass_rate_card(self):
        reporter = HtmlReporter()
        ec = EvalCase(input="q", output="a")
        reporter.add(ec, [Score(name="m", value=0.9, threshold=0.7)], group="g1")
        reporter.add(ec, [Score(name="m", value=0.3, threshold=0.7)], group="g2")
        html = reporter.generate()
        assert "Pass Rate" in html
        assert "50%" in html


@pytest.mark.unit
class TestHtmlSink:
    def test_sink_accumulates_and_finalizes(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sink_report.html"
            sink = HtmlSink(path, title="Sink Test")

            ec = EvalCase(
                input="prompt",
                output="response",
                tags={"group": "test_001", "quality": "good"},
            )
            scores = [Score(name="accuracy", value=0.9, threshold=0.7)]
            sink.write(scores, ec)

            ec2 = EvalCase(
                input="prompt",
                output="bad response",
                tags={"group": "test_001", "quality": "bad"},
            )
            scores2 = [Score(name="accuracy", value=0.2, threshold=0.7)]
            sink.write(scores2, ec2)

            result_path = sink.finalize()
            assert Path(result_path).exists()
            content = Path(result_path).read_text()
            assert "Sink Test" in content
            assert "test_001" in content

    def test_sink_uses_tags_for_grouping(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "tags_report.html"
            sink = HtmlSink(path)

            ec = EvalCase(
                input="q",
                output="a",
                tags={"group": "my_group", "variant": "v1", "label": "My Label"},
            )
            sink.write([Score(name="m", value=0.8, threshold=0.5)], ec)
            sink.finalize()
            content = Path(path).read_text()
            assert "my_group" in content
            assert "My Label" in content
