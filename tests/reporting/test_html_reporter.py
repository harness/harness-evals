"""Tests for the HTML reporting module."""

import re
import tempfile
from pathlib import Path

import pytest

from harness_evals import EvalCase, Score
from harness_evals.reporting import HtmlReporter, HtmlSink


def _dim_score(name: str, value: float, dimension: str) -> Score:
    return Score(name=name, value=value, threshold=0.7, metadata={"dimension": dimension})


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


@pytest.mark.unit
class TestDimensionRadar:
    def _report_with_dimensions(self) -> HtmlReporter:
        reporter = HtmlReporter(title="Radar")
        ec = EvalCase(input="q", output="a")
        reporter.add(
            ec,
            [
                _dim_score("exact_match", 1.0, "correctness"),
                _dim_score("faithfulness", 0.8, "groundedness"),
                _dim_score("pii", 0.0, "safety"),
                _dim_score("trajectory", 0.6, "trajectory"),
                _dim_score("latency", 0.9, "performance"),
            ],
            group="g1",
            variant="good",
        )
        return reporter

    def test_radar_svg_is_self_contained(self):
        html = self._report_with_dimensions().generate()
        svgs = re.findall(r"<svg.*?</svg>", html, re.S)
        assert svgs, "expected an inline SVG radar chart"
        svg = svgs[0]
        # No external dependencies: no scripts, images, links, or remote fetches.
        # (The SVG xmlns is a namespace identifier, not a network reference.)
        assert "<script" not in svg
        assert "<img" not in svg
        assert "src=" not in svg
        assert "href=" not in svg
        # The only allowed URL is the SVG namespace declaration.
        assert svg.count("http") == 1
        assert 'xmlns="http://www.w3.org/2000/svg"' in svg

    def test_radar_labels_all_five_dimensions(self):
        html = self._report_with_dimensions().generate()
        for label in ("Correctness", "Groundedness", "Trajectory", "Performance"):
            assert label in html
        # Safety axis is annotated with its violation count (ADR-003).
        assert "Safety (1 viol.)" in html

    def test_safety_axis_rendered_red(self):
        html = self._report_with_dimensions().generate()
        svg = re.findall(r"<svg.*?</svg>", html, re.S)[0]
        assert "#dc2626" in svg  # safety spoke/label drawn in red

    def test_unknown_bucket_omitted_from_axes_but_footnoted(self):
        reporter = HtmlReporter(title="Radar")
        ec = EvalCase(input="q", output="a")
        reporter.add(
            ec,
            [
                _dim_score("exact_match", 1.0, "correctness"),
                _dim_score("faithfulness", 0.8, "groundedness"),
                Score(name="mystery", value=0.5, threshold=0.7),  # no dimension -> unknown
            ],
            group="g1",
            variant="good",
        )
        html = reporter.generate()
        svg = re.findall(r"<svg.*?</svg>", html, re.S)[0]
        # unknown must not appear as an axis label...
        assert "unknown" not in svg.lower()
        # ...but is acknowledged in a footnote.
        assert "no declared dimension" in html
        assert "1 metric(s)" in html

    def test_no_radar_when_no_dimensioned_scores(self):
        reporter = HtmlReporter(title="Radar")
        ec = EvalCase(input="q", output="a")
        # Only an undeclared-dimension score -> everything is "unknown".
        reporter.add(ec, [Score(name="m", value=0.9, threshold=0.7)], group="g1", variant="good")
        html = reporter.generate()
        assert "Dimension breakdown" not in html
        assert "<svg" not in html

    def test_radar_renders_with_two_dimensions(self):
        # Fewer than 3 axes: a polygon would be degenerate, so the grid falls
        # back to circles and the data to a connecting line — still valid SVG.
        reporter = HtmlReporter(title="Radar")
        ec = EvalCase(input="q", output="a")
        reporter.add(
            ec,
            [
                _dim_score("exact_match", 1.0, "correctness"),
                _dim_score("faithfulness", 0.5, "groundedness"),
            ],
            group="g1",
            variant="good",
        )
        html = reporter.generate()
        svg = re.findall(r"<svg.*?</svg>", html, re.S)[0]
        assert "Correctness" in html and "Groundedness" in html
        assert "<polygon" not in svg  # no polygon with <3 axes
        assert "<circle" in svg  # circular grid + data markers instead

    def test_radar_renders_with_single_dimension(self):
        reporter = HtmlReporter(title="Radar")
        ec = EvalCase(input="q", output="a")
        reporter.add(ec, [_dim_score("exact_match", 0.8, "correctness")], group="g1", variant="good")
        html = reporter.generate()
        svg = re.findall(r"<svg.*?</svg>", html, re.S)[0]
        assert "Correctness" in html
        assert "<circle" in svg  # single data marker + circular grid

    def test_radar_aggregates_mean_across_results(self):
        # Two cases, correctness 1.0 and 0.0 -> the axis mean must be 0.50.
        reporter = HtmlReporter(title="Radar")
        for val in (1.0, 0.0):
            reporter.add(
                EvalCase(input="q", output="a"),
                [_dim_score("exact_match", val, "correctness")],
                group="g1",
                variant="good",
            )
        html = reporter.generate()
        # Legend row shows the aggregated mean across both results.
        assert "0.50" in html

    def test_non_canonical_dimension_ordered_after_canonical(self):
        from harness_evals.summary import order_dimensions

        order = order_dimensions(["custom", "safety", "correctness"])
        # Canonical dims come first in ADR-009 order; extras are appended.
        assert order == ["correctness", "safety", "custom"]

    def test_radar_overlays_one_polygon_per_variant(self):
        # Two variants must render as two distinct polygons (not one averaged
        # shape) in their variant colors, with a variant key in the legend.
        reporter = HtmlReporter(title="AB")
        ec = EvalCase(input="q", output="a")
        reporter.add(
            ec,
            [
                _dim_score("em", 0.9, "correctness"),
                _dim_score("f", 0.8, "groundedness"),
                _dim_score("pii", 1.0, "safety"),
            ],
            group="g1",
            variant="good",
        )
        reporter.add(
            ec,
            [
                _dim_score("em", 0.1, "correctness"),
                _dim_score("f", 0.2, "groundedness"),
                _dim_score("pii", 0.0, "safety"),
            ],
            group="g1",
            variant="bad",
        )
        html = reporter.generate()
        svg = re.findall(r"<svg.*?</svg>", html, re.S)[0]
        data_polys = re.findall(r'<polygon[^>]*fill="(#[0-9a-fA-F]+)" fill-opacity', svg)
        # One data polygon per variant, in the variant colors (good=green, bad=red).
        assert data_polys == ["#16a34a", "#dc2626"]
        # Variant key + per-variant safety counts in the legend.
        assert "good" in html and "bad" in html
        assert "safety 0 viol." in html and "safety 1 viol." in html
        # Shared safety axis stays a plain label (counts live in the key).
        assert ">Safety<" in svg and "Safety (" not in svg

    def test_radar_labels_stay_within_viewbox(self):
        # The viewBox must contain every axis label — no clipping by the SVG
        # viewport, even for long labels like "Safety (N viol.)".
        from harness_evals.reporting.html_reporter import _RADAR_CHAR_W, _radar_svg

        axis_labels = [
            ("Correctness", False),
            ("Groundedness", False),
            ("Safety (1234 viol.)", True),  # deliberately long
            ("Trajectory", False),
            ("Performance", False),
        ]
        series = [("#2563eb", [0.9, 0.72, 0.0, 0.55, 0.9])]
        svg = _radar_svg(axis_labels, series)
        w, h = (int(v) for v in re.search(r'viewBox="0 0 (\d+) (\d+)"', svg).groups())
        text_re = re.compile(r'<text x="([\d.]+)" y="([\d.]+)" text-anchor="(\w+)"[^>]*>([^<]+)</text>')
        found = 0
        for mx, my, anchor, label in text_re.findall(svg):
            found += 1
            x, y, tw = float(mx), float(my), len(label) * _RADAR_CHAR_W
            if anchor == "start":
                left, right = x, x + tw
            elif anchor == "end":
                left, right = x - tw, x
            else:
                left, right = x - tw / 2, x + tw / 2
            assert left >= 0 and right <= w, f"{label!r} spans [{left:.1f},{right:.1f}] outside width {w}"
            assert 0 <= y <= h, f"{label!r} y={y} outside height {h}"
        assert found == len(axis_labels)
