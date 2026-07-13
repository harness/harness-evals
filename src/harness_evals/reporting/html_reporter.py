"""Generic HTML report generator for harness-evals.

Groups eval results by test case, shows variants side-by-side,
auto-categorizes metrics, and generates an executive summary.

Usage::

    reporter = HtmlReporter(title="My Evals")

    # Add results as you evaluate
    reporter.add(eval_case, scores, group="kg_001", variant="good", label="What depends on auth?")
    reporter.add(eval_case, scores, group="kg_001", variant="bad", label="What depends on auth?")

    # Optionally classify metrics into sections
    reporter.set_metric_categories({
        "Deterministic": ["exact_match", "contains", "query_validity"],
        "LLM Judge": ["relevance", "actionability"],
    })

    # Generate
    reporter.generate("report.html")
"""

from __future__ import annotations

import html as html_mod
import math
from dataclasses import dataclass
from pathlib import Path

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score
from harness_evals.summary import SAFETY_DIMENSION, UNKNOWN_DIMENSION, order_dimensions, summarize


@dataclass
class EvalResult:
    """One evaluated test case with its scores and grouping metadata."""

    eval_case: EvalCase
    scores: list[Score]
    group: str = ""
    variant: str = ""
    label: str = ""

    @staticmethod
    def _extract_metadata(
        metadata: dict,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Split metadata into normalised scores ([0, 1]) and raw info values."""
        scores: dict[str, float] = {}
        info: dict[str, float] = {}
        for k, v in metadata.items():
            if isinstance(v, (int, float)):
                fv = float(v)
                (scores if 0.0 <= fv <= 1.0 else info)[k] = fv
            elif isinstance(v, dict):
                for nk, nv in v.items():
                    if isinstance(nv, (int, float)):
                        fv = float(nv)
                        (scores if 0.0 <= fv <= 1.0 else info)[nk] = fv
        return scores, info

    @property
    def scores_dict(self) -> dict[str, float]:
        """Normalised scores (0–1 range) for percentage display."""
        result: dict[str, float] = {}
        for s in self.scores:
            result[s.name] = s.value
            if s.metadata:
                sub_scores, _ = self._extract_metadata(s.metadata)
                result.update(sub_scores)
        return result

    @property
    def info_dict(self) -> dict[str, float]:
        """Raw informational values (e.g. latency_ms) for display as-is."""
        result: dict[str, float] = {}
        for s in self.scores:
            if s.metadata:
                _, info = self._extract_metadata(s.metadata)
                result.update(info)
        return result


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

_THRESHOLDS = (0.7, 0.5)


def _color(v: float) -> str:
    if v >= _THRESHOLDS[0]:
        return "#16a34a"
    if v >= _THRESHOLDS[1]:
        return "#ca8a04"
    return "#dc2626"


def _bg(v: float) -> str:
    if v >= _THRESHOLDS[0]:
        return "#f0fdf4"
    if v >= _THRESHOLDS[1]:
        return "#fefce8"
    return "#fef2f2"


# ---------------------------------------------------------------------------
# Dimension radar chart (ADR-009)
# ---------------------------------------------------------------------------

_RADAR_COLOR = "#2563eb"
_RADAR_SAFETY_COLOR = "#dc2626"


# Approximate text metrics for -apple-system at font-size 11, used only to size
# the viewBox around the labels. Slightly generous so estimates never run short.
_RADAR_CHAR_W = 6.8
_RADAR_LINE_H = 13.0
_RADAR_LABEL_GAP = 1.16  # label distance from center, in units of plot radius


def _radar_label_anchor(cos: float) -> str:
    """Text anchor for an axis label given the cosine of its angle."""
    if abs(cos) < 0.3:
        return "middle"
    return "start" if cos > 0 else "end"


def _radar_svg(
    axis_labels: list[tuple[str, bool]],
    series: list[tuple[str, list[float]]],
    *,
    radius: float = 108,
    pad: float = 10,
) -> str:
    """Build a self-contained SVG radar chart (no external dependencies).

    ``axis_labels`` is a list of ``(label, is_safety)`` tuples, one per axis
    (the safety axis is drawn in red per ADR-003). ``series`` is a list of
    ``(color, values)`` tuples — one polygon per series (e.g. one per variant),
    with ``values`` aligned to the axes and clamped to ``[0, 1]``. Requires at
    least one axis.

    The viewBox is fitted to the bounding box of the plot circle **plus the
    estimated extent of every label**, so no axis label is ever clipped by the
    SVG viewport regardless of its length (e.g. "Safety (1234 viol.)").
    """
    n = len(axis_labels)
    # Start at the top (−90°) and go clockwise.
    angles = [-math.pi / 2 + 2 * math.pi * i / n for i in range(n)]

    # First pass: label geometry in a plot-centered coordinate system (origin at
    # the plot center). Track the bounding box of the 100% ring and every label
    # so the canvas can be sized to contain them.
    labels: list[tuple[float, float, str]] = []  # (lx, ly, anchor) per axis
    min_x = min_y = -radius
    max_x = max_y = radius
    for i, (label, _is_safety) in enumerate(axis_labels):
        cos, sin = math.cos(angles[i]), math.sin(angles[i])
        lx, ly = radius * _RADAR_LABEL_GAP * cos, radius * _RADAR_LABEL_GAP * sin
        anchor = _radar_label_anchor(cos)
        text_w = len(label) * _RADAR_CHAR_W
        if anchor == "start":
            left, right = lx, lx + text_w
        elif anchor == "end":
            left, right = lx - text_w, lx
        else:
            left, right = lx - text_w / 2, lx + text_w / 2
        min_x, max_x = min(min_x, left), max(max_x, right)
        min_y, max_y = min(min_y, ly - _RADAR_LINE_H / 2), max(max_y, ly + _RADAR_LINE_H / 2)
        labels.append((lx, ly, anchor))

    min_x, min_y = min_x - pad, min_y - pad
    max_x, max_y = max_x + pad, max_y + pad
    width, height = round(max_x - min_x), round(max_y - min_y)
    # Plot center, shifted so the whole bounding box has non-negative coords.
    cx, cy = -min_x, -min_y

    def point(value: float, i: int) -> tuple[float, float]:
        return (cx + radius * value * math.cos(angles[i]), cy + radius * value * math.sin(angles[i]))

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="Dimension radar chart" '
        f'xmlns="http://www.w3.org/2000/svg">'
    ]

    # Concentric grid rings at 25/50/75/100%.
    for ring in (0.25, 0.5, 0.75, 1.0):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(ring, i) for i in range(n)))
        parts.append(
            f'<polygon points="{pts}" fill="none" stroke="#e2e8f0" stroke-width="1" />'
            if n >= 3
            else f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius * ring:.1f}" fill="none" stroke="#e2e8f0" />'
        )

    # Axis spokes + labels.
    for i, (label, is_safety) in enumerate(axis_labels):
        ex, ey = point(1.0, i)
        color = _RADAR_SAFETY_COLOR if is_safety else "#94a3b8"
        parts.append(
            f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{color}" stroke-width="1" />'
        )
        lx0, ly0, anchor = labels[i]
        lx, ly = cx + lx0, cy + ly0
        label_color = _RADAR_SAFETY_COLOR if is_safety else "#475569"
        weight = 700 if is_safety else 600
        parts.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" dominant-baseline="middle" '
            f'font-size="11" font-weight="{weight}" fill="{label_color}" '
            f'font-family="-apple-system, sans-serif">{html_mod.escape(label)}</text>'
        )

    # One data polygon per series (e.g. per variant), overlaid on shared axes.
    for color, values in series:
        data_pts = [point(max(0.0, min(1.0, v)), i) for i, v in enumerate(values)]
        pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in data_pts)
        if n >= 3:
            parts.append(
                f'<polygon points="{pts_str}" fill="{color}" fill-opacity="0.15" stroke="{color}" stroke-width="2" />'
            )
        elif n == 2:
            (x1, y1), (x2, y2) = data_pts
            parts.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="2" />'
            )
        # Marker dots so a single-axis chart is still visible.
        for x, y in data_pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}" />')

    parts.append("</svg>")
    return "".join(parts)


_VARIANT_COLORS = {
    "good": "#16a34a",
    "mediocre": "#ca8a04",
    "bad": "#dc2626",
    "pass": "#16a34a",
    "fail": "#dc2626",
    "baseline": "#6366f1",
    "control": "#6366f1",
    "experiment": "#0ea5e9",
}


def _variant_color(variant: str) -> str:
    """Stable display color for a variant (slate fallback for unknown names)."""
    return _VARIANT_COLORS.get(variant.lower(), "#64748b")


def _variant_badge(variant: str) -> str:
    c = _variant_color(variant)
    return (
        f'<span style="background:{c};color:white;padding:3px 10px;border-radius:12px;'
        f'font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px">'
        f"{html_mod.escape(variant)}</span>"
    )


# ---------------------------------------------------------------------------
# HTML building blocks
# ---------------------------------------------------------------------------


def _score_cell(v: float) -> str:
    c = _color(v)
    bg = _bg(v)
    pct = max(0, min(100, v * 100))
    return (
        f'<td style="padding:8px 12px;background:{bg};text-align:center">'
        f'<div style="font-size:20px;font-weight:700;color:{c}">{v:.0%}</div>'
        f'<div style="width:100%;height:4px;background:#e5e7eb;border-radius:2px;margin-top:4px">'
        f'<div style="width:{pct:.0f}%;height:100%;background:{c};border-radius:2px"></div>'
        f"</div></td>"
    )


def _info_cell(v: float) -> str:
    """Render a raw informational value (not a percentage)."""
    if v >= 1000:
        display = f"{v:,.0f}"
    elif v >= 1:
        display = f"{v:,.1f}"
    else:
        display = f"{v:.4f}"
    return (
        f'<td style="padding:8px 12px;text-align:center">'
        f'<div style="font-size:16px;font-weight:600;color:#475569">{display}</div></td>'
    )


def _metric_row(
    label: str,
    values: dict[str, float | None],
    variants: list[str],
    is_info: bool = False,
) -> str:
    cells = (
        f'<td style="padding:8px 12px;font-size:13px;font-weight:500;'
        f'color:#374151;white-space:nowrap">{html_mod.escape(label)}</td>'
    )
    render = _info_cell if is_info else _score_cell
    for v in variants:
        val = values.get(v)
        if val is not None:
            cells += render(val)
        else:
            cells += '<td style="padding:8px;text-align:center;color:#9ca3af">\u2014</td>'
    return f'<tr style="border-bottom:1px solid #f3f4f6">{cells}</tr>'


def _section_header(title: str, colspan: int) -> str:
    return (
        f'<tr><td colspan="{colspan}" style="padding:8px 12px;font-size:11px;font-weight:700;'
        f"color:#64748b;text-transform:uppercase;letter-spacing:1px;background:#f1f5f9;"
        f'border-top:1px solid #e2e8f0">{html_mod.escape(title)}</td></tr>'
    )


def _summary_card(title: str, value: float, count: int) -> str:
    c = _color(value)
    bg = _bg(value)
    return (
        f'<div style="background:{bg};border:1px solid #e2e8f0;border-radius:12px;'
        f'padding:16px 20px;min-width:150px;flex:1;box-shadow:0 1px 3px rgba(0,0,0,0.04)">'
        f'<div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px">'
        f"{html_mod.escape(title)}</div>"
        f'<div style="font-size:32px;font-weight:800;color:{c};margin:4px 0">{value:.0%}</div>'
        f'<div style="font-size:12px;color:#94a3b8">{count} test cases</div>'
        f"</div>"
    )


# ---------------------------------------------------------------------------
# HtmlReporter
# ---------------------------------------------------------------------------


class HtmlReporter:
    """Collects eval results and generates a grouped HTML report.

    Results are grouped by ``group`` (e.g., test case id) and compared
    across ``variant`` values (e.g., good/mediocre/bad).
    """

    def __init__(self, title: str = "Eval Report") -> None:
        self.title = title
        self._results: list[EvalResult] = []
        self._metric_categories: dict[str, list[str]] | None = None
        self._overall_key: str = "overall"
        self._description: str | None = None
        self._variant_descriptions: dict[str, str] = {}
        self._how_to_read: str | None = None

    def add(
        self,
        eval_case: EvalCase,
        scores: list[Score],
        *,
        group: str = "",
        variant: str = "",
        label: str = "",
    ) -> None:
        """Add one evaluated case to the report."""
        self._results.append(
            EvalResult(
                eval_case=eval_case,
                scores=scores,
                group=group or _default_group(eval_case),
                variant=variant,
                label=label or _default_label(eval_case),
            )
        )

    def add_result(self, result: EvalResult) -> None:
        """Add a pre-built EvalResult."""
        self._results.append(result)

    def set_metric_categories(self, categories: dict[str, list[str]]) -> None:
        """Define how metrics are grouped in the report.

        Keys are section titles, values are lists of metric names.
        Metrics not listed in any category go into "Other".
        """
        self._metric_categories = categories

    def set_overall_key(self, key: str) -> None:
        """Set which metric/metadata key is the 'overall' score (default: 'overall')."""
        self._overall_key = key

    def set_description(self, description: str) -> None:
        """Set the report description/methodology text (supports HTML)."""
        self._description = description

    def set_variant_descriptions(self, descriptions: dict[str, str]) -> None:
        """Explain what each variant means. e.g. {"good": "Hand-crafted ideal output", ...}"""
        self._variant_descriptions = descriptions

    def set_how_to_read(self, text: str) -> None:
        """Set a 'How to read this report' section (supports HTML)."""
        self._how_to_read = text

    def generate(self, path: str | Path | None = None) -> str:
        """Generate the HTML report. Optionally write to file. Returns HTML string."""
        html_content = self._render()
        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(html_content)
        return html_content

    # --- Internal rendering ---

    def _render(self) -> str:
        groups = self._group_results()
        all_variants = self._discover_variants()
        all_metrics = self._discover_metrics()
        categories = self._resolve_categories(all_metrics)

        body_parts = []
        if self._description or self._how_to_read or self._variant_descriptions:
            body_parts.append(self._render_methodology(all_variants))
        body_parts.append(self._render_summary(groups, all_variants))
        body_parts.append(self._render_dimension_radar())
        body_parts.append(self._render_groups(groups, all_variants, categories))

        body = "\n".join(body_parts)
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html_mod.escape(self.title)}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: #f1f5f9; padding: 32px; color: #0f172a; }}
        table {{ border-collapse: collapse; }}
        tr:hover {{ background: rgba(0,0,0,0.01); }}
    </style>
</head>
<body>
    <div style="max-width:1100px;margin:0 auto">
        <div style="margin-bottom:32px">
            <h1 style="font-size:28px;font-weight:800;color:#0f172a;margin-bottom:4px">
                {html_mod.escape(self.title)}</h1>
            <p style="color:#64748b;font-size:14px">
                {len(self._results)} evaluations across {len(self._group_results())} test groups</p>
        </div>
        {body}
        <div style="text-align:center;padding:24px;color:#94a3b8;font-size:12px">
            Generated by harness-evals
        </div>
    </div>
</body>
</html>"""

    def _render_dimension_radar(self) -> str:
        """Per-dimension radar (ADR-009) from the collected scores.

        With two or more variants, each variant is drawn as its own polygon on
        shared axes so they can be compared (rather than averaged into one shape).
        With a single variant, one polygon is drawn with a per-dimension legend.
        The ``unknown`` bucket is excluded from the axes and noted in a footnote
        (ADR-009); the safety axis/violations are surfaced separately (ADR-003).
        Returns ``""`` when there are no dimensioned scores to plot.
        """
        if not self._results:
            return ""
        combined = summarize([r.scores for r in self._results])
        plotted = [d for d in order_dimensions(list(combined.by_dimension)) if d != UNKNOWN_DIMENSION]
        if not plotted:
            return ""

        has_safety = any(combined.by_dimension[d].is_safety for d in plotted)
        variants = self._discover_variants()

        if len(variants) >= 2:
            svg, legend = self._radar_multi_variant(plotted, variants, has_safety)
        else:
            svg, legend = self._radar_single(plotted, combined)

        footnote = ""
        unknown = combined.by_dimension.get(UNKNOWN_DIMENSION)
        if unknown is not None:
            footnote = (
                f'<p style="margin-top:12px;font-size:12px;color:#94a3b8;font-style:italic">'
                f"{unknown.metric_count} metric(s) with no declared dimension omitted "
                f'from the chart (bucketed as "{UNKNOWN_DIMENSION}").</p>'
            )

        return (
            f'<div style="background:white;border:1px solid #e2e8f0;border-radius:12px;'
            f'padding:24px;margin-bottom:24px;box-shadow:0 1px 3px rgba(0,0,0,0.04)">'
            f'<h2 style="font-size:18px;font-weight:700;color:#0f172a;margin-bottom:16px">'
            f"Dimension breakdown</h2>"
            f'<div style="display:flex;flex-wrap:wrap;gap:24px;align-items:center">'
            f'<div style="flex:0 0 auto">{svg}</div>'
            f'<div style="flex:1 1 240px;min-width:240px">{legend}</div>'
            f"</div>{footnote}</div>"
        )

    def _radar_single(self, plotted: list[str], combined) -> tuple[str, str]:
        """One polygon + a per-dimension legend (mean · pass-rate · n)."""
        by_dim = combined.by_dimension
        axis_labels: list[tuple[str, bool]] = []
        values: list[float] = []
        for dim in plotted:
            ds = by_dim[dim]
            label = f"Safety ({combined.safety_violations} viol.)" if ds.is_safety else dim.capitalize()
            axis_labels.append((label, ds.is_safety))
            values.append(ds.mean)
        svg = _radar_svg(axis_labels, [(_RADAR_COLOR, values)])

        rows = ""
        for dim in plotted:
            ds = by_dim[dim]
            color = _RADAR_SAFETY_COLOR if ds.is_safety else _color(ds.mean)
            detail = f"{combined.safety_violations} violation(s)" if ds.is_safety else f"{ds.pass_rate:.0%} pass"
            rows += (
                f'<div style="display:flex;justify-content:space-between;gap:16px;font-size:13px;'
                f'padding:4px 0;border-bottom:1px solid #f1f5f9">'
                f'<span style="color:#475569">{html_mod.escape(dim.capitalize())}</span>'
                f'<span style="color:{color};font-weight:600">{ds.mean:.2f} · {detail} '
                f'<span style="color:#94a3b8;font-weight:400">(n={ds.metric_count})</span></span>'
                f"</div>"
            )
        return svg, rows

    def _radar_multi_variant(self, plotted: list[str], variants: list[str], has_safety: bool) -> tuple[str, str]:
        """One polygon per variant on shared axes + a variant color key."""
        # Axis labels are shared; per-variant safety counts go in the key, so the
        # safety axis label stays a plain "Safety" here.
        axis_labels = [(dim.capitalize(), dim == SAFETY_DIMENSION) for dim in plotted]

        series: list[tuple[str, list[float]]] = []
        rows = ""
        for v in variants:
            vres = summarize([r.scores for r in self._results if r.variant == v])
            # A variant that didn't exercise a dimension plots 0.0 on that axis.
            values = [vres.by_dimension[d].mean if d in vres.by_dimension else 0.0 for d in plotted]
            color = _variant_color(v)
            series.append((color, values))
            safety_note = f" · safety {vres.safety_violations} viol." if has_safety else ""
            rows += (
                f'<div style="display:flex;align-items:center;gap:8px;font-size:13px;padding:4px 0;'
                f'border-bottom:1px solid #f1f5f9">'
                f'<span style="width:12px;height:12px;border-radius:3px;background:{color};flex:0 0 auto"></span>'
                f'<span style="color:#475569;font-weight:600">{html_mod.escape(v)}</span>'
                f'<span style="color:#94a3b8">{safety_note}</span>'
                f"</div>"
            )
        return _radar_svg(axis_labels, series), rows

    def _render_methodology(self, variants: list[str]) -> str:
        parts = []

        if self._description:
            parts.append(
                f'<div style="margin-bottom:16px;font-size:14px;line-height:1.6;color:#374151">'
                f"{self._description}</div>"
            )

        if self._how_to_read:
            parts.append(
                f'<div style="margin-bottom:16px">'
                f'<h3 style="font-size:14px;font-weight:700;color:#0f172a;margin-bottom:8px">'
                f"How to read this report</h3>"
                f'<div style="font-size:13px;line-height:1.6;color:#374151">{self._how_to_read}</div>'
                f"</div>"
            )

        if self._variant_descriptions and variants:
            items = ""
            for v in variants:
                desc = self._variant_descriptions.get(v, "")
                if desc:
                    items += (
                        f'<div style="display:flex;align-items:baseline;gap:10px;margin-bottom:6px">'
                        f"{_variant_badge(v)}"
                        f'<span style="font-size:13px;color:#374151">{desc}</span>'
                        f"</div>"
                    )
            if items:
                parts.append(
                    f'<div style="margin-bottom:4px">'
                    f'<h3 style="font-size:14px;font-weight:700;color:#0f172a;margin-bottom:10px">'
                    f"What the columns mean</h3>"
                    f"{items}</div>"
                )

        if not parts:
            return ""

        return (
            f'<div style="background:white;border:1px solid #e2e8f0;border-radius:12px;'
            f'padding:24px;margin-bottom:24px;box-shadow:0 1px 3px rgba(0,0,0,0.04)">'
            f"{''.join(parts)}</div>"
        )

    def _group_results(self) -> dict[str, list[EvalResult]]:
        groups: dict[str, list[EvalResult]] = {}
        for r in self._results:
            groups.setdefault(r.group, []).append(r)
        return groups

    def _discover_variants(self) -> list[str]:
        seen: dict[str, int] = {}
        for r in self._results:
            if r.variant and r.variant not in seen:
                seen[r.variant] = len(seen)
        return sorted(seen, key=lambda v: seen[v])

    def _discover_metrics(self) -> list[str]:
        seen: dict[str, int] = {}
        for r in self._results:
            for key in r.scores_dict:
                if key not in seen:
                    seen[key] = len(seen)
        return sorted(seen, key=lambda m: seen[m])

    def _resolve_categories(self, all_metrics: list[str]) -> list[tuple[str, list[str]]]:
        if self._metric_categories:
            categorized = set()
            result = []
            for cat_name, keys in self._metric_categories.items():
                present = [k for k in keys if k in set(all_metrics)]
                if present:
                    result.append((cat_name, present))
                    categorized.update(present)
            uncategorized = [m for m in all_metrics if m not in categorized and m != self._overall_key]
            if uncategorized:
                result.append(("Other", uncategorized))
            return result
        # Auto-categorize: just one flat list excluding overall
        non_overall = [m for m in all_metrics if m != self._overall_key]
        if non_overall:
            return [("Metrics", non_overall)]
        return []

    def _render_summary(self, groups: dict[str, list[EvalResult]], variants: list[str]) -> str:
        cards = []
        if variants:
            for v in variants:
                results_for_v = [r for r in self._results if r.variant == v]
                overalls = [r.scores_dict.get(self._overall_key) for r in results_for_v]
                overalls = [o for o in overalls if o is not None]
                if overalls:
                    avg = sum(overalls) / len(overalls)
                    cards.append(_summary_card(v.title(), avg, len(overalls)))
        else:
            overalls = [r.scores_dict.get(self._overall_key) for r in self._results]
            overalls = [o for o in overalls if o is not None]
            if overalls:
                avg = sum(overalls) / len(overalls)
                cards.append(_summary_card("Overall", avg, len(overalls)))

        # Pass/fail summary
        total = len(self._results)
        passed = sum(1 for r in self._results if all(s.passed for s in r.scores))
        if total > 0:
            pass_rate = passed / total
            cards.append(_summary_card("Pass Rate", pass_rate, total))

        narrative = self._render_narrative(variants)

        return f"""
        <div style="margin-bottom:32px">
            <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:24px;
                        overflow-x:auto;-webkit-overflow-scrolling:touch">{"".join(cards)}</div>
            {narrative}
        </div>"""

    def _render_narrative(self, variants: list[str]) -> str:
        findings = []

        # Quality discrimination across variants
        if len(variants) >= 2:
            first_v = variants[0]
            last_v = variants[-1]
            first_overalls = [r.scores_dict.get(self._overall_key, 0) for r in self._results if r.variant == first_v]
            last_overalls = [r.scores_dict.get(self._overall_key, 0) for r in self._results if r.variant == last_v]
            if first_overalls and last_overalls:
                avg_first = sum(first_overalls) / len(first_overalls)
                avg_last = sum(last_overalls) / len(last_overalls)
                spread = abs(avg_first - avg_last)
                high_v = first_v if avg_first >= avg_last else last_v
                low_v = last_v if avg_first >= avg_last else first_v
                high_avg = max(avg_first, avg_last)
                low_avg = min(avg_first, avg_last)
                if spread >= 0.3:
                    findings.append(
                        f"<strong>Quality discrimination is strong</strong> \u2014 "
                        f"<em>{high_v}</em> averages <strong style='color:#16a34a'>{high_avg:.0%}</strong> "
                        f"vs <em>{low_v}</em> at <strong style='color:#dc2626'>{low_avg:.0%}</strong> "
                        f"(spread: {spread:.0%})."
                    )
                else:
                    findings.append(
                        f"<strong>Weak discrimination</strong> \u2014 "
                        f"<em>{high_v}</em> ({high_avg:.0%}) vs <em>{low_v}</em> ({low_avg:.0%}). "
                        f"Consider tuning metrics or thresholds."
                    )

        # Metrics that consistently fail
        all_metrics = self._discover_metrics()
        for m in all_metrics:
            if m == self._overall_key:
                continue
            vals = [r.scores_dict.get(m) for r in self._results if m in r.scores_dict]
            vals = [v for v in vals if v is not None]
            if vals:
                avg = sum(vals) / len(vals)
                fail_rate = sum(1 for v in vals if v < 0.5) / len(vals)
                if fail_rate >= 0.6:
                    findings.append(
                        f"<strong>{m.replace('_', ' ').title()}</strong> fails in "
                        f"{fail_rate:.0%} of cases (avg: {avg:.0%}) \u2014 "
                        f"this may indicate a systematic gap."
                    )

        # High-performing metrics
        perfect = [
            m
            for m in all_metrics
            if m != self._overall_key
            and all(r.scores_dict.get(m, 0) >= 0.99 for r in self._results if m in r.scores_dict)
        ]
        if perfect:
            findings.append(
                f"<strong>{len(perfect)} metric(s) score 100% across all cases</strong>: "
                f"{', '.join(m.replace('_', ' ') for m in perfect[:5])}."
            )

        if not findings:
            return ""

        items = "".join(f'<li style="margin-bottom:6px;line-height:1.5">{f}</li>' for f in findings)
        return (
            f'<div style="background:white;border:1px solid #e2e8f0;border-radius:12px;'
            f'padding:24px;margin-bottom:24px;box-shadow:0 1px 3px rgba(0,0,0,0.04)">'
            f'<h3 style="margin:0 0 12px 0;font-size:16px;color:#0f172a">Key Findings</h3>'
            f'<ul style="margin:0;padding-left:20px;color:#374151;font-size:13px">{items}</ul>'
            f"</div>"
        )

    def _render_groups(
        self,
        groups: dict[str, list[EvalResult]],
        variants: list[str],
        categories: list[tuple[str, list[str]]],
    ) -> str:
        sections = []
        for group_key, results in groups.items():
            label = results[0].label or group_key
            by_variant = {r.variant: r.scores_dict for r in results}
            by_variant_info = {r.variant: r.info_dict for r in results}

            # Variant column headers
            cols = variants if variants else list(by_variant.keys())
            n_cols = len(cols) + 1

            header = (
                '<tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0">'
                '<th style="padding:10px 12px;text-align:left;font-size:12px;color:#64748b;'
                'min-width:200px">Metric</th>'
            )
            for v in cols:
                header += f'<th style="padding:10px 12px;text-align:center;min-width:120px">{_variant_badge(v)}</th>'
            header += "</tr>"

            # Overall row
            overall_cells = (
                '<td style="padding:10px 12px;font-size:14px;font-weight:700;color:#0f172a">Overall Score</td>'
            )
            for v in cols:
                val = by_variant.get(v, {}).get(self._overall_key)
                if val is not None:
                    c = _color(val)
                    bg = _bg(val)
                    overall_cells += (
                        f'<td style="padding:10px 12px;background:{bg};text-align:center;'
                        f'border-bottom:2px solid {c}">'
                        f'<div style="font-size:28px;font-weight:800;color:{c}">{val:.0%}</div></td>'
                    )
                else:
                    overall_cells += '<td style="padding:10px;text-align:center;color:#9ca3af">\u2014</td>'

            # Metric rows by category
            metric_rows = ""
            for cat_name, keys in categories:
                cat_rows = ""
                for key in keys:
                    values = {v: by_variant.get(v, {}).get(key) for v in cols}
                    if any(val is not None for val in values.values()):
                        display_name = key.replace("_", " ").title()
                        cat_rows += _metric_row(display_name, values, cols)
                if cat_rows:
                    metric_rows += _section_header(cat_name, n_cols) + cat_rows

            # Info rows — raw values displayed as-is (e.g. latency_ms)
            all_info_keys: list[str] = []
            seen: set[str] = set()
            for info in by_variant_info.values():
                for k in info:
                    if k not in seen:
                        all_info_keys.append(k)
                        seen.add(k)
            if all_info_keys:
                info_rows = ""
                for key in all_info_keys:
                    values = {v: by_variant_info.get(v, {}).get(key) for v in cols}
                    if any(val is not None for val in values.values()):
                        unit = ""
                        if key.endswith("_ms"):
                            unit = " (ms)"
                        elif key.endswith("_s"):
                            unit = " (s)"
                        display_name = key.replace("_", " ").title() + unit
                        info_rows += _metric_row(display_name, values, cols, is_info=True)
                if info_rows:
                    metric_rows += _section_header("Info", n_cols) + info_rows

            sections.append(f"""
            <div style="background:white;border:1px solid #e2e8f0;border-radius:12px;padding:0;
                        margin-bottom:20px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.04)">
                <div style="padding:16px 20px;background:linear-gradient(135deg,#f8fafc,#f1f5f9);
                            border-bottom:1px solid #e2e8f0">
                    <div style="font-size:11px;color:#64748b;text-transform:uppercase;
                                letter-spacing:0.5px;margin-bottom:4px">
                        {html_mod.escape(group_key)}</div>
                    <div style="font-size:15px;font-weight:600;color:#0f172a">
                        {html_mod.escape(label)}</div>
                </div>
                <div style="overflow-x:auto;-webkit-overflow-scrolling:touch">
                    <table style="width:100%;border-collapse:collapse;min-width:500px">
                        {header}
                        <tr>{overall_cells}</tr>
                        {metric_rows}
                    </table>
                </div>
            </div>""")

        return "\n".join(sections)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def _default_group(ec: EvalCase) -> str:
    if isinstance(ec.input, str):
        return ec.input[:60]
    return str(ec.input)[:60]


def _default_label(ec: EvalCase) -> str:
    if isinstance(ec.input, str):
        return ec.input
    return str(ec.input)
