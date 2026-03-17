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
from dataclasses import dataclass
from pathlib import Path

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score


@dataclass
class EvalResult:
    """One evaluated test case with its scores and grouping metadata."""

    eval_case: EvalCase
    scores: list[Score]
    group: str = ""
    variant: str = ""
    label: str = ""

    @property
    def scores_dict(self) -> dict[str, float]:
        result = {}
        for s in self.scores:
            result[s.name] = s.value
            if s.metadata:
                for k, v in s.metadata.items():
                    if isinstance(v, (int, float)):
                        result[k] = float(v)
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


def _variant_badge(variant: str) -> str:
    c = _VARIANT_COLORS.get(variant.lower(), "#64748b")
    return (
        f'<span style="background:{c};color:white;padding:3px 10px;border-radius:12px;'
        f'font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px">'
        f'{html_mod.escape(variant)}</span>'
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
        f'</div></td>'
    )


def _metric_row(label: str, values: dict[str, float | None], variants: list[str]) -> str:
    cells = (
        f'<td style="padding:8px 12px;font-size:13px;font-weight:500;'
        f'color:#374151;white-space:nowrap">{html_mod.escape(label)}</td>'
    )
    for v in variants:
        val = values.get(v)
        if val is not None:
            cells += _score_cell(val)
        else:
            cells += '<td style="padding:8px;text-align:center;color:#9ca3af">\u2014</td>'
    return f'<tr style="border-bottom:1px solid #f3f4f6">{cells}</tr>'


def _section_header(title: str, colspan: int) -> str:
    return (
        f'<tr><td colspan="{colspan}" style="padding:8px 12px;font-size:11px;font-weight:700;'
        f'color:#64748b;text-transform:uppercase;letter-spacing:1px;background:#f1f5f9;'
        f'border-top:1px solid #e2e8f0">{html_mod.escape(title)}</td></tr>'
    )


def _summary_card(title: str, value: float, count: int) -> str:
    c = _color(value)
    bg = _bg(value)
    return (
        f'<div style="background:{bg};border:1px solid #e2e8f0;border-radius:12px;'
        f'padding:16px 20px;min-width:150px;flex:1;box-shadow:0 1px 3px rgba(0,0,0,0.04)">'
        f'<div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px">'
        f'{html_mod.escape(title)}</div>'
        f'<div style="font-size:32px;font-weight:800;color:{c};margin:4px 0">{value:.0%}</div>'
        f'<div style="font-size:12px;color:#94a3b8">{count} test cases</div>'
        f'</div>'
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
        body_parts.append(self._render_groups(groups, all_variants, categories))

        body = "\n".join(body_parts)
        return f'''<!DOCTYPE html>
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
</html>'''

    def _render_methodology(self, variants: list[str]) -> str:
        parts = []

        if self._description:
            parts.append(
                f'<div style="margin-bottom:16px;font-size:14px;line-height:1.6;color:#374151">'
                f'{self._description}</div>'
            )

        if self._how_to_read:
            parts.append(
                f'<div style="margin-bottom:16px">'
                f'<h3 style="font-size:14px;font-weight:700;color:#0f172a;margin-bottom:8px">'
                f'How to read this report</h3>'
                f'<div style="font-size:13px;line-height:1.6;color:#374151">{self._how_to_read}</div>'
                f'</div>'
            )

        if self._variant_descriptions and variants:
            items = ""
            for v in variants:
                desc = self._variant_descriptions.get(v, "")
                if desc:
                    items += (
                        f'<div style="display:flex;align-items:baseline;gap:10px;margin-bottom:6px">'
                        f'{_variant_badge(v)}'
                        f'<span style="font-size:13px;color:#374151">{desc}</span>'
                        f'</div>'
                    )
            if items:
                parts.append(
                    f'<div style="margin-bottom:4px">'
                    f'<h3 style="font-size:14px;font-weight:700;color:#0f172a;margin-bottom:10px">'
                    f'What the columns mean</h3>'
                    f'{items}</div>'
                )

        if not parts:
            return ""

        return (
            f'<div style="background:white;border:1px solid #e2e8f0;border-radius:12px;'
            f'padding:24px;margin-bottom:24px;box-shadow:0 1px 3px rgba(0,0,0,0.04)">'
            f'{"".join(parts)}</div>'
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

        return f'''
        <div style="margin-bottom:32px">
            <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:24px;
                        overflow-x:auto;-webkit-overflow-scrolling:touch">{"".join(cards)}</div>
            {narrative}
        </div>'''

    def _render_narrative(self, variants: list[str]) -> str:
        findings = []

        # Quality discrimination across variants
        if len(variants) >= 2:
            first_v = variants[0]
            last_v = variants[-1]
            first_overalls = [r.scores_dict.get(self._overall_key, 0)
                              for r in self._results if r.variant == first_v]
            last_overalls = [r.scores_dict.get(self._overall_key, 0)
                             for r in self._results if r.variant == last_v]
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
        perfect = [m for m in all_metrics if m != self._overall_key
                   and all(r.scores_dict.get(m, 0) >= 0.99
                           for r in self._results if m in r.scores_dict)]
        if perfect:
            findings.append(
                f"<strong>{len(perfect)} metric(s) score 100% across all cases</strong>: "
                f"{', '.join(m.replace('_', ' ') for m in perfect[:5])}."
            )

        if not findings:
            return ""

        items = "".join(
            f'<li style="margin-bottom:6px;line-height:1.5">{f}</li>' for f in findings
        )
        return (
            f'<div style="background:white;border:1px solid #e2e8f0;border-radius:12px;'
            f'padding:24px;margin-bottom:24px;box-shadow:0 1px 3px rgba(0,0,0,0.04)">'
            f'<h3 style="margin:0 0 12px 0;font-size:16px;color:#0f172a">Key Findings</h3>'
            f'<ul style="margin:0;padding-left:20px;color:#374151;font-size:13px">{items}</ul>'
            f'</div>'
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

            # Variant column headers
            cols = variants if variants else list(by_variant.keys())
            n_cols = len(cols) + 1

            header = (
                '<tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0">'
                '<th style="padding:10px 12px;text-align:left;font-size:12px;color:#64748b;'
                'min-width:200px">Metric</th>'
            )
            for v in cols:
                header += (
                    f'<th style="padding:10px 12px;text-align:center;min-width:120px">'
                    f'{_variant_badge(v)}</th>'
                )
            header += '</tr>'

            # Overall row
            overall_cells = (
                '<td style="padding:10px 12px;font-size:14px;font-weight:700;color:#0f172a">'
                'Overall Score</td>'
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
                    overall_cells += (
                        '<td style="padding:10px;text-align:center;color:#9ca3af">\u2014</td>'
                    )

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

            sections.append(f'''
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
            </div>''')

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
