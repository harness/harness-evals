"""DP-4: aggregate ``results.json`` into an accuracy/calibration/cost report.

For each dimension (department, severity, is_escalation) and approach
(decision, judge), computes:

- accuracy: mean(correct)
- Brier score: mean((confidence - correct) ** 2) — a correctness-Brier, not a
  pure calibration measure: it conflates accuracy and confidence, since
  ``correct`` feeds both terms. Lower is better, computed as plain arithmetic
  here (not via a judged metric), per the plan.
- mean latency (seconds) and mean total tokens
- a calibration bucket table: for confidence buckets [0.5-0.6), ..., [0.9-1.0],
  the mean reported confidence vs. the observed accuracy in that bucket. A
  well-calibrated approach has bucket confidence ~= bucket accuracy. Buckets
  with n=1-2 are shown for transparency but are not meaningful evidence of
  calibration quality.

Confidence is NOT collected the same way on both sides: the decision
primitive's confidence comes from TypeSafe's own probability distribution
over choices (continuous), while the judge's confidence is a verbalized
number in a prompt response (empirically clusters on a handful of round
values like 0.85/0.9/0.95/0.98/1.0). This asymmetry can advantage the
finer-grained side on Brier score and must not be read as an apples-to-apples
comparison of "how well-calibrated is this model" in general.

Neither side has a real per-token USD price available in this codebase for
the models used (TypeSafe has no public pricing; the Anthropic model used
here isn't in the local pricing table) — cost is compared via token counts
instead, noted explicitly in the printed report, not glossed over.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

DIMENSIONS = ("department", "severity", "is_escalation")
APPROACHES = ("decision", "judge")
BUCKET_EDGES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]


def _bucket_label(i: int) -> str:
    lo, hi = BUCKET_EDGES[i], min(BUCKET_EDGES[i + 1], 1.0)
    return f"{lo:.1f}-{hi:.1f}"


def _bucket_index(confidence: float) -> int:
    if confidence < BUCKET_EDGES[0]:
        return 0
    for i in range(len(BUCKET_EDGES) - 1):
        if BUCKET_EDGES[i] <= confidence < BUCKET_EDGES[i + 1]:
            return i
    return len(BUCKET_EDGES) - 2


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def summarize(rows: list[dict], dimension: str, approach: str) -> dict:
    subset = [r for r in rows if r["dimension"] == dimension and r["approach"] == approach]
    n = len(subset)
    accuracy = sum(1 for r in subset if r["correct"]) / n
    brier = sum((r["confidence"] - (1.0 if r["correct"] else 0.0)) ** 2 for r in subset) / n
    mean_latency = sum(r["latency_s"] for r in subset) / n
    tokened = [r for r in subset if r["input_tokens"] is not None and r["output_tokens"] is not None]
    mean_tokens = sum(r["input_tokens"] + r["output_tokens"] for r in tokened) / len(tokened) if tokened else None
    costed = [r for r in subset if r["cost_usd"] is not None]
    mean_cost = sum(r["cost_usd"] for r in costed) / len(costed) if costed else None

    buckets: dict[int, list[dict]] = defaultdict(list)
    for r in subset:
        buckets[_bucket_index(r["confidence"])].append(r)
    calibration = []
    for i in sorted(buckets):
        items = buckets[i]
        mean_conf = sum(r["confidence"] for r in items) / len(items)
        obs_acc = sum(1 for r in items if r["correct"]) / len(items)
        calibration.append(
            {"bucket": _bucket_label(i), "n": len(items), "mean_confidence": mean_conf, "observed_accuracy": obs_acc}
        )

    return {
        "n": n,
        "accuracy": accuracy,
        "brier_score": brier,
        "mean_latency_s": mean_latency,
        "mean_tokens": mean_tokens,
        "mean_cost_usd": mean_cost,
        "calibration": calibration,
    }


def print_report(rows: list[dict]) -> None:
    for dimension in DIMENSIONS:
        print(f"\n{'=' * 70}\n{dimension}\n{'=' * 70}")
        for approach in APPROACHES:
            s = summarize(rows, dimension, approach)
            label = "decision primitive" if approach == "decision" else "LLM judge"
            print(f"\n-- {label} (n={s['n']}) --")
            print(f"  accuracy:       {s['accuracy']:.3f}")
            print(f"  brier score:    {s['brier_score']:.4f}  (lower = better calibrated)")
            print(f"  mean latency:   {s['mean_latency_s']:.3f}s")
            if s["mean_tokens"] is not None:
                print(f"  mean tokens:    {s['mean_tokens']:.0f}")
            if s["mean_cost_usd"] is not None:
                print(f"  mean cost:      ${s['mean_cost_usd']:.6f}")
            print("  calibration (confidence bucket -> observed accuracy):")
            for row in s["calibration"]:
                print(
                    f"    {row['bucket']:>9} (n={row['n']:>2}): "
                    f"mean_confidence={row['mean_confidence']:.3f}  observed_accuracy={row['observed_accuracy']:.3f}"
                )


def _bar(value: float, max_value: float, width_px: int = 160) -> str:
    pct = 0.0 if max_value <= 0 else min(1.0, value / max_value)
    return f'<div class="bar-track" style="width:{width_px}px"><div class="bar-fill" style="width:{pct * 100:.1f}%"></div></div>'


def _calibration_rows_html(calibration: list[dict]) -> str:
    return "\n".join(
        f"<tr><td>{row['bucket']}</td><td>{row['n']}</td>"
        f"<td>{row['mean_confidence']:.3f}</td><td>{row['observed_accuracy']:.3f}</td></tr>"
        for row in calibration
    )


def _approach_card_html(label: str, s: dict, max_latency: float) -> str:
    tokens_row = f"<tr><td>Mean tokens</td><td>{s['mean_tokens']:.0f}</td></tr>" if s["mean_tokens"] is not None else ""
    cost_row = (
        f"<tr><td>Mean cost</td><td>${s['mean_cost_usd']:.6f}</td></tr>" if s["mean_cost_usd"] is not None else ""
    )
    return f"""
    <div class="card">
      <h3>{label} <span class="n">(n={s["n"]})</span></h3>
      <table class="metrics">
        <tr><td>Accuracy</td><td>{s["accuracy"]:.3f}</td></tr>
        <tr><td>Brier score</td><td>{s["brier_score"]:.4f}</td></tr>
        <tr><td>Mean latency</td><td>{s["mean_latency_s"]:.3f}s {_bar(s["mean_latency_s"], max_latency)}</td></tr>
        {tokens_row}
        {cost_row}
      </table>
      <table class="calibration">
        <thead><tr><th>Confidence bucket</th><th>n</th><th>Mean confidence</th><th>Observed accuracy</th></tr></thead>
        <tbody>{_calibration_rows_html(s["calibration"])}</tbody>
      </table>
    </div>
    """


def _summary_table_html(rows: list[dict]) -> str:
    body = []
    for dimension in DIMENSIONS:
        summaries = {a: summarize(rows, dimension, a) for a in APPROACHES}
        d, j = summaries["decision"], summaries["judge"]
        acc_d_win = d["accuracy"] > j["accuracy"]
        acc_j_win = j["accuracy"] > d["accuracy"]
        brier_d_win = d["brier_score"] < j["brier_score"]
        brier_j_win = j["brier_score"] < d["brier_score"]
        lat_d_win = d["mean_latency_s"] < j["mean_latency_s"]
        lat_j_win = j["mean_latency_s"] < d["mean_latency_s"]
        body.append(
            f"<tr><td>{dimension}</td><td>Decision primitive</td>"
            f"<td class='{'win' if acc_d_win else ''}'>{d['accuracy']:.3f}</td>"
            f"<td class='{'win' if brier_d_win else ''}'>{d['brier_score']:.3f}</td>"
            f"<td class='{'win' if lat_d_win else ''}'>{d['mean_latency_s']:.3f}s</td></tr>"
        )
        body.append(
            f"<tr><td></td><td>LLM judge</td>"
            f"<td class='{'win' if acc_j_win else ''}'>{j['accuracy']:.3f}</td>"
            f"<td class='{'win' if brier_j_win else ''}'>{j['brier_score']:.3f}</td>"
            f"<td class='{'win' if lat_j_win else ''}'>{j['mean_latency_s']:.3f}s</td></tr>"
        )
    return "\n".join(body)


def render_html(rows: list[dict]) -> str:
    dimension_sections = []
    for dimension in DIMENSIONS:
        summaries = {approach: summarize(rows, dimension, approach) for approach in APPROACHES}
        max_latency = max(s["mean_latency_s"] for s in summaries.values())
        cards = "\n".join(
            _approach_card_html("Decision primitive" if a == "decision" else "LLM judge", summaries[a], max_latency)
            for a in APPROACHES
        )
        dimension_sections.append(f'<section><h2>{dimension}</h2><div class="cards">{cards}</div></section>')

    summary_rows = _summary_table_html(rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Decision primitives vs. LLM-as-judge — efficacy report</title>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 2rem auto; max-width: 900px; color: #1a1a1a; line-height: 1.5; padding: 0 1rem; }}
  h1 {{ font-size: 1.6rem; }}
  h2 {{ text-transform: capitalize; border-bottom: 2px solid #333; padding-bottom: 0.25rem; margin-top: 2.5rem; }}
  h2.plain {{ text-transform: none; }}
  .subtitle {{ color: #555; margin-top: -0.5rem; }}
  .cards {{ display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 1rem; }}
  .card {{ flex: 1; min-width: 320px; border: 1px solid #ddd; border-radius: 8px; padding: 1rem; }}
  .card h3 {{ margin-top: 0; }}
  .n {{ color: #888; font-size: 0.85em; font-weight: normal; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 0.75rem; font-size: 0.9rem; }}
  table.metrics td {{ padding: 3px 6px; }}
  table.metrics td:first-child {{ color: #555; }}
  table.calibration th, table.calibration td {{ border: 1px solid #eee; padding: 4px 6px; text-align: right; }}
  table.calibration th:first-child, table.calibration td:first-child {{ text-align: left; }}
  table.summary th, table.summary td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: right; }}
  table.summary th:nth-child(1), table.summary th:nth-child(2),
  table.summary td:nth-child(1), table.summary td:nth-child(2) {{ text-align: left; }}
  table.summary td.win {{ background: #eaf5ea; font-weight: 600; }}
  .bar-track {{ display: inline-block; height: 10px; background: #eee; border-radius: 4px; vertical-align: middle; margin-left: 6px; }}
  .bar-fill {{ height: 10px; background: #4a7cff; border-radius: 4px; }}
  .note {{ color: #666; font-size: 0.9rem; }}
  .panel {{ border-left: 4px solid #ccc; padding: 0.25rem 1rem; margin: 1rem 0; background: #fafafa; }}
  .panel.use {{ border-color: #4caf50; }}
  .panel.avoid {{ border-color: #d9534f; }}
  .panel.caveat {{ border-color: #f0ad4e; }}
  .bottom-line {{ border: 2px solid #333; border-radius: 8px; padding: 1rem 1.25rem; margin-top: 1rem; background: #fff; }}
  code {{ background: #f2f2f2; padding: 1px 5px; border-radius: 3px; }}
  ul {{ padding-left: 1.25rem; }}
</style>
</head>
<body>
<h1>Decision primitives vs. LLM-as-judge — efficacy report</h1>
<p class="subtitle">Support-ticket triage, 18 hand-labeled cases, 3 dimensions (department / severity /
is_escalation), both approaches run live against the same inputs.</p>

<h2 class="plain">What's being compared</h2>
<p><strong>Decision primitives</strong> — <code>ChoiceMetric</code>/<code>ScoreMetric</code>/<code>NoulMetric</code>
against a live <code>TypeSafeDecisionProvider</code> (model <code>jev-1.13.0</code>) — vs.
<strong>LLM-as-judge</strong> — a custom prompt + JSON schema against a live <code>AnthropicLLM</code>
(<code>claude-sonnet-4-5-20250929</code>). Same criteria wording, same 18 tickets, both self-report a 0-1
confidence — but the two confidences are not collected the same way. The decision primitive's confidence
comes from TypeSafe's own probability distribution over choices; the judge's confidence is a verbalized
number in a prompt response, which empirically clusters on a handful of round values (0.85/0.9/0.95/1.0).
This asymmetry caveats every Brier-score comparison below.</p>

<h2 class="plain">Summary</h2>
<table class="summary">
  <thead><tr><th>Dimension</th><th>Approach</th><th>Accuracy</th><th>Brier score (lower=better)</th><th>Mean latency</th></tr></thead>
  <tbody>{summary_rows}</tbody>
</table>
<p class="note">Green = better on that column for that dimension. Latency is the one metric that wins
consistently for the decision primitive across every dimension (~7-9x faster). Accuracy splits one
dimension each way (<code>department</code> to the decision primitive, <code>severity</code> to the judge).
The Brier score (a correctness-Brier, not an independent calibration measure — see caveats) follows the
same accuracy split, plus a decision-primitive edge on <code>is_escalation</code>; but both approaches are
already at 100% accuracy there, so treat that Brier gap as noise rather than a real calibration win.</p>
<p class="note"><strong>Cost:</strong> not comparable — the Anthropic pricing table in this codebase has no
rate for <code>claude-sonnet-4-5-20250929</code>, so <code>cost_usd</code> came back empty for the judge side.
Token counts (per-dimension cards below) are the fallback proxy; they don't move in one consistent
direction — the decision primitive uses ~28% more tokens on <code>department</code> (432 vs 338) but fewer
on <code>is_escalation</code> (349 vs 288) — so the latency gap is real generation/round-trip time, not
explained by a token-volume difference either way.</p>

<div class="panel caveat">
<h2 class="plain" style="margin-top:0.25rem;border:none;padding:0">Caveats — read before trusting the numbers</h2>
<ul>
  <li><strong>n=18.</strong> One flipped case moves accuracy ~5.5 points. Don't read the accuracy deltas
  above as a precise, reproducible gap.</li>
  <li><strong>One task family, one rubric style.</strong> Fixed-category/fixed-rubric decisions on short
  (1-3 sentence) text only. Nothing here covers long documents, multi-turn context, or open-ended judgment.</li>
  <li><strong>No cost comparison</strong> — unresolved, not "roughly equal."</li>
  <li><strong>Single judge model.</strong> A different model or prompt could change accuracy without
  changing the latency conclusion.</li>
  <li><strong>Confidence isn't collected the same way on both sides.</strong> The decision primitive's
  confidence is a real probability from TypeSafe; the judge's is a verbalized, clustered self-report. Brier
  scores compare these two differently-shaped numbers, not two instances of the same measurement.</li>
  <li><strong>Brier score here conflates accuracy and calibration.</strong> It's a correctness-Brier
  (confidence vs. right/wrong), not an independent calibration signal — a dimension where one approach
  "wins" both accuracy and Brier is really one underlying win, not two.</li>
  <li><strong>n=1-2 calibration buckets are not meaningful evidence</strong> of calibration quality — they're
  shown for transparency, not as a signal to act on.</li>
</ul>
</div>

<div class="panel use">
<h2 class="plain" style="margin-top:0.25rem;border:none;padding:0">Where to use decision primitives</h2>
<ul>
  <li>Fixed-rubric classification/rating on short inputs where latency matters — ticket routing, inline
  content flags, per-turn conversation tagging, hot-path or high-volume decisions.</li>
  <li>Decisions where the criteria are genuinely enumerable — a closed category set, or an ordered scale
  with a description per level.</li>
  <li>Batched/composite checks — <code>DecisionCompositeMetric</code> folds several sub-checks sharing one
  input into a single provider call. This report only tested one unbatched call per dimension, so the claim
  that batching compounds the latency advantage further is a reasonable expectation, not something this
  report measured — treat it as speculative until tested.</li>
</ul>
</div>

<div class="panel">
<h2 class="plain" style="margin-top:0.25rem;border:none;padding:0">How to use them</h2>
<ul>
  <li>Write <code>criteria</code> as a closed, mutually-exclusive set (Choice), an ordered rubric with a
  description per level (Score), or one unambiguous yes/no condition (Noul) — not an open instruction.</li>
  <li>Use <code>mode="correctness"</code> against a golden <code>expected</code> when labels exist (as
  here); use <code>mode="confidence"</code> when they don't, and treat the self-reported probability as a
  calibration signal rather than pass/fail.</li>
  <li>Spot-check calibration per use case — this report shows it varies by dimension, so it doesn't
  transfer automatically.</li>
</ul>
</div>

<div class="panel avoid">
<h2 class="plain" style="margin-top:0.25rem;border:none;padding:0">Where NOT to use decision primitives</h2>
<ul>
  <li><strong>Open-ended judgment</strong> — free-text critique, "is this response helpful," rubric-free
  quality assessment. That's what <code>GEval</code>/<code>RubricJudge</code> are for; decision primitives
  don't attempt it and were not tested against them for that kind of task.</li>
  <li><strong>Tasks needing an explanation/rationale.</strong> Decision primitives return a
  choice/score/bool plus a confidence, not natural-language reasoning. If a human reviewer or audit trail
  needs the "why," use an LLM judge.</li>
  <li><strong>As the sole basis for a go/no-go call at this sample size.</strong> This is a first data
  point, not a benchmark suite — don't cite these exact numbers in a procurement or design decision.</li>
</ul>
</div>

<div class="bottom-line">
<strong>Bottom line:</strong> for this one task family — short, fixed-rubric classification — decision
primitives were dramatically faster and roughly accuracy-competitive (better on <code>department</code>,
worse on <code>severity</code>, both at 100% on <code>is_escalation</code>). That's a real, if narrow,
efficiency argument for structured, high-volume classification work. It is not evidence they should replace LLM-as-judge metrics generally, and says nothing about
open-ended judgment tasks, where LLM-as-judge remains the only tool of the two. No production-wiring
decision follows from this pass.
</div>

<h2>Per-dimension detail</h2>
{"".join(dimension_sections)}
</body>
</html>
"""


if __name__ == "__main__":
    results_path = Path(__file__).parent / "results.json"
    rows = load_rows(results_path)
    print_report(rows)

    html_path = Path(__file__).parent / "report.html"
    html_path.write_text(render_html(rows))
    print(f"\nWrote HTML report to {html_path}")
