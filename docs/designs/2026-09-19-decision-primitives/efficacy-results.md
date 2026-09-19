# Decision primitives vs. LLM-as-judge — efficacy report

## 1. What we're evaluating

**Decision primitives** are a new way to get a structured decision out of a
model — pick one of N categories, rate on an ordered scale, or answer
yes/no — via TypeSafe's System One, exposed here as three metrics:
`ChoiceMetric`, `ScoreMetric`, `NoulMetric`.

The alternative most teams already use is **LLM-as-judge**: prompt an LLM
with the same criteria and a JSON schema, and parse its answer.

This report asks one narrow question: on a fixed-rubric classification task,
does the decision primitive actually do better than just asking Claude?

## 2. How the test was run

18 hand-labeled support tickets (labeled by a human reading the text, not
generated), each scored on three dimensions:

| Dimension | Type | Values |
|---|---|---|
| `department` | Choice | returns / shipping / billing |
| `severity` | Score (ordered rubric) | 0=low, 1=medium, 2=high |
| `is_escalation` | Yes/No | is the customer demanding a human/refund right now |

Both approaches were run live, against the same 18 tickets, with the same
criteria wording, and asked to self-report a confidence (0-1) in their own
answer:

- **Decision primitive**: `ChoiceMetric`/`ScoreMetric`/`NoulMetric` against a
  live `TypeSafeDecisionProvider` (model `jev-1.13.0`).
- **LLM judge**: a custom prompt + JSON schema against a live `AnthropicLLM`
  (`claude-sonnet-4-5-20250929`).

**The two confidences are not collected the same way.** The decision
primitive's confidence comes from TypeSafe's own probability distribution
over choices. The judge's confidence is a verbalized number in a prompt
response, which empirically clusters on a handful of round values
(0.85/0.9/0.95/1.0). Every Brier-score comparison below inherits this
asymmetry — it is not an apples-to-apples measurement of "how well-calibrated
is this model" in general.

108 total calls (18 cases × 3 dimensions × 2 approaches). Raw data:
`examples/decision_vs_llm_judge/results.json`. Reproduce with
`python examples/decision_vs_llm_judge/run_comparison.py` (needs
`TYPESAFE_API_KEY` + `ANTHROPIC_API_KEY`), then
`python examples/decision_vs_llm_judge/report.py` (also writes `report.html`).

## 3. Results

| Dimension | Approach | Accuracy | Brier score (lower = "better," see note below) | Mean latency | Mean tokens |
|---|---|---|---|---|---|
| department | decision primitive | **0.889** (16/18) | **0.087** | **0.185s** | 432 |
| department | LLM judge | 0.778 (14/18) | 0.169 | 1.363s | 338 |
| severity | decision primitive | 0.778 (14/18) | 0.183 | **0.174s** | 392 |
| severity | LLM judge | **0.833** (15/18) | **0.131** | 1.319s | 366 |
| is_escalation | decision primitive | 1.000 (18/18) | **0.002** | **0.147s** | 349 |
| is_escalation | LLM judge | 1.000 (18/18) | 0.005 | 1.297s | 288 |

**Bolded** = better on that metric for that dimension.

**The Brier score here is a correctness-Brier, not an independent
calibration measure** — it's computed as `(confidence - correct) ** 2`, so it
conflates accuracy and confidence. A dimension where one approach "wins" both
accuracy and Brier is really one underlying win (the right/wrong outcome),
not two independent ones. Combined with the confidence-asymmetry caveat
above, don't read the Brier column as a clean calibration signal on its own.

**Cost:** could not be compared in dollars — the Anthropic pricing table in
this codebase has no entry for `claude-sonnet-4-5-20250929`, so `cost_usd`
came back empty for the judge side. Token counts (above) are the fallback
proxy, and they don't move in one consistent direction — the decision
primitive uses ~28% more tokens on `department` (432 vs 338) but fewer on
`is_escalation` (349 vs 288) — so the latency gap below is real
generation/round-trip time, not explained by a token-volume difference
either way.

### The one consistent finding: latency

The decision primitive was **7-9x faster on every single dimension**
(0.15-0.19s vs. 1.3-1.4s mean). This held regardless of which approach won
on accuracy. It is the only result in this report that didn't flip
direction between dimensions.

### Accuracy: mixed, not a clean win either way

- `department`: decision primitive won on accuracy (and, since Brier
  conflates with accuracy here, also on Brier).
- `severity`: LLM judge won on accuracy (and likewise on Brier).
- `is_escalation`: both were effectively perfect — the ticket language
  ("get me a manager or I'm disputing the charge") was unambiguous enough
  that neither approach was tested by this dimension. The decision
  primitive's slightly lower Brier score there (0.002 vs 0.005) is noise at
  100% accuracy for both, not a real calibration edge — with only n=2 cases
  in the low-confidence bucket for each approach, these bucket-level numbers
  aren't meaningful evidence either way.

## 4. Caveats — read before trusting the numbers

- **n=18.** One flipped case moves accuracy by ~5.5 points. The
  `department`/`severity` differences above (2 cases) are within noise for a
  sample this small — do not read "0.889 vs 0.778" as a precise, reproducible
  gap.
- **One task family, one rubric style.** All three dimensions are
  fixed-category or fixed-rubric decisions with short (1-3 sentence) input
  text. Nothing here was tested on long documents, multi-turn context, or
  open-ended judgment.
- **No cost comparison.** The dollar-cost question this report set out to
  answer is unresolved, not "roughly equal" — the pricing lookup simply
  didn't have data for the model used.
- **Single judge model.** Only one Claude model was tested. A different
  judge model or a longer/few-shot prompt could change the accuracy numbers
  without changing the latency conclusion.
- **Confidence isn't collected the same way on both sides.** The decision
  primitive's confidence is a real probability from TypeSafe; the judge's is
  a verbalized, clustered self-report. See section 3 for how this caveats
  the Brier-score comparisons.
- **Brier score conflates accuracy and calibration here.** It is not an
  independent calibration signal — see section 3.

## 5. Where to use decision primitives

- **Fixed-rubric classification/rating on short inputs where latency
  matters** — e.g. real-time ticket routing, inline content flags, per-turn
  conversation tagging, anything running in a hot path or at high volume
  where a 7-9x latency cut compounds.
- **Decisions where the criteria are genuinely enumerable** — a closed set
  of categories, or an ordered scale with a description per level. If you
  can write the rubric as "pick one of these N things," it's a candidate.
- **Batched/composite checks** — `DecisionCompositeMetric` (DP-3) folds
  several decision sub-checks sharing one input into a single provider call.
  This report only tested one unbatched call per dimension, so the
  expectation that batching compounds the latency/cost advantage further is
  speculative, not something this report measured.

## 6. How to use decision primitives

- Write `criteria` as a closed, mutually-exclusive set (Choice), an ordered
  rubric with a description per level (Score), or a single unambiguous
  yes/no condition (Noul) — not an open-ended instruction.
- Use `mode="correctness"` against a golden `expected` value when you have
  labels (as in this test); use `mode="confidence"` when you don't and want
  the self-reported probability as a calibration signal instead of a
  pass/fail.
- Treat the reported confidence as a calibration signal to monitor, not a
  guarantee — this report shows calibration quality varies by dimension, so
  spot-check it per use case rather than assuming it transfers.

## 7. Where NOT to use decision primitives

- **Open-ended judgment tasks** — free-text critique, "is this response
  helpful," rubric-free quality assessment, anything where the "criteria"
  can't be written as a fixed set of categories/levels. That's what
  `GEval`/`RubricJudge`/other LLM-judge metrics are built for, and this
  report did not test decision primitives against them for that kind of
  task — because they don't attempt it.
- **Tasks needing explanation/reasoning output** — decision primitives
  return a choice/score/bool plus a confidence, not a natural-language
  rationale. If you need the "why" (e.g. for a human reviewer or an audit
  trail), an LLM judge is the only option of the two.
- **Any decision as the sole basis for a go/no-go call at this sample size.**
  This report is a first data point, not a benchmark suite. Don't cite
  "0.889 vs 0.778" as a settled accuracy delta in a design doc or a
  procurement decision.

## 8. Bottom line

For this one task family — short, fixed-rubric classification — decision
primitives were dramatically faster and roughly accuracy-competitive (better
on `department`, worse on `severity`, both at 100% on `is_escalation`).
That's a real, if narrow, efficiency argument for using them on structured,
high-volume classification work. It is not evidence they should replace LLM-as-judge metrics generally,
and it says nothing about open-ended judgment tasks, where LLM-as-judge
remains the only tool of the two.

No production-wiring decision follows from this pass — that's deferred to a
`metrics-guide.md` follow-up once more task families are covered.
