# ADR-009: Every metric belongs to exactly one of five evaluation dimensions

## Status

Accepted

## Context

harness-evals ships 60+ metrics across 11 categories (deterministic, RAG, safety, agent, etc.). Categories describe *how* a metric works — they don't tell you *what aspect of quality* it measures. When a user assembles a suite of metrics and runs it against a target, the results are a flat list of scores. There's no higher-level signal that answers: "where is my agent strong, and where is it weak?"

We need a classification that:
- Groups scores into a small number of orthogonal quality axes
- Enables a single radar chart that works for any target (LLM, agent, RAG pipeline, chatbot)
- Is set by harness-evals at metric definition time, not configured by users
- Is stable — new dimensions should be rare (years, not months)

## Decision

Every metric maps to exactly **one** of five **dimensions**:

| Dimension | Question it answers | Examples |
|-----------|-------------------|----------|
| **Correctness** | Did it get the right answer? | ExactMatch, GEval, TaskCompletion, GoalAccuracy, AnswerCorrectness |
| **Groundedness** | Is the output supported by evidence? | Faithfulness, ContextPrecision, ContextRecall, AnswerRelevancy |
| **Safety** | Did it violate any policy or constraint? | PII, Toxicity, PromptInjection, Hallucination |
| **Trajectory** | Did it take an efficient, logical path? | PlanAdherence, StepEfficiency, ToolCorrectness, TurnEfficiency |
| **Performance** | Was it fast and cost-effective? | Latency, TokenCost, CostEfficiency, RetryCount |

### Properties

1. **Exhaustive** — every metric belongs to a dimension. No metric is unclassified.
2. **Exclusive** — each metric belongs to exactly one dimension. If a metric feels like it spans two, either the metric is compound (split it) or the dimensions overlap (tighten definitions).
3. **Intrinsic** — the dimension is set by the metric author at definition time. It is not user-configurable. Community contributors adding new metrics choose the dimension in their PR.
4. **Stable** — adding a new dimension requires an ADR and community consensus. The bar is: (a) it can't be measured by extending an existing dimension, (b) a low score triggers a different remediation action than any existing dimension, and (c) it applies to most targets.

### Dimension definitions

**Correctness**: The output matches what was expected. Ground-truth comparison, task completion, goal accuracy. Answers: "is it right?"

**Groundedness**: The output is faithful to the context it was given. No fabrication, no unsupported claims, no hallucination beyond the provided evidence. Answers: "is it supported?" This is distinct from correctness — an agent can hallucinate an answer that happens to be correct, or give a grounded answer that's wrong because the context was wrong.

**Safety**: The output and behavior comply with policies. PII protection, toxicity, prompt injection resistance, authorization boundaries. Safety scores are hard constraints (per ADR-003) — they are never averaged into an overall score, but they still belong to the Safety dimension for radar chart display.

**Trajectory**: The path the agent took to reach its answer was efficient and logical. Plan quality, step efficiency, tool selection, turn efficiency. Answers: "did it get there well?" A perfect score means the agent took the optimal path. A low score means it wandered, looped, used wrong tools, or took unnecessary steps.

**Performance**: The operational cost of producing the output. Latency, token usage, dollar cost, retry count. Answers: "was it cheap and fast?" This dimension lets you compare two agents that both get the right answer but differ in resource consumption.

### What about...

**Security?** Folds into Safety. SQL injection, command injection, SSRF, unauthorized actions — these are policy violations, same as PII leaks or toxicity.

**Robustness / Consistency?** Not a dimension. Robustness is measured by running the *same* eval multiple times and observing variance. It shows up as confidence intervals on each dimension's score, not as its own axis. An agent that's inconsistent on correctness has a wide error bar on the Correctness axis.

**Tool use efficiency?** Subset of Trajectory. Tool selection, argument correctness, and error handling are all part of "did it take a good path?"

**Context utilization?** Splits across Groundedness (did it use the context faithfully?) and Trajectory (did it retrieve the right context in the first place?).

## Implementation

### Phase 1: Dimension enum and metric tagging

Add a `Dimension` enum to `core/metric.py` and a `dimension` property to `BaseMetric`:

```python
from enum import Enum

class Dimension(str, Enum):
    CORRECTNESS = "correctness"
    GROUNDEDNESS = "groundedness"
    SAFETY = "safety"
    TRAJECTORY = "trajectory"
    PERFORMANCE = "performance"
```

Each metric sets its dimension in `__init__()`:

```python
class ExactMatchMetric(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(name="exact_match", dimension=Dimension.CORRECTNESS, **kwargs)
```

`SafetyMetric` base class defaults to `Dimension.SAFETY`.

### Phase 2: Radar chart aggregation

The radar chart score for a dimension is the mean of all metric scores in that dimension:

```python
dimension_score = mean(score.value for score in scores if score.dimension == dim)
```

Safety dimension follows ADR-003: displayed on the radar chart but *never* averaged into an overall composite score. The radar chart shows the shape — there is no single number.

### Phase 3: Metric guide update

The metrics-guide.md authoring template includes dimension selection. When authoring a new metric, the author must choose one dimension and justify it in the PR.

## Rationale

Five axes is the sweet spot for a radar chart — enough to show meaningful differentiation, few enough to read at a glance. Each dimension triggers a different remediation:

- Low **Correctness** → better training data, improved prompts, stronger model
- Low **Groundedness** → better retrieval, citation enforcement, RAG pipeline fixes
- Low **Safety** → policy rules, guardrails, output filtering
- Low **Trajectory** → better planning prompts, tool documentation, agent architecture
- Low **Performance** → caching, model downgrade, prompt compression, batching

If a low score on two dimensions would trigger the same fix, they probably shouldn't be separate dimensions.

## Trade-offs

- **Rigid classification**: Some metrics (e.g., Hallucination) could arguably live in either Safety or Groundedness. We pick one and commit. Hallucination is currently a `SafetyMetric` by base class, so it stays in Safety.
- **No user override**: Users cannot reclassify a metric at runtime. This is intentional — it keeps radar charts comparable across teams and orgs. If a user wants a metric in a different dimension, that's a signal the metric is measuring something different and should be a separate metric. Reclassification happens through a PR to the repo.
- **Future dimensions**: The bar for adding a 6th dimension is deliberately high. Pentagons are easy to read; hexagons are harder; heptagons are noise.

## Consequences

- Every metric in harness-evals gets a `dimension` field (one-time migration of ~60 metrics)
- The radar chart becomes a first-class output of any suite run
- ADR-003's "reported separately" rule for safety is formalized as a dimension property
- New metric PRs must specify and justify their dimension
