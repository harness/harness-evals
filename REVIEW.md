# PLAN.md Review

## Overall Assessment

**The plan is strong and ready to execute.** The strategy-to-plan translation is faithful, the abstractions are clean, and the phased approach is pragmatic. The reliability framework from Rabanser et al. is integrated substantively — not decoratively — and represents genuine differentiation against every framework in the space.

Below I evaluate the plan against established industry patterns from DeepEval, RAGAS, and promptfoo.

---

## What's Good

### 1. Research grounding is genuine differentiation

The Rabanser et al. integration isn't superficial. You've mapped 12 metrics to specific phases, designed `ReliabilityMetric` as a proper base class, and made multi-run evaluation native via `TestCase.runs`. No existing framework (DeepEval, RAGAS, promptfoo, OpenAI Evals, autoevals) has first-class reliability measurement. DeepEval's `GEval` supports `n_eval` for judge consistency, but that's measuring the *evaluator's* variance, not the *agent's*.

### 2. "One metric = one file" is the right constraint

This is what makes the repo AI-agent-friendly. An agent reads AGENTS.md, looks at `exact_match.py`, and creates a new metric. Self-documenting pattern.

Industry comparison: DeepEval uses one *directory* per metric (with `template.py`, `schema.py` alongside the metric). That's heavier. promptfoo uses assertion *types* as strings in YAML — not extensible without modifying core. The harness-evals approach (one file per metric, flat within category subdirectories) is the cleanest for contribution.

### 3. Phase 1 is LLM-free

Smart. Removes the biggest friction from first use. DeepEval requires an LLM key for most useful metrics (only `ExactMatchMetric` and `JsonSchemaMetric` are deterministic). RAGAS requires an LLM for everything. promptfoo's deterministic assertions (`equals`, `contains`, `regex`) work without keys but the framework is config-driven, not programmatic.

harness-evals Phase 1 gives you structural JSON diff, schema validation, latency thresholds, and reliability metrics — all without an API key. That's a meaningfully better first-run experience than any competitor.

### 4. Clean OSS/commercial boundary

Scoring in OSS, workflow/governance in the paid product. This mirrors DeepEval's model (OSS library + Confident AI platform) but with a clearer separation — DeepEval has telemetry and cloud hooks baked into the OSS `__init__.py`. harness-evals should keep the OSS layer completely standalone.

### 5. Core abstractions are minimal and correct

`TestCase`, `BaseMetric`, `Score`, `assert_test` — this matches the established pattern from DeepEval (`LLMTestCase`, `BaseMetric`, `assert_test`) while being cleaner. The right surface area for Phase 1.

---

## Issues and Recommendations

### 1. Add `evaluate()` — the non-assertion variant

**Priority: High. This is table stakes.**

The plan only mentions `assert_test`. Both DeepEval and RAGAS provide a non-failing `evaluate()` function, and it's the more commonly used entry point:

- **DeepEval**: `evaluate(test_cases=[...], metrics=[...])` returns `TestResult` objects without raising
- **RAGAS**: `evaluate(dataset, metrics, llm)` returns `EvaluationResult` with `.to_pandas()`
- **promptfoo**: `eval` command defaults to reporting, `--ci` flag enables failure exit codes

Developers explore with `evaluate()`, then gate with `assert_test()`. Both are needed from day one.

```python
# Exploration mode — returns scores, doesn't fail
scores = evaluate(test_case, metrics=[...])

# Gate mode — raises AssertionError on failure
assert_test(test_case, metrics=[...])
```

The `runner.py` file is already in the plan. Make `evaluate()` explicit in the core abstractions section.

### 2. `TestCase.runs` — the recursive type needs a guard, not a new class

The plan defines:

```python
runs: list["TestCase"] | None = None
```

Each run is a `TestCase` that could have its own `runs`, creating unbounded recursion. However, introducing a separate `MultiRunTestCase` class (as I initially considered) would break the single-type interface that makes `BaseMetric.measure(test_case)` clean.

**Recommendation**: Keep `runs` on `TestCase` but document and enforce that nested runs are ignored. This is simpler than a type split:

```python
@dataclass
class TestCase:
    # ... fields ...
    runs: list["TestCase"] | None = None  # Nested runs on sub-cases are ignored
```

`ReliabilityMetric.measure()` already handles this correctly — it reads `test_case.runs` and passes individual `TestCase` objects to `measure_runs()`, never recursing into their `.runs`.

### 3. `confidence` should stay in `metadata`, not be a top-level field

The plan puts `confidence` as both a top-level `TestCase` field AND a standard metadata key. Pick one.

**Recommendation**: Keep it in `metadata`. Reasons:
- It's only used by Phase 2 predictability metrics — no need to pollute the Phase 1 dataclass
- DeepEval doesn't have `confidence` at all; RAGAS doesn't either. This is novel territory — keep it in the flexible `metadata` dict until the pattern stabilizes
- Standard metadata keys convention already handles this cleanly

### 4. Sinks need a minimal interface specification

The directory structure shows `sinks/stdout.py` and `sinks/json_sink.py`, but the plan doesn't describe how sinks connect to `assert_test` or `evaluate`. This matters for the commercial product integration — Harness AI Evals will need to send scores to its own backend.

Industry patterns:
- **DeepEval**: No pluggable sinks in OSS. Console output + hardcoded Confident AI API. Custom output = process the return value yourself.
- **RAGAS**: Returns `EvaluationResult` with `.to_pandas()`. No sink abstraction.
- **promptfoo**: Output format determined by `-o` file extension (`.json`, `.xml`, `.csv`, `.html`).

**Recommendation**: Keep it simple. Sinks are an optional parameter on `evaluate()`:

```python
scores = evaluate(test_case, metrics=[...], sinks=[StdoutSink(), JsonSink("results.json")])
```

Minimal interface:

```python
class BaseSink(ABC):
    @abstractmethod
    def write(self, scores: list[Score], test_case: TestCase) -> None: ...
```

This is cleaner than DeepEval's approach (no sink abstraction at all) and gives the commercial product a clean extension point without forking.

### 5. Align Phase 1 metric count with the strategy doc

The strategy doc promises ~14 metrics for Phase 1:
- 4 deterministic: ExactMatch, Contains, Regex, NumericDiff
- 3 structural: JsonDiff (3 tiers count as 1), SchemaValidation
- 4 operational: Latency, TokenCost, CostEfficiency, RetryCount
- 2 reliability: OutcomeConsistency, ResourceConsistency

The PLAN.md details only 6 reference metrics and the directory structure only shows files for those 6. The missing ones (Contains, Regex, NumericDiff, TokenCost, CostEfficiency, RetryCount) are trivial — each is 20-40 lines.

**Recommendation**: Add them to the directory structure. They're easy wins that fill out the metric inventory and give more examples for contributors. Each is a single file.

### 6. Add `py.typed` and note type annotation strategy

For a library meant to be `pip install`ed and adopted into a product, PEP 561 compliance matters. Add `py.typed` marker to `src/harness_evals/` and note that all public APIs use type annotations. This is free and makes IDE integration significantly better.

DeepEval uses Pydantic models (typed by default). RAGAS uses dataclasses with annotations. harness-evals should match this standard.

### 7. Move the citation/reference section out of PLAN.md

Lines 376-431 are documentation guidance — useful content, but not implementation plan. Move to `docs/references.md` or `CITATIONS.md` to keep PLAN.md focused on what to build.

### 8. Consider `__all__` exports

The plan says `__init__.py` exports `TestCase, assert_test, Score`. Make this explicit with `__all__`. DeepEval does this — small public API surface at the top level, metrics imported from `deepeval.metrics`. harness-evals should follow the same pattern:

```python
# harness_evals/__init__.py
from harness_evals.core.test_case import TestCase
from harness_evals.core.score import Score
from harness_evals.core.runner import assert_test, evaluate

__all__ = ["TestCase", "Score", "assert_test", "evaluate"]
```

Metrics imported separately: `from harness_evals.metrics import ExactMatch, JsonDiff`

---

## Industry Pattern Comparison

| Aspect | DeepEval | RAGAS | promptfoo | harness-evals (plan) | Assessment |
|--------|----------|-------|-----------|---------------------|------------|
| **Test case** | `LLMTestCase` (Pydantic) | `SingleTurnSample` (dataclass) | YAML `vars` + `assert` | `TestCase` (dataclass) | Good — dataclass is simpler than Pydantic, more Pythonic than YAML |
| **Metric interface** | `measure(test_case) -> float` | `_single_turn_ascore(sample) -> float` | Assertion type string | `measure(test_case) -> float` | Good — matches DeepEval's proven pattern |
| **Non-assertion eval** | `evaluate()` returns results | `evaluate()` returns results | Default mode is reporting | Not in plan | **Gap — add `evaluate()`** |
| **Multi-run** | Not native (loop yourself) | Not native | `--repeat` flag | Native `runs` field + `ReliabilityMetric` | **Differentiator** |
| **Output sinks** | Console + hardcoded cloud API | `.to_pandas()` return value | CLI `-o` flag (JSON/XML/CSV/HTML) | `sinks/` directory | Good direction — needs interface spec |
| **Metric organization** | One directory per metric (heavy) | Flat in `metrics/` | Assertion types (not extensible) | One file per metric in category subdirs | **Best of both** — categorized but not heavy |
| **LLM-free metrics** | ~2 (ExactMatch, JsonSchema) | 0 | ~8 deterministic assertions | ~13 in Phase 1 | **Differentiator** |
| **CI/CD native** | Not really (pytest only) | Not really (call in pytest) | Yes (`--ci`, JUnit, exit codes) | Yes (pytest + JUnit + baseline) | Matches promptfoo, better than DeepEval/RAGAS |
| **Reliability metrics** | None | None | None | 12 metrics across 4 dimensions | **Unique in the space** |

---

## Verdict

The plan is architecturally sound and well-positioned against the industry. The main adjustments are:

1. **Add `evaluate()`** — table stakes, every framework has it
2. **Specify sink interface** — minimal ABC, needed for commercial integration
3. **Add the missing Phase 1 metrics** to the directory structure — align with strategy doc's ~14 count
4. **Keep `confidence` in metadata** — don't promote to top-level until Phase 2 validates the pattern
5. **Guard `runs` recursion** — document, don't restructure
6. **Move citations section** — keep PLAN.md focused on what to build

None of these are blockers. The plan is ready to execute.
