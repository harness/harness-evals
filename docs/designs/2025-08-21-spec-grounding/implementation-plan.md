# Spec Grounding — harness-evals implementation plan

## Scope and ownership

`harness-evals` owns the reusable, source-agnostic scoring behavior. It receives one spec as
`EvalCase.context[0]`, classifies requirements and claims, computes a deterministic score, and returns a
structured `Score`. It must not know about document sources, connectors, online evaluation configs, spec IDs,
database caches, or multi-spec binding selectors.

The control plane invokes this metric once per applicable binding and performs the cross-binding rollup.

## Deliverables

- `SpecGroundingMetric`, catalog kind `spec_grounding`, dimension `GROUNDEDNESS`.
- Pure scoring and explanation types/functions that need no LLM.
- Separate, reusable requirement extraction.
- Two judge calls per evaluation after requirements have been extracted.
- Immutable `with_requirements(...)` binding for control-plane caching.
- Structured metadata suitable for a rich score explanation UI.
- Optional three-way faithfulness verdict without changing existing faithfulness scores.

## Progress

| PR | Status | Notes |
|---|---|---|
| HE-1 | **Merged** (PR #84) | Pure scoring + tests |
| HE-2 | In progress | Metric, catalog, factory, `0.20.0` |
| HE-3 | Not started | Independent |

### HE-1 — Pure scoring model and reason rendering

**Status:** merged (PR #84).

**Goal:** lock down arithmetic and edge-case semantics independently of prompts and model variability.

Shipped files:

- `src/harness_evals/metrics/grounding/__init__.py`
- `src/harness_evals/metrics/grounding/scoring.py`
- `tests/metrics/grounding/test_scoring.py`

Types:

- `RequirementStatus`: `satisfied | missing | violated`
- `ClaimStatus`: `supported | unsupported | contradicted`
- `RequirementResult`: text, status, evidence, optional spec heading
- `ClaimResult`: text, status, evidence
- `SpecGroundingResult`: final value, sub-scores, effective weights, sensitivities, cap state

Public pure functions:

- `compute_score(requirements, claims, weights, contradiction_is_fatal=False)`
- `build_metadata(result)`
- `render_reason(result)`

Rules to encode:

- `coverage = satisfied / applicable`.
- `faithfulness = supported / total_claims`.
- `consistency = 1 - contradicted / total_claims`.
- Default weights: coverage `0.4`, faithfulness `0.3`, consistency `0.3`.
- Reject negative weights and an all-zero map.
- With `applicable == 0`, omit coverage from the rollup and renormalize remaining positive weights.
- With coverage as the only positive weight and `applicable == 0`, return `0.0`; never divide by zero.
- With `applicable == 0` and `total_claims == 0`, return `0.0` with “nothing to ground”; an empty/refusal
  response must not become a perfect score.
- With `total_claims == 0`, omit faithfulness and consistency from the rollup and renormalize remaining
  positive weights (usually coverage). Sub-scores still record the guarded `1.0` ratios; they must not inflate
  the composite. If no positive weight remains, return `0.0`.
- With `contradiction_is_fatal=True`, any contradiction caps the final value at `0.0`. Already-capped results
  omit improvement/risk deltas. Pre-cap results still report improvements, but the contradiction risk is
  `-value` (any contradiction drops the score to `0.0`), not the per-claim faithfulness/consistency cost.
- Deltas use the effective post-renormalization weight sum. Contradiction risk is the next reachable
  drop: faithfulness+consistency if any supported claim remains, consistency only if only unsupported claims
  remain, and omitted when every claim is already contradicted.
- Bare status inputs may support arithmetic-only tests, but reason rendering requires typed records containing
  text and evidence.

**Approach decisions made during HE-1**

- There is no fourth `not_applicable` status. `compute_score` treats every supplied requirement as applicable
  (`satisfied | missing | violated`). Call A (HE-2) must omit out-of-scope requirements before scoring.
- `applicable == 0` and `total_claims == 0` returns `value = 0.0` with `nothing_to_ground` and omits deltas.
  That is a deliberate tightening of the merged RFC's `applicable == 0` renormalize path, which would otherwise
  treat empty faithfulness/consistency guards as a perfect `1.0`. Sub-scores still record the guarded ratios
  (`coverage 0.0`, `faithfulness 1.0`, `consistency 1.0`); only the composite is forced to zero.
- `applicable == 0` with claims still drops coverage and renormalizes over remaining positive weights.
- `total_claims == 0` with applicable requirements drops faithfulness/consistency and renormalizes over
  coverage, so a 0-of-N or 1-of-2 output cannot ride empty-claim guards past the default `0.7` threshold.
- Reason text does **not** include spec title/version/source. The control plane adds that envelope later.
- `build_metadata` includes `effective_weights`, `capped`, and `nothing_to_ground` in addition to the RFC
  result sketch. Display rounding in `render_reason` is two decimal places (`0.10`, not `0.100`).
- Unknown weight keys are rejected. Missing keys are allowed. Zero-valued keys are omitted from
  `effective_weights`.

Tests:

- Exact default arithmetic (`0.50`, `0.80`, `1.00` → `0.74`).
- Every zero-denominator combination.
- Missing weight keys and zero-valued keys.
- Floating-point clamping at `Score` boundaries.
- Fatal-cap reason and metadata.
- Deterministic ordering of requirements, claims, improvements, and risks.
- Reason snapshot/golden tests with requirement and claim text.

**Automated checks**

```bash
pytest tests/metrics/grounding/test_scoring.py -v
ruff check src/harness_evals/metrics/grounding tests/metrics/grounding
ruff format --check src/harness_evals/metrics/grounding tests/metrics/grounding
```

**Manual check**

Run a short Python snippet with typed requirement/claim records and verify:

1. the printed value is `0.74`;
2. the reason names missing requirements and unsupported claims;
3. the metadata deltas add back to the score using the documented effective weights.

**Merge gate:** no LLM calls, no catalog changes, and no service-specific imports. Satisfied: `metrics/__init__.py`,
`catalog.py`, and `factory.py` are untouched.

---

### HE-2 — `SpecGroundingMetric`, prompts, factory, and release

**Goal:** expose a production metric that uses HE-1 and can reuse pre-extracted requirements.

Add/update:

- `src/harness_evals/metrics/grounding/spec_grounding.py`
- `src/harness_evals/metrics/grounding/__init__.py`
- `src/harness_evals/metrics/__init__.py`
- `src/harness_evals/catalog.py`
- `tests/metrics/grounding/test_spec_grounding.py`
- `tests/metrics/test_factory.py`, `tests/test_plugins.py`, and/or `tests/test_recommender.py`
- `README.md`
- `docs/metrics-guide.md`
- `AGENTS.md` project-structure tree
- `CHANGELOG.md`
- `pyproject.toml`

Metric contract:

```python
SpecGroundingMetric(
    llm,
    threshold=0.7,
    weights=None,
    requirements=None,
    contradiction_is_fatal=False,
)
```

Required behavior:

- Require non-empty `EvalCase.context`; use only `context[0]`.
- `extract_requirements(spec_text, llm)` is a separate async operation.
- If `requirements` is absent, extract from `context[0]`.
- If `requirements` is supplied, skip extraction.
- `with_requirements(requirements)` returns a copy; it must not mutate a shared metric instance.
- Call A receives extracted requirements plus input/output and returns applicability/satisfaction.
- Call B receives output plus the full spec and returns claims with three-way verdicts.
- Judge responses are validated and malformed/duplicate/missing classifications have explicit behavior.
- Return one `Score(name="spec_grounding", ...)` with HE-1 metadata.
- Keep spec identity out of SDK metadata; the control plane adds binding/spec envelopes.

Catalog/factory work:

- Export `SpecGroundingMetric`.
- Add `"spec_grounding"` to `_build_registry()`.
- The current catalog sets `CatalogEntry.name = cls.__name__`; add an opt-in class/catalog display-name
  override and set this metric to `spec_grounding` so downstream consumers persist the expected score label
  without renaming every existing built-in.
- Verify catalog dimension, default threshold, and `requires_llm=True`.
- `_catalog_registry()` discovers the LLM metric. The factory reserves the
  metric's declared runtime-only options so persisted `requirements` are
  rejected, while `weights` and `contradiction_is_fatal` pass through.
- Do not expose `requirements` as persisted metric config; the control plane binds it at runtime.
- Do not import `FaithfulnessMetric`; `docs/metrics-guide.md` forbids cross-metric imports.

Release:

- Minor version bump, because this adds a built-in metric (from current `0.19.x`, this is expected to be
  `0.20.0`).
- Matching changelog entry.
- Publishing is tag-triggered by `.github/workflows/publish.yml`: after merge, create the matching `vX.Y.Z` tag
  and wait for the workflow/PyPI publication. A version change on `main` alone does not publish. Downstream
  consumers should depend on the published version, not an unpublished `main` SHA.

Tests use a fake `BaseLLM` with canned JSON and assert exact call counts:

- cold: extraction + A + B = three calls;
- pre-extracted: A + B = two calls;
- `with_requirements` leaves original instance unchanged;
- missing context returns a clear `0.0` score;
- malformed classifications do not produce a value outside `[0, 1]`;
- fatal contradiction metadata matches HE-1;
- factory construction and catalog discovery.

**Automated checks**

```bash
pytest tests/metrics/grounding/test_spec_grounding.py -v
pytest tests/metrics/test_factory.py tests/test_plugins.py tests/test_recommender.py -v
ruff check src/ tests/
ruff format --check src/ tests/
pytest tests/ -v
```

**Manual check**

Install the built wheel into a clean virtual environment, instantiate the metric through
`build_metric(...)`, and score one `EvalCase` with pasted context using a real judge connector. Capture:

- three calls without requirements;
- two calls after `extract_requirements`;
- reason and metadata JSON;
- package/catalog version.

**Merge gate:** publish the SDK version before downstream consumers start integration. A `0.20.0` minor is a
breaking range change for any consumer that currently pins below that version.

---

### HE-3 — Faithfulness three-way verdict (parallel, non-blocking)

**Goal:** expose contradicted claims in the existing metric without changing its numerical meaning.

Update:

- `src/harness_evals/metrics/rag/faithfulness.py`
- `tests/metrics/test_rag.py`
- `CHANGELOG.md`
- `pyproject.toml` only if not released together with HE-2

Rules:

- Judge verdict becomes `supported | unsupported | contradicted`.
- Both unsupported and contradicted count as unsupported in the existing formula.
- Existing score values remain byte-for-byte equivalent for equivalent judge classifications.
- Add `contradicted_claims` and evidence to metadata.

**Automated checks**

```bash
pytest tests/metrics/test_rag.py -v
```

**Manual check**

Score a statement opposite to its context and verify the score is unchanged from the old unsupported
classification while metadata labels it contradicted.

**Dependency:** none. This PR may merge before or after HE-1/HE-2 and does not gate the v1 milestone.

## Milestone mapping

| Milestone | Required SDK PRs | Exit condition |
|---|---|---|
| M0 — scoring contract frozen | HE-1 | Pure arithmetic/reason tests pass and edge cases are decided |
| M1 — releasable metric | HE-2 | Published package contains `spec_grounding` and factory/catalog can construct it |
| M2 — richer existing metrics | HE-3 | Faithfulness reports contradictions without score regressions |

## Downstream handoff

Provide the control plane / downstream consumers with:

- published package version;
- import paths for `SpecGroundingMetric` and `extract_requirements`;
- immutable `with_requirements` behavior;
- exact `Score.metadata` schema;
- fake-LLM fixtures they can mirror in contract tests.

Do not add multi-binding logic here. A standalone SDK caller evaluates one spec; the control plane selects
bindings, loads pinned content, caches requirements, and aggregates binding scores.
