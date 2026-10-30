# Admission design simplicity review

- [x] Review the configured-admission design and recent fixes for simpler equivalent alternatives.
- [x] Remove duplicate tool configuration checks; retain validation at configuration boundaries.
- [x] Extend mutation regressions and run focused/full checks without provider credentials.
- [x] Push the bounded cleanup to the existing PR branch; do not merge or release.

Removed repeated MCP expected-trace and tool-correctness mode checks from
contract assembly; JSON input and live-instance reconstruction already validate
these configurations. Eight additional malformed-mutation cases retain strict
JSON and no-execution coverage. No public contract or rule types changed.
Full tests passed on Python 3.10, 3.11, 3.12, and 3.14: 4,175 passed and four
live-provider tests skipped per version. Ruff lint and format checks passed.

# Admission/scoring parity follow-up

- [x] Reject unknown tool arguments for strict argument matching without rejecting observed empty objects.
- [x] Diagnose unestablished execution order and preserve uniquely linked tool-result identity.
- [x] Reject malformed consumed turn telemetry while retaining supported missing-value skips.
- [x] Require finite positive configured and active composite weights.
- [x] Keep mutable regex configuration and cached scorer state consistent.
- [x] Add adapter-to-admission-to-scoring regressions and run focused/full checks.
- [x] Push the fixes to the existing feature branch without merging or publishing.

Full tests: 4,167 passed and four live-provider tests skipped on Python 3.10,
3.11, 3.12, and 3.14. All-extras installs used declared dependencies; Python
3.10–3.12 dev tools were pinned to lockfile versions. Ruff lint/format and
isolated Python 3.12 wheel/source builds passed. Greedy subset argument
pairing requires established order; tool-name subset matching does not.
Sanitized unknown Group telemetry retains supported partial scoring, while
malformed values still consumed by operational scorers fail admission.

# Workflow and release readiness follow-up

- [x] Remove the PR's added `docs/` file; retain the published LLD outside the repository.
- [x] Align README and AGENTS with configured admission and the checked-in workflow commands.
- [x] Install declared dependencies and run CI/publish test checks on Python 3.10, 3.11, and 3.12.
- [x] Verify the release version guard, isolated wheel/source build, and packaged public APIs.
- [x] Push the follow-up to the existing PR without tagging, publishing, or merging.

Each workflow matrix version passed the locked all-extras installation, pinned
Ruff lint/format, and exact workflow test command: 3,869 passed, 4 live-provider
tests skipped. The additional full suite passed 3,989 tests with the same four
skips on each version. Python 3.12 package validation and isolated wheel/source
build passed. Matching and mismatched release-tag guard cases behaved correctly
without creating tags. Wheel/source contents, metadata, version/changelog, and
installed core-only wheel imports, catalog JSON, admission, and CLI smoke checks
passed outside the source checkout. No dependency or lockfile changes were needed.
Local checks do not exercise PyPI publishing permissions or GitHub release writes.

# Configured metric validation

- [x] Write and publish the low-level design separating discovery, configuration, admission, and scoring.
- [x] Reproduce configuration-specific declaration errors and correct projection precision.
- [x] Add a versioned JSON configured-metric contract and pure contract resolution.
- [x] Add shared mode/scope/execution and nested/alternative evidence validation with safe reasons.
- [x] Explicitly review a bounded initial Group-compatible set and reject other Group metrics.
- [x] Integrate contract discovery into the SDK catalog without enforcing backend/frontend policy.
- [x] Add configuration, no-I/O, serialization, known-empty, Group, and regression tests.
- [x] Update documentation/changelog/version and run complete checks.
- [x] Open the SDK PR without merging or publishing a release.

Validation: 3,989 passed and 4 live-provider tests skipped on both Python 3.12
and Python 3.14. Ruff lint/format, package validation, wheel/source builds, and
whitespace checks pass. The configured admission API is opt-in; no backend or
frontend enforcement was changed.

# Previous task

- [x] Split transcript generation into random and module-coverage datasets.
- [x] Balance coverage round-robin by module and environment.
- [x] Keep the two samples non-overlapping.
- [x] Regenerate and verify both review datasets.
- [x] Add canonical conversation JSON and backfill all 45 cached sessions.
- [x] Export validated EvalCase batches of 5, 10, and 30 conversations.
- [x] Implement the full-conversation usefulness/quality judge.
- [x] Add the configurable pre-captured EvalCase runner and review outputs.
- [x] Add focused tests and validate the first calibration batch offline.

## Review

Generated 15 random and 30 module-coverage conversations. Each dataset has
matching Markdown/tool-sidecar counts and a labels CSV. The samples have zero
conversation overlap; module coverage spans 14 modules.

All 45 sessions now have matching Markdown, full tool sidecar, and canonical
conversation JSON files. The module-coverage EvalCase batches contain 5, 10,
and 30 validated rows, with no structurally ineligible rows. The five-session
runner input validates without invoking an agent or LLM. Focused pipeline tests
pass, as do the relevant core and plugin regression suites. A paid judge run
was intentionally not started because no OpenAI credential was supplied.

## Team skill: production conversations to evals

- [x] Encode selection, portability, elicitation, judging, and validation rules.
- [x] Add reusable golden/override templates and review checklist.
- [x] Validate skill structure and document the team entry point.

The project skill `build-conversation-eval-dataset` now captures the end-to-end
curation workflow and links to a reusable reference. Metadata, file references,
and whitespace validate; focused converter and elicitation tests pass.
