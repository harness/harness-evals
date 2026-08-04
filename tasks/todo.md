# Current task

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
