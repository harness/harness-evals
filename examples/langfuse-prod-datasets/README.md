# Production Conversation Quality Dataset

This directory contains a reproducible pipeline for sampling Harness Agent v3
conversations from production Langfuse traces and turning them into portable
live-eval goldens. Early steps only prepare and categorize fetched
conversations — they do **not** run the live agent.

Unless a section says otherwise, run the commands below from
`examples/langfuse-prod-datasets/`.

## Artifacts

Each selected session has three files:

- `*.md` — human-readable transcript
- `*.tools.json` — complete tool requests and responses
- `*.conversation.json` — canonical cleaned conversation used by later steps

Each `review-batches/*.jsonl` file is an **offline judge-input batch**: multiple
pre-captured conversations packaged for categorization. These are not live
goldens and are not used by `harness-evals run`.

## 1. Fetch a fresh sample

```bash
python scripts/build_agent_transcripts.py \
  --random-count 15 \
  --module-count 30 \
  --use-cache
```

Future runs write Markdown, tool sidecars, and canonical conversation files in
the same pass.

## 2. Backfill existing samples without refetching

```bash
python scripts/build_agent_transcripts.py --backfill-conversations
```

This reads `random/labels.csv` and `module-coverage/labels.csv`, then rebuilds
canonical conversations from `cache/traces/*.json`. If a cached trace is
missing, the command fails and lists the missing traces. Fetch only those traces
with:

```bash
python scripts/build_agent_transcripts.py \
  --backfill-conversations \
  --fetch-missing
```

## 3. Build offline review batches

Package canonical conversations into judge-input JSONL for the offline
categorization step. This does not invoke the agent and does not create live
goldens.

```bash
python scripts/build_review_batches.py --source module-coverage --limit 5
python scripts/build_review_batches.py --source module-coverage --limit 10
python scripts/build_review_batches.py --source module-coverage --limit 30
```

The exporter writes eligible rows under `review-batches/` and a sibling
`*.ineligible.jsonl` file for structurally unusable conversations. Eligible
rows are validated with `EvalCase.from_dict()` so the judge runner can reuse
SDK helpers.

## 4. Validate a review batch without categorizing

```bash
python scripts/run_conversation_quality_eval.py \
  --input review-batches/module-coverage-005.jsonl \
  --validate-only
```

This optional check validates the input rows and configuration. It does not call
an LLM and does not assign categories.

## 5. Categorize conversations with the offline judge

```bash
export OPENAI_API_KEY=...
python scripts/run_conversation_quality_eval.py \
  --input review-batches/module-coverage-005.jsonl \
  --provider openai \
  --model gpt-4o
```

The judge reads each complete conversation and assigns:

- `usefulness`: `useful` or `useless`
- `quality` (agent outcome):
  - `good` — agent substantially satisfied the request
  - `bad` — agent materially failed
  - `unclear` — evidence is insufficient to decide good vs bad
  - `not_applicable` — only when usefulness is useless
- `golden_readiness` (portability; independent of quality):
  - `ready` — portable with only org/project placeholders
  - `needs_rewrite` — keep it, but rewrite production-specific entity refs first
    (can apply to good *or* bad outcomes)

`final_category` is kept as a compatibility alias of agent `quality` (or
`useless` when usefulness is useless). It is **not** where portability lives.

The runner writes:

- `results.jsonl` — detailed categories, component scores, confidence, reasoning,
  and cited evidence
- `review.csv` — blank human quality / golden_readiness / notes columns
- `summary.json` — quality/readiness/module distribution and run configuration

Human notes are not included in the judge prompt.

## 6. Export categorization results to Excel

Combine one or more judge result files into a review workbook:

```bash
python scripts/export_categorization_workbook.py \
  --results results/module-coverage-100/results.jsonl \
  --results results/random-100/results.jsonl \
  --output results/conversation-categorization.xlsx
```

The workbook contains:

- `Insights` — category and golden-readiness summaries, prompt themes,
  high-error patterns, and module/environment breakdowns
- `All Results` — all categorized conversations
- One sheet per supplied result dataset

The exporter requires `openpyxl`.

## 6b. Track golden inventory (`goldens.csv`)

`goldens.csv` is the inventory of prompts/sessions that are already in
`examples/prod-conversation.goldens.jsonl` plus new candidates classified as
`golden_readiness=ready` in recent categorization runs. Agent `quality` does not
filter this inventory: `bad` and `unclear` rows can be portable negative or
regression goldens.

Regenerate / refresh it after new categorization:

```bash
python scripts/export_goldens_csv.py \
  --results results/module-coverage-200/results.jsonl \
  --results results/random-200/results.jsonl \
  --output goldens.csv
```

The script backfills judge scores, reasoning, canonical files, scenario type,
and provisional `golden_id` values from:

- `prod-conversation.goldens.jsonl` / `.manifest.jsonl`
- any `results/*/results.jsonl` that contain those conversation IDs

| Column | Meaning |
|---|---|
| `status` | `in_goldens` = already in the JSONL; `candidate` = judged `ready`, pending promotion |
| `date_added` | When the row entered this inventory |
| `date_promoted` | When a candidate was written into `prod-conversation.goldens.jsonl` (blank until promoted) |
| `dataset_source` | `prod-conversation.goldens.jsonl`, `module-coverage-200`, `random-200`, etc. |
| `portability_action` | Manifest action for promoted rows; `pending_review` for candidates |

Fill `notes` during human review. When a candidate is promoted, set
`status=in_goldens`, fill `date_promoted`, and rebuild the JSONL via
`build_conversation_goldens.py` (then re-run `export_goldens_csv.py`).

## 7. Build live conversation goldens

Turn the categorized conversations into environment-portable
`ConversationGolden` rows that the live Harness SSE conversation runner can
replay against any Harness project.

```bash
python scripts/build_conversation_goldens.py \
  --review results/module-coverage-030/review.csv \
  --conversations module-coverage
```

Outputs (under the repo `examples/` directory):

- `prod-conversation.goldens.jsonl` — validated `ConversationGolden` rows
- `prod-conversation.goldens.manifest.jsonl` — one record per source
  conversation with the decision (`emitted` / `excluded`), the portability
  `action`, and the `reason`

### What the converter does

- **Filters** on agent `quality`: keeps `good` / `bad` / `unclear`, drops
  `useless`. Pipeline error-analysis conversations are also dropped (they need a
  specific failed execution that will not exist in the eval environment). Legacy
  review rows labeled `needs_improvement` map to `quality=good` +
  `golden_readiness=needs_rewrite`.
- **Ignores review-gate injections.** Synthetic platform messages ("The user
  approved the entity ...", "The user provided the following values ...") are
  elicitation continuations, not real user turns, so they never become scripted
  turns and never inflate `max_turns`.
- **Makes rows portable.** Production org/project identifiers are replaced with
  `${HARNESS_ORG}` / `${HARNESS_PROJECT}` placeholders. Read-only or write/mutation
  rows that depend on a production-specific named resource **fail closed** — they
  are excluded unless a curated entry exists in `conversation-golden-overrides.json`.
- **Scans for secrets/PII.** Emails, Harness/Bearer tokens, `api_key=` pairs, and
  non-documentation URLs in any emitted field cause the row to be excluded.
- **Validates** every emitted row with `ConversationGolden.from_dict()`.

There is no fixed yield. Inspect the manifest to see the included/excluded counts
by reason; the run prints the same breakdown.

### Curated overrides

`conversation-golden-overrides.json` holds explicit, reviewable portability
rewrites keyed by full `conversation_id`. Each entry either sets
`"exclude": true` with a `reason`, or supplies a portable `scenario`,
`expected_outcome`, and `turns`/`initial_prompt`. Never put concrete identifiers
in `elicitation_hints`.

## 8. Run the goldens against a live agent

Use a **disposable eval project** — write rows create real entities.

Run this command from the repository root (`harness-evals/`), not from this
dataset directory:

```bash
cd ../..
export SSE_ENDPOINT_URL=http://localhost:8000/stream
export HARNESS_ACCOUNT=...
export HARNESS_ORG=...
export HARNESS_PROJECT=<disposable-eval-project>
export TOKEN=...
export OPENAI_API_KEY=...
export EVAL_RUN_SUFFIX="$(date +%s)"   # unique-ify created entity names
PYTHONPATH=. poetry run harness-evals run examples/prod-conversation.eval.yaml
```

`examples/prod-conversation.eval.yaml` omits `conversation.mode` so each golden's
own mode (`scripted`) is respected, grades outcomes with the
`outcome_goal_accuracy` plugin metric (which reads the golden's curated
`expected_outcome`), and requires every per-row `sse_checks` assertion to pass
(`sse_events_match` threshold `1.0`).
