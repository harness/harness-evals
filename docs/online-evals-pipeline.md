# Online Evals Pipeline

Manual Harness CI pipeline that scores production agent traces with **TaskCompletion** and publishes the scores as JUnit.

## Scope (v1)

| In | Out (later) |
|----|-------------|
| Explicit `trace_ids` input | `list_traces` / lookback window |
| `TaskCompletionMetric` + `OpenAILLM` | Larger metric sets |
| Manual trigger | Cron / trigger-based runs |
| Branch-local install from cloned repo | Published package install after release |

## How it works

```
Manual run (trace_ids, org_id, project_id, branch)
  → CI stage (python:3.11-slim)
      1. Clone this repository at the requested branch ref
      2. pip install -e ".[harness,llm]"
      3. python scripts/online_eval.py
      4. Write /harness/results/junit.xml + scores.json
  → Harness Tests tab (JUnit)
```

## Local usage

```bash
export HARNESS_API_KEY=...
export HARNESS_ACCOUNT_ID=...
export HARNESS_BASE_URL=https://qa.harness.io
export OPENAI_API_KEY=...

pip install -e ".[harness,llm]"

python scripts/online_eval.py \
  --trace-ids <id1>,<id2> \
  --org-id SrikarOrg \
  --project-id SrikarProject \
  --output-dir results
```

## Harness Pipeline

- **Org / Project:** `SrikarOrg` / `SrikarProject`
- **Identifier:** `online_evals_trace_task_completion`
- **YAML:** `.harness/online_eval.yaml`
- **Secrets:** runtime inputs `harness_api_key` and `openai_api_key`
- **Repo inputs:** `repo_connector`, `repo_name`, and `branch`
- **Default branch:** `feat/harness-otel-eval-case-source`
- **Eval inputs:** `trace_ids`, `org_id`, `project_id`, `threshold`, `model`

Because the importer and script are not published yet, the pipeline clones the feature branch and installs the package in editable mode from the workspace. After release, this can switch back to a normal package install.

## Follow-ups

- Discover traces via `list_traces(lookback_hours=...)`
- Expand metric set and model selection
- Add baselines or score regression gates
