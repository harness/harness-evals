# AI Platform UA — local harness-evals run stack (template)

Copy to `local-run/aiplat-ua/` (gitignored) before running:

```bash
cp -R local-run/aiplat-ua.template local-run/aiplat-ua
```

Goldens live in `.harness/evals/datasets/aiplat-ua/conversation/` (committed).
This template provides eval YAML + `plugins/` for the harness-evals CLI.

## Run (readonly)

```bash
export SSE_ENDPOINT_URL="https://<env>/gateway/intelligence/chat/v2"
export HARNESS_ACCOUNT="<account>"
export HARNESS_ORG="<org>"
export HARNESS_PROJECT="<shared-eval-project>"
export TOKEN="<pat-or-sat>"
export OPENAI_API_KEY="<key>"

export PYTHONPATH=local-run/aiplat-ua:.
poetry run harness-evals run local-run/aiplat-ua/conversation-readonly.eval.yaml
```

Results: `local-run/aiplat-ua/output/conversation-readonly-results.jsonl`

## Run (write)

```bash
export HARNESS_PROJECT="<disposable-eval-project>"
export EVAL_RUN_SUFFIX="$(date +%s)"
poetry run harness-evals run local-run/aiplat-ua/conversation-write.eval.yaml
```
