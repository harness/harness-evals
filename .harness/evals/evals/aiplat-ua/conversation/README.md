# Conversation evals (phase 2)

Datasets (committed):

- `../../datasets/aiplat-ua/conversation/conversation-readonly.jsonl` — 3 goldens
- `../../datasets/aiplat-ua/conversation/conversation-write.jsonl` — 8 goldens
- `../../datasets/aiplat-ua/conversation/conversation.jsonl` — combined

Run via harness-evals CLI (local-run stack, gitignored):

```bash
cp -r local-run/aiplat-ua.template local-run/aiplat-ua   # or sync goldens from datasets/
export PYTHONPATH=local-run/aiplat-ua:.
poetry run harness-evals run local-run/aiplat-ua/conversation-readonly.eval.yaml
```

See `local-run/aiplat-ua/README.md` and `scripts/langfuse-trace-curation/README.md`.
