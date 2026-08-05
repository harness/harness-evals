# Harness-internal AI Evals entities (aiplat-ua)

Git-backed eval definitions for **trace-derived** Unified Agent and Knowledge Graph
prod evals. Paired with the public OSS SDK in this repo; **not** published to PyPI.

| Surface | Purpose | Registered on control plane |
|---------|---------|----------------------------|
| `examples/` | Generic OSS examples | N/A |
| `qa-chat` | (lives in qpe-evals-library) | Yes |
| `aiplat-ua` | Prod trace / prod account evals | No — local-run only |

## Layout

```
.harness/evals/
├── datasets/aiplat-ua/
│   ├── knowledge-graph/     # kg-prod-* JSONL
│   └── conversation/        # readonly + write ConversationGolden JSONL
├── evals/aiplat-ua/
│   ├── knowledge-graph/     # kg-prod-* eval YAML
│   └── conversation/        # phase 2 — harness-evals CLI (see README)
├── suites/aiplat-ua/
│   ├── aiplat-ua-kg.yaml
│   └── aiplat-ua-conversation.yaml
└── targets/prod/
    └── kg-unified-agent.yaml
```

## Related paths

- `scripts/langfuse-trace-curation/` — Langfuse export → golden builder
- `local-run/aiplat-ua/` — gitignored harness-evals CLI run stack (`plugins/`, eval YAML)
