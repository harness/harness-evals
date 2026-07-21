# Conversation Golden Datasets

Conversation goldens describe multi-turn scenarios for conversational agents. One JSONL row is one eval scenario, not one HTTP request. Runtime systems such as `ConversationSimulator` and conversational targets may make several calls while satisfying that single row.

## Minimal golden

Start here. Only the fields required to describe the task and judge the result:

```json
{
  "scenario": "Deploy the latest build to staging",
  "expected_outcome": "Deployment confirmed and completed"
}
```

With simulate mode and an explicit first user message:

```json
{
  "scenario": "Deploy the latest build to staging",
  "expected_outcome": "Deployment confirmed and completed",
  "mode": "simulate",
  "max_turns": 1,
  "initial_prompt": "Deploy the latest build to staging"
}
```

## Field guide

| Field | Purpose |
|-------|---------|
| `id` | Stable row identifier for reports and debugging. |
| `scenario` | Natural-language task the simulated user is trying to accomplish. |
| `expected_outcome` | Outcome metrics judge the final conversation against this. |
| `mode` | Conversation mode: `simulate`, `scripted`, `replay`, or `graph`. |
| `max_turns` | Number of scenario-level user turns. Human-input continuations do not count as scenario turns. |
| `max_elicitation_rounds` | Per-user-turn cap for human-in-the-loop responses. |
| `initial_prompt` | Exact first user message. If omitted, the simulator LLM generates one from `scenario`. |
| `user_persona` | Persona the simulated user (and elicitation responder) should follow. |
| `context` | Background facts the simulator can use when answering agent questions. |
| `elicitation_hints` | Optional semantic preferences for deterministic human-input answers. |
| `metadata.elicitation_script` | Deterministic replay steps for tests only. |
| `metadata.sse_checks` | Per-row checks consumed by stream trajectory metrics such as `sse_events_match`. |

## LLM-driven elicitation (no hints)

Add `context`, `user_persona`, and trajectory checks when the agent asks different
questions each run. Put expectations in natural language and let
`conversation.simulator_llm` answer elicitations.

```json
{
  "id": "k8s-connector-create",
  "scenario": "Create a Kubernetes connector named testconnector in the ${HARNESS_PROJECT} project",
  "expected_outcome": "Connector 'testconnector' created with Inherit from Delegate auth, project scope, delegate selector '${HARNESS_DELEGATE_SELECTOR:-hello}'",
  "mode": "simulate",
  "max_turns": 1,
  "max_elicitation_rounds": 6,
  "initial_prompt": "Create a k8s connector named testconnector",
  "user_persona": "Platform engineer who provides reasonable defaults when the agent asks",
  "context": [
    "Org: ${HARNESS_ORG}, Project: ${HARNESS_PROJECT}",
    "Prefer Inherit from Delegate auth and project-level scope",
    "Use delegate selector tag ${HARNESS_DELEGATE_SELECTOR:-hello}"
  ],
  "metadata": {
    "sse_checks": [
      {"event": "elicitation_form", "exists": true},
      {"event": "elicitation_yaml", "exists": true},
      {"event": "entity_mutation", "path": "$.resource_type", "equals": "connector"},
      {"event": "entity_mutation", "path": "$.identifier", "equals": "testconnector"},
      {"event": "assistant_message", "exists": true}
    ]
  },
  "tags": {"domain": "connectors", "resource": "k8s"}
}
```

For Harness SSE agents, pair this golden with:

- `conversation.simulator_llm` (for example `{provider: openai, name: gpt-4o-mini}`)
- `conversation.elicitation_adapter: harness_sse` and `plugins: [examples.harness_sse_elicitation_adapter]`

See `examples/harness-agent.goldens.jsonl` and `examples/harness-agent-conversation.eval.yaml`.
Set `HARNESS_ORG` and `HARNESS_PROJECT` before loading the example so the
dataset can run against any Harness project. Optional values can use
``${VAR:-default}`` syntax in the golden file (for example
``${HARNESS_DELEGATE_SELECTOR:-hello}``).

Tradeoffs: less golden boilerplate and more resilient to changing agent prompts, but
answers are non-deterministic and cost LLM tokens.

## Deterministic elicitation hints

When you need stable CI replay, add `elicitation_hints`. Map live agent question text
to intent keys with matchers, then supply the desired answer per intent:

```json
"elicitation_hints": {
  "intents": {
    "connector_name": "testconnector",
    "delegate_selector": "${HARNESS_DELEGATE_SELECTOR:-hello}"
  },
  "matchers": [
    {"intent": "connector_name", "question_contains": ["name", "identifier", "connector"]},
    {"intent": "delegate_selector", "question_contains": ["delegate", "selector", "tag"]}
  ],
  "yaml": {"default_action": "accept"}
}
```

The `yaml.default_action` key applies to Harness `elicitation_yaml` approvals (typically
`accept`).

## How human-input answers are chosen

At runtime, `HumanInputSimulator` receives the live pending payload from the target
(`pending_human_input` in assistant message metadata) and resolves an answer via:

1. `metadata.elicitation_script` — exact replay for tests
2. `elicitation_hints.matchers` → `intents` — when hints are present
3. Protocol adapter — for `harness_sse`, uses `simulator_llm` when hints are omitted
4. Generic LLM fallback — non-Harness adapters without a protocol adapter

Use `metadata.elicitation_script` only when a test must replay exact fixture inputs.

## Harness SSE agent (full example)

One golden row can drive many HTTP calls. A captured six-turn Harness SSE flow looks
like this:

| Turn | Runtime input | Runtime result |
|------|---------------|----------------|
| `turn1` | User prompt: `Create a k8s connector` | `elicitation_form` for auth and scope |
| `turn2` | `system_event` with `form_values` | `elicitation_free_text` for connector name |
| `turn3` | `system_event` with `free_text: testconnector` | `elicitation_yaml` approval |
| `turn4` | `system_event` accepting YAML | `elicitation_free_text` for delegate selector |
| `turn5` | `system_event` with `free_text: ${HARNESS_DELEGATE_SELECTOR:-hello}` | Updated `elicitation_yaml` approval |
| `turn6` | `system_event` accepting updated YAML | `entity_mutation` and final `assistant_message` |

The eval YAML wires Harness-specific target fields:

- `conversation.elicitation_adapter: harness_sse` with `plugins: [examples.harness_sse_elicitation_adapter]`
- `human_input_events`, `session_fields`, and `human_input_body_template` on `conversational_streaming_http`

Deterministic golden with full `elicitation_hints` for that flow:

```json
{
  "id": "k8s-connector-create",
  "scenario": "Create a Kubernetes connector in the ${HARNESS_PROJECT} project",
  "expected_outcome": "Connector 'testconnector' created with Inherit from Delegate auth, project scope, delegate selector '${HARNESS_DELEGATE_SELECTOR:-hello}'",
  "mode": "simulate",
  "max_turns": 1,
  "max_elicitation_rounds": 6,
  "initial_prompt": "Create a k8s connector",
  "elicitation_hints": {
    "intents": {
      "auth_method": "Inherit from Delegate",
      "scope": "Project (Recommended)",
      "project_id": "${HARNESS_PROJECT}",
      "connector_name": "testconnector",
      "delegate_selector": "${HARNESS_DELEGATE_SELECTOR:-hello}"
    },
    "matchers": [
      {"intent": "auth_method", "question_contains": ["auth", "authentication"]},
      {"intent": "project_id", "question_contains": ["which project", "project in the", "project should"]},
      {"intent": "scope", "question_contains": ["what scope", "at what scope", "scope should"]},
      {"intent": "delegate_selector", "question_contains": ["delegate", "selector", "tag"]},
      {"intent": "connector_name", "question_contains": ["name would you like", "name for the", "identifier"]}
    ],
    "yaml": {"default_action": "accept"}
  }
}
```

Matchers are evaluated in order, and the first match wins. Put specific prompts
such as project selection before broad prompts such as scope, and avoid generic
terms like `project` in a scope matcher when project identifiers are separate
answers.

See `examples/harness-agent-conversation.eval.yaml` for the full target and adapter
configuration. For the hint-free variant shipped in the repo, see
`examples/harness-agent.goldens.jsonl`.
