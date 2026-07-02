# Streaming HTTP Target Example

This example shows how to evaluate an HTTP endpoint that returns Server-Sent Events (SSE) instead of a single buffered JSON response.

It uses `StreamingHttpTarget` with a local Harness AI chat endpoint, but the target itself is generic. It can be used with any endpoint that responds with `text/event-stream`.

## Files

- `streaming-http.eval.yaml` - runnable eval config using `type: streaming_http`.
- `streaming-http.goldens.jsonl` - one sample input and expected substring.
- `sse_trajectory_metric.py` - example custom metric that prints a summary of captured SSE events.

## Purpose

Use this pattern when the target system streams intermediate events such as:

- assistant messages or thoughts
- tool requests and tool results
- progress/status updates
- review or approval events
- usage or metadata events

The streaming target converts the stream into a normal `EvalCase`:

- selected final output goes into `EvalCase.output`
- selected captured events go into `EvalCase.metadata["sse_events"]`
- metrics evaluate the resulting `EvalCase`

## Key Config Fields

```yaml
target:
  type: streaming_http
  url: "${HARNESS_AI_CHAT_URL}?orgIdentifier=${HARNESS_ORG_ID}&projectIdentifier=${HARNESS_PROJECT_ID}"
  body_template:
    prompt: null
    stream: true
    context:
      is_v2: true
    harness_context:
      account_id: "${HARNESS_ACCOUNT_ID}"
      org_id: "${HARNESS_ORG_ID}"
      project_id: "${HARNESS_PROJECT_ID}"
  input_path: $.prompt
  output_event: elicitation_yaml
  output_path: $.content.yaml
  capture_events:
    - stream_metadata
    - assistant_tool_request
    - assistant_tool_result
    - elicitation_yaml
    - model_usage
```

`body_template` is the JSON request body template. The golden input is inserted at `input_path`.

`output_event` selects which SSE event should become the gradeable output. In this example, the chat run pauses on an `elicitation_yaml` review event, so the eval grades the generated YAML rather than a final assistant message.

`output_path` extracts the exact field to grade from the selected event payload.

`capture_events` controls what is preserved in `EvalCase.metadata["sse_events"]`. Only listed event names are captured.

## Local Setup

Start your local streaming service first, then export the required scope variables:

```bash
cd /Users/saranyajena/Documents/harness/harness-evals

export HARNESS_AI_CHAT_URL=http://localhost:8000/chat/unified
export HARNESS_ACCOUNT_ID=<account-id>
export HARNESS_ORG_ID=<org-id>
export HARNESS_PROJECT_ID=<project-id>
```

Install the package if this is a fresh checkout:

```bash
poetry install --all-extras
```

Run the eval:

```bash
PYTHONPATH=. poetry run harness-evals run examples/streaming-http.eval.yaml
```

`PYTHONPATH=.` is needed because `sse_trajectory_metric.py` is an example plugin module, not an installed package module.

Validate config only, without making an HTTP call:

```bash
poetry run harness-evals run examples/streaming-http.eval.yaml --validate
```

## Metrics

This example uses two metrics:

```yaml
metrics:
  - {kind: contains, threshold: 1.0}
  - {kind: sse_trajectory, threshold: 1.0}
```

`contains` is the actual content check. It verifies that `EvalCase.output` contains the golden's `expected` substring. In this example, `EvalCase.output` is the YAML extracted from `elicitation_yaml.content.yaml`.

`sse_trajectory` is an observability metric. It passes when at least one SSE event was captured and prints a summary of event counts, tool names, and review ids in the score reason. It is useful for proving the stream was captured and for debugging the agent trajectory.

## Expected Output

A successful run should show two passing scores:

```text
--- Eval: input='Create a user table with name and age column , database schema...' ---
  [PASS] contains: 1.00 (threshold=1.0)
  [PASS] sse_trajectory: 1.00 (threshold=1.0) - captured: assistant_tool_request=..., elicitation_yaml=1, ...
```

If `sse_trajectory` fails, check:

- the endpoint is running and reachable
- the response content type is `text/event-stream`
- `stream: true` is present in the request body
- `capture_events` includes the event names emitted by the service

If `contains` fails, check:

- `output_event` points to the event containing the final gradeable output
- `output_path` extracts the correct field from that event payload
- the golden `expected` substring matches the generated output

## Scope

This example only captures and evaluates the first streamed response. It does not send follow-up HITL resume events back to the service. Multi-turn HITL resume logic can be layered above `StreamingHttpTarget` in a higher-level harness.
