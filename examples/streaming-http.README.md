# Streaming HTTP Target Example

This example shows how to evaluate an HTTP endpoint that returns Server-Sent Events (SSE) instead of a single buffered JSON response.

`StreamingHttpTarget` is generic and vendor-neutral. It works with any endpoint that responds with `text/event-stream` — chat/agent services, token streams, progress streams, etc. Non-streaming responses fall back to buffered JSON/text parsing, matching `HttpTarget`.

## Files

- `streaming-http.eval.yaml` - runnable eval config using `type: streaming_http`.
- `streaming-http.goldens.jsonl` - one sample input and expected substring.
- `sse_trajectory_metric.py` - example custom metric that prints a summary of captured SSE events.

## Purpose

Use this pattern when the target system streams intermediate events such as:

- assistant messages or token deltas
- tool requests and tool results
- progress/status updates
- usage or metadata events

The streaming target converts the stream into a normal `EvalCase`:

- the selected final output goes into `EvalCase.output`
- captured events (all by default, or a configured subset) go into `EvalCase.metadata["sse_events"]`
- metrics evaluate the resulting `EvalCase`

## Assumed Event Stream

This example assumes a service that streams events shaped like this:

```text
event: tool_call
data: {"name": "search", "args": {"q": "capital of France"}}

event: token
data: The capital

event: token
data:  of France is Paris.

event: final
data: {"answer": "The capital of France is Paris."}

event: usage
data: {"total_tokens": 42}
```

Adapt the event names in the config to whatever your service actually emits.

## Key Config Fields

```yaml
target:
  type: streaming_http
  url: "${SSE_ENDPOINT_URL}"
  body_template:
    prompt: "{{input}}"
    stream: true
  output_event: final
  output_path: $.answer
  capture_events:
    - tool_call
    - usage
```

`body_template` is the JSON request body template. `{{input}}` placeholders are replaced per request with the golden's input. Use `{{input.field}}` (dotted paths, including list indices like `{{input.items.0}}`) to scatter fields of a structured input across the body, and `{{metadata.key}}` for golden metadata. A whole-string placeholder (`"{{input}}"`) keeps the value's native type; an embedded one (`"Hello {{input.name}}"`) is string-interpolated. A placeholder that doesn't resolve raises an error rather than silently sending null.

The same `{{...}}` placeholders work in `headers` values (e.g. `Authorization: "Bearer {{input.token}}"`), so per-golden auth tokens or session ids can be carried in a header. Header values are always string-interpolated; header names are never templated.

`output_event` selects which SSE event becomes the gradeable output. Here the `final` event carries the answer. If you omit `output_event`, the target grades the last JSON `data` payload instead (or, for a plain token stream with no JSON, the concatenated text of `data` lines).

`output_path` extracts the exact field to grade from the selected event payload (`$.answer`).

`capture_events` controls what is preserved in `EvalCase.metadata["sse_events"]` for metrics to evaluate. If omitted, **all** events are captured by default; provide an explicit list to restrict capture to those event names (an empty list `[]` captures nothing). `output_event` is independent — it only selects the primary `EvalCase.output`; every other event still reaches metrics via `sse_events`.

The URL is read from `SSE_ENDPOINT_URL` so no host or credentials are baked into the config — target params support `${VAR}` interpolation (recursively, including nested `headers`/`body_template` values).

## Local Setup

Point the target at your streaming service:

```bash
cd harness-evals
export SSE_ENDPOINT_URL=http://localhost:8000/stream
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

`contains` is the content check. It verifies that `EvalCase.output` contains the golden's `expected` substring. Here `EvalCase.output` is the answer extracted from `final.answer`.

`sse_trajectory` is an observability metric. It passes when at least one SSE event was captured and prints a summary of event counts and tool names in the score reason. It is useful for proving the stream was captured and for debugging the agent trajectory.

## Expected Output

A successful run should show two passing scores:

```text
--- Eval: input='What is the capital of France?' ---
  [PASS] contains: 1.00 (threshold=1.0)
  [PASS] sse_trajectory: 1.00 (threshold=1.0) - captured: tool_call=1, usage=1 | tools: search
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

This example captures and evaluates a single streamed response. For multi-turn
conversations, drive an `agent_fn` through `harness_evals.conversation`
(`ConversationSimulator` / `evaluate_conversation`) and evaluate with the
conversation metrics (`turn_relevancy`, `conversation_completeness`, etc.).
