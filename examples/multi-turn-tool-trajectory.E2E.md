# Multi-turn tool / trajectory — end-to-end verification

Use this checklist to verify that multi-turn tool selection and trajectory
scoring work against a REPLAY transcript.

## Run the demo

```bash
pip install -e "."
python examples/multi_turn_tool_eval.py
```

Expect:

- `pass` case: `tool_correctness` and `tool_argument_match` both PASS (1.0)
- `fail` case: both FAIL

Goldens live at `examples/data/multi_turn_tool_goldens.jsonl`.

## What the demo covers

- `ConversationGolden` with authored `turns` runs in `ConversationMode.REPLAY`,
  so no agent is invoked and the stored transcript is scored as-is.
- Tool calls embedded on assistant turns populate `EvalCase.tool_calls`, with
  `mcp__server__tool` names normalized to their short form so they compare
  against a golden's `expected_tool_calls`.
- `ConversationGolden.expected_tool_calls` also drives `EvalCase.expected_tools`
  for `tool_correctness`. An authored empty list means "no tools should be
  called" and is scored, not skipped.

## Scoring a transcript with no tool calls

`EvalCase.tool_calls` distinguishes an empty trajectory from a missing one:

- `[]` — assistant turns were captured and no tools were requested. Trajectory
  metrics score this normally, so a golden expecting tools correctly FAILs.
- `None` — no assistant behavior was captured, so the trajectory is unknown and
  metrics configured with `skip_when_missing=True` skip the case.

## Regression tests

```bash
pytest tests/conversation/test_simulator.py tests/examples/test_multi_turn_tool_eval.py -v
```
