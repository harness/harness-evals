from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink

_MAX_TOOL_RESULT_CHARS = 500


def _truncate_tool_result_items(items: list) -> list:
    new_items = []
    for item in items:
        if not isinstance(item, dict) or "result" not in item:
            new_items.append(item)
            continue
        result = item.get("result")
        if isinstance(result, str) and len(result) > _MAX_TOOL_RESULT_CHARS:
            item = {
                **item,
                "result": result[:_MAX_TOOL_RESULT_CHARS]
                + f"... [truncated, {len(result)} chars total]",
            }
        new_items.append(item)
    return new_items


def _truncate_tool_results(events: dict) -> dict:
    """Shrink large MCP tool payloads for readable JSONL traces."""
    tool_results = events.get("assistant_tool_result")
    if not isinstance(tool_results, list):
        return events
    compact = dict(events)
    trimmed: list = []
    for payload in tool_results:
        if not isinstance(payload, dict):
            trimmed.append(payload)
            continue
        items = payload.get("v")
        if not isinstance(items, list):
            trimmed.append(payload)
            continue
        trimmed.append({**payload, "v": _truncate_tool_result_items(items)})
    compact["assistant_tool_result"] = trimmed
    return compact


def _truncate_timeline(timeline: list[Any]) -> list[Any]:
    trimmed: list[Any] = []
    for entry in timeline:
        if not isinstance(entry, dict):
            trimmed.append(entry)
            continue
        event = entry.get("event")
        payload = entry.get("payload")
        if event != "assistant_tool_result" or not isinstance(payload, dict):
            trimmed.append(entry)
            continue
        items = payload.get("v")
        if not isinstance(items, list):
            trimmed.append(entry)
            continue
        trimmed.append(
            {
                **entry,
                "payload": {**payload, "v": _truncate_tool_result_items(items)},
            }
        )
    return trimmed


def _eval_case_for_json(
    eval_case: EvalCase,
    *,
    omit_messages: bool = False,
    sse_as_timeline: bool = False,
) -> dict:
    data = eval_case.to_dict()
    if omit_messages:
        data.pop("messages", None)

    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        metadata = dict(metadata)
        if omit_messages:
            if sse_as_timeline:
                timeline = metadata.pop("sse_timeline", None)
                metadata.pop("sse_events", None)
                metadata.pop("sse_event_names", None)
                if isinstance(timeline, list) and timeline:
                    metadata["sse_events"] = _truncate_timeline(timeline)
            elif isinstance(metadata.get("sse_events"), dict):
                metadata["sse_events"] = _truncate_tool_results(metadata["sse_events"])
                metadata.pop("sse_timeline", None)
        else:
            for key in ("sse_events", "sse_timeline", "sse_event_names"):
                metadata.pop(key, None)
        data["metadata"] = metadata

    if not omit_messages:
        messages = data.get("messages")
        if isinstance(messages, list):
            cleaned_messages = []
            for msg in messages:
                if not isinstance(msg, dict):
                    cleaned_messages.append(msg)
                    continue
                msg_meta = msg.get("metadata")
                if isinstance(msg_meta, dict):
                    drop_keys = {"sse_events", "sse_timeline"}
                    if any(key in msg_meta for key in drop_keys):
                        msg = dict(msg)
                        msg["metadata"] = {
                            k: v for k, v in msg_meta.items() if k not in drop_keys
                        }
                cleaned_messages.append(msg)
            data["messages"] = cleaned_messages
    return data


class JsonSink(BaseSink):
    """Append scores as JSON lines to a file. One JSON object per write() call."""

    def __init__(
        self,
        path: str,
        include_eval_case: bool = False,
        omit_messages: bool = False,
        sse_as_timeline: bool = False,
        overwrite: bool = False,
    ) -> None:
        self.path = Path(path)
        self.include_eval_case = include_eval_case
        self.omit_messages = omit_messages
        self.sse_as_timeline = sse_as_timeline
        self.overwrite = overwrite
        self._write_count = 0

    def write(self, scores: list[Score], eval_case: EvalCase) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "input": eval_case.input,
            "scores": [s.to_dict() for s in scores],
        }
        if self.include_eval_case:
            record["eval_case"] = _eval_case_for_json(
                eval_case,
                omit_messages=self.omit_messages,
                sse_as_timeline=self.sse_as_timeline,
            )
        mode = "w" if self.overwrite and self._write_count == 0 else "a"
        with open(self.path, mode) as f:
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()
        self._write_count += 1

    def finalize(self) -> None:
        if self._write_count == 0:
            return
        resolved = self.path.resolve()
        print(
            f"Wrote {self._write_count} JSONL record(s) to {resolved}",
            file=sys.stderr,
        )
