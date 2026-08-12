from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink
from harness_evals.summary import summarize, summary_to_dict

_MAX_TOOL_RESULT_CHARS = 500
_DEBUG_METADATA_KEYS = (
    "golden_id",
    "elicitation_trace",
    "elicitation_rounds",
    "elicitation_error",
    "source_conversation_id",
)


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
                "result": result[:_MAX_TOOL_RESULT_CHARS] + f"... [truncated, {len(result)} chars total]",
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


def _truncate_text(text: str, max_len: int = _MAX_TOOL_RESULT_CHARS) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 35] + f"... [truncated, {len(text)} chars total]"


def _messages_for_json(messages: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Chronological trace for manual debugging; truncate bulky tool payloads."""
    if not messages:
        return []
    cleaned: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            cleaned.append(msg)
            continue
        msg = dict(msg)
        msg_meta = msg.get("metadata")
        if isinstance(msg_meta, dict):
            drop_keys = {"sse_events", "sse_timeline"}
            if any(key in msg_meta for key in drop_keys):
                msg["metadata"] = {k: v for k, v in msg_meta.items() if k not in drop_keys}
        role = msg.get("role")
        content = msg.get("content")
        if role == "tool" and isinstance(content, str) and len(content) > _MAX_TOOL_RESULT_CHARS:
            msg["content"] = _truncate_text(content)
        cleaned.append(msg)
    return cleaned


def _debug_metadata(metadata: dict[str, Any] | None, messages: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Small eval-run context for debugging; golden config (sse_checks) stays in the dataset."""
    debug: dict[str, Any] = {}
    if isinstance(metadata, dict):
        for key in _DEBUG_METADATA_KEYS:
            if key in metadata:
                debug[key] = metadata[key]
    if isinstance(messages, list):
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            msg_meta = msg.get("metadata")
            if not isinstance(msg_meta, dict):
                continue
            conversation_id = msg_meta.get("conversation_id")
            if conversation_id:
                debug["conversation_id"] = conversation_id
                break
    return debug


def _eval_case_for_json(
    eval_case: EvalCase,
    *,
    omit_messages: bool = False,
    sse_as_timeline: bool = False,
) -> dict:
    """Legacy compact eval_case blob (omit_messages / sse_as_timeline modes)."""
    data = eval_case.to_dict()
    data.pop("tool_calls", None)
    data.pop("expected_tool_calls", None)
    if omit_messages:
        data.pop("messages", None)

    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        metadata = dict(metadata)
        metadata.pop("sse_checks", None)
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
        data["metadata"] = metadata
    return data


def _resolve_output_path(path: str, *, unique_per_run: bool) -> Path:
    """Return the output path, optionally suffixing a UTC timestamp before the extension."""
    resolved = Path(path)
    if not unique_per_run:
        return resolved
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = resolved.suffix or ".jsonl"
    stem = resolved.stem if resolved.suffix else resolved.name
    return resolved.parent / f"{stem}-{stamp}{suffix}"


class JsonSink(BaseSink):
    """Append scores as JSON lines to a file. One JSON object per write() call."""

    def __init__(
        self,
        path: str,
        include_eval_case: bool = False,
        omit_messages: bool = False,
        sse_as_timeline: bool = False,
        overwrite: bool = False,
        unique_per_run: bool = False,
        include_summary: bool = True,
    ) -> None:
        self.path = _resolve_output_path(path, unique_per_run=unique_per_run)
        self.include_eval_case = include_eval_case
        self.omit_messages = omit_messages
        self.sse_as_timeline = sse_as_timeline
        self.overwrite = overwrite
        self.include_summary = include_summary
        self._write_count = 0
        self._all_scores: list[list[Score]] = []

    def write(self, scores: list[Score], eval_case: EvalCase) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
            "input": eval_case.input,
            "scores": [s.to_dict() for s in scores],
        }
        if self.include_eval_case:
            if self.omit_messages:
                record["output"] = eval_case.output
                record["eval_case"] = _eval_case_for_json(
                    eval_case,
                    omit_messages=True,
                    sse_as_timeline=self.sse_as_timeline,
                )
            else:
                messages = _messages_for_json(eval_case.to_dict().get("messages"))
                record["output"] = eval_case.output
                record["messages"] = messages
                debug = _debug_metadata(eval_case.metadata, messages)
                if debug:
                    record["debug"] = debug
        mode = "w" if self.overwrite and self._write_count == 0 else "a"
        with open(self.path, mode) as f:
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()
        self._write_count += 1
        if self.include_summary:
            self._all_scores.append(list(scores))

    def finalize(self) -> None:
        if self._write_count == 0:
            return
        if self.include_summary and self._all_scores:
            summary_record = summary_to_dict(summarize(self._all_scores))
            with open(self.path, "a") as f:
                f.write(json.dumps(summary_record, default=str) + "\n")
                f.flush()
            self._all_scores.clear()
        resolved = self.path.resolve()
        print(
            f"Wrote {self._write_count} JSONL record(s) to {resolved}"
            + (" (+ summary)" if self.include_summary else ""),
            file=sys.stderr,
        )
