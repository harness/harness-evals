#!/usr/bin/env python3
"""End-to-end: prod + harness-agent-v3 conversations -> clean Markdown transcripts + CSV.

Single self-contained pipeline (no dependency on the other scripts):

  1. DISCOVER  Fetch traces from Langfuse filtered by prod environments,
               trace name (unified_agent_chat), and agent tag (agent:harness-agent-v3),
               within a look-back window. Collect distinct session_ids.
  2. SAMPLE    Build two non-overlapping datasets:
               - random/: unbiased random conversations
               - module-coverage/: round-robin coverage across module + environment
  3. RESOLVE   For each sampled session, GET /sessions/{id} to fetch ALL turns
               (including HITL/resume turns that may not carry the v3 tag).
  4. HYDRATE   Fetch each trace with observations (fresh by default; --use-cache to reuse).
  5. CLEAN     Reconstruct the real conversation flow from the richest LLM-turn
               message array + the final agent answer. All telemetry
               noise (resourceAttributes, usage/cost details, gRPC spans, etc.) is dropped.
  6. WRITE     Each dataset gets Markdown transcripts, canonical conversation JSON,
               full tool sidecars, and labels.csv for review.

Auth: LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY, or ../../ai-evals/.mcp.json.

Usage:
  python scripts/build_agent_transcripts.py
  python scripts/build_agent_transcripts.py --random-count 15 --module-count 30 --use-cache
  python scripts/build_agent_transcripts.py --random-count 10 --module-count 20 --seed 123
  python scripts/build_agent_transcripts.py --backfill-conversations
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------ configuration

PROD_ENVS = ("prod0", "prod1", "prod2", "prod3", "eu1")
TRACE_NAME = "unified_agent_chat"
AGENT_TAG = "agent:harness-agent-v3"
DEFAULT_WINDOW_DAYS = 30

CACHE_DIR = ROOT / "cache" / "traces"
DISCOVERY_CACHE = ROOT / "cache" / "discovered-sessions.json"
LEGACY_OUT_DIR = ROOT / "transcripts"
RANDOM_DIR = ROOT / "random"
MODULE_COVERAGE_DIR = ROOT / "module-coverage"

TOOL_RESP_LIMIT = 500
CANONICAL_TOOL_RESP_LIMIT = 2_000
CONVERSATION_SCHEMA_VERSION = "1.0"
SYSTEM_PROMPT_MARKER = "You are Harness AI"
CONTEXT_MARKERS = ("## Current Harness Context", "## Additional Context", "## User Context")
FEEDBACK_PREFIXES = ('{"reasons"',)


def load_langfuse_config() -> tuple[str, str, str]:
    host = os.environ.get("LANGFUSE_HOST", "https://langfuse-prod.harness.io").rstrip("/")
    public = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret = os.environ.get("LANGFUSE_SECRET_KEY", "")
    if public and secret:
        return host, public, secret
    mcp_json = ROOT.parents[2] / "ai-evals" / ".mcp.json"
    if mcp_json.is_file():
        cfg = json.loads(mcp_json.read_text())
        env = cfg.get("mcpServers", {}).get("langfuse", {}).get("env", {})
        return (
            env.get("LANGFUSE_HOST", host).rstrip("/"),
            env.get("LANGFUSE_PUBLIC_KEY", ""),
            env.get("LANGFUSE_SECRET_KEY", ""),
        )
    raise SystemExit(
        "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY, or ensure ai-evals/.mcp.json exists."
    )


# ------------------------------------------------------------------ Langfuse client


class LangfuseClient:
    def __init__(self, host: str, public_key: str, secret_key: str) -> None:
        self.host = host.rstrip("/")
        token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        self.headers = {"Authorization": f"Basic {token}"}

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = self.host + path
        if params:
            # doseq=True so list values (e.g. environment, tags) repeat correctly
            url += "?" + urllib.parse.urlencode(params, doseq=True)
        req = urllib.request.Request(url, headers=self.headers)
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as err:
                last_err = err
                if err.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))
        raise last_err  # type: ignore[misc]

    def list_traces(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._get("/api/public/traces", params)

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self._get(f"/api/public/sessions/{urllib.parse.quote(session_id, safe='')}")

    def fetch_trace(self, trace_id: str) -> dict[str, Any]:
        trace = self._get(f"/api/public/traces/{trace_id}")
        observations = list(trace.get("observations") or [])
        if not observations:
            observations = self._observations_paginated(trace_id)
        observations.sort(key=lambda o: o.get("startTime") or o.get("start_time") or "")
        trace["observations"] = observations
        return normalize_trace(trace)

    def _observations_paginated(self, trace_id: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        page = 1
        while True:
            payload = self._get(
                "/api/public/observations",
                {"traceId": trace_id, "page": page, "limit": 100},
            )
            out.extend(payload.get("data") or [])
            meta = payload.get("meta") or {}
            if page >= (meta.get("totalPages") or 1):
                break
            page += 1
        return out


def normalize_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Normalize Langfuse REST (camelCase) trace to the snake_case shape used here."""
    if trace.get("session_id") or not trace.get("sessionId"):
        return trace
    return {
        "id": trace.get("id"),
        "timestamp": trace.get("timestamp"),
        "name": trace.get("name"),
        "session_id": trace.get("sessionId"),
        "user_id": trace.get("userId"),
        "metadata": trace.get("metadata") or {},
        "tags": trace.get("tags") or [],
        "environment": trace.get("environment"),
        "total_cost": trace.get("totalCost") or trace.get("total_cost"),
        "observations": trace.get("observations") or [],
    }


# ------------------------------------------------------------------ discovery + scope


def get_env(trace: dict[str, Any]) -> str:
    env = (trace.get("environment") or "").strip()
    if env:
        return env
    for tag in trace.get("tags") or []:
        if isinstance(tag, str) and tag.startswith("environment:"):
            return tag.split(":", 1)[1]
    return "unknown"


def get_module(trace: dict[str, Any]) -> str:
    md = trace.get("metadata") or {}
    if isinstance(md, dict):
        for key in ("product", "agent.module", "harness.module"):
            val = md.get(key)
            if val:
                return str(val).lower()
    for tag in trace.get("tags") or []:
        if isinstance(tag, str) and tag.startswith("product:"):
            return tag.split(":", 1)[1].lower()
    return "none"


def scope_fields(trace: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    md = trace.get("metadata") or {}
    if not isinstance(md, dict):
        return None, None, None
    account = md.get("account_id") or md.get("harness.account.id")
    org = md.get("org_id") or md.get("harness.org.id")
    project = md.get("project_id") or md.get("harness.project.id")
    return (
        str(account) if account else None,
        str(org) if org else None,
        str(project) if project else None,
    )


def discover_sessions(
    client: LangfuseClient,
    *,
    since: str,
    envs: tuple[str, ...],
    tag: str,
    name: str,
    max_pages: int,
    page_size: int,
) -> dict[str, dict[str, Any]]:
    """Return {session_id: representative_trace} for prod + agent-tag traces."""
    sessions: dict[str, dict[str, Any]] = {}
    for env in envs:
        for page in range(1, max_pages + 1):
            params = {
                "environment": env,
                "name": name,
                "tags": tag,
                "fromTimestamp": since,
                "page": page,
                "limit": page_size,
            }
            payload = client.list_traces(params)
            batch = payload.get("data") or []
            if not batch:
                break
            for trace in batch:
                sid = trace.get("sessionId") or trace.get("session_id")
                if not sid:
                    continue
                sessions.setdefault(str(sid), normalize_trace(dict(trace)))
            meta = payload.get("meta") or {}
            if page >= (meta.get("totalPages") or page):
                break
            time.sleep(0.05)
        print(f"  {env}: {len(sessions)} sessions cumulative", flush=True)
    return sessions


# ------------------------------------------------------------------ parsing (clean)


def _obs_messages(obs: dict[str, Any]) -> list[dict[str, Any]]:
    inp = obs.get("input")
    if not isinstance(inp, dict):
        return []
    msgs = inp.get("messages")
    if isinstance(msgs, str):
        try:
            msgs = json.loads(msgs)
        except json.JSONDecodeError:
            return []
    return msgs if isinstance(msgs, list) else []


def _output_assistant_msg(obs: dict[str, Any]) -> dict[str, Any] | None:
    """The assistant message produced by this LLM call (not present in its own input)."""
    out = obs.get("output")
    if isinstance(out, dict) and out.get("role") == "assistant" and out.get("content") is not None:
        return {"role": "assistant", "content": out["content"]}
    return None


def richest_observation(trace: dict[str, Any]) -> dict[str, Any] | None:
    """The LLM observation with the longest cumulative message array.

    harness-agent-v3 traces store the full Anthropic message array under
    ``llm_turn_*.input.messages`` (older builds used ``provider_call.input.messages``);
    the richest (last) turn holds every prior user/assistant/tool message.
    """
    best: dict[str, Any] | None = None
    best_count = -1
    for obs in trace.get("observations") or []:
        name = obs.get("name") or ""
        if not (name.startswith("llm_turn_") or name.startswith("provider_call")):
            continue
        msgs = _obs_messages(obs)
        if not msgs:
            continue
        inp = obs.get("input") or {}
        count = inp.get("message_count") or len(msgs)
        if count >= best_count:
            best_count = count
            best = obs
    return best


def richest_messages(trace: dict[str, Any]) -> list[dict[str, Any]]:
    obs = richest_observation(trace)
    if obs is None:
        return []
    msgs = list(_obs_messages(obs))
    final_msg = _output_assistant_msg(obs)  # the LLM's own reply isn't in its input
    if final_msg is not None:
        msgs.append(final_msg)
    return msgs


def final_answer(trace: dict[str, Any]) -> str:
    # Preferred: explicit final answer recorded by the agent.
    candidates: list[tuple[str, str]] = []
    for obs in trace.get("observations") or []:
        if (obs.get("name") or "") != "process_chat_response":
            continue
        out = obs.get("output") or {}
        data = out.get("output_data") if isinstance(out, dict) else None
        if isinstance(data, str) and data.strip():
            ts = obs.get("start_time") or obs.get("startTime") or ""
            candidates.append((ts, data.strip()))
    if candidates:
        candidates.sort(key=lambda c: c[0])
        return candidates[-1][1]
    # Fallback: the last LLM turn's assistant text.
    obs = richest_observation(trace)
    if obs is not None:
        msg = _output_assistant_msg(obs)
        if msg is not None:
            return _text_of(msg.get("content"))
    return ""


def _content_blocks(content: Any) -> list[dict[str, Any]]:
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    parts = [b.get("text", "") for b in _content_blocks(content) if b.get("type") == "text"]
    return "\n".join(p for p in parts if p and p.strip()).strip()


def _is_tool_result_msg(content: Any) -> bool:
    return any(b.get("type") == "tool_result" for b in _content_blocks(content))


def _is_plain_user_text(msg: dict[str, Any]) -> bool:
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    blocks = _content_blocks(content)
    return any(b.get("type") == "text" for b in blocks) and not _is_tool_result_msg(content)


def _is_system(text: str) -> bool:
    return SYSTEM_PROMPT_MARKER in text[:400]


def clean_user_text(text: str) -> str:
    text = (text or "").strip()
    cut = len(text)
    for marker in CONTEXT_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    brace = text.find("\n\n{")
    if brace != -1 and ("currentUrl" in text[brace:] or "current_url" in text[brace:]):
        cut = min(cut, brace)
    return text[:cut].strip()


def build_turn(trace: dict[str, Any], seq: int) -> dict[str, Any]:
    msgs = richest_messages(trace)

    last_user_idx = -1
    for i, m in enumerate(msgs):
        if _is_plain_user_text(m) and not _is_system(_text_of(m.get("content"))):
            last_user_idx = i
    tail = msgs[last_user_idx:] if last_user_idx >= 0 else msgs

    results: dict[str, str] = {}
    for m in tail:
        if m.get("role") != "user":
            continue
        for b in _content_blocks(m.get("content")):
            if b.get("type") == "tool_result":
                content = b.get("content")
                if isinstance(content, list):
                    content = "\n".join(
                        c.get("text", "") if isinstance(c, dict) else str(c) for c in content
                    )
                results[b.get("tool_use_id", "")] = (
                    content if isinstance(content, str) else json.dumps(content)
                )

    user_text = clean_user_text(_text_of(tail[0].get("content"))) if tail else ""

    events: list[dict[str, Any]] = []
    for m in tail[1:] if tail else []:
        if m.get("role") != "assistant":
            continue
        for b in _content_blocks(m.get("content")):
            btype = b.get("type")
            if btype == "text":
                txt = (b.get("text") or "").strip()
                if not txt or txt == "{}" or txt.startswith(FEEDBACK_PREFIXES):
                    continue
                events.append({"kind": "assistant_text", "text": txt})
            elif btype == "tool_use":
                tid = b.get("id", "")
                events.append(
                    {
                        "kind": "tool",
                        "id": tid,
                        "name": b.get("name", "?"),
                        "request": b.get("input", {}),
                        "response": results.get(tid, ""),
                    }
                )

    final = final_answer(trace)
    if events and events[-1]["kind"] == "assistant_text" and events[-1]["text"] == final:
        final = ""

    return {
        "sequence": seq,
        "trace_id": trace.get("id"),
        "user_text": user_text,
        "events": events,
        "final": final,
        "cost_usd": trace.get("total_cost") or trace.get("totalCost") or 0.0,
    }


# ------------------------------------------------------------------ rendering


def _truncate(text: str, limit: int = TOOL_RESP_LIMIT) -> tuple[str, bool]:
    text = text or ""
    return (text, False) if len(text) <= limit else (text[:limit] + " …", True)


def _compact_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))
    except (TypeError, ValueError):
        return str(obj)


def render_markdown(meta: dict[str, Any], turns: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    total_cost = sum(t["cost_usd"] for t in turns)
    tool_count = sum(1 for t in turns for e in t["events"] if e["kind"] == "tool")
    scope = "/".join(x for x in (meta.get("org_id"), meta.get("project_id")) if x)

    lines += [
        f"# Conversation `{meta['conversation_id']}`",
        "",
        f"- **env:** {meta.get('env')}  ·  **module:** {meta.get('module')}  ·  "
        f"**scope:** {scope or 'account'}",
        f"- **turns:** {len(turns)}  ·  **tool calls:** {tool_count}  ·  "
        f"**total cost:** ${total_cost:.4f}",
        f"- **first seen:** {meta.get('first_timestamp', '')}",
        "",
        "> System prompt on every turn: _[constant system prompt]_ (omitted for readability)",
        "",
        "---",
        "",
    ]

    for turn in turns:
        n = turn["sequence"]
        lines += [f"## Turn {n} · user", "", turn["user_text"] or "_[no user text]_", ""]
        for e in turn["events"]:
            if e["kind"] == "assistant_text":
                lines += [f"**Turn {n} · assistant**", "", e["text"], ""]
            else:
                resp, truncated = _truncate(e["response"])
                note = "  _(full response in `.tools.json`)_" if truncated else ""
                lines += [
                    f"**Turn {n} · assistant → tool `{e['name']}`**",
                    "",
                    "_request_",
                    "```json",
                    _compact_json(e["request"]),
                    "```",
                    f"_response_{note}",
                    "```json",
                    resp or "_[empty]_",
                    "```",
                    "",
                ]
        if turn["final"]:
            lines += [f"## Turn {n} · assistant (final)", "", turn["final"], ""]
        lines += ["---", ""]

    return "\n".join(lines).rstrip() + "\n"


def build_tool_sidecar(conversation_id: str, turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for turn in turns:
        for e in turn["events"]:
            if e["kind"] == "tool":
                rows.append(
                    {
                        "conversation_id": conversation_id,
                        "turn": turn["sequence"],
                        "tool_use_id": e["id"],
                        "name": e["name"],
                        "request": e["request"],
                        "response": e["response"],
                    }
                )
    return rows


def _bounded_tool_response(text: str, limit: int = CANONICAL_TOOL_RESP_LIMIT) -> tuple[str, bool]:
    text = text or ""
    if len(text) <= limit:
        return text, False
    return (
        text[:limit]
        + "\n\n[tool response truncated; complete payload is available in the .tools.json sidecar]",
        True,
    )


def build_canonical_conversation(
    meta: dict[str, Any],
    turns: list[dict[str, Any]],
    *,
    sample_type: str,
    transcript_file: str,
    tools_file: str,
) -> dict[str, Any]:
    """Build the reusable machine-readable representation of one session."""
    messages: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    truncated_tool_use_ids: list[str] = []

    for turn in turns:
        if turn["user_text"]:
            messages.append(
                {
                    "role": "user",
                    "content": turn["user_text"],
                    "metadata": {"turn": turn["sequence"], "trace_id": turn["trace_id"]},
                }
            )

        for event in turn["events"]:
            if event["kind"] == "assistant_text":
                messages.append(
                    {
                        "role": "assistant",
                        "content": event["text"],
                        "metadata": {"turn": turn["sequence"], "trace_id": turn["trace_id"]},
                    }
                )
                continue

            response, truncated = _bounded_tool_response(event["response"])
            tool_call = {
                "name": event["name"],
                "input": event["request"],
                "output": response,
            }
            tool_calls.append(tool_call)
            message_metadata = {
                "turn": turn["sequence"],
                "trace_id": turn["trace_id"],
                "tool_use_id": event["id"],
                "tool_response_truncated": truncated,
            }
            if truncated:
                truncated_tool_use_ids.append(event["id"])
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Called tool `{event['name']}`.",
                    "tool_calls": [tool_call],
                    "metadata": message_metadata,
                }
            )

        if turn["final"]:
            messages.append(
                {
                    "role": "assistant",
                    "content": turn["final"],
                    "metadata": {"turn": turn["sequence"], "trace_id": turn["trace_id"], "final": True},
                }
            )

    first_user = next((m["content"] for m in messages if m["role"] == "user" and m.get("content")), "")
    final_assistant = next(
        (m["content"] for m in reversed(messages) if m["role"] == "assistant" and m.get("content")),
        "",
    )
    trace_ids = [str(turn["trace_id"]) for turn in turns if turn.get("trace_id")]

    return {
        "schema_version": CONVERSATION_SCHEMA_VERSION,
        "conversation_id": meta["conversation_id"],
        "sample_type": sample_type,
        "input": first_user,
        "output": final_assistant,
        "messages": messages,
        "tool_calls": tool_calls,
        "metadata": {
            "environment": meta.get("env"),
            "module": meta.get("module"),
            "org_id": meta.get("org_id"),
            "project_id": meta.get("project_id"),
            "first_timestamp": meta.get("first_timestamp"),
            "trace_ids": trace_ids,
            "num_turns": len(turns),
            "num_messages": len(messages),
            "num_tool_calls": len(tool_calls),
            "total_cost_usd": round(sum(turn["cost_usd"] for turn in turns), 6),
            "canonical_tool_response_limit": CANONICAL_TOOL_RESP_LIMIT,
            "truncated_tool_use_ids": truncated_tool_use_ids,
            "transcript_file": transcript_file,
            "tools_file": tools_file,
        },
    }


# ------------------------------------------------------------------ hydration cache


def get_trace(
    client: LangfuseClient,
    trace_id: str,
    *,
    use_cache: bool,
    save_cache: bool,
) -> dict[str, Any]:
    cache_path = CACHE_DIR / f"{trace_id}.json"
    if use_cache and cache_path.is_file():
        return json.loads(cache_path.read_text())
    trace = client.fetch_trace(trace_id)
    if save_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n")
    return trace


def short_id(session_id: str) -> str:
    return session_id.split("-", 1)[0][:8]


def slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in (text or "none").lower()).strip("-") or "none"


def sample_random(
    sessions: dict[str, dict[str, Any]],
    count: int,
    *,
    rng: random.Random,
) -> list[str]:
    session_ids = sorted(sessions)
    return rng.sample(session_ids, min(count, len(session_ids)))


def sample_module_coverage(
    sessions: dict[str, dict[str, Any]],
    count: int,
    *,
    rng: random.Random,
    exclude: set[str],
) -> list[str]:
    """Round-robin across modules, then environments within each module."""
    buckets: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for sid, trace in sessions.items():
        if sid in exclude:
            continue
        buckets[get_module(trace)][get_env(trace)].append(sid)

    for env_buckets in buckets.values():
        for session_ids in env_buckets.values():
            rng.shuffle(session_ids)

    modules = sorted(buckets)
    rng.shuffle(modules)
    env_order: dict[str, list[str]] = {}
    env_index: dict[str, int] = {}
    for module in modules:
        envs = sorted(buckets[module])
        rng.shuffle(envs)
        env_order[module] = envs
        env_index[module] = 0

    selected: list[str] = []
    while len(selected) < count:
        added = False
        for module in modules:
            envs = env_order[module]
            for _ in range(len(envs)):
                index = env_index[module] % len(envs)
                env_index[module] += 1
                bucket = buckets[module][envs[index]]
                if bucket:
                    selected.append(bucket.pop())
                    added = True
                    break
            if len(selected) >= count:
                break
        if not added:
            break
    return selected


def selection_counts(
    selected: list[str],
    sessions: dict[str, dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for sid in selected:
        trace = sessions[sid]
        counts[f"{get_env(trace)}|{get_module(trace)}"] += 1
    return dict(sorted(counts.items()))


def clear_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for old in path.glob("*"):
        if old.is_file():
            old.unlink()


def clear_legacy_root_outputs() -> None:
    """Remove files produced by the earlier single-directory layout."""
    if not LEGACY_OUT_DIR.is_dir():
        return
    for pattern in ("*.md", "*.tools.json", "*.conversation.json", "labels.csv"):
        for old in LEGACY_OUT_DIR.glob(pattern):
            old.unlink()
    try:
        LEGACY_OUT_DIR.rmdir()
    except OSError:
        pass


def write_labels_csv(rows: list[dict[str, Any]], out_dir: Path) -> Path | None:
    if not rows:
        return None
    csv_path = out_dir / "labels.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["transcript_file"]))
    return csv_path


def load_review_fields(out_dir: Path) -> dict[str, dict[str, str]]:
    """Keep human labels and notes when regenerating the same sample."""
    labels_path = out_dir / "labels.csv"
    if not labels_path.is_file():
        return {}
    with labels_path.open(newline="") as handle:
        return {
            row["conversation_id"]: {
                "label": row.get("label") or "",
                "notes": row.get("notes") or "",
            }
            for row in csv.DictReader(handle)
            if row.get("conversation_id")
        }


def _load_cached_trace(trace_id: str) -> dict[str, Any] | None:
    cache_path = CACHE_DIR / f"{trace_id}.json"
    if not cache_path.is_file():
        return None
    return json.loads(cache_path.read_text())


def backfill_canonical_conversations(
    out_dir: Path,
    *,
    dataset_name: str,
    client: LangfuseClient | None = None,
) -> tuple[int, list[str]]:
    """Create canonical files for an existing sample from labels.csv + trace cache."""
    labels_path = out_dir / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing sample index: {labels_path}")

    created = 0
    missing_trace_ids: list[str] = []
    with labels_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    for row in rows:
        trace_ids = [trace_id for trace_id in (row.get("trace_ids") or "").split(";") if trace_id]
        traces: list[dict[str, Any]] = []
        row_missing: list[str] = []
        for trace_id in trace_ids:
            trace = _load_cached_trace(trace_id)
            if trace is None and client is not None:
                trace = client.fetch_trace(trace_id)
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                (CACHE_DIR / f"{trace_id}.json").write_text(
                    json.dumps(trace, indent=2, ensure_ascii=False) + "\n"
                )
            if trace is None:
                row_missing.append(trace_id)
            else:
                traces.append(trace)
        if row_missing:
            missing_trace_ids.extend(row_missing)
            continue

        traces.sort(key=lambda trace: trace.get("timestamp") or "")
        turns = [build_turn(trace, sequence) for sequence, trace in enumerate(traces, start=1)]
        transcript_file = row["transcript_file"]
        stem = Path(transcript_file).stem
        tools_file = f"{stem}.tools.json"
        meta = {
            "conversation_id": row["conversation_id"],
            "env": row.get("env"),
            "module": row.get("module"),
            "org_id": row.get("org_id") or None,
            "project_id": row.get("project_id") or None,
            "first_timestamp": row.get("first_timestamp"),
        }
        canonical = build_canonical_conversation(
            meta,
            turns,
            sample_type=dataset_name,
            transcript_file=transcript_file,
            tools_file=tools_file,
        )
        expected_turns = int(row.get("num_turns") or len(trace_ids))
        expected_tools = int(row.get("num_tool_calls") or canonical["metadata"]["num_tool_calls"])
        if canonical["conversation_id"] != row["conversation_id"]:
            raise ValueError(f"Conversation ID mismatch for {transcript_file}")
        if canonical["metadata"]["num_turns"] != expected_turns:
            raise ValueError(
                f"Turn count mismatch for {row['conversation_id']}: "
                f"{canonical['metadata']['num_turns']} != {expected_turns}"
            )
        if canonical["metadata"]["num_tool_calls"] != expected_tools:
            raise ValueError(
                f"Tool count mismatch for {row['conversation_id']}: "
                f"{canonical['metadata']['num_tool_calls']} != {expected_tools}"
            )
        (out_dir / f"{stem}.conversation.json").write_text(
            json.dumps(canonical, indent=2, ensure_ascii=False) + "\n"
        )
        row["conversation_file"] = f"{stem}.conversation.json"
        row["tools_file"] = tools_file
        created += 1

    if "conversation_file" not in fieldnames:
        insert_at = fieldnames.index("transcript_file") + 1 if "transcript_file" in fieldnames else len(fieldnames)
        fieldnames.insert(insert_at, "conversation_file")
    if "tools_file" not in fieldnames:
        insert_at = fieldnames.index("conversation_file") + 1
        fieldnames.insert(insert_at, "tools_file")
    with labels_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return created, sorted(set(missing_trace_ids))


def build_dataset(
    client: LangfuseClient,
    sessions: dict[str, dict[str, Any]],
    selected: list[str],
    *,
    dataset_name: str,
    out_dir: Path,
    use_cache: bool,
    save_cache: bool,
) -> list[dict[str, Any]]:
    review_fields = load_review_fields(out_dir)
    clear_output_dir(out_dir)
    rows: list[dict[str, Any]] = []
    for i, sid in enumerate(sorted(selected), start=1):
        rep = sessions[sid]
        env, module = get_env(rep), get_module(rep)
        _account_id, org_id, project_id = scope_fields(rep)

        try:
            session = client.get_session(sid)
        except urllib.error.HTTPError as err:
            print(f"  ! session {sid} fetch failed ({err.code}); skipping", flush=True)
            continue
        traces = session.get("traces") or []
        traces.sort(key=lambda trace: trace.get("timestamp") or "")
        if not traces:
            continue

        turns: list[dict[str, Any]] = []
        for seq, trace_summary in enumerate(traces, start=1):
            trace_id = trace_summary.get("id")
            if not trace_id:
                continue
            trace = get_trace(
                client,
                trace_id,
                use_cache=use_cache,
                save_cache=save_cache,
            )
            if not org_id:
                _account_id, org_id, project_id = scope_fields(trace)
            if module == "none":
                module = get_module(trace)
            turns.append(build_turn(trace, seq))

        if not turns:
            continue

        meta = {
            "conversation_id": sid,
            "env": env,
            "module": module,
            "org_id": org_id,
            "project_id": project_id,
            "first_timestamp": traces[0].get("timestamp"),
        }
        filename = f"{i:02d}-{slug(env)}-{slug(module)}-{short_id(sid)}"
        transcript_file = f"{filename}.md"
        tools_file = f"{filename}.tools.json"
        (out_dir / transcript_file).write_text(render_markdown(meta, turns))
        (out_dir / tools_file).write_text(
            json.dumps(build_tool_sidecar(sid, turns), indent=2, ensure_ascii=False) + "\n"
        )
        canonical = build_canonical_conversation(
            meta,
            turns,
            sample_type=dataset_name,
            transcript_file=transcript_file,
            tools_file=tools_file,
        )
        (out_dir / f"{filename}.conversation.json").write_text(
            json.dumps(canonical, indent=2, ensure_ascii=False) + "\n"
        )

        tool_count = sum(1 for turn in turns for event in turn["events"] if event["kind"] == "tool")
        rows.append(
            {
                "sample_type": dataset_name,
                "conversation_id": sid,
                "env": env,
                "module": module,
                "org_id": org_id or "",
                "project_id": project_id or "",
                "num_turns": len(turns),
                "num_tool_calls": tool_count,
                "total_cost_usd": round(sum(turn["cost_usd"] for turn in turns), 6),
                "first_timestamp": meta["first_timestamp"],
                "trace_ids": ";".join(str(turn["trace_id"]) for turn in turns),
                "transcript_file": transcript_file,
                "conversation_file": f"{filename}.conversation.json",
                "tools_file": tools_file,
                "label": review_fields.get(sid, {}).get("label", ""),
                "notes": review_fields.get(sid, {}).get("notes", ""),
            }
        )
        print(
            f"  [{i}/{len(selected)}] {filename}  ({len(turns)} turns, {tool_count} tools)",
            flush=True,
        )

    write_labels_csv(rows, out_dir)
    return rows


# ------------------------------------------------------------------ main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-count", type=int, default=15, help="Size of random/ sample")
    parser.add_argument(
        "--module-count",
        type=int,
        default=30,
        help="Size of module-coverage/ sample",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--envs", default=",".join(PROD_ENVS), help="Comma-separated prod envs")
    parser.add_argument("--tag", default=AGENT_TAG, help="Agent tag filter")
    parser.add_argument("--name", default=TRACE_NAME, help="Trace name filter")
    parser.add_argument("--max-pages", type=int, default=12, help="Max trace pages per env")
    parser.add_argument("--page-size", type=int, default=100, help="Traces per page (max 100)")
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse hydrated traces from cache/ if present (default: fresh fetch)",
    )
    parser.add_argument(
        "--no-save-cache",
        action="store_true",
        help="Do not write fetched traces to the cache dir",
    )
    parser.add_argument(
        "--backfill-conversations",
        action="store_true",
        help="Write .conversation.json for existing samples from labels.csv + cached traces",
    )
    parser.add_argument(
        "--fetch-missing",
        action="store_true",
        help="With --backfill-conversations, fetch only traces missing from the cache",
    )
    args = parser.parse_args()

    if args.fetch_missing and not args.backfill_conversations:
        parser.error("--fetch-missing requires --backfill-conversations")

    if args.backfill_conversations:
        client: LangfuseClient | None = None
        if args.fetch_missing:
            host, public, secret = load_langfuse_config()
            client = LangfuseClient(host, public, secret)

        total_created = 0
        all_missing: list[str] = []
        for out_dir, dataset_name in (
            (RANDOM_DIR, "random"),
            (MODULE_COVERAGE_DIR, "module-coverage"),
        ):
            created, missing = backfill_canonical_conversations(
                out_dir,
                dataset_name=dataset_name,
                client=client,
            )
            total_created += created
            all_missing.extend(missing)
            print(f"Backfilled {created} canonical conversations in {out_dir}", flush=True)
        if all_missing:
            print(
                f"Missing {len(set(all_missing))} cached traces. "
                "Rerun with --backfill-conversations --fetch-missing.",
                file=sys.stderr,
            )
            return 1
        print(f"Backfilled {total_created} canonical conversations.", flush=True)
        return 0

    host, public, secret = load_langfuse_config()
    client = LangfuseClient(host, public, secret)
    envs = tuple(e.strip() for e in args.envs.split(",") if e.strip())
    since = (
        (datetime.now(timezone.utc) - timedelta(days=args.window_days))
        .isoformat()
        .replace("+00:00", "Z")
    )

    # 1. DISCOVER
    if args.use_cache and DISCOVERY_CACHE.is_file():
        sessions = json.loads(DISCOVERY_CACHE.read_text())
        print(
            f"Loaded {len(sessions)} discovered conversations from {DISCOVERY_CACHE}.",
            flush=True,
        )
    else:
        print(f"Discovering prod + {args.tag} traces since {since} …", flush=True)
        sessions = discover_sessions(
            client,
            since=since,
            envs=envs,
            tag=args.tag,
            name=args.name,
            max_pages=args.max_pages,
            page_size=min(args.page_size, 100),
        )
        DISCOVERY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        DISCOVERY_CACHE.write_text(json.dumps(sessions, ensure_ascii=False) + "\n")
    if not sessions:
        print("No matching sessions found.", flush=True)
        return 1
    print(f"Found {len(sessions)} distinct conversations.", flush=True)

    # 2. SAMPLE: unbiased random + non-overlapping balanced module coverage.
    random_rng = random.Random(args.seed)
    module_rng = random.Random(args.seed + 1)
    random_selected = sample_random(sessions, args.random_count, rng=random_rng)
    module_selected = sample_module_coverage(
        sessions,
        args.module_count,
        rng=module_rng,
        exclude=set(random_selected),
    )
    print(
        f"Random sample: {len(random_selected)}; "
        f"module-coverage sample: {len(module_selected)} (seed={args.seed}).",
        flush=True,
    )
    print("Random distribution:")
    print(json.dumps(selection_counts(random_selected, sessions), indent=2))
    print("Module-coverage distribution:")
    print(json.dumps(selection_counts(module_selected, sessions), indent=2))

    # 3-6. RESOLVE -> HYDRATE -> CLEAN -> WRITE
    clear_legacy_root_outputs()
    print(f"\nBuilding random sample in {RANDOM_DIR} …", flush=True)
    random_rows = build_dataset(
        client,
        sessions,
        random_selected,
        dataset_name="random",
        out_dir=RANDOM_DIR,
        use_cache=args.use_cache,
        save_cache=not args.no_save_cache,
    )
    print(f"\nBuilding module coverage in {MODULE_COVERAGE_DIR} …", flush=True)
    module_rows = build_dataset(
        client,
        sessions,
        module_selected,
        dataset_name="module-coverage",
        out_dir=MODULE_COVERAGE_DIR,
        use_cache=args.use_cache,
        save_cache=not args.no_save_cache,
    )

    if not random_rows and not module_rows:
        print("No transcripts built.", flush=True)
        return 1

    print(f"\nBuilt {len(random_rows)} random transcripts in {RANDOM_DIR}")
    print(f"Built {len(module_rows)} module-coverage transcripts in {MODULE_COVERAGE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
