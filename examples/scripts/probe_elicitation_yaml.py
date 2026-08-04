#!/usr/bin/env python3
"""Probe live Harness SSE for elicitation_yaml on a single conversation turn.

Review (AIPLAT-952): Dev helper used while curating write-flow goldens — confirms
elicitation_yaml / pending_human_input appear in the SSE stream before encoding checks in JSONL.

Usage (env vars required):
  export SSE_ENDPOINT_URL="https://qa.harness.io/gateway/harness-intelligence/api/v2/chat?orgIdentifier=AI_Devops&projectIdentifier=AICHAT"
  export HARNESS_ACCOUNT="..."
  export TOKEN="..."   # PAT or session token
  export HARNESS_ORG="AI_Devops"
  export HARNESS_PROJECT="AICHAT"
  python examples/scripts/probe_elicitation_yaml.py

Or pass a custom prompt:
  python examples/scripts/probe_elicitation_yaml.py --prompt "Create a pipeline ..."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from typing import Any


def _parse_sse(raw: str) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    current_event = ""
    current_data: list[str] = []
    for line in raw.splitlines():
        if line.startswith("event: "):
            if current_event and current_data:
                events.append((current_event, "\n".join(current_data)))
            current_event = line[7:]
            current_data = []
        elif line.startswith("data: "):
            current_data.append(line[6:])
        elif line == "" and current_event and current_data:
            events.append((current_event, "\n".join(current_data)))
            current_event = ""
            current_data = []
    if current_event and current_data:
        events.append((current_event, "\n".join(current_data)))
    return events


def _default_prompt() -> str:
    suffix = os.environ.get("EVAL_RUN_SUFFIX") or str(int(time.time()))
    project = os.environ.get("HARNESS_PROJECT") or "AICHAT"
    return (
        f"Create a pipeline named eval_org_standards_{suffix} in {project} "
        "with one CI stage and a Run step that echoes 'hello world'."
    )


def _build_body(prompt: str) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "mode": "standard",
        "stream": True,
        "context": {"is_v2": True},
        "harness_context": {
            "account_id": os.environ["HARNESS_ACCOUNT"],
            "org_id": os.environ.get("HARNESS_ORG", ""),
            "project_id": os.environ.get("HARNESS_PROJECT", ""),
        },
    }


def _auth_headers() -> dict[str, str]:
    token = os.environ["TOKEN"]
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Harness-account": os.environ["HARNESS_ACCOUNT"],
    }
    # PAT works as Bearer; some deployments also accept x-api-key.
    if token.startswith("pat.") or token.startswith("sat."):
        headers["Authorization"] = f"Bearer {token}"
        headers["x-api-key"] = token
    else:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe SSE stream for elicitation_yaml")
    parser.add_argument("--prompt", default=None, help="User prompt (default: row 1 turn 1)")
    parser.add_argument("--timeout", type=int, default=180, help="HTTP timeout seconds")
    args = parser.parse_args()

    missing = [k for k in ("SSE_ENDPOINT_URL", "HARNESS_ACCOUNT", "TOKEN") if not os.environ.get(k)]
    if missing:
        print(f"Missing env: {', '.join(missing)}", file=sys.stderr)
        return 2

    prompt = args.prompt or _default_prompt()
    body = _build_body(prompt)
    url = os.environ["SSE_ENDPOINT_URL"]
    data = json.dumps(body).encode("utf-8")

    print(f"POST {url}")
    print(f"Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
    print("---")

    req = urllib.request.Request(url, data=data, headers=_auth_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            content_type = resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        print(f"HTTP {exc.code}: {detail}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Request failed: {exc.reason}", file=sys.stderr)
        return 1

    print(f"Content-Type: {content_type}")
    events = _parse_sse(raw)
    counts = Counter(name for name, _ in events)

    print(f"Total SSE events: {len(events)}")
    print("Event counts:")
    for name, count in counts.most_common():
        marker = "  <--" if name == "elicitation_yaml" else ""
        print(f"  {name}: {count}{marker}")

    yaml_events = [(i, data) for i, (name, data) in enumerate(events) if name == "elicitation_yaml"]
    create_tools = []
    for _i, (_name, data) in enumerate(events):
        if _name != "assistant_tool_request":
            continue
        try:
            parsed = json.loads(data)
            for tool in parsed.get("v") or []:
                if "harness_create" in str(tool.get("name", "")):
                    create_tools.append(tool.get("name"))
        except json.JSONDecodeError:
            pass

    print(f"\nharness_create tool requests: {len(create_tools)}")
    print(f"elicitation_yaml events: {len(yaml_events)}")

    if yaml_events:
        _idx, payload = yaml_events[0]
        try:
            parsed = json.loads(payload)
            entity = parsed.get("entity_info") or {}
            print("\nFirst elicitation_yaml summary:")
            print(f"  review_id: {parsed.get('review_id')}")
            print(f"  title: {parsed.get('title')}")
            print(f"  entity_type: {entity.get('entity_type')}")
            print(f"  identifier: {entity.get('identifier')}")
            print(f"  request_action: {entity.get('request_action')}")
            yaml_text = (parsed.get("content") or {}).get("yaml") or ""
            print(f"  yaml_chars: {len(yaml_text)}")
        except json.JSONDecodeError:
            print("  (could not parse elicitation_yaml payload as JSON)")
        return 0

    print("\nNo elicitation_yaml event in stream.")
    print("Last 8 event types:", [name for name, _ in events[-8:]])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
