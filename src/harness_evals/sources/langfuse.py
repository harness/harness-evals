"""Langfuse source — hydrate EvalCases from Langfuse traces.

Requires: pip install harness-evals[langfuse]
"""

from __future__ import annotations

try:
    from langfuse import Langfuse
except ImportError as _err:
    raise ImportError(
        "LangfuseSource requires the langfuse package. Install with: pip install harness-evals[langfuse]"
    ) from _err

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.types import Message, ToolCall


class LangfuseSource:
    """Hydrate EvalCases from Langfuse traces.

    Uses the Langfuse SDK to fetch trace + observation data and maps it
    to an ``EvalCase`` with typed ``messages``, ``tool_calls``, and
    operational fields.

    Sets ``metadata["langfuse_trace_id"]`` so that a ``LangfuseSink``
    can write scores back to the same trace.

    Example::

        from langfuse import Langfuse
        from harness_evals.sources.langfuse import LangfuseSource
        from harness_evals import evaluate
        from harness_evals.metrics import LatencyMetric

        source = LangfuseSource(Langfuse())
        ec = source.from_trace("trace-abc-123")
        scores = evaluate(ec, metrics=[LatencyMetric(max_ms=3000)])
    """

    def __init__(self, client: Langfuse) -> None:
        self._client = client

    def from_trace(self, trace_id: str) -> EvalCase:
        """Fetch a single Langfuse trace and convert it to an EvalCase.

        Maps:
          - trace.input / trace.output -> input / output
          - generation observations -> messages (with tool_calls embedded)
          - tool observations -> tool_calls
          - aggregated usage -> token_count, cost_usd
          - trace duration -> latency_ms
          - trace.tags -> tags
          - trace.metadata -> metadata (includes langfuse_trace_id)
        """
        trace = self._client.api.trace.get(trace_id)

        observations = self._client.api.observations.get_many(
            trace_id=trace_id,
            fields="core,basic,io,usage",
        )
        obs_list = observations.data if hasattr(observations, "data") else []

        messages: list[Message] = []
        tool_calls: list[ToolCall] = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0

        for obs in obs_list:
            obs_type = getattr(obs, "type", None)
            if isinstance(obs_type, str):
                obs_type = obs_type.lower()

            if obs_type == "generation":
                self._process_generation(obs, messages, tool_calls)

                usage = getattr(obs, "usage_details", None) or {}
                total_input_tokens += usage.get("input", 0) or 0
                total_output_tokens += usage.get("output", 0) or 0
                total_cost += getattr(obs, "total_cost", 0) or 0

            elif obs_type == "tool":
                tc = ToolCall(
                    name=getattr(obs, "name", None) or "unknown_tool",
                    input=_to_dict_or_none(getattr(obs, "input", None)),
                    output=_to_str_or_dict_or_none(getattr(obs, "output", None)),
                )
                tool_calls.append(tc)

        trace_input = getattr(trace, "input", None) or ""
        trace_output = getattr(trace, "output", None) or ""

        latency_ms: float | None = None
        start_time = getattr(trace, "start_time", None)
        end_time = getattr(trace, "end_time", None)
        if start_time and end_time:
            latency_ms = (end_time - start_time).total_seconds() * 1000

        total_tokens = total_input_tokens + total_output_tokens

        trace_tags = getattr(trace, "tags", None)
        tags: dict[str, str] | None = None
        if trace_tags and isinstance(trace_tags, list):
            tags = {t: "true" for t in trace_tags}

        trace_meta = getattr(trace, "metadata", None) or {}
        if isinstance(trace_meta, dict):
            metadata = {**trace_meta, "langfuse_trace_id": trace_id}
        else:
            metadata = {"langfuse_trace_id": trace_id}

        return EvalCase(
            input=trace_input,
            output=trace_output,
            messages=messages or None,
            tool_calls=tool_calls or None,
            latency_ms=latency_ms,
            token_count=total_tokens if total_tokens > 0 else None,
            cost_usd=total_cost if total_cost > 0 else None,
            tags=tags,
            metadata=metadata,
        )

    @staticmethod
    def _process_generation(
        obs: object,
        messages: list[Message],
        tool_calls: list[ToolCall],
    ) -> None:
        """Extract messages and tool calls from a generation observation."""
        obs_input = getattr(obs, "input", None)
        obs_output = getattr(obs, "output", None)

        if isinstance(obs_input, list):
            for entry in obs_input:
                if isinstance(entry, dict) and "role" in entry:
                    msg_tool_calls = None
                    if "tool_calls" in entry and entry["tool_calls"]:
                        msg_tool_calls = [
                            ToolCall(
                                name=tc.get("function", {}).get("name", tc.get("name", "")),
                                input=tc.get("function", {}).get("arguments") if "function" in tc else tc.get("input"),
                            )
                            for tc in entry["tool_calls"]
                            if isinstance(tc, dict)
                        ]
                    messages.append(
                        Message(
                            role=entry.get("role", "unknown"),
                            content=entry.get("content"),
                            tool_calls=msg_tool_calls,
                        )
                    )

        if isinstance(obs_output, dict):
            role = obs_output.get("role", "assistant")
            content = obs_output.get("content")
            msg_tool_calls = None
            if "tool_calls" in obs_output and obs_output["tool_calls"]:
                msg_tool_calls = []
                for tc in obs_output["tool_calls"]:
                    if isinstance(tc, dict):
                        tc_obj = ToolCall(
                            name=tc.get("function", {}).get("name", tc.get("name", "")),
                            input=tc.get("function", {}).get("arguments") if "function" in tc else tc.get("input"),
                        )
                        msg_tool_calls.append(tc_obj)
                        tool_calls.append(tc_obj)
            messages.append(Message(role=role, content=content, tool_calls=msg_tool_calls or None))
        elif isinstance(obs_output, str):
            messages.append(Message(role="assistant", content=obs_output))


def _to_dict_or_none(val: object) -> dict | None:
    if isinstance(val, dict):
        return val
    return None


def _to_str_or_dict_or_none(val: object) -> str | dict | None:
    if isinstance(val, (str, dict)):
        return val
    return None
