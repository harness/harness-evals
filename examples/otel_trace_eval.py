"""Evaluate a GenAI trace using OTELEvalCaseSource.

Loads an ecommerce support agent trace from a JSON file and produces a single
EvalCase with the full conversation trajectory. This single case is evaluated
with both:
  - Whole-conversation metrics (turn efficiency, latency)
  - Per-turn assertions (via messages and metadata["turns"])

Run: python examples/otel_trace_eval.py

Trace structure (see ecommerce_support_trace.json):
  1. Customer asks about order status → agent calls lookup_order tool
  2. Tool returns shipping details
  3. Agent responds with order details and ETA

Follows: https://github.com/open-telemetry/semantic-conventions-genai
"""

from __future__ import annotations

import json
from pathlib import Path

from harness_evals import evaluate
from harness_evals.importers.otel import OTELEvalCaseSource
from harness_evals.metrics import ContainsMetric, LatencyMetric
from harness_evals.metrics.conversation import TurnEfficiencyMetric
from harness_evals.sinks import StdoutSink

TRACE_FILE = Path(__file__).parent / "ecommerce_support_trace.json"


def main() -> None:
    with open(TRACE_FILE) as f:
        spans = json.load(f)

    # ──────────────────────────────────────────────────────────────────
    # Build a single EvalCase from the full trace
    # ──────────────────────────────────────────────────────────────────
    ec = OTELEvalCaseSource.from_span_json(spans)

    # Enrich with evaluation expectations
    ec.metadata = ec.metadata or {}
    ec.metadata["expected_turns"] = 5  # user + 2 assistant + tool_call + tool
    ec.metadata["chatbot_role"] = "ecommerce support agent for ShopFast"
    ec.expected = "ORD-98712"  # Final output must reference the order

    # ──────────────────────────────────────────────────────────────────
    # Display the trace
    # ──────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("OTel GenAI Trace → Single EvalCase")
    print("=" * 70)
    print()
    print(f"  Trace ID:  {ec.meta('trace_id')}")
    print(f"  Provider:  {ec.meta('provider')}")
    print(f"  Model:     {ec.meta('model')}")
    print(f"  Latency:   {ec.latency_ms:.0f}ms")
    print(f"  Tokens:    {ec.token_count}")
    print()
    print(f"  Input:     {ec.input}")
    print(f"  Output:    {str(ec.output)[:80]}...")
    print()

    # Show the conversation trajectory
    print("  Conversation trajectory:")
    for i, m in enumerate(ec.messages or []):
        content = (m.content or "")[:60]
        tc = f" → {[t.name for t in m.tool_calls]}" if m.tool_calls else ""
        print(f"    [{i}] {m.role:10s} {content}{tc}")
    print()

    # Show per-turn breakdown from metadata
    print("  Per-turn breakdown:")
    for turn in ec.meta("turns") or []:
        print(
            f"    {turn['span_id']}: "
            f"{turn['input_tokens']}+{turn['output_tokens']} tokens, "
            f"{turn['latency_ms']:.0f}ms"
        )
    print()

    # ──────────────────────────────────────────────────────────────────
    # Evaluate — one case, multiple metric types
    # ──────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print()

    metrics = [
        # Whole-trace metrics
        LatencyMetric(max_ms=5000, threshold=0.3),
        TurnEfficiencyMetric(threshold=0.6),
        # Per-output assertion: final response must mention the order ID
        ContainsMetric(case_sensitive=False),
        # LLM-judged conversation metrics (uncomment with a BaseLLM instance):
        # ConversationCoherenceMetric(llm=my_llm),
        # ConversationResolutionMetric(llm=my_llm),
        # RoleAdherenceMetric(llm=my_llm),
        # ToolUseMetric(llm=my_llm),
    ]

    scores = evaluate(ec, metrics=metrics, sinks=[StdoutSink()])
    print()

    passed = [s for s in scores if s.passed]
    failed = [s for s in scores if not s.passed]
    print(f"  Summary: {len(passed)}/{len(scores)} passed")
    if failed:
        print(f"  Failed:  {[s.name for s in failed]}")
    print()


if __name__ == "__main__":
    main()
