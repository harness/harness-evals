"""Multi-turn tool / trajectory eval demo (ConversationGolden REPLAY).

Scores canned support-agent transcripts for correct tool selection and
arguments. Runs offline with no API key for deterministic metrics.

Run:
  python examples/multi_turn_tool_eval.py

Optional LLM-judged metrics (needs OPENAI_API_KEY + ``pip install -e ".[llm]"``):
  USE_LLM=1 python examples/multi_turn_tool_eval.py

Product validation checklist
----------------------------
1. Author goldens whose turns include assistant ``tool_calls`` plus
   ``expected_tool_calls``.
2. Attach ``tool_correctness`` (subsequence) and ``tool_argument_match``.
3. Run the eval; confirm each result exposes ``messages`` and ``tool_calls``.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from harness_evals.conversation import load_conversation_dataset
from harness_evals.core.runner import evaluate_dataset
from harness_evals.core.types import Message
from harness_evals.metrics.agent.tool_argument_match import ToolArgumentMatchMetric
from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric
from harness_evals.sinks.stdout import StdoutSink

GOLDENS_PATH = Path(__file__).parent / "data" / "multi_turn_tool_goldens.jsonl"


async def _noop_agent(messages: list[Message]) -> Message:
    """REPLAY never calls the agent; evaluate_dataset still requires a callable."""
    return Message(role="assistant", content="")


def _build_metrics() -> list:
    metrics: list = [
        ToolCorrectnessMetric(mode="subsequence", threshold=1.0),
        ToolArgumentMatchMetric(pair="subsequence", arg_match="subset", threshold=1.0),
    ]
    if os.environ.get("USE_LLM", "0") == "1":
        from harness_evals.llm.openai import OpenAILLM
        from harness_evals.metrics.agent.step_efficiency import StepEfficiencyMetric
        from harness_evals.metrics.conversation.goal_accuracy import GoalAccuracyMetric
        from harness_evals.metrics.conversation.tool_use import ToolUseMetric

        llm = OpenAILLM(model=os.environ.get("EVAL_MODEL", "gpt-4o-mini"))
        metrics.extend(
            [
                StepEfficiencyMetric(llm=llm, threshold=0.7),
                ToolUseMetric(llm=llm, threshold=0.7),
                GoalAccuracyMetric(llm=llm, threshold=0.7),
            ]
        )
    return metrics


async def main() -> None:
    goldens = load_conversation_dataset(GOLDENS_PATH)
    results = await evaluate_dataset(
        goldens,
        _noop_agent,
        metrics=_build_metrics(),
        sinks=[StdoutSink()],
        simulator_llm=None,
    )

    print("\n=== Summary ===")
    for golden, scores in zip(goldens, results, strict=True):
        passed = all(s.passed for s in scores)
        label = (golden.tags or {}).get("case", golden.id or golden.scenario)
        print(f"  {label}: {'PASS' if passed else 'FAIL'} ({len(scores)} metrics)")
        for score in scores:
            print(f"    - {score.name}: value={score.value:.2f} passed={score.passed}")


if __name__ == "__main__":
    asyncio.run(main())
