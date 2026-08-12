"""Tests for stderr progress reporting during eval runs."""

from __future__ import annotations

import asyncio

import pytest

from harness_evals.conversation.golden import ConversationGolden, ConversationMode
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.golden import Golden
from harness_evals.core.runner import evaluate_dataset
from harness_evals.core.types import Message
from harness_evals.metrics.deterministic.exact_match import ExactMatchMetric
from harness_evals.progress import eval_case_label, make_stderr_progress_handlers


@pytest.mark.unit
class TestEvalCaseLabel:
    def test_prefers_golden_id_from_metadata(self):
        ec = EvalCase(input="", output="x", metadata={"golden_id": "row-abc"})
        assert eval_case_label(ec) == "row-abc"


@pytest.mark.unit
class TestStderrProgressHandlers:
    async def test_conversation_run_prints_done_before_next_running_with_concurrency_one(self, capsys):
        """With concurrency=1, item N must print done before item N+1 prints running."""
        goldens = [
            ConversationGolden(
                id=f"golden-{i}",
                scenario=f"S{i}",
                expected_outcome="done",
                mode=ConversationMode.SCRIPTED,
                turns=[Message(role="user", content="Hi")],
            )
            for i in range(2)
        ]

        async def slow_agent(_messages):
            await asyncio.sleep(0.02)
            return Message(role="assistant", content="ok")

        on_progress, on_result = make_stderr_progress_handlers()
        await evaluate_dataset(
            goldens,
            slow_agent,
            metrics=[ExactMatchMetric()],
            simulator_llm=None,
            concurrency=1,
            on_progress=on_progress,
            on_result=on_result,
        )

        err = capsys.readouterr().err
        assert "[1/2] running — golden-0" in err
        assert "[1/2] done —" in err
        assert "[2/2] running — golden-1" in err
        assert "[2/2] done —" in err
        assert err.index("[1/2] done") < err.index("[2/2] running")

    async def test_single_turn_prints_done_per_item(self, capsys):
        async def agent(golden: Golden) -> EvalCase:
            await asyncio.sleep(0.01)
            return EvalCase(input=golden.input, output=golden.expected or "")

        goldens = [Golden(input=f"q{i}", expected=f"q{i}") for i in range(2)]
        on_progress, on_result = make_stderr_progress_handlers()

        await evaluate_dataset(
            goldens,
            agent,
            metrics=[ExactMatchMetric()],
            concurrency=1,
            on_progress=on_progress,
            on_result=on_result,
        )

        err = capsys.readouterr().err
        assert "[1/2] done —" in err
        assert "[2/2] done —" in err
