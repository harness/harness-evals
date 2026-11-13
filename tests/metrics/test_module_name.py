"""Module name is an LLM label of the user prompt."""

from __future__ import annotations

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import OnlineScope
from harness_evals.core.types import Message
from harness_evals.metrics.llm_judge.module_name import ModuleNameMetric
from harness_evals.validation.contracts import resolve_metric_contract


class _Judge:
    def __init__(self, module: str) -> None:
        self.module = module
        self.prompts: list[str] = []

    async def generate_json(self, prompt: str, schema: dict) -> dict[str, str]:
        self.prompts.append(prompt)
        return {"module": self.module}


@pytest.mark.unit
def test_module_name_uses_the_user_prompt() -> None:
    judge = _Judge("ccm")
    case = EvalCase(
        input="ignored when messages exist",
        output="done",
        metadata={"module": "cd", "agent.module": "ci"},
        tags={"product": "idp"},
        messages=[
            Message(
                role="user",
                content="<current_context><module>cd</module></current_context> Why did yesterday's cloud spend jump?",
            ),
            Message(role="assistant", content="The CCM perspective grew."),
            Message(role="user", content="Break that spend down by cluster."),
        ],
    )

    score = ModuleNameMetric(judge).measure(case)

    assert score is not None
    assert score.reason == "ccm"
    assert score.value == 1.0
    assert score.metadata == {"module": "ccm"}
    assert score.passed
    assert len(judge.prompts) == 1
    sent = judge.prompts[0]
    assert "cloud spend" in sent
    assert "Break that spend down by cluster." in sent
    assert "<current_context>" not in sent
    assert "cd" not in sent.split("User request:", 1)[-1]


@pytest.mark.unit
def test_module_name_falls_back_to_input() -> None:
    judge = _Judge("sto")
    score = ModuleNameMetric(judge).measure(EvalCase(input="Scan this image for vulnerabilities.", output=""))
    assert score is not None
    assert score.reason == "sto"
    assert "ccm: Cloud Cost Management" in judge.prompts[0]
    assert "Scan this image" in judge.prompts[0]


@pytest.mark.unit
def test_module_name_other_is_a_label() -> None:
    score = ModuleNameMetric(_Judge("other")).measure(EvalCase(input="What is the weather?", output=""))
    assert score is not None
    assert score.reason == "other"
    assert score.value == 0.0
    assert score.passed


@pytest.mark.unit
def test_module_name_skips_empty_prompt() -> None:
    judge = _Judge("ci")
    score = ModuleNameMetric(judge).measure(
        EvalCase(input="  ", output="", messages=[Message(role="assistant", content="hello")])
    )
    assert score is None
    assert judge.prompts == []


@pytest.mark.unit
def test_module_name_rejects_an_unlisted_label() -> None:
    score = ModuleNameMetric(_Judge("platform")).measure(EvalCase(input="Show my pipelines.", output=""))
    assert score is None


@pytest.mark.unit
def test_module_name_is_reviewed_for_online_traces() -> None:
    contract = resolve_metric_contract("module_name", {})
    assert contract.reviewed
    assert contract.resources == ("llm",)
    assert OnlineScope.TRACE in contract.online_scopes
    assert OnlineScope.SPAN in contract.online_scopes
    assert OnlineScope.GROUP not in contract.online_scopes
