"""Normalize the three input scenarios into a common description for the recommender engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from harness_evals.errors import HarnessEvalsError


class ScenarioType(str, Enum):
    PROMPT = "prompt"
    HTTP_ENDPOINT = "http_endpoint"
    TRACES = "traces"


@dataclass
class ScenarioInput:
    type: ScenarioType
    content: str


def load_scenario(
    prompt: str | None = None,
    endpoint: str | None = None,
    traces: str | None = None,
) -> ScenarioInput:
    if sum(x is not None for x in [prompt, endpoint, traces]) != 1:
        raise HarnessEvalsError("Provide exactly one of --prompt, --endpoint, or --traces.")

    if prompt is not None:
        path = Path(prompt)
        content = path.read_text() if path.exists() else prompt
        return ScenarioInput(type=ScenarioType.PROMPT, content=content)

    if endpoint is not None:
        return ScenarioInput(type=ScenarioType.HTTP_ENDPOINT, content=endpoint)

    path = Path(traces)
    content = path.read_text() if path.exists() else traces
    return ScenarioInput(type=ScenarioType.TRACES, content=content)
