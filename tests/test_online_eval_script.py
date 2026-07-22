from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree

import pytest
from scripts import online_eval

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score


class FakeSource:
    def __init__(self, cases: list[EvalCase]) -> None:
        self.cases = cases
        self.requested_trace_ids: list[str] | None = None

    async def fetch_traces(self, trace_ids: list[str]) -> list[EvalCase]:
        self.requested_trace_ids = trace_ids
        return self.cases


class StaticMetric(BaseMetric):
    def __init__(self, value: float = 0.9, threshold: float = 0.7) -> None:
        super().__init__(name="task_completion", dimension=Dimension.CORRECTNESS, threshold=threshold)
        self.value = value

    def measure(self, eval_case: EvalCase) -> Score:
        return Score(
            name=self.name,
            value=self.value,
            threshold=self.threshold,
            reason="looks complete",
        )


def _args(tmp_path: Path, trace_ids: str = "trace-1") -> argparse.Namespace:
    return argparse.Namespace(
        trace_ids=trace_ids,
        org_id="SrikarOrg",
        project_id="SrikarProject",
        threshold=0.7,
        fail_below=0.0,
        model="gpt-4o-mini",
        metrics="task_completion",
        geval_criteria="Evaluate answer quality.",
        latency_threshold_ms=30_000.0,
        output_dir=str(tmp_path),
    )


@pytest.mark.unit
def test_parse_trace_ids_accepts_commas_and_newlines() -> None:
    assert online_eval._parse_trace_ids(" trace-1,\ntrace-2, ") == ["trace-1", "trace-2"]


@pytest.mark.unit
async def test_run_scores_trace_and_writes_junit_and_scores_json(tmp_path: Path) -> None:
    source = FakeSource(
        [
            EvalCase(
                input="Deploy the app",
                output="The app was deployed successfully.",
                metadata={"trace_id": "trace-1"},
            )
        ]
    )

    rc = await online_eval._run(
        _args(tmp_path),
        source_factory=lambda _args: source,
        metrics_factory=lambda _args: [StaticMetric()],
    )

    assert rc == 0
    assert source.requested_trace_ids == ["trace-1"]

    scores = json.loads((tmp_path / "scores.json").read_text())
    assert scores["trace_count"] == 1
    assert scores["metrics"]["task_completion"]["mean"] == 0.9
    assert scores["traces"][0]["scores"]["task_completion"]["passed"] is True

    suite = ElementTree.parse(tmp_path / "junit.xml").getroot()
    assert suite.attrib["tests"] == "1"
    assert suite.attrib["failures"] == "0"
    assert suite.find("testcase").attrib["name"] == "task_completion"


@pytest.mark.unit
async def test_run_writes_failing_junit_when_no_trace_is_usable(tmp_path: Path) -> None:
    source = FakeSource(
        [
            EvalCase(
                input="",
                output="",
                metadata={"trace_id": "trace-1", "error": "QueryService returned 403"},
            )
        ]
    )

    rc = await online_eval._run(
        _args(tmp_path),
        source_factory=lambda _args: source,
        metrics_factory=lambda _args: [StaticMetric()],
    )

    assert rc == 1

    scores = json.loads((tmp_path / "scores.json").read_text())
    assert scores["trace_count"] == 0
    assert scores["skipped"] == [{"trace_id": "trace-1", "reason": "QueryService returned 403"}]

    suite = ElementTree.parse(tmp_path / "junit.xml").getroot()
    assert suite.attrib["tests"] == "1"
    assert suite.attrib["failures"] == "1"
    testcase = suite.find("testcase")
    assert testcase.attrib["name"] == "trace_fetch"
    assert "QueryService returned 403" in testcase.find("failure").text
