"""JUnit XML sink — write scores as JUnit XML for CI integration."""

from __future__ import annotations

from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink


class JUnitSink(BaseSink):
    """Write scores as JUnit XML compatible with CI systems.

    Each ``write()`` call adds test cases to the suite. Call ``finalize()``
    to flush the accumulated results to disk. Each metric score becomes a
    ``<testcase>``; failures (score below threshold) become ``<failure>``
    elements.

    Compatible with Harness CI, GitHub Actions, Jenkins, and GitLab CI.
    """

    def __init__(self, path: str, suite_name: str = "harness-evals") -> None:
        self.path = Path(path)
        self.suite_name = suite_name
        self._testcases: list[Element] = []
        self._failures = 0
        self._tests = 0

    def write(self, scores: list[Score], eval_case: EvalCase) -> None:
        input_preview = str(eval_case.input)[:120]
        for score in scores:
            self._tests += 1
            tc = Element(
                "testcase",
                name=score.name,
                classname=input_preview,
            )
            if not score.passed:
                self._failures += 1
                failure = SubElement(
                    tc,
                    "failure",
                    message=f"{score.name}: {score.value:.4f} < {score.threshold:.4f}",
                    type="MetricFailure",
                )
                failure.text = score.reason or ""
            self._testcases.append(tc)

    def finalize(self) -> None:
        """Write accumulated test cases to the JUnit XML file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        suite = Element(
            "testsuite",
            name=self.suite_name,
            tests=str(self._tests),
            failures=str(self._failures),
        )
        for tc in self._testcases:
            suite.append(tc)

        tree = ElementTree(suite)
        with open(self.path, "wb") as f:
            tree.write(f, encoding="utf-8", xml_declaration=True)
