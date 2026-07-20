"""JUnit XML sink — write scores as JUnit XML for CI integration."""

from __future__ import annotations

import re
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink

# Characters that are illegal in XML 1.0 even when escaped. ElementTree escapes
# <, >, &, and quotes but writes these control chars verbatim, producing a file
# that strict JUnit parsers (Jenkins, GitHub Actions, GitLab) reject as
# malformed. LLM output and score reasons routinely contain them. The legal set
# is tab, newline, carriage return, and everything from 0x20 up (minus the
# surrogate/FFFE-FFFF gaps).
_XML_ILLEGAL_RE = re.compile("[^\x09\x0a\x0d\x20-퟿-�\U00010000-\U0010ffff]")


def _xml_safe(text: str) -> str:
    """Strip characters that are not legal in XML 1.0 text/attribute content."""
    return _XML_ILLEGAL_RE.sub("", text)


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
        input_preview = _xml_safe(str(eval_case.input)[:120])
        for score in scores:
            self._tests += 1
            attrs: dict[str, str] = {
                "name": _xml_safe(score.name),
                "classname": input_preview,
            }
            if score.scoring_duration_ms is not None:
                attrs["time"] = f"{score.scoring_duration_ms / 1000.0:.3f}"
            tc = Element("testcase", **attrs)
            if not score.passed:
                self._failures += 1
                failure = SubElement(
                    tc,
                    "failure",
                    message=f"{score.name}: {score.value:.4f} < {score.threshold:.4f}",
                    type="MetricFailure",
                )
                failure.text = _xml_safe(score.reason or "")
            self._testcases.append(tc)

    def finalize(self) -> None:
        """Write accumulated test cases to the JUnit XML file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        total_time = sum(float(tc.attrib["time"]) for tc in self._testcases if "time" in tc.attrib)
        suite_attrs: dict[str, str] = {
            "name": self.suite_name,
            "tests": str(self._tests),
            "failures": str(self._failures),
            "time": f"{total_time:.3f}",
        }
        suite = Element("testsuite", **suite_attrs)
        for tc in self._testcases:
            suite.append(tc)

        tree = ElementTree(suite)
        with open(self.path, "wb") as f:
            tree.write(f, encoding="utf-8", xml_declaration=True)
