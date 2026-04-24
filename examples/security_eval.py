"""Run security remediation metrics end-to-end.

This example uses MockLLM to simulate judge responses, demonstrating the
full pipeline: EvalCase -> 7 metrics -> composite RQI -> stdout output.

Run: python examples/security_eval.py
"""

from harness_evals import EvalCase, evaluate
from harness_evals.metrics.security import (
    ActionabilityMetric,
    CodeQualityMetric,
    CodeSafetyMetric,
    ExplanationQualityMetric,
    RootCauseAnalysisMetric,
    SecurityCompletenessMetric,
    VulnerabilityCorrectnessMetric,
    remediation_quality_index,
)
from harness_evals.sinks import StdoutSink

# Simulate an LLM judge that returns realistic scores.
# In production, replace with OpenAILLM or HarnessAILLM.
from harness_evals.llm.base import BaseLLM


class MockJudge(BaseLLM):
    """Returns pre-configured scores in order."""

    def __init__(self, responses: list[dict]):
        self._responses = responses
        self._idx = 0

    async def generate(self, prompt: str, **kw) -> str:
        return ""

    async def generate_json(self, prompt: str, schema: dict, **kw) -> dict:
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return resp


llm = MockJudge(
    responses=[
        {"reasoning": "Fix correctly targets XSS root cause with escape()", "score": 9},
        {"reasoning": "Handles main case but missing CSP and output encoding", "score": 6},
        {"reasoning": "No new vulnerabilities, minimal scope", "score": 8},
        {"reasoning": "Correct Python, idiomatic use of markupsafe", "score": 9},
        {"reasoning": "Specific to CWE-79 but missing attack scenario", "score": 7},
        {"reasoning": "Taint path: request.args -> template render identified", "score": 8},
        {"reasoning": "Copy-paste ready with import and fix", "score": 9},
    ]
)

# Real vulnerability case
xss_case = EvalCase(
    input=(
        "CWE-79: Reflected XSS in user_profile.py line 42.\n"
        "User input from request.args['name'] is rendered directly in the "
        "Jinja2 template: {{ name }} without escaping."
    ),
    output=(
        "## Vulnerability: CWE-79 Reflected XSS\n\n"
        "**Root Cause**: The `request.args['name']` parameter is inserted into "
        "the HTML template without sanitization, allowing script injection.\n\n"
        "**Attack**: `https://app.com/profile?name=<script>document.cookie</script>`\n\n"
        "### Fix (user_profile.py line 42)\n"
        "```python\n"
        "from markupsafe import escape\n"
        "name = escape(request.args.get('name', ''))\n"
        "```\n\n"
        "### Additional hardening\n"
        "- Add `Content-Security-Policy: default-src 'self'` header\n"
        "- Enable Jinja2 autoescape: `Environment(autoescape=True)`"
    ),
)

metrics = [
    VulnerabilityCorrectnessMetric(llm=llm, threshold=0.5),
    SecurityCompletenessMetric(llm=llm, threshold=0.5),
    CodeSafetyMetric(llm=llm, threshold=0.5),
    CodeQualityMetric(llm=llm, threshold=0.5),
    ExplanationQualityMetric(llm=llm, threshold=0.5),
    RootCauseAnalysisMetric(llm=llm, threshold=0.5),
    ActionabilityMetric(llm=llm, threshold=0.5),
]

print("=" * 70)
print("Security Remediation Evaluation — CWE-79 XSS Fix")
print("=" * 70)

scores = evaluate(xss_case, metrics=metrics, sinks=[StdoutSink()])

print("\n" + "-" * 70)
print("Composite Score")
print("-" * 70)

rqi = remediation_quality_index(scores)
status = "PASS" if rqi.passed else "FAIL"
print(f"  {status} RemediationQualityIndex: {rqi.value:.3f} (threshold={rqi.threshold})")
print(f"  Matched: {len(rqi.metadata['matched_metrics'])}/7 metrics")
print(f"  Reason: {rqi.reason}")
