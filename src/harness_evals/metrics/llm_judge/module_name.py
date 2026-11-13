"""LLM judge that labels a session with the module the user asked about."""

from __future__ import annotations

import re

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension, MetricInput, MetricInputs
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM

# Names and scope come from the Harness Platform module table:
# https://developer.harness.io/harness-platform
MODULE_DESCRIPTIONS: dict[str, str] = {
    "code": "Code Repository: git repositories and pull requests",
    "cd": "Continuous Delivery & GitOps: deployments, services, environments, and CD pipelines",
    "release": "Release Orchestration: multi-service releases and approval gates",
    "ci": "Continuous Integration: builds, tests, publishing artifacts, and CI pipelines",
    "idp": "Internal Developer Portal: service catalog, scorecards, self-service",
    "iacm": "Infrastructure as Code Management: Terraform, workspaces, drift, cost estimation",
    "dbops": "Database DevOps: database schema changes and deployments",
    "artifact": "Artifact Registry: build artifacts and container images",
    "fme": "Feature Management & Experimentation: A/B tests and feature experiments",
    "ff": "Feature Flags: progressive rollouts without redeploying",
    "chaos": "Resilience Testing: chaos experiments and resilience tests",
    "aitest": "AI Test Automation: generating and running tests",
    "aisre": "AI Site Reliability Engineering: incident detection and root cause analysis",
    "aisec": "AI Security: AI assets, threats, and AI endpoint testing",
    "aasp": "API & Application Discovery: cataloging APIs and applications",
    "aarp": "Application & API Runtime Protection: blocking attacks at runtime",
    "aast": "Application & API Security Testing: security tests during development and CI",
    "sto": "Security Testing Orchestration: security scan results across the pipeline",
    "scs": "Supply Chain Security: SBOMs, SLSA, and supply chain policy",
    "sast": "SAST & SCA: code vulnerabilities and open-source dependency risk",
    "ccm": "Cloud Cost Management, including Cloud & AI Cost Management: cloud and AI spend, budgets, recommendations",
    "aidi": "AI DLC Insights: developer productivity, DORA metrics, and engineering bottlenecks",
    "other": "None of the above, such as delegates, secrets, templates, and policy as code. Pipeline requests belong to ci or cd",
}
MODULES = tuple(MODULE_DESCRIPTIONS)
_MODULE_SET = frozenset(MODULES)
_CONTEXT_BLOCK = re.compile(r"<current_context\b[^>]*>.*?</current_context>", re.IGNORECASE | re.DOTALL)
_MODULE_LIST = "\n".join(f"- {code}: {description}" for code, description in MODULE_DESCRIPTIONS.items())
_PROMPT = """Classify the user request into one Harness module. Use only the request text.

Modules:
{modules}

User request:
{prompt}

Respond with JSON: {{"module": "<one code from the list>"}}
"""
_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["module"],
    "properties": {"module": {"type": "string", "enum": list(MODULES)}},
}


def _user_prompt(eval_case: EvalCase) -> str:
    """User text only, with UI context blocks removed."""
    parts: list[str] = []
    for message in eval_case.messages or []:
        if getattr(message, "role", None) == "user" and isinstance(message.content, str):
            parts.append(message.content)
    text = "\n".join(parts).strip()
    if not text and isinstance(eval_case.input, str):
        text = eval_case.input
    text = _CONTEXT_BLOCK.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


class ModuleNameMetric(BaseMetric):
    """Label a session from the user prompt.

    An LLM picks one module code. ``reason`` and ``metadata["module"]`` are
    that code. ``value`` is 1.0 for a named module and 0.0 for ``other``.
    The default threshold is 0.0, so the label does not fail the session.
    No user text, or a code outside the list, skips the case.
    """

    eval_inputs = MetricInputs(any_of=({MetricInput.INPUT}, {MetricInput.MESSAGES}))

    def __init__(self, llm: BaseLLM, threshold: float = 0.0, **kwargs: object) -> None:
        super().__init__(name="module_name", dimension=Dimension.CORRECTNESS, threshold=threshold, **kwargs)
        self.llm = llm

    def measure(self, eval_case: EvalCase) -> Score | None:
        return _run_async(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score | None:
        prompt = _user_prompt(eval_case)
        if not prompt:
            return None
        result = await self.llm.generate_json(_PROMPT.format(modules=_MODULE_LIST, prompt=prompt), _RESPONSE_SCHEMA)
        if not isinstance(result, dict):
            return None
        module = str(result.get("module", "")).strip().lower()
        if module not in _MODULE_SET:
            return None
        return self._score(module)

    def _score(self, module: str) -> Score:
        return Score(
            name=self.name,
            value=0.0 if module == "other" else 1.0,
            threshold=self.threshold,
            reason=module,
            metadata={"module": module},
        )
