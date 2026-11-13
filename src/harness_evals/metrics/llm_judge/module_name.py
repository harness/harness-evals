"""LLM judge that labels a session with the module the user asked about."""

from __future__ import annotations

import re

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension, MetricInput, MetricInputs
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM

# Names and scope come from the Harness Platform module table and each module's overview docs:
# https://developer.harness.io/harness-platform
MODULE_DESCRIPTIONS: dict[str, str] = {
    "code": (
        "Code Repository: git repositories, branches, commits, pull requests, code review, repo rules and "
        "webhooks hosted in Harness Code"
    ),
    "cd": (
        "Continuous Delivery & GitOps: deploying services to environments and infrastructure (Kubernetes, ECS, "
        "Helm, serverless, VMs), services, environments, infrastructure definitions, rollbacks, freeze windows, "
        "GitOps agents and applications, and deployment or provisioning pipelines"
    ),
    "release": (
        "Release Orchestration: coordinating multi-service releases with phases, tasks, approvals and a release "
        "calendar"
    ),
    "ci": (
        "Continuous Integration: building and testing code, pushing images and artifacts, build caching, test "
        "intelligence, build infrastructure, and CI pipelines"
    ),
    "idp": (
        "Internal Developer Portal: software catalog entities, scorecards, and self-service workflows such "
        "as golden paths and software templates that scaffold new services, repos and environments, plus "
        "Backstage plugins and developer onboarding"
    ),
    "worker_agents": (
        "Worker agents: creating, updating, or deleting custom worker agents, including their uid, "
        "LLM connector, MCP connector, and task prompt"
    ),
    "iacm": (
        "Infrastructure as Code Management: IaCM workspaces for OpenTofu, Terraform, Terragrunt and Ansible, "
        "state, plan/apply runs, drift detection, IaC pull request automation, and cost estimates of IaC "
        "changes"
    ),
    "dbops": "Database DevOps: database schema changes, migrations, rollbacks and database instances",
    "artifact": (
        "Artifact Registry: storing and managing artifacts such as container images, Helm charts and "
        "Maven/npm/PyPI packages, registries, upstream proxies, lifecycle rules and quarantine"
    ),
    "fme": (
        "Feature Management & Experimentation: A/B tests, experiments, treatments, metrics and "
        "feature impact measurement"
    ),
    "ff": (
        "Feature Flags: creating and toggling flags, targets and target groups, and progressive rollouts "
        "without redeploying"
    ),
    "chaos": "Resilience Testing: chaos experiments, faults, probes, game days and resilience scores",
    "aitest": (
        "AI Test Automation: generating, running and self-healing tests, and test suites created from natural language"
    ),
    "aisre": (
        "AI Site Reliability Engineering: incidents, alerts, runbooks, root cause analysis and "
        "change-to-incident correlation"
    ),
    "aisec": (
        "AI Security: discovering AI assets, LLM and MCP endpoint threats, prompt injection, and AI "
        "endpoint security testing"
    ),
    "aasp": (
        "API & Application Discovery: cataloging APIs and applications, shadow APIs, API inventory and "
        "sensitive data exposure in APIs"
    ),
    "aarp": (
        "Application & API Runtime Protection: blocking attacks at runtime, WAF, bot and fraud protection, "
        "and runtime threat policies"
    ),
    "aast": (
        "Application & API Security Testing: security tests such as DAST and API security tests run "
        "during development and CI"
    ),
    "sto": (
        "Security Testing Orchestration: running security scanners in pipelines (Snyk, Trivy, Semgrep, "
        "Checkmarx and similar) and reviewing the vulnerabilities (CVEs, CVSS scores, severities) found in "
        "container images, code, dependencies and apps, plus exemptions, remediation and scan governance"
    ),
    "scs": (
        "Supply Chain Security: SBOMs, SLSA provenance, artifact signing and attestation, repository "
        "posture against CIS and OWASP CI/CD risks, and open-source dependency or license policy"
    ),
    "sast": (
        "SAST & SCA: static analysis of source code and software composition analysis of dependencies, "
        "such as code vulnerabilities, secrets in code and vulnerable open-source packages in repositories"
    ),
    "ccm": (
        "Cloud Cost Management, including Cloud & AI Cost Management: cloud and AI spend across AWS, "
        "Azure, GCP, Kubernetes and AI providers; Perspectives, Cost Explorer, budgets, cost categories, "
        "anomalies, recommendations, AutoStopping, savings and showback"
    ),
    "aidi": (
        "AI DLC Insights: developer productivity, AI coding assistant adoption and impact, DORA metrics, "
        "lead time, and engineering bottlenecks"
    ),
    "other": (
        "Last resort when no module above fits, such as delegates, secrets, connectors, users and RBAC, "
        "policy as code, audit trails, and general questions about Harness. Pipeline requests belong to ci or cd"
    ),
}
MODULES = tuple(MODULE_DESCRIPTIONS)
_MODULE_SET = frozenset(MODULES)
_CONTEXT_BLOCK = re.compile(r"<current_context\b[^>]*>.*?</current_context>", re.IGNORECASE | re.DOTALL)
_MODULE_LIST = "\n".join(f"- {code}: {description}" for code, description in MODULE_DESCRIPTIONS.items())
_PROMPT = """Classify the user request into the one Harness module the user is working in. Use only the request text.

Modules:
{modules}

User request:
{prompt}

Respond with JSON: {{"module": "<one code from the list>", "reason": "<one short sentence>"}}
"""
_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["module", "reason"],
    "properties": {
        "module": {"type": "string", "enum": list(MODULES)},
        "reason": {"type": "string"},
    },
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

    An LLM picks one module code and a one-sentence reason. ``metadata["module"]`` is
    the code; ``reason`` is ``<code>: <sentence>``. ``value`` is 1.0 for a named module and 0.0 for ``other``.
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
        reason = result.get("reason")
        reason = " ".join(reason.split()) if isinstance(reason, str) else ""
        return self._score(module, reason)

    def _score(self, module: str, reason: str) -> Score:
        return Score(
            name=self.name,
            value=0.0 if module == "other" else 1.0,
            threshold=self.threshold,
            reason=f"{module}: {reason}" if reason else module,
            metadata={"module": module, "reason": reason} if reason else {"module": module},
        )
