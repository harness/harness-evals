"""Core recommender engine — calls a ``BaseLLM`` with the full metric catalog and returns recommendations."""

from __future__ import annotations

import logging

from harness_evals.llm.base import BaseLLM
from harness_evals.recommender.scenarios import ScenarioInput

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an AI evaluation expert with deep knowledge of the harness-evals metric catalog.
Given a description of what someone is building or has, recommend the right evaluations for it.

You will be given:
1. A scenario type: prompt (someone has a prompt), http_endpoint (someone has a deployed endpoint), or traces (someone has prior interaction traces)
2. The scenario content
3. The full list of available metrics from the harness-evals catalog

Return a JSON object with exactly these fields:
{
  "dimensions_covered": [
    {"dimension": "<name>", "applies": true/false, "rationale": "<one sentence>"}
  ],
  "recommended_metrics": [
    {"name": "<exact catalog metric kind>", "dimension": "<dimension>", "rationale": "<why it applies>", "threshold": <float 0.0-1.0>}
  ],
  "recommended_dataset": [
    {"input": "<realistic input>", "expected": "<expected output>", "context": null, "metric_tested": "<metric name>"}
  ],
  "recommended_actions": "<plain text next steps using harness-evals run>"
}

Rules:
- Only use metric names that appear exactly in the provided catalog list.
- Provide 3 to 5 entries in recommended_dataset.
- Be specific to the scenario described, not generic.
- Return only valid JSON, no prose before or after.
"""


# JSON Schema for the recommendation response. Passed to ``generate_json()`` so
# the provider enforces structured output. Kept strict-output friendly: no
# free-form object properties, nullable fields expressed as type unions.
RECOMMENDATION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "dimensions_covered": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "dimension": {"type": "string"},
                    "applies": {"type": "boolean"},
                    "rationale": {"type": "string"},
                },
            },
        },
        "recommended_metrics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "dimension": {"type": "string"},
                    "rationale": {"type": "string"},
                    "threshold": {"type": "number"},
                },
            },
        },
        "recommended_dataset": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "expected": {"type": "string"},
                    "context": {"type": ["array", "null"], "items": {"type": "string"}},
                    "metric_tested": {"type": "string"},
                },
            },
        },
        "recommended_actions": {"type": "string"},
    },
}


def _build_catalog_text() -> str:
    from harness_evals.catalog import catalog

    entries = catalog()
    lines = []
    for entry in entries:
        lines.append(f"- {entry.kind} ({entry.dimension.value}): {entry.description}")
    return "\n".join(lines)


def _build_user_prompt(scenario: ScenarioInput) -> str:
    catalog_text = _build_catalog_text()
    return f"""Scenario type: {scenario.type.value}

Scenario content:
{scenario.content}

Available metrics in the harness-evals catalog:
{catalog_text}

Return your recommendation as a JSON object following the schema in your instructions."""


async def recommend(scenario: ScenarioInput, llm: BaseLLM) -> dict:
    """Ask *llm* to recommend evals for *scenario*.

    The CLI is responsible for building *llm* via ``build_llm(ModelSpec(...))``,
    so this function stays provider-agnostic and works with any ``BaseLLM``.
    """

    prompt = _build_user_prompt(scenario)
    # Log the scenario type and prompt size only — never the prompt content
    # (which may contain proprietary system prompts) or any credentials.
    logger.info(
        "Requesting recommendation: scenario_type=%s prompt_chars=%d",
        scenario.type.value,
        len(prompt),
    )
    recommendation = await llm.generate_json(
        prompt,
        RECOMMENDATION_SCHEMA,
        system_prompt=SYSTEM_PROMPT,
    )
    logger.info(
        "Received recommendation: metrics=%d dataset_cases=%d",
        len(recommendation.get("recommended_metrics", [])),
        len(recommendation.get("recommended_dataset", [])),
    )
    return recommendation
