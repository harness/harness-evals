"""Core recommender engine — calls an LLM with the full metric catalog and returns recommendations."""

from __future__ import annotations
import json
from harness_evals.recommender.scenarios import ScenarioInput


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
    {"input": "<realistic input>", "expected": "<expected output>", "context": null, "expected_tools": null, "expected_tool_calls": null, "metadata": {}, "tags": {}, "metric_tested": "<metric name>"}
  ],
  "recommended_actions": "<plain text next steps using harness-evals run>"
}

Rules:
- Only use metric names that appear exactly in the provided catalog list.
- Provide 3 to 5 entries in recommended_dataset.
- Be specific to the scenario described, not generic.
- Return only valid JSON, no prose before or after.
"""


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


def recommend(
    scenario: ScenarioInput,
    api_key: str,
    provider: str = "anthropic",
    model: str | None = None,
) -> dict:
    if provider == "anthropic":
        return _recommend_anthropic(scenario, api_key, model or "claude-sonnet-4-20250514")
    elif provider == "openai":
        return _recommend_openai(scenario, api_key, model or "gpt-4o")
    else:
        raise ValueError(f"Unsupported provider: {provider}. Choose 'anthropic' or 'openai'.")


def _recommend_anthropic(scenario: ScenarioInput, api_key: str, model: str) -> dict:
    try:
        import anthropic
    except ImportError:
        raise ImportError("Install anthropic: pip install anthropic")

    from harness_evals._async_compat import _run_async

    async def _call_anthropic():
        client = anthropic.AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_prompt(scenario)}],
        )
        if not response.content:
            raise ValueError("Anthropic returned empty response.")
        raw = response.content[0].text
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
            clean = clean.strip()
        return json.loads(clean)

    return _run_async(_call_anthropic())


def _recommend_openai(scenario: ScenarioInput, api_key: str, model: str) -> dict:
    try:
        import openai
    except ImportError:
        raise ImportError("Install openai: pip install openai")

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(scenario)},
        ],
    )
    raw = response.choices[0].message.content
    if not raw or not raw.strip():
        raise ValueError(f"OpenAI returned empty response. Check your API key and account credits.")
    # Strip markdown code fences if present
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        raise ValueError(f"OpenAI returned non-JSON response: {raw[:200]}")
