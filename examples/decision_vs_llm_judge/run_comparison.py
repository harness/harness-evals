"""DP-4: run the decision-primitive vs. LLM-as-judge efficacy comparison.

For every case in ``dataset.CASES`` and every one of the three labeled
dimensions (department / severity / is_escalation), runs both approaches on
the *same* input with deliberately comparable instructions:

- **Decision primitive**: ``ChoiceMetric``/``ScoreMetric``/``NoulMetric``
  against a live ``TypeSafeDecisionProvider``, ``mode="correctness"``.
- **LLM judge**: a custom JSON-schema prompt against a live ``AnthropicLLM``,
  asked to classify/rate the same dimension and self-report a confidence.

Records per (case, dimension, approach): predicted value, whether it matched
the golden label, self-reported confidence, latency, and token usage/cost.
Writes the raw rows to ``results.json`` for ``report.py`` to aggregate.

Requires ``TYPESAFE_API_KEY`` and ``ANTHROPIC_API_KEY`` in the environment.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from dataset import CASES, DEPARTMENTS, SEVERITY_LEVELS, TriageCase

from harness_evals.core.eval_case import EvalCase
from harness_evals.decision.typesafe import TypeSafeDecisionProvider
from harness_evals.llm.anthropic import AnthropicLLM
from harness_evals.llm.usage import collect_token_usage
from harness_evals.metrics.decision.choice import ChoiceMetric
from harness_evals.metrics.decision.noul import NoulMetric
from harness_evals.metrics.decision.score import ScoreMetric

DEPARTMENT_INSTRUCTIONS = "Classify which department should handle this support ticket."
DEPARTMENT_CRITERIA = {
    "returns": "Product returns, refunds for returned items, exchange requests.",
    "shipping": "Shipping delays, lost/damaged packages, delivery address issues, carrier tracking.",
    "billing": "Charges, invoices, duplicate/incorrect payments, subscription billing.",
}

SEVERITY_INSTRUCTIONS = "Rate how severe/urgent this support ticket is."
SEVERITY_CRITERIA = [
    "low: informational or non-urgent question, no real problem to fix",
    "medium: a real problem affecting the customer but not urgent or high-stakes",
    "high: significant financial impact, repeated unresolved contact, or urgent business impact",
]

ESCALATION_INSTRUCTIONS = (
    "Is the customer explicitly demanding to speak to a human/manager, or "
    "threatening a dispute/chargeback, right now in this message?"
)


@dataclass
class ResultRow:
    case_id: str
    dimension: str
    approach: str  # "decision" | "judge"
    predicted: str
    correct: bool
    confidence: float
    latency_s: float
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None


def _department_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": list(DEPARTMENTS)},
            "confidence": {"type": "number"},
        },
        "required": ["label", "confidence"],
    }


def _severity_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "label": {"type": "integer", "enum": [0, 1, 2]},
            "confidence": {"type": "number"},
        },
        "required": ["label", "confidence"],
    }


def _escalation_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "is_escalation": {"type": "boolean"},
            "confidence": {"type": "number"},
        },
        "required": ["is_escalation", "confidence"],
    }


async def _judge_department(
    llm: AnthropicLLM, text: str
) -> tuple[str, float, float, int | None, int | None, float | None]:
    prompt = (
        f"{DEPARTMENT_INSTRUCTIONS}\n\n"
        f"Departments:\n"
        + "\n".join(f"- {name}: {desc}" for name, desc in DEPARTMENT_CRITERIA.items())
        + f"\n\nTicket:\n{text}\n\n"
        "Respond with the department label and your confidence (0-1) that this "
        "classification is correct."
    )
    t0 = time.perf_counter()
    with collect_token_usage() as usage:
        result = await llm.generate_json(prompt, schema=_department_schema())
    latency = time.perf_counter() - t0
    return (
        result["label"],
        float(result["confidence"]),
        latency,
        usage.input_tokens,
        usage.output_tokens,
        usage.cost_usd,
    )


async def _judge_severity(
    llm: AnthropicLLM, text: str
) -> tuple[int, float, float, int | None, int | None, float | None]:
    prompt = (
        f"{SEVERITY_INSTRUCTIONS}\n\n"
        f"Rubric (0=low, 1=medium, 2=high):\n"
        + "\n".join(f"- {i}: {desc}" for i, desc in enumerate(SEVERITY_CRITERIA))
        + f"\n\nTicket:\n{text}\n\n"
        "Respond with the severity level (0, 1, or 2) and your confidence (0-1) "
        "that this rating is correct."
    )
    t0 = time.perf_counter()
    with collect_token_usage() as usage:
        result = await llm.generate_json(prompt, schema=_severity_schema())
    latency = time.perf_counter() - t0
    return (
        int(result["label"]),
        float(result["confidence"]),
        latency,
        usage.input_tokens,
        usage.output_tokens,
        usage.cost_usd,
    )


async def _judge_escalation(
    llm: AnthropicLLM, text: str
) -> tuple[bool, float, float, int | None, int | None, float | None]:
    prompt = (
        f"{ESCALATION_INSTRUCTIONS}\n\n"
        f"Ticket:\n{text}\n\n"
        "Respond with is_escalation (true/false) and your confidence (0-1) that "
        "this determination is correct."
    )
    t0 = time.perf_counter()
    with collect_token_usage() as usage:
        result = await llm.generate_json(prompt, schema=_escalation_schema())
    latency = time.perf_counter() - t0
    return (
        bool(result["is_escalation"]),
        float(result["confidence"]),
        latency,
        usage.input_tokens,
        usage.output_tokens,
        usage.cost_usd,
    )


async def run(cases: list[TriageCase]) -> list[ResultRow]:
    provider = TypeSafeDecisionProvider()
    llm = AnthropicLLM(model="claude-sonnet-4-5-20250929")

    choice_metric = ChoiceMetric(
        provider=provider,
        instructions=DEPARTMENT_INSTRUCTIONS,
        criteria=DEPARTMENT_CRITERIA,
        mode="correctness",
        state_field="output",
    )
    severity_metric = ScoreMetric(
        provider=provider,
        instructions=SEVERITY_INSTRUCTIONS,
        criteria=SEVERITY_CRITERIA,
        mode="correctness",
        state_field="output",
    )
    escalation_metric = NoulMetric(
        provider=provider,
        instructions=ESCALATION_INSTRUCTIONS,
        mode="correctness",
        state_field="output",
    )

    rows: list[ResultRow] = []

    for case in cases:
        # --- department ---
        ec = EvalCase(input=case.case_id, output=case.text, expected=case.department)
        t0 = time.perf_counter()
        score = await choice_metric.a_measure(ec)
        latency = time.perf_counter() - t0
        rows.append(
            ResultRow(
                case.case_id,
                "department",
                "decision",
                str(score.metadata["choice"]),
                score.value == 1.0,
                float(score.metadata["confidence"]),
                latency,
                score.metadata.get("input_tokens"),
                score.metadata.get("output_tokens"),
                None,
            )
        )

        label, conf, latency, in_tok, out_tok, cost = await _judge_department(llm, case.text)
        rows.append(
            ResultRow(
                case.case_id,
                "department",
                "judge",
                label,
                label == case.department,
                conf,
                latency,
                in_tok,
                out_tok,
                cost,
            )
        )

        # --- severity ---
        ec = EvalCase(input=case.case_id, output=case.text, expected=case.severity)
        t0 = time.perf_counter()
        score = await severity_metric.a_measure(ec)
        latency = time.perf_counter() - t0
        predicted_level = round(score.metadata["score"])
        rows.append(
            ResultRow(
                case.case_id,
                "severity",
                "decision",
                SEVERITY_LEVELS[max(0, min(2, predicted_level))],
                predicted_level == case.severity,
                float(score.metadata["confidence"]),
                latency,
                score.metadata.get("input_tokens"),
                score.metadata.get("output_tokens"),
                None,
            )
        )

        level, conf, latency, in_tok, out_tok, cost = await _judge_severity(llm, case.text)
        rows.append(
            ResultRow(
                case.case_id,
                "severity",
                "judge",
                SEVERITY_LEVELS[level],
                level == case.severity,
                conf,
                latency,
                in_tok,
                out_tok,
                cost,
            )
        )

        # --- is_escalation ---
        ec = EvalCase(input=case.case_id, output=case.text, expected=case.is_escalation)
        t0 = time.perf_counter()
        score = await escalation_metric.a_measure(ec)
        latency = time.perf_counter() - t0
        noul = float(score.metadata["noul"])
        predicted_bool = noul >= 0.5
        confidence_in_direction = noul if predicted_bool else 1.0 - noul
        rows.append(
            ResultRow(
                case.case_id,
                "is_escalation",
                "decision",
                str(predicted_bool),
                predicted_bool == case.is_escalation,
                confidence_in_direction,
                latency,
                score.metadata.get("input_tokens"),
                score.metadata.get("output_tokens"),
                None,
            )
        )

        is_esc, conf, latency, in_tok, out_tok, cost = await _judge_escalation(llm, case.text)
        rows.append(
            ResultRow(
                case.case_id,
                "is_escalation",
                "judge",
                str(is_esc),
                is_esc == case.is_escalation,
                conf,
                latency,
                in_tok,
                out_tok,
                cost,
            )
        )

        print(f"done: {case.case_id}")

    return rows


if __name__ == "__main__":
    results = asyncio.run(run(CASES))
    out_path = Path(__file__).parent / "results.json"
    out_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\nWrote {len(results)} rows to {out_path}")
