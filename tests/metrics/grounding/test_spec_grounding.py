"""Tests for SpecGroundingMetric with a canned BaseLLM."""

from __future__ import annotations

from typing import cast

import pytest

from harness_evals import EvalCase
from harness_evals.catalog import catalog
from harness_evals.core.metric import Dimension
from harness_evals.llm.base import BaseLLM
from harness_evals.metrics.factory import build_metric
from harness_evals.metrics.grounding.scoring import RequirementResult
from harness_evals.metrics.grounding.spec_grounding import (
    ExtractedRequirement,
    SpecGroundingMetric,
    extract_requirements,
)

pytestmark = pytest.mark.unit


class RecordingLLM(BaseLLM):
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.schemas: list[dict] = []

    async def generate(self, prompt: str, **kwargs: object) -> str:
        return ""

    async def generate_json(self, prompt: str, schema: dict, **kwargs: object) -> dict:
        self.prompts.append(prompt)
        self.schemas.append(schema)
        if not self._responses:
            raise AssertionError(f"unexpected LLM call: {prompt[:80]!r}")
        return cast(dict, self._responses.pop(0))


def _extract_payload() -> dict:
    return {
        "requirements": [
            {"text": "Must offer a refund", "spec_heading": "Refunds"},
            {"text": "Must quote the order id", "spec_heading": "Orders"},
            {"text": "Must greet the user", "spec_heading": "Tone"},
            {"text": "Must confirm the warehouse", "spec_heading": "Fulfillment"},
        ]
    }


def _call_a_payload() -> dict:
    return {
        "requirements": [
            {
                "text": "Must offer a refund",
                "applicable": True,
                "status": "satisfied",
                "evidence": "issued a refund",
            },
            {
                "text": "Must quote the order id",
                "applicable": True,
                "status": "satisfied",
                "evidence": "order 123",
            },
            {
                "text": "Must greet the user",
                "applicable": True,
                "status": "missing",
                "evidence": "no greeting",
            },
            {
                "text": "Must confirm the warehouse",
                "applicable": True,
                "status": "missing",
                "evidence": "no warehouse",
            },
        ]
    }


def _call_b_payload() -> dict:
    return {
        "claims": [
            {"text": "Refund issued", "status": "supported", "evidence": "refunds allowed"},
            {"text": "Order 123 exists", "status": "supported", "evidence": "order id"},
            {"text": "Order 123 exists copy", "status": "supported", "evidence": "order id"},
            {"text": "Ships same day", "status": "supported", "evidence": "shipping"},
            {"text": "Free overnight", "status": "unsupported", "evidence": "not in spec"},
        ]
    }


def _case() -> EvalCase:
    return EvalCase(
        input="I want a refund for order 123",
        output="Refund issued for order 123. Ships same day. Free overnight.",
        context=["# Refunds\nMust offer a refund.\n# Orders\nMust quote the order id."],
    )


def test_cold_path_makes_three_calls() -> None:
    llm = RecordingLLM([_extract_payload(), _call_a_payload(), _call_b_payload()])
    score = SpecGroundingMetric(llm=llm).measure(_case())
    assert len(llm.prompts) == 3
    assert score.name == "spec_grounding"
    assert score.value == 0.74
    assert score.metadata["sub_scores"]["coverage"] == 0.5
    assert "Must greet the user" in (score.reason or "")


def test_pre_extracted_skips_extraction() -> None:
    llm = RecordingLLM([_call_a_payload(), _call_b_payload()])
    reqs = [
        ExtractedRequirement("Must offer a refund", "Refunds"),
        ExtractedRequirement("Must quote the order id", "Orders"),
        ExtractedRequirement("Must greet the user", "Tone"),
        ExtractedRequirement("Must confirm the warehouse", "Fulfillment"),
    ]
    metric = SpecGroundingMetric(llm=llm, requirements=reqs)
    score = metric.measure(_case())
    assert len(llm.prompts) == 2
    assert "Specification" not in llm.prompts[0]
    assert score.value == 0.74


async def test_extract_requirements_function() -> None:
    llm = RecordingLLM([_extract_payload()])
    reqs = await extract_requirements("Must offer a refund", llm)
    assert [r.text for r in reqs] == [
        "Must offer a refund",
        "Must quote the order id",
        "Must greet the user",
        "Must confirm the warehouse",
    ]
    assert len(llm.prompts) == 1
    assert llm.schemas[0]["properties"]["requirements"]["items"]["required"] == ["text", "spec_heading"]


async def test_extract_requirements_rejects_malformed_payload() -> None:
    llm = RecordingLLM([{}])
    with pytest.raises(ValueError, match="requirements list"):
        await extract_requirements("Must offer a refund", llm)


@pytest.mark.parametrize("row", [42, {}, {"text": ""}, {"text": "..."}, {"text": None}])
async def test_extract_requirements_rejects_malformed_rows(row: object) -> None:
    llm = RecordingLLM([{"requirements": [row]}])
    with pytest.raises(ValueError, match="non-empty text"):
        await extract_requirements("Must offer a refund", llm)


def test_with_requirements_does_not_mutate_original() -> None:
    llm = RecordingLLM([_extract_payload(), _call_a_payload(), _call_b_payload(), _call_a_payload(), _call_b_payload()])
    original = SpecGroundingMetric(llm=llm)
    bound = original.with_requirements(
        [
            "Must offer a refund",
            "Must quote the order id",
            "Must greet the user",
            "Must confirm the warehouse",
        ]
    )
    assert original.requirements == ()
    assert bound is not original
    assert len(bound.requirements) == 4

    cold = original.measure(_case())
    warm = bound.measure(_case())
    assert len(llm.prompts) == 5
    assert cold.value == 0.74
    assert warm.value == 0.74


def test_string_requirement_is_one_requirement_not_characters() -> None:
    llm = RecordingLLM(
        [
            {
                "requirements": [
                    {"text": "Must offer a refund", "applicable": True, "status": "satisfied", "evidence": "ok"}
                ]
            },
            {"claims": []},
        ]
    )
    metric = SpecGroundingMetric(llm=llm, requirements="Must offer a refund")
    score = metric.measure(_case())
    assert [row["text"] for row in score.metadata["requirements"]] == ["Must offer a refund"]


def test_with_requirements_preserves_factory_score_name() -> None:
    metric = build_metric(
        "llm",
        {"kind": "spec_grounding", "metadata": {"llm": object()}},
        score_name="policy_grounding",
    )
    bound = metric.with_requirements(["Must offer a refund"])
    assert metric.name == "policy_grounding"
    assert bound.name == "policy_grounding"


def test_missing_context_is_zero_without_llm() -> None:
    llm = RecordingLLM([])
    score = SpecGroundingMetric(llm=llm).measure(EvalCase(input="q", output="a"))
    assert score.value == 0.0
    assert "context[0]" in (score.reason or "")
    assert llm.prompts == []


def test_malformed_classifications_stay_in_unit_interval() -> None:
    llm = RecordingLLM(
        [
            {
                "requirements": [
                    {"text": "Must offer a refund", "applicable": True, "status": "nope"},
                    {"text": "unknown extra", "applicable": True, "status": "satisfied"},
                    {"text": "Must offer a refund", "applicable": True, "status": "satisfied"},
                ]
            },
            {
                "claims": [
                    {"text": "Refund issued", "status": "banana", "evidence": ""},
                    {"text": "Refund issued", "status": "contradicted", "evidence": ""},
                ]
            },
        ]
    )
    reqs = [ExtractedRequirement("Must offer a refund")]
    score = SpecGroundingMetric(llm=llm, requirements=reqs).measure(_case())
    assert 0.0 <= score.value <= 1.0
    assert score.metadata["requirements"][0]["status"] == "missing"
    assert score.metadata["claims"][0]["status"] == "unsupported"
    assert len(score.metadata["claims"]) == 1
    assert score.metadata["extra"]["call_a"] == {
        "unmatched_requirements": 0,
        "unmatched_judge_rows": 1,
        "normalized_text_matches": 0,
        "duplicate_judge_rows": 1,
    }


@pytest.mark.parametrize("payload", [{}, {"requirements": None}, []])
def test_malformed_classification_payload_fails_closed(payload: object) -> None:
    llm = RecordingLLM([payload])
    with pytest.raises(ValueError, match="Requirement classification response"):
        SpecGroundingMetric(llm=llm, requirements=["Must offer a refund"]).measure(_case())


def test_fatal_contradiction_matches_he1() -> None:
    llm = RecordingLLM(
        [
            {
                "requirements": [
                    {"text": "Must offer a refund", "applicable": True, "status": "satisfied", "evidence": "ok"},
                ]
            },
            {"claims": [{"text": "Never refunds", "status": "contradicted", "evidence": "spec allows refunds"}]},
        ]
    )
    score = SpecGroundingMetric(
        llm=llm,
        requirements=[ExtractedRequirement("Must offer a refund")],
        contradiction_is_fatal=True,
    ).measure(_case())
    assert score.value == 0.0
    assert score.metadata["capped"] is True
    assert score.metadata["contradiction_is_fatal"] is True


def test_out_of_scope_requirements_are_omitted() -> None:
    llm = RecordingLLM(
        [
            {
                "requirements": [
                    {"text": "Must offer a refund", "applicable": True, "status": "satisfied", "evidence": "ok"},
                    {"text": "Must greet the user", "applicable": False, "status": "missing"},
                ]
            },
            {"claims": [{"text": "Refund issued", "status": "supported", "evidence": "ok"}]},
        ]
    )
    score = SpecGroundingMetric(
        llm=llm,
        requirements=["Must offer a refund", "Must greet the user"],
    ).measure(_case())
    assert [row["text"] for row in score.metadata["requirements"]] == ["Must offer a refund"]
    assert "Return one row for every listed requirement" in llm.prompts[0]
    assert "never leave a row out" in llm.prompts[0]
    assert llm.schemas[0]["properties"]["requirements"]["items"]["required"] == [
        "text",
        "applicable",
        "status",
        "evidence",
        "spec_heading",
    ]


def test_requirement_echo_drift_is_normalized_and_diagnosed() -> None:
    llm = RecordingLLM(
        [
            {
                "requirements": [
                    {
                        "text": "  MUST   OFFER a REFUND. ",
                        "applicable": True,
                        "status": "satisfied",
                        "evidence": "ok",
                    }
                ]
            },
            {"claims": []},
        ]
    )
    score = SpecGroundingMetric(llm=llm, requirements=["Must offer a refund"]).measure(_case())
    assert score.metadata["requirements"][0]["status"] == "satisfied"
    assert score.metadata["extra"]["call_a"] == {
        "unmatched_requirements": 0,
        "unmatched_judge_rows": 0,
        "normalized_text_matches": 1,
        "duplicate_judge_rows": 0,
    }


def test_requirement_echo_strips_prompt_list_and_heading_wrapper() -> None:
    llm = RecordingLLM(
        [
            {
                "requirements": [
                    {
                        "text": "- Must offer a refund (heading: Refunds)",
                        "applicable": True,
                        "status": "satisfied",
                        "evidence": "ok",
                    }
                ]
            },
            {"claims": []},
        ]
    )
    score = SpecGroundingMetric(
        llm=llm,
        requirements=[ExtractedRequirement("Must offer a refund", "Refunds")],
    ).measure(_case())
    assert score.metadata["requirements"][0]["status"] == "satisfied"
    assert score.metadata["extra"]["call_a"]["unmatched_requirements"] == 0
    assert score.metadata["sub_scores"]["coverage"] == 1.0


def test_omitted_call_a_row_scores_missing_and_is_diagnosed() -> None:
    llm = RecordingLLM(
        [
            {
                "requirements": [
                    {
                        "text": "Must offer a refund",
                        "applicable": True,
                        "status": "satisfied",
                        "evidence": "ok",
                    }
                ]
            },
            {"claims": []},
        ]
    )
    score = SpecGroundingMetric(
        llm=llm,
        requirements=["Must offer a refund", "Must quote the order id"],
    ).measure(_case())
    assert score.metadata["extra"]["call_a"]["unmatched_requirements"] == 1
    by_text = {row["text"]: row for row in score.metadata["requirements"]}
    assert by_text["Must quote the order id"]["status"] == "missing"
    assert score.metadata["sub_scores"]["coverage"] == 0.5


def test_normalized_requirement_duplicates_keep_first() -> None:
    llm = RecordingLLM(
        [
            {
                "requirements": [
                    {"text": "Must offer a refund.", "applicable": True, "status": "missing"},
                    {"text": "MUST OFFER A REFUND", "applicable": True, "status": "satisfied"},
                ]
            },
            {"claims": []},
        ]
    )
    score = SpecGroundingMetric(llm=llm, requirements=["Must offer a refund"]).measure(_case())
    assert score.metadata["requirements"][0]["status"] == "missing"
    assert score.metadata["extra"]["call_a"]["duplicate_judge_rows"] == 1


@pytest.mark.parametrize("applicable", ["false", None, 0, 1])
def test_invalid_applicability_cannot_inflate_coverage(applicable: object) -> None:
    llm = RecordingLLM(
        [
            {
                "requirements": [
                    {
                        "text": "Must offer a refund",
                        "applicable": applicable,
                        "status": "satisfied",
                        "evidence": "ok",
                    }
                ]
            },
            {"claims": [{"text": "Refund issued", "status": "supported", "evidence": "ok"}]},
        ]
    )
    score = SpecGroundingMetric(llm=llm, requirements=["Must offer a refund"]).measure(_case())
    assert score.metadata["requirements"][0]["status"] == "missing"


def test_factory_and_catalog() -> None:
    entries = {entry.kind: entry for entry in catalog()}
    entry = entries["spec_grounding"]
    assert entry.name == "spec_grounding"
    assert entry.dimension == Dimension.GROUNDEDNESS
    assert entry.requires_llm is True
    assert entry.default_threshold == 0.7
    assert entry.category == "grounding"
    assert entry.metric_class is SpecGroundingMetric

    metric = build_metric(
        "llm",
        {
            "kind": "spec_grounding",
            "options": {
                "weights": {"coverage": 1.0, "faithfulness": 0.0, "consistency": 0.0},
                "contradiction_is_fatal": True,
            },
            "metadata": {"llm": object()},
        },
        threshold=0.8,
    )
    assert isinstance(metric, SpecGroundingMetric)
    assert metric.threshold == 0.8
    assert metric.contradiction_is_fatal is True
    assert metric.weights == {"coverage": 1.0, "faithfulness": 0.0, "consistency": 0.0}
    assert metric.requirements == ()


def test_factory_rejects_persisting_unknown_options() -> None:
    with pytest.raises(TypeError, match="unknown option"):
        build_metric(
            "llm",
            {
                "kind": "spec_grounding",
                "options": {"not_a_real_option": True},
                "metadata": {"llm": object()},
            },
        )


def test_factory_rejects_persisting_requirements() -> None:
    with pytest.raises(TypeError, match="runtime-only.*requirements"):
        build_metric(
            "llm",
            {
                "kind": "spec_grounding",
                "options": {"requirements": ["stale cached requirement"]},
                "metadata": {"llm": object()},
            },
        )


@pytest.mark.parametrize("value", ["false", "true", 0, 1])
def test_factory_rejects_non_boolean_fatal_flag(value: object) -> None:
    with pytest.raises(TypeError, match="must be a boolean"):
        build_metric(
            "llm",
            {
                "kind": "spec_grounding",
                "options": {"contradiction_is_fatal": value},
                "metadata": {"llm": object()},
            },
        )


def test_zero_faithfulness_and_consistency_skips_call_b() -> None:
    llm = RecordingLLM(
        [
            {
                "requirements": [
                    {"text": "Must offer a refund", "applicable": True, "status": "satisfied", "evidence": "ok"},
                ]
            }
        ]
    )
    score = SpecGroundingMetric(
        llm=llm,
        weights={"coverage": 1.0, "faithfulness": 0.0, "consistency": 0.0},
        requirements=["Must offer a refund"],
    ).measure(_case())
    assert len(llm.prompts) == 1
    assert score.value == 1.0
    assert score.metadata["claims"] == []


def test_fatal_flag_still_checks_claims_when_claim_weights_are_zero() -> None:
    llm = RecordingLLM(
        [
            {
                "requirements": [
                    {"text": "Must offer a refund", "applicable": True, "status": "satisfied", "evidence": "ok"}
                ]
            },
            {"claims": [{"text": "Refunds are forbidden", "status": "contradicted", "evidence": "refunds allowed"}]},
        ]
    )
    score = SpecGroundingMetric(
        llm=llm,
        weights={"coverage": 1.0, "faithfulness": 0.0, "consistency": 0.0},
        requirements=["Must offer a refund"],
        contradiction_is_fatal=True,
    ).measure(_case())
    assert len(llm.prompts) == 2
    assert score.value == 0.0
    assert score.metadata["capped"] is True


def test_zero_coverage_skips_extraction_and_call_a() -> None:
    llm = RecordingLLM([{"claims": [{"text": "Refund issued", "status": "supported", "evidence": "ok"}]}])
    score = SpecGroundingMetric(
        llm=llm,
        weights={"coverage": 0.0, "faithfulness": 1.0, "consistency": 0.0},
    ).measure(_case())
    assert len(llm.prompts) == 1
    assert "Extract factual claims" in llm.prompts[0]
    assert score.value == 1.0
    assert score.metadata["requirements"] == []


def test_empty_bound_requirements_do_not_trigger_extraction() -> None:
    llm = RecordingLLM([{"claims": [{"text": "Refund issued", "status": "supported", "evidence": "ok"}]}])
    score = SpecGroundingMetric(llm=llm, requirements=[]).measure(_case())
    assert len(llm.prompts) == 1
    assert "Extract factual claims" in llm.prompts[0]
    assert score.value == 1.0


def test_malformed_claim_payload_fails_closed() -> None:
    llm = RecordingLLM(
        [
            {
                "requirements": [
                    {"text": "Must offer a refund", "applicable": True, "status": "satisfied", "evidence": "ok"}
                ]
            },
            {},
        ]
    )
    with pytest.raises(ValueError, match="claims list"):
        SpecGroundingMetric(llm=llm, requirements=["Must offer a refund"]).measure(_case())


@pytest.mark.parametrize("row", [42, {}, {"text": ""}, {"text": None}])
def test_malformed_claim_rows_fail_closed(row: object) -> None:
    llm = RecordingLLM(
        [
            {
                "requirements": [
                    {"text": "Must offer a refund", "applicable": True, "status": "satisfied", "evidence": "ok"}
                ]
            },
            {"claims": [row]},
        ]
    )
    with pytest.raises(ValueError, match="non-empty text"):
        SpecGroundingMetric(llm=llm, requirements=["Must offer a refund"]).measure(_case())


def test_requirement_result_binding_reclassifies_text_for_each_case() -> None:
    llm = RecordingLLM([_call_a_payload(), _call_b_payload()])
    metric = SpecGroundingMetric(llm=llm).with_requirements(
        [
            RequirementResult(text="Must offer a refund", status="missing"),
            RequirementResult(text="Must quote the order id", status="missing"),
            RequirementResult(text="Must greet the user", status="missing"),
            RequirementResult(text="Must confirm the warehouse", status="missing"),
        ]
    )
    # Cached requirements supply text/headings only; Call A determines the
    # per-case applicability and status.
    assert metric.measure(_case()).value == 0.74


def test_normalized_claim_duplicates_keep_first() -> None:
    llm = RecordingLLM(
        [
            {
                "claims": [
                    {"text": "Refund issued", "status": "supported", "evidence": "ok"},
                    {"text": "refund issued.", "status": "unsupported", "evidence": "echo drift"},
                ]
            }
        ]
    )
    score = SpecGroundingMetric(
        llm=llm,
        weights={"coverage": 0.0, "faithfulness": 1.0, "consistency": 0.0},
    ).measure(_case())
    assert len(score.metadata["claims"]) == 1
    assert score.metadata["claims"][0]["text"] == "Refund issued"
    assert score.metadata["claims"][0]["status"] == "supported"
    assert score.metadata["sub_scores"]["faithfulness"] == 1.0
    assert score.value == 1.0


def test_claim_verdict_alias() -> None:
    llm = RecordingLLM(
        [
            {
                "requirements": [
                    {"text": "Must offer a refund", "applicable": True, "status": "satisfied", "evidence": "ok"}
                ]
            },
            {"claims": [{"text": "Refund issued", "verdict": "supported", "evidence": "ok"}]},
        ]
    )
    score = SpecGroundingMetric(llm=llm, requirements=["Must offer a refund"]).measure(_case())
    assert score.metadata["claims"][0]["status"] == "supported"
    assert llm.schemas[1]["properties"]["claims"]["items"]["required"] == ["text", "status", "evidence"]


@pytest.mark.parametrize("status", [None, ""])
def test_falsy_claim_status_falls_back_to_fatal_verdict(status: object) -> None:
    llm = RecordingLLM(
        [
            {
                "requirements": [
                    {"text": "Must offer a refund", "applicable": True, "status": "satisfied", "evidence": "ok"}
                ]
            },
            {
                "claims": [
                    {
                        "text": "Refunds are forbidden",
                        "status": status,
                        "verdict": "contradicted",
                        "evidence": "refunds allowed",
                    }
                ]
            },
        ]
    )
    score = SpecGroundingMetric(
        llm=llm,
        requirements=["Must offer a refund"],
        weights={"coverage": 1.0, "faithfulness": 0.0, "consistency": 0.0},
        contradiction_is_fatal=True,
    ).measure(_case())
    assert score.value == 0.0
    assert score.metadata["claims"][0]["status"] == "contradicted"
    assert score.metadata["capped"] is True
