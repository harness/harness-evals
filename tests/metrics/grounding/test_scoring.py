"""Pure spec-grounding arithmetic, reason rendering, and metadata."""

from __future__ import annotations

from dataclasses import replace

import pytest

from harness_evals.core.score import Score
from harness_evals.metrics.grounding import (
    ClaimResult,
    Improvement,
    RequirementResult,
    Risk,
    build_metadata,
    compute_score,
    render_reason,
    validate_weights,
)
from harness_evals.metrics.grounding.scoring import _clamp01, _join_en

# RFC worked example: coverage 0.50, faithfulness 0.80, consistency 1.00 → 0.74
_WORKED_REQUIREMENTS = [
    RequirementResult(text="Returns 429 with Retry-After on rate limit", status="satisfied"),
    RequirementResult(text="Error body includes a machine-readable code", status="satisfied"),
    RequirementResult(text="Retries use exponential backoff with jitter", status="missing"),
    RequirementResult(
        text="Idempotency-Key is required on POST /payments",
        status="missing",
    ),
]
_WORKED_CLAIMS = [
    ClaimResult(text="rate-limited clients receive 429", status="supported"),
    ClaimResult(text="error bodies include a code", status="supported"),
    ClaimResult(text="the API is JSON", status="supported"),
    ClaimResult(text="POST /payments is idempotent when keyed", status="supported"),
    ClaimResult(
        text="the default timeout is 30 seconds",
        status="unsupported",
        evidence="not stated in the spec",
    ),
]

pytestmark = pytest.mark.unit


def test_validate_weights_defaults() -> None:
    assert validate_weights(None) == {"coverage": 0.4, "faithfulness": 0.3, "consistency": 0.3}


def test_validate_weights_rejects_negative() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        validate_weights({"coverage": -0.1, "faithfulness": 1.0})


def test_validate_weights_rejects_all_zero() -> None:
    with pytest.raises(ValueError, match="At least one weight must be > 0"):
        validate_weights({"coverage": 0.0, "faithfulness": 0.0, "consistency": 0.0})


def test_validate_weights_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="Unknown weight key"):
        validate_weights({"coverage": 1.0, "style": 0.2})


def test_validate_weights_allows_missing_keys() -> None:
    assert validate_weights({"faithfulness": 1.0}) == {"faithfulness": 1.0}


def test_validate_weights_rejects_non_dict() -> None:
    with pytest.raises(TypeError, match="weights must be a dict"):
        validate_weights([0.4, 0.3, 0.3])  # type: ignore[arg-type]


def test_validate_weights_rejects_non_numeric() -> None:
    with pytest.raises(ValueError, match="must be a number"):
        validate_weights({"coverage": None})  # type: ignore[dict-item]


def test_validate_weights_rejects_nan_and_inf() -> None:
    with pytest.raises(ValueError, match="must be a finite number"):
        validate_weights({"coverage": float("nan")})
    with pytest.raises(ValueError, match="must be a finite number"):
        validate_weights({"coverage": float("inf")})


def test_worked_example_bare_statuses() -> None:
    result = compute_score(
        requirements=["satisfied", "satisfied", "missing", "missing"],
        claims=["supported"] * 4 + ["unsupported"],
    )
    assert result.value == pytest.approx(0.74)
    assert result.sub_scores == {
        "coverage": pytest.approx(0.5),
        "faithfulness": pytest.approx(0.8),
        "consistency": pytest.approx(1.0),
    }
    assert result.sensitivity == {
        "per_requirement": pytest.approx(0.1),
        "per_claim_faithfulness": pytest.approx(0.06),
        "per_claim_consistency": pytest.approx(0.06),
    }


def test_worked_example_reason_and_metadata() -> None:
    result = compute_score(_WORKED_REQUIREMENTS, _WORKED_CLAIMS)
    reason = render_reason(result)
    assert reason.splitlines()[0] == "spec_grounding 0.74 — coverage 0.50, faithfulness 0.80, consistency 1.00"
    assert "MISSING  Retries use exponential backoff with jitter" in reason
    assert "MISSING  Idempotency-Key is required on POST /payments" in reason
    assert 'UNSUPPORTED  "the default timeout is 30 seconds" — not stated in the spec' in reason
    assert "each missing requirement is worth +0.10; addressing all 2 → 0.94" in reason
    assert "supporting the 1 unsupported claim is worth +0.06" in reason
    assert "each output claim that contradicts the spec costs -0.12" in reason
    assert "-0.06 faithfulness, -0.06 consistency" in reason

    metadata = build_metadata(result)
    assert sorted(metadata) == [
        "capped",
        "claims",
        "contradiction_is_fatal",
        "effective_weights",
        "extra",
        "improvements",
        "nothing_to_ground",
        "requirements",
        "risks",
        "sensitivity",
        "sub_scores",
        "value",
        "weights",
    ]
    assert metadata["requirements"][0] == {
        "text": "Returns 429 with Retry-After on rate limit",
        "status": "satisfied",
        "evidence": "",
        "spec_heading": None,
    }
    assert metadata["value"] == pytest.approx(0.74)
    assert metadata["improvements"][0]["delta"] == pytest.approx(0.1)
    assert metadata["improvements"][-1]["action"].startswith("Support unsupported claim:")
    assert metadata["risks"][0]["delta"] == pytest.approx(-0.12)
    restored = (
        result.value
        + metadata["improvements"][0]["delta"]
        + metadata["improvements"][1]["delta"]
        + metadata["improvements"][2]["delta"]
    )
    assert restored == pytest.approx(0.74 + 0.1 + 0.1 + 0.06)
    assert metadata["extra"] == {}
    tagged = replace(result, extra={"spec_id": "S1"})
    assert build_metadata(tagged)["extra"] == {"spec_id": "S1"}


def test_render_reason_rejects_bare_status_records() -> None:
    result = compute_score(["satisfied"], ["supported"])
    with pytest.raises(ValueError, match="bare status strings are arithmetic-only"):
        render_reason(result)


def test_applicable_zero_renormalizes_over_remaining_weights() -> None:
    result = compute_score(
        requirements=[],
        claims=["supported", "supported", "unsupported"],
    )
    # coverage dropped; faithfulness 2/3, consistency 1.0 → (0.3*2/3 + 0.3*1) / 0.6 = 5/6
    assert "coverage" not in result.effective_weights
    assert result.sub_scores["coverage"] == 0.0
    assert result.value == pytest.approx((0.3 * (2 / 3) + 0.3 * 1.0) / 0.6)
    assert result.sensitivity["per_requirement"] is None
    assert result.sensitivity["per_claim_faithfulness"] == pytest.approx(0.3 / 0.6 / 3)
    reason = render_reason(
        compute_score(
            requirements=[],
            claims=[
                ClaimResult(text="p", status="supported"),
                ClaimResult(text="q", status="supported"),
                ClaimResult(text="s", status="unsupported"),
            ],
        )
    )
    assert "Scored on faithfulness and consistency; coverage is excluded (no applicable requirements)." in reason


def test_applicable_zero_coverage_only_weights_short_circuit() -> None:
    result = compute_score(
        requirements=[],
        claims=["supported"],
        weights={"coverage": 1.0, "faithfulness": 0.0, "consistency": 0.0},
    )
    assert result.value == 0.0
    assert result.effective_weights == {}
    reason = render_reason(
        compute_score(
            requirements=[],
            claims=[ClaimResult(text="ok", status="supported")],
            weights={"coverage": 1.0, "faithfulness": 0.0, "consistency": 0.0},
        )
    )
    assert "No applicable requirements" in reason
    assert "Score is 0.0 because the only weighted dimension (coverage) has no applicable requirements." in reason


def test_applicable_zero_omits_coverage_key_without_error() -> None:
    result = compute_score(
        requirements=[],
        claims=["supported"],
        weights={"faithfulness": 1.0},
    )
    assert result.value == pytest.approx(1.0)
    assert "coverage" not in result.effective_weights


def test_nothing_to_ground_is_zero_not_perfect() -> None:
    result = compute_score(requirements=[], claims=[])
    assert result.value == 0.0
    assert result.nothing_to_ground is True
    assert result.sub_scores["faithfulness"] == 1.0
    assert result.sub_scores["consistency"] == 1.0
    assert result.sensitivity == {
        "per_requirement": None,
        "per_claim_faithfulness": None,
        "per_claim_consistency": None,
    }
    assert result.improvements == ()
    assert result.risks == ()
    assert "Nothing to ground" in render_reason(result)


def test_no_claims_with_applicable_requirements_uses_coverage() -> None:
    result = compute_score(
        requirements=[
            RequirementResult(text="A", status="satisfied"),
            RequirementResult(text="B", status="missing"),
        ],
        claims=[],
    )
    assert result.sub_scores["faithfulness"] == 1.0
    assert result.sub_scores["consistency"] == 1.0
    assert "faithfulness" not in result.effective_weights
    assert "consistency" not in result.effective_weights
    assert result.value == pytest.approx(0.5)
    assert result.sensitivity["per_requirement"] == pytest.approx(0.5)
    assert result.sensitivity["per_claim_faithfulness"] is None
    assert result.sensitivity["per_claim_consistency"] is None
    reason = render_reason(result)
    assert "No factual claims were identified" in reason
    assert "Scored on coverage only; faithfulness and consistency are excluded (no output claims)." in reason
    assert "To decrease:" not in reason


def test_no_claims_unsatisfied_coverage_does_not_pass_default_threshold() -> None:
    result = compute_score(
        requirements=[
            RequirementResult(text="A", status="missing"),
            RequirementResult(text="B", status="missing"),
        ],
        claims=[],
    )
    assert result.value == 0.0
    score = Score(name="spec_grounding", value=result.value, threshold=0.7)
    assert score.passed is False


def test_contradiction_is_fatal_caps_value_and_omits_deltas() -> None:
    result = compute_score(
        requirements=[RequirementResult(text="Must not invent timeouts", status="satisfied")],
        claims=[
            ClaimResult(text="timeout is 30s", status="contradicted", evidence="spec says 10s"),
        ],
        contradiction_is_fatal=True,
    )
    assert result.value == 0.0
    assert result.capped is True
    assert result.sub_scores["consistency"] == 0.0
    assert result.improvements == ()
    assert result.risks == ()
    assert all(value is None for value in result.sensitivity.values())
    reason = render_reason(result)
    assert reason.startswith("spec_grounding 0.00")
    assert "Score is 0.0 because a claim contradicts the spec" in reason
    assert 'CONTRADICTED  "timeout is 30s" — spec says 10s' in reason
    assert "To increase:" not in reason
    metadata = build_metadata(result)
    assert metadata["improvements"] == []
    assert metadata["risks"] == []
    assert metadata["capped"] is True


def test_contradiction_is_fatal_pre_cap_risk_is_full_value() -> None:
    result = compute_score(
        requirements=[RequirementResult(text="A", status="satisfied")],
        claims=[ClaimResult(text="x", status="supported")],
        contradiction_is_fatal=True,
    )
    assert result.value == pytest.approx(1.0)
    assert result.capped is False
    assert len(result.risks) == 1
    assert isinstance(result.risks[0], Risk)
    assert result.risks[0].action == "Contradict a spec requirement"
    assert result.risks[0].delta == pytest.approx(-1.0)
    reason = render_reason(result)
    assert "any output claim that contradicts the spec drops the score to 0.00" in reason
    assert "faithfulness" not in reason.split("To decrease:")[-1]
    metadata = build_metadata(result)
    assert metadata["risks"][0]["action"] == "Contradict a spec requirement"
    assert metadata["risks"][0]["delta"] == pytest.approx(-1.0)


def test_contradiction_without_fatal_flag_does_not_cap() -> None:
    result = compute_score(
        requirements=[RequirementResult(text="timeouts are 10s", status="satisfied")],
        claims=[ClaimResult(text="timeout is 30s", status="contradicted")],
    )
    assert result.capped is False
    assert result.value == pytest.approx(0.4)
    assert result.risks == ()
    reason = render_reason(result)
    assert "To decrease:" not in reason


def test_unsupported_claim_risk_is_consistency_only() -> None:
    result = compute_score(
        requirements=[RequirementResult(text="A", status="satisfied")],
        claims=[ClaimResult(text="u", status="unsupported")],
    )
    assert result.value == pytest.approx(0.7)
    assert result.risks[0].delta == pytest.approx(-0.3)
    reason = render_reason(result)
    assert "contradicting an unsupported claim costs -0.30 consistency" in reason
    flipped = compute_score(
        requirements=[RequirementResult(text="A", status="satisfied")],
        claims=[ClaimResult(text="u", status="contradicted")],
    )
    assert flipped.value == pytest.approx(result.value + result.risks[0].delta)


def test_supported_claim_risk_includes_faithfulness_and_consistency() -> None:
    result = compute_score(
        requirements=[RequirementResult(text="A", status="satisfied")],
        claims=[ClaimResult(text="x", status="supported")],
    )
    assert result.value == pytest.approx(1.0)
    assert result.risks[0].delta == pytest.approx(-0.6)


def test_violated_requirement_is_applicable_not_satisfied() -> None:
    result = compute_score(
        requirements=[
            RequirementResult(text="Use jitter", status="violated", evidence="used constant backoff"),
            RequirementResult(text="Send Retry-After", status="satisfied"),
        ],
        claims=[ClaimResult(text="retried", status="supported")],
    )
    assert result.applicable == 2
    assert result.satisfied == 1
    assert result.sub_scores["coverage"] == pytest.approx(0.5)
    reason = render_reason(result)
    assert "VIOLATED  Use jitter" in reason
    assert isinstance(result.improvements[0], Improvement)
    assert result.improvements[0].action.startswith("Address violated requirement:")


def test_zero_weight_dimension_is_omitted_from_effective_weights() -> None:
    result = compute_score(
        ["satisfied"],
        ["unsupported"],
        weights={"coverage": 1.0, "faithfulness": 0.0, "consistency": 0.0},
    )
    assert result.effective_weights == {"coverage": 1.0}
    assert result.value == pytest.approx(1.0)
    assert result.sensitivity["per_claim_faithfulness"] is None
    assert result.sensitivity["per_claim_consistency"] is None
    assert result.risks == ()
    reason = render_reason(
        compute_score(
            [RequirementResult(text="A", status="satisfied")],
            [ClaimResult(text="u", status="unsupported")],
            weights={"coverage": 1.0, "faithfulness": 0.0, "consistency": 0.0},
        )
    )
    assert "Scored on coverage only; faithfulness and consistency are excluded (zero weight)." in reason


def test_omitted_weight_keys_are_explained_like_explicit_zeros() -> None:
    reason = render_reason(
        compute_score(
            [RequirementResult(text="Must not invent timeouts", status="satisfied")],
            [
                ClaimResult(
                    text="timeout is 30s",
                    status="contradicted",
                    evidence="spec says 10s",
                )
            ],
            weights={"coverage": 1.0},
        )
    )
    assert "Scored on coverage only; faithfulness and consistency are excluded (zero weight)." in reason
    assert 'CONTRADICTED  "timeout is 30s" — spec says 10s' in reason


def test_improvement_and_claim_order_follows_input() -> None:
    result = compute_score(
        [
            RequirementResult(text="second missing", status="missing"),
            RequirementResult(text="first satisfied", status="satisfied"),
            RequirementResult(text="violated item", status="violated"),
        ],
        [
            ClaimResult(text="unsupported later", status="unsupported"),
            ClaimResult(text="ok", status="supported"),
            ClaimResult(text="contradicted first-class", status="contradicted"),
        ],
    )
    actions = [item.action for item in result.improvements]
    assert actions == [
        "Address missing requirement: second missing",
        "Address violated requirement: violated item",
        "Support unsupported claim: unsupported later",
        "Resolve contradicted claim: contradicted first-class",
    ]


def test_score_value_is_safe_for_score_dataclass() -> None:
    result = compute_score(_WORKED_REQUIREMENTS, _WORKED_CLAIMS)
    score = Score(name="spec_grounding", value=result.value, threshold=0.7, reason=render_reason(result))
    assert score.value == pytest.approx(0.74)
    assert score.passed is True

    empty = compute_score([], [])
    clamped = Score(name="spec_grounding", value=empty.value, threshold=0.7)
    assert clamped.value == 0.0
    assert clamped.passed is False


def test_unknown_status_strings_are_rejected() -> None:
    with pytest.raises(TypeError, match="RequirementResult"):
        compute_score(["n/a"], ["supported"])
    with pytest.raises(TypeError, match="ClaimResult"):
        compute_score(["satisfied"], ["maybe"])


def test_invalid_status_on_result_records_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid requirement status"):
        compute_score(
            [RequirementResult(text="A", status="maybe")],  # type: ignore[arg-type]
            [],
        )
    with pytest.raises(ValueError, match="Invalid claim status"):
        compute_score(
            [],
            [ClaimResult(text="x", status="maybe")],  # type: ignore[arg-type]
        )


def test_faithfulness_only_with_no_claims_explains_empty_rollup() -> None:
    result = compute_score(
        [RequirementResult(text="A", status="satisfied")],
        [],
        weights={"faithfulness": 1.0},
    )
    assert result.value == 0.0
    assert result.effective_weights == {}
    reason = render_reason(result)
    assert "Score is 0.0 because no weighted dimension remains after dropping empty ones." in reason


def test_coverage_only_supported_claim_omits_decrease_lines() -> None:
    result = compute_score(
        [RequirementResult(text="A", status="satisfied")],
        [ClaimResult(text="x", status="supported")],
        weights={"coverage": 1.0, "faithfulness": 0.0, "consistency": 0.0},
    )
    assert result.value == pytest.approx(1.0)
    assert result.risks == ()
    assert "To decrease:" not in render_reason(result)


def test_claims_header_ends_the_sentence_when_every_claim_is_supported() -> None:
    result = compute_score(
        [RequirementResult(text="A", status="satisfied")],
        [ClaimResult(text="x", status="supported"), ClaimResult(text="y", status="supported")],
    )
    reason = render_reason(result)
    assert "2 of 2 output claims are supported by the spec." in reason
    assert "supported by the spec:" not in reason


def test_claims_header_introduces_the_list_when_a_claim_is_flagged() -> None:
    result = compute_score(
        [RequirementResult(text="A", status="satisfied")],
        [ClaimResult(text="x", status="supported"), ClaimResult(text="y", status="unsupported")],
    )
    lines = render_reason(result).splitlines()
    header = lines.index("1 of 2 output claims are supported by the spec:")
    assert 'UNSUPPORTED  "y"' in lines[header + 1]


def test_clamp01_and_join_en_helpers() -> None:
    assert _clamp01(-0.2) == 0.0
    assert _clamp01(1.2) == 1.0
    assert _clamp01(0.4) == 0.4
    assert _join_en(["coverage"]) == "coverage"
    assert _join_en(["faithfulness", "consistency"]) == "faithfulness and consistency"
    assert _join_en(["coverage", "faithfulness", "consistency"]) == ("coverage, faithfulness, and consistency")
