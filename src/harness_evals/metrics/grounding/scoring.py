"""Pure spec-grounding arithmetic, reason rendering, and metadata.

The LLM classifies requirements and claims. This module turns those
classifications into a score in ``[0, 1]``, a deterministic explanation, and
structured metadata. There is no I/O and no LLM call here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RequirementStatus = Literal["satisfied", "missing", "violated"]
ClaimStatus = Literal["supported", "unsupported", "contradicted"]
WeightKey = Literal["coverage", "faithfulness", "consistency"]

REQUIREMENT_STATUSES: frozenset[str] = frozenset({"satisfied", "missing", "violated"})
CLAIM_STATUSES: frozenset[str] = frozenset({"supported", "unsupported", "contradicted"})
WEIGHT_KEYS: tuple[WeightKey, ...] = ("coverage", "faithfulness", "consistency")

DEFAULT_WEIGHTS: dict[str, float] = {
    "coverage": 0.4,
    "faithfulness": 0.3,
    "consistency": 0.3,
}

_SCORE_NAME = "spec_grounding"


@dataclass(frozen=True)
class RequirementResult:
    """One classified spec requirement."""

    text: str
    status: RequirementStatus
    evidence: str = ""
    spec_heading: str | None = None


@dataclass(frozen=True)
class ClaimResult:
    """One classified output claim."""

    text: str
    status: ClaimStatus
    evidence: str = ""


@dataclass(frozen=True)
class Improvement:
    action: str
    delta: float


@dataclass(frozen=True)
class Risk:
    action: str
    delta: float


@dataclass(frozen=True)
class SpecGroundingResult:
    """Deterministic spec-grounding score plus explanation inputs."""

    value: float
    sub_scores: dict[str, float]
    weights: dict[str, float]
    effective_weights: dict[str, float]
    applicable: int
    satisfied: int
    total_claims: int
    supported: int
    contradicted: int
    requirements: tuple[RequirementResult, ...]
    claims: tuple[ClaimResult, ...]
    sensitivity: dict[str, float | None]
    improvements: tuple[Improvement, ...]
    risks: tuple[Risk, ...]
    contradiction_is_fatal: bool = False
    capped: bool = False
    nothing_to_ground: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def validate_weights(weights: dict[str, float] | None) -> dict[str, float]:
    """Return a copy of ``weights``, defaulting only when ``weights`` is ``None``.

    Every supplied key must be one of coverage / faithfulness / consistency.
    Missing keys are allowed. Every value must be finite and ``>= 0``. The
    sum of supplied values (or defaults when ``weights`` is ``None``) must
    be ``> 0``.
    """
    if weights is None:
        return dict(DEFAULT_WEIGHTS)

    if not isinstance(weights, dict):
        raise TypeError("weights must be a dict")

    normalized: dict[str, float] = {}
    for key, raw in weights.items():
        if key not in WEIGHT_KEYS:
            raise ValueError(f"Unknown weight key {key!r}; expected one of {WEIGHT_KEYS}")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Weight {key!r} must be a number, got {raw!r}") from exc
        if value != value or value == float("inf") or value == float("-inf"):  # NaN / inf
            raise ValueError(f"Weight {key!r} must be a finite number, got {raw!r}")
        if value < 0.0:
            raise ValueError(f"Weight {key!r} must be >= 0, got {value}")
        normalized[key] = value

    if sum(normalized.values()) <= 0.0:
        raise ValueError("At least one weight must be > 0")
    return normalized


def compute_score(
    requirements: list[RequirementResult | RequirementStatus | str],
    claims: list[ClaimResult | ClaimStatus | str],
    weights: dict[str, float] | None = None,
    contradiction_is_fatal: bool = False,
) -> SpecGroundingResult:
    """Score classified requirements and claims.

    Bare status strings are accepted for arithmetic-only tests. Reason
    rendering requires :class:`RequirementResult` / :class:`ClaimResult`
    records with text (and typically evidence).
    """
    parsed_requirements = tuple(_coerce_requirement(item) for item in requirements)
    parsed_claims = tuple(_coerce_claim(item) for item in claims)
    validated_weights = validate_weights(weights)

    applicable = len(parsed_requirements)
    satisfied = sum(1 for req in parsed_requirements if req.status == "satisfied")
    coverage = (satisfied / applicable) if applicable else 0.0

    total_claims = len(parsed_claims)
    supported = sum(1 for claim in parsed_claims if claim.status == "supported")
    contradicted = sum(1 for claim in parsed_claims if claim.status == "contradicted")
    faithfulness = (supported / total_claims) if total_claims else 1.0
    consistency = (1.0 - contradicted / total_claims) if total_claims else 1.0

    sub_scores = {
        "coverage": coverage,
        "faithfulness": faithfulness,
        "consistency": consistency,
    }

    effective_weights = {key: weight for key, weight in validated_weights.items() if weight > 0.0}
    if applicable == 0:
        effective_weights.pop("coverage", None)
    if total_claims == 0:
        effective_weights.pop("faithfulness", None)
        effective_weights.pop("consistency", None)

    nothing_to_ground = applicable == 0 and total_claims == 0
    if nothing_to_ground:
        value = 0.0
    else:
        effective_w = sum(effective_weights.values())
        if effective_w > 0.0:
            value = sum(weight * sub_scores[key] for key, weight in effective_weights.items()) / effective_w
        else:
            value = 0.0

    capped = bool(contradiction_is_fatal and contradicted > 0)
    if capped:
        value = 0.0

    value = _clamp01(value)

    suppress_deltas = capped or nothing_to_ground
    effective_w = sum(effective_weights.values())
    sensitivity = _sensitivity(
        effective_weights=effective_weights,
        effective_w=effective_w,
        applicable=applicable,
        total_claims=total_claims,
        suppress=suppress_deltas,
    )
    improvements = () if suppress_deltas else _improvements(parsed_requirements, parsed_claims, sensitivity)
    risks = (
        ()
        if suppress_deltas
        else _risks(
            total_claims,
            supported,
            contradicted,
            sensitivity,
            contradiction_is_fatal=contradiction_is_fatal,
            value=value,
        )
    )

    return SpecGroundingResult(
        value=value,
        sub_scores=sub_scores,
        weights=validated_weights,
        effective_weights=effective_weights,
        applicable=applicable,
        satisfied=satisfied,
        total_claims=total_claims,
        supported=supported,
        contradicted=contradicted,
        requirements=parsed_requirements,
        claims=parsed_claims,
        sensitivity=sensitivity,
        improvements=improvements,
        risks=risks,
        contradiction_is_fatal=contradiction_is_fatal,
        capped=capped,
        nothing_to_ground=nothing_to_ground,
    )


def build_metadata(result: SpecGroundingResult) -> dict[str, Any]:
    """Structured per-spec payload nested later by the control plane."""
    return {
        "value": result.value,
        "sub_scores": dict(result.sub_scores),
        "weights": dict(result.weights),
        "effective_weights": dict(result.effective_weights),
        "requirements": [
            {
                "text": req.text,
                "status": req.status,
                "evidence": req.evidence,
                "spec_heading": req.spec_heading,
            }
            for req in result.requirements
        ],
        "claims": [
            {
                "text": claim.text,
                "status": claim.status,
                "evidence": claim.evidence,
            }
            for claim in result.claims
        ],
        "sensitivity": dict(result.sensitivity),
        "improvements": [{"action": item.action, "delta": item.delta} for item in result.improvements],
        "risks": [{"action": item.action, "delta": item.delta} for item in result.risks],
        "contradiction_is_fatal": result.contradiction_is_fatal,
        "capped": result.capped,
        "nothing_to_ground": result.nothing_to_ground,
        "extra": dict(result.extra),
    }


def render_reason(result: SpecGroundingResult) -> str:
    """Human-readable explanation derived only from counts and weights."""
    _require_reason_text(result)

    lines = [
        (
            f"{_SCORE_NAME} {_fmt(result.value)} — "
            f"coverage {_fmt(result.sub_scores['coverage'])}, "
            f"faithfulness {_fmt(result.sub_scores['faithfulness'])}, "
            f"consistency {_fmt(result.sub_scores['consistency'])}"
        )
    ]

    if result.capped:
        lines.append("Score is 0.0 because a claim contradicts the spec.")
        contradicted = [claim for claim in result.claims if claim.status == "contradicted"]
        if contradicted:
            lines.append("")
            for claim in contradicted:
                lines.append(_claim_line(claim))
        return "\n".join(lines)

    if result.nothing_to_ground:
        lines.append("Nothing to ground: no applicable requirements and no output claims.")
        return "\n".join(lines)

    lines.append("")
    if not result.effective_weights:
        lines.append(_empty_effective_weights_reason(result))
    else:
        dropped_line = _dropped_dimension_reason(result)
        if dropped_line:
            lines.append(dropped_line)
    if result.applicable == 0:
        lines.append("No applicable requirements.")
    else:
        lines.append(f"Covered {result.satisfied} of {result.applicable} applicable requirements:")
        for req in result.requirements:
            lines.append(_requirement_line(req))

    lines.append("")
    if result.total_claims == 0:
        lines.append("No factual claims were identified in the output.")
    else:
        flagged = [claim for claim in result.claims if claim.status != "supported"]
        header = f"{result.supported} of {result.total_claims} output claims are supported by the spec"
        lines.append(f"{header}:" if flagged else f"{header}.")
        for claim in flagged:
            lines.append(_claim_line(claim))

    increase_lines = _increase_lines(result)
    decrease_lines = _decrease_lines(result)
    if increase_lines or decrease_lines:
        lines.append("")
    if increase_lines:
        lines.extend(increase_lines)
    if decrease_lines:
        lines.extend(decrease_lines)

    return "\n".join(lines)


def _coerce_requirement(item: RequirementResult | RequirementStatus | str) -> RequirementResult:
    if isinstance(item, RequirementResult):
        if item.status not in REQUIREMENT_STATUSES:
            raise ValueError(f"Invalid requirement status {item.status!r}")
        return item
    if isinstance(item, str) and item in REQUIREMENT_STATUSES:
        return RequirementResult(text="", status=item)  # type: ignore[arg-type]
    raise TypeError(f"Expected RequirementResult or status string, got {item!r}")


def _coerce_claim(item: ClaimResult | ClaimStatus | str) -> ClaimResult:
    if isinstance(item, ClaimResult):
        if item.status not in CLAIM_STATUSES:
            raise ValueError(f"Invalid claim status {item.status!r}")
        return item
    if isinstance(item, str) and item in CLAIM_STATUSES:
        return ClaimResult(text="", status=item)  # type: ignore[arg-type]
    raise TypeError(f"Expected ClaimResult or status string, got {item!r}")


def _require_reason_text(result: SpecGroundingResult) -> None:
    missing_req = [req for req in result.requirements if not req.text.strip()]
    missing_claim = [claim for claim in result.claims if not claim.text.strip()]
    if missing_req or missing_claim:
        raise ValueError(
            "render_reason requires RequirementResult and ClaimResult records with text; "
            "bare status strings are arithmetic-only"
        )


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _sensitivity(
    *,
    effective_weights: dict[str, float],
    effective_w: float,
    applicable: int,
    total_claims: int,
    suppress: bool,
) -> dict[str, float | None]:
    if suppress or effective_w <= 0.0:
        return {
            "per_requirement": None,
            "per_claim_faithfulness": None,
            "per_claim_consistency": None,
        }

    per_requirement: float | None = None
    if applicable > 0 and "coverage" in effective_weights:
        per_requirement = effective_weights["coverage"] / effective_w / applicable

    per_claim_faithfulness: float | None = None
    per_claim_consistency: float | None = None
    if total_claims > 0:
        if "faithfulness" in effective_weights:
            per_claim_faithfulness = effective_weights["faithfulness"] / effective_w / total_claims
        if "consistency" in effective_weights:
            per_claim_consistency = effective_weights["consistency"] / effective_w / total_claims

    return {
        "per_requirement": per_requirement,
        "per_claim_faithfulness": per_claim_faithfulness,
        "per_claim_consistency": per_claim_consistency,
    }


def _improvements(
    requirements: tuple[RequirementResult, ...],
    claims: tuple[ClaimResult, ...],
    sensitivity: dict[str, float | None],
) -> tuple[Improvement, ...]:
    items: list[Improvement] = []
    per_req = sensitivity["per_requirement"]
    if per_req is not None:
        for req in requirements:
            if req.status == "missing":
                items.append(Improvement(action=f"Address missing requirement: {req.text}", delta=per_req))
            elif req.status == "violated":
                items.append(Improvement(action=f"Address violated requirement: {req.text}", delta=per_req))

    per_faith = sensitivity["per_claim_faithfulness"]
    per_cons = sensitivity["per_claim_consistency"]
    for claim in claims:
        if claim.status == "unsupported" and per_faith is not None:
            items.append(Improvement(action=f"Support unsupported claim: {claim.text}", delta=per_faith))
        elif claim.status == "contradicted":
            delta = (per_faith or 0.0) + (per_cons or 0.0)
            if delta > 0.0:
                items.append(Improvement(action=f"Resolve contradicted claim: {claim.text}", delta=delta))
    return tuple(items)


def _risks(
    total_claims: int,
    supported: int,
    contradicted: int,
    sensitivity: dict[str, float | None],
    *,
    contradiction_is_fatal: bool,
    value: float,
) -> tuple[Risk, ...]:
    remaining = total_claims - contradicted
    if remaining <= 0 or value <= 0.0:
        return ()
    if contradiction_is_fatal:
        return (Risk(action="Contradict a spec requirement", delta=-value),)
    per_faith = sensitivity["per_claim_faithfulness"] or 0.0
    per_cons = sensitivity["per_claim_consistency"] or 0.0
    # Flipping a supported claim loses faithfulness and consistency. Flipping an
    # already-unsupported claim only loses consistency. Already-contradicted claims
    # cannot drop the score further.
    delta = -(per_faith + per_cons) if supported > 0 else -per_cons
    if delta >= 0.0:
        return ()
    return (Risk(action="Contradict a spec requirement", delta=delta),)


def _empty_effective_weights_reason(result: SpecGroundingResult) -> str:
    only_coverage = (
        result.weights.get("coverage", 0.0) > 0.0
        and result.weights.get("faithfulness", 0.0) <= 0.0
        and result.weights.get("consistency", 0.0) <= 0.0
    )
    if only_coverage:
        return "Score is 0.0 because the only weighted dimension (coverage) has no applicable requirements."
    return "Score is 0.0 because no weighted dimension remains after dropping empty ones."


def _dropped_dimension_reason(result: SpecGroundingResult) -> str | None:
    """Explain a partial rollup when empty or zero-weight dimensions were dropped."""
    supplied = set(WEIGHT_KEYS)
    effective = set(result.effective_weights)
    if not effective or effective == supplied:
        return None
    scored = [key for key in WEIGHT_KEYS if key in effective]
    dropped = [key for key in WEIGHT_KEYS if key in supplied - effective]
    only = " only" if len(scored) == 1 else ""
    by_reason: dict[str, list[str]] = {}
    for key in dropped:
        if result.weights.get(key, 0.0) <= 0.0:
            why = "zero weight"
        elif key == "coverage":
            why = "no applicable requirements"
        else:
            why = "no output claims"
        by_reason.setdefault(why, []).append(key)
    excluded = "; ".join(
        f"{_join_en(keys)} {'is' if len(keys) == 1 else 'are'} excluded ({why})" for why, keys in by_reason.items()
    )
    return f"Scored on {_join_en(scored)}{only}; {excluded}."


def _join_en(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{names[0]}, {names[1]}, and {names[2]}"


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def _requirement_line(req: RequirementResult) -> str:
    if req.status == "satisfied":
        return f"  ✓ {req.text}"
    label = "MISSING" if req.status == "missing" else "VIOLATED"
    return f"  ✗ {label}  {req.text}"


def _claim_line(claim: ClaimResult) -> str:
    label = "UNSUPPORTED" if claim.status == "unsupported" else "CONTRADICTED"
    evidence = f" — {claim.evidence}" if claim.evidence else ""
    return f'  ! {label}  "{claim.text}"{evidence}'


def _increase_lines(result: SpecGroundingResult) -> list[str]:
    per_req = result.sensitivity["per_requirement"]
    missing = [req for req in result.requirements if req.status in {"missing", "violated"}]
    unsupported = [claim for claim in result.claims if claim.status == "unsupported"]
    contradicted = [claim for claim in result.claims if claim.status == "contradicted"]
    per_faith = result.sensitivity["per_claim_faithfulness"]
    per_cons = result.sensitivity["per_claim_consistency"]

    parts: list[str] = []
    if per_req is not None and missing:
        raised = result.value + per_req * len(missing)
        verb = "missing/violated requirement" if any(r.status == "violated" for r in missing) else "missing requirement"
        parts.append(f"each {verb} is worth +{_fmt(per_req)}; addressing all {len(missing)} → {_fmt(_clamp01(raised))}")
    if per_faith is not None and unsupported:
        noun = "claim" if len(unsupported) == 1 else "claims"
        parts.append(
            f"supporting the {len(unsupported)} unsupported {noun} is worth +{_fmt(per_faith * len(unsupported))}"
        )
    if (per_faith is not None or per_cons is not None) and contradicted:
        delta = (per_faith or 0.0) + (per_cons or 0.0)
        noun = "claim" if len(contradicted) == 1 else "claims"
        parts.append(
            f"resolving the {len(contradicted)} contradicted {noun} is worth +{_fmt(delta * len(contradicted))}"
        )
    if not parts:
        return []

    first, *rest = parts
    lines = [f"To increase: {first}."]
    for part in rest:
        lines.append(f"             {part}.")
    return lines


def _decrease_lines(result: SpecGroundingResult) -> list[str]:
    remaining = result.total_claims - result.contradicted
    if remaining <= 0 or result.value <= 0.0:
        return []
    if result.contradiction_is_fatal:
        return ["To decrease: any output claim that contradicts the spec drops the score to 0.00."]
    per_faith = result.sensitivity["per_claim_faithfulness"]
    per_cons = result.sensitivity["per_claim_consistency"]
    faith = per_faith or 0.0
    cons = per_cons or 0.0
    if result.supported > 0:
        combined = faith + cons
        if combined <= 0.0:
            return []
        breakdown = [
            f"-{_fmt(component)} {name}"
            for name, component, present in (
                ("faithfulness", faith, per_faith is not None),
                ("consistency", cons, per_cons is not None),
            )
            if present
        ]
        return [
            f"To decrease: each output claim that contradicts the spec costs -{_fmt(combined)}",
            f"             ({', '.join(breakdown)}).",
        ]
    if cons <= 0.0:
        return []
    return [
        f"To decrease: contradicting an unsupported claim costs -{_fmt(cons)} consistency.",
    ]
