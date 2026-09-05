"""Spec-grounding metric — judge classification plus HE-1 arithmetic.

Receives one spec as ``EvalCase.context[0]``. It does not know about document
sources, spec IDs, or multi-spec bindings.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, replace

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM
from harness_evals.metrics.grounding.scoring import (
    CLAIM_STATUSES,
    REQUIREMENT_STATUSES,
    ClaimResult,
    ClaimStatus,
    RequirementResult,
    RequirementStatus,
    build_metadata,
    compute_score,
    render_reason,
    validate_weights,
)

_SCORE_NAME = "spec_grounding"

_EXTRACT_PROMPT = """Extract the atomic requirements from the specification below.
Do not use the user input or the assistant output. List only requirements the
specification itself states. Use Markdown ATX headings as optional grouping.

**Specification**:
{spec}

Respond with JSON:
{{"requirements": [{{"text": "requirement", "spec_heading": "optional heading or null"}}]}}
"""

_CALL_A_PROMPT = """You are classifying which extracted specification requirements apply to this
interaction, and whether the assistant output satisfies each applicable one.

Return one row for every listed requirement. Mark requirements that are out of
scope for this input with applicable: false — never leave a row out.
Do not invent requirements. Classify only the listed items.
Set text to the requirement wording only — do not copy the leading dash or
(heading: …) wrapper.

Statuses for applicable requirements:
- satisfied: the output meets the requirement
- missing: the output does not address the requirement
- violated: the output conflicts with the requirement

**Input**:
{input}

**Output**:
{output}

**Extracted requirements**:
{requirements}

Respond with JSON:
{{"requirements": [{{"text": "requirement text", "applicable": true, "status": "satisfied|missing|violated", "evidence": "short quote", "spec_heading": "optional"}}]}}
"""

_CALL_B_PROMPT = """Extract factual claims from the assistant output and classify each against
the specification.

Verdicts:
- supported: the specification confirms or implies the claim
- unsupported: the specification does not mention the claim
- contradicted: the specification states the opposite

**Specification**:
{spec}

**Output**:
{output}

Respond with JSON:
{{"claims": [{{"text": "claim", "status": "supported|unsupported|contradicted", "evidence": "short quote"}}]}}
"""

_EXTRACT_SCHEMA = {
    "type": "object",
    "required": ["requirements"],
    "properties": {
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["text", "spec_heading"],
                "properties": {
                    "text": {"type": "string"},
                    "spec_heading": {"type": ["string", "null"]},
                },
            },
        }
    },
}

_CALL_A_SCHEMA = {
    "type": "object",
    "required": ["requirements"],
    "properties": {
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["text", "applicable", "status", "evidence", "spec_heading"],
                "properties": {
                    "text": {"type": "string"},
                    "applicable": {"type": "boolean"},
                    "status": {"type": "string", "enum": ["satisfied", "missing", "violated"]},
                    "evidence": {"type": "string"},
                    "spec_heading": {"type": ["string", "null"]},
                },
            },
        }
    },
}

_CALL_B_SCHEMA = {
    "type": "object",
    "required": ["claims"],
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["text", "status", "evidence"],
                "properties": {
                    "text": {"type": "string"},
                    "status": {"type": "string", "enum": ["supported", "unsupported", "contradicted"]},
                    "evidence": {"type": "string"},
                },
            },
        }
    },
}


@dataclass(frozen=True)
class ExtractedRequirement:
    """A spec requirement before per-trace applicability classification."""

    text: str
    spec_heading: str | None = None


@dataclass(frozen=True)
class _CallAResult:
    requirements: list[RequirementResult]
    unmatched_requirements: int
    unmatched_judge_rows: int
    normalized_text_matches: int
    duplicate_judge_rows: int

    def metadata(self) -> dict[str, int]:
        return {
            "unmatched_requirements": self.unmatched_requirements,
            "unmatched_judge_rows": self.unmatched_judge_rows,
            "normalized_text_matches": self.normalized_text_matches,
            "duplicate_judge_rows": self.duplicate_judge_rows,
        }


def _requirement_text(item: object) -> str:
    if isinstance(item, str):
        return item.strip()
    text = getattr(item, "text", None)
    if isinstance(text, str):
        return text.strip()
    if isinstance(item, dict):
        return str(item.get("text") or "").strip()
    return str(item).strip()


def _requirement_heading(item: object) -> str | None:
    heading = getattr(item, "spec_heading", None)
    if isinstance(heading, str) and heading.strip():
        return heading.strip()
    if isinstance(item, dict):
        raw = item.get("spec_heading")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


_LIST_PREFIX = re.compile(r"^[-*]\s+")
_HEADING_SUFFIX = re.compile(r"\s*\(heading:.*$", re.IGNORECASE)


def _requirement_key(text: str) -> str:
    """Normalize judge-echoed requirement text for deterministic matching."""
    normalized = " ".join(text.split()).casefold()
    normalized = _LIST_PREFIX.sub("", normalized, count=1)
    normalized = _HEADING_SUFFIX.sub("", normalized)
    while normalized and unicodedata.category(normalized[-1]).startswith("P"):
        normalized = normalized[:-1].rstrip()
    return normalized


def normalize_extracted_requirements(
    requirements: Sequence[str | ExtractedRequirement | RequirementResult | dict[str, object]] | None,
) -> tuple[ExtractedRequirement, ...]:
    """Deduplicate requirements by normalized text; first occurrence wins."""
    if requirements is None:
        return ()
    items: Sequence[str | ExtractedRequirement | RequirementResult | dict[str, object]]
    items = (requirements,) if isinstance(requirements, str) else requirements
    seen: set[str] = set()
    normalized: list[ExtractedRequirement] = []
    for item in items:
        text = _requirement_text(item)
        key = _requirement_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        normalized.append(ExtractedRequirement(text=text, spec_heading=_requirement_heading(item)))
    return tuple(normalized)


async def extract_requirements(spec_text: str, llm: BaseLLM) -> tuple[ExtractedRequirement, ...]:
    """List requirements from spec text only (one judge call)."""
    result = await llm.generate_json(_EXTRACT_PROMPT.format(spec=spec_text), _EXTRACT_SCHEMA)
    raw = result.get("requirements") if isinstance(result, dict) else None
    if not isinstance(raw, list):
        raise ValueError("Requirement extraction response must contain a requirements list")
    if any(
        not isinstance(item, dict) or not isinstance(item.get("text"), str) or not _requirement_key(item["text"])
        for item in raw
    ):
        raise ValueError("Each extracted requirement must be an object with non-empty text")
    return normalize_extracted_requirements(raw)


class SpecGroundingMetric(BaseMetric):
    """Coverage, faithfulness, and consistency of output against one spec.

    Uses ``EvalCase.context[0]`` as the spec. Pre-extracted requirements skip
    the extraction judge call via ``requirements=`` or ``with_requirements``.
    """

    catalog_name = _SCORE_NAME
    factory_reserved_options = frozenset({"requirements"})

    def __init__(
        self,
        llm: BaseLLM,
        threshold: float = 0.7,
        weights: dict[str, float] | None = None,
        requirements: Sequence[str | ExtractedRequirement | RequirementResult] | None = None,
        contradiction_is_fatal: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(name=_SCORE_NAME, dimension=Dimension.GROUNDEDNESS, threshold=threshold, **kwargs)
        if not isinstance(contradiction_is_fatal, bool):
            raise TypeError("contradiction_is_fatal must be a boolean")
        self.llm = llm
        self.weights = validate_weights(weights)
        self.contradiction_is_fatal = contradiction_is_fatal
        self._requirements_bound = requirements is not None
        self.requirements = normalize_extracted_requirements(requirements)

    def with_requirements(
        self, requirements: Sequence[str | ExtractedRequirement | RequirementResult]
    ) -> SpecGroundingMetric:
        """Return a copy bound to pre-extracted requirements; do not mutate self."""
        bound = SpecGroundingMetric(
            llm=self.llm,
            threshold=self.threshold,
            weights=dict(self.weights),
            requirements=requirements,
            contradiction_is_fatal=self.contradiction_is_fatal,
        )
        bound.name = self.name
        return bound

    def measure(self, eval_case: EvalCase) -> Score:
        return _run_async(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        spec = _spec_text(eval_case)
        if spec is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="No spec provided in context[0] — cannot score spec grounding",
            )

        call_a: _CallAResult | None = None
        if self.weights.get("coverage", 0.0) > 0.0:
            extracted = self.requirements
            if not self._requirements_bound:
                extracted = await extract_requirements(spec, self.llm)
            call_a = await self._call_a(eval_case, extracted)
        claims = await self._call_b(eval_case, spec)

        result = compute_score(
            call_a.requirements if call_a else [],
            claims,
            weights=self.weights,
            contradiction_is_fatal=self.contradiction_is_fatal,
        )
        if call_a is not None:
            result = replace(result, extra={"call_a": call_a.metadata()})
        return Score(
            name=self.name,
            value=result.value,
            threshold=self.threshold,
            reason=render_reason(result),
            metadata=build_metadata(result),
        )

    async def _call_a(
        self,
        eval_case: EvalCase,
        extracted: tuple[ExtractedRequirement, ...],
    ) -> _CallAResult:
        if not extracted:
            return _CallAResult([], 0, 0, 0, 0)
        listed = "\n".join(
            f"- {item.text}" + (f" (heading: {item.spec_heading})" if item.spec_heading else "") for item in extracted
        )
        raw = await self.llm.generate_json(
            _CALL_A_PROMPT.format(
                input=_stringify(eval_case.input),
                output=_stringify(eval_case.output),
                requirements=listed,
            ),
            _CALL_A_SCHEMA,
        )
        if not isinstance(raw, dict):
            raise ValueError("Requirement classification response must be an object containing a requirements list")
        return _apply_call_a(extracted, raw)

    async def _call_b(self, eval_case: EvalCase, spec: str) -> list[ClaimResult]:
        faith = self.weights.get("faithfulness", 0.0)
        consist = self.weights.get("consistency", 0.0)
        if faith <= 0.0 and consist <= 0.0 and not self.contradiction_is_fatal:
            return []
        raw = await self.llm.generate_json(
            _CALL_B_PROMPT.format(spec=spec, output=_stringify(eval_case.output)),
            _CALL_B_SCHEMA,
        )
        if not isinstance(raw, dict):
            raise ValueError("Claim verification response must be an object containing a claims list")
        return _parse_claims(raw)


def _spec_text(eval_case: EvalCase) -> str | None:
    if not eval_case.context:
        return None
    spec = eval_case.context[0]
    if not isinstance(spec, str) or not spec.strip():
        return None
    return spec


def _stringify(value: object) -> str:
    if isinstance(value, str):
        return value
    return str(value)


def _apply_call_a(
    extracted: tuple[ExtractedRequirement, ...],
    raw: dict[str, object],
) -> _CallAResult:
    """Map judge rows onto extracted requirements.

    Matching is case/whitespace/trailing-punctuation insensitive. Missing rows
    become ``missing``. Unknown texts are ignored. Duplicate normalized texts
    keep the first row. Invalid statuses or applicability values become
    ``missing``. ``applicable: false`` omits the requirement from scoring.
    """
    rows = raw.get("requirements")
    if not isinstance(rows, list):
        raise ValueError("Requirement classification response must contain a requirements list")
    by_text: dict[str, dict[str, object]] = {}
    extracted_keys = {_requirement_key(req.text) for req in extracted}
    unmatched_judge_rows = 0
    duplicate_judge_rows = 0
    for item in rows:
        if not isinstance(item, dict):
            unmatched_judge_rows += 1
            continue
        text = _requirement_text(item)
        key = _requirement_key(text)
        if not key:
            unmatched_judge_rows += 1
            continue
        if key in by_text:
            duplicate_judge_rows += 1
            continue
        by_text[key] = item
        if key not in extracted_keys:
            unmatched_judge_rows += 1

    classified: list[RequirementResult] = []
    unmatched_requirements = 0
    normalized_text_matches = 0
    for req in extracted:
        row = by_text.get(_requirement_key(req.text))
        if row is None:
            unmatched_requirements += 1
            classified.append(
                RequirementResult(text=req.text, status="missing", evidence="", spec_heading=req.spec_heading)
            )
            continue
        if _requirement_text(row) != req.text:
            normalized_text_matches += 1
        applicable = row.get("applicable")
        if applicable is False:
            continue
        status = _requirement_status(row.get("status")) if applicable is True else "missing"
        heading = _requirement_heading(row) or req.spec_heading
        evidence = row.get("evidence")
        classified.append(
            RequirementResult(
                text=req.text,
                status=status,
                evidence=evidence.strip() if isinstance(evidence, str) else "",
                spec_heading=heading,
            )
        )
    return _CallAResult(
        classified,
        unmatched_requirements,
        unmatched_judge_rows,
        normalized_text_matches,
        duplicate_judge_rows,
    )


def _requirement_status(value: object) -> RequirementStatus:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in REQUIREMENT_STATUSES:
            return lowered  # type: ignore[return-value]
    return "missing"


def _parse_claims(raw: dict[str, object]) -> list[ClaimResult]:
    """Parse Call B claims. Invalid verdicts become ``unsupported``. Duplicate
    claim texts (after the same normalization as Call A) keep the first row.
    Missing/non-list payloads fail closed.
    """
    rows = raw.get("claims")
    if not isinstance(rows, list):
        raise ValueError("Claim verification response must contain a claims list")
    seen: set[str] = set()
    claims: list[ClaimResult] = []
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError("Each claim must be an object with non-empty text")
        raw_text = item.get("text")
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError("Each claim must be an object with non-empty text")
        text = raw_text.strip()
        key = _requirement_key(text)
        if key in seen:
            continue
        seen.add(key)
        evidence = item.get("evidence")
        claims.append(
            ClaimResult(
                text=text,
                status=_claim_status(item.get("status") or item.get("verdict")),
                evidence=evidence.strip() if isinstance(evidence, str) else "",
            )
        )
    return claims


def _claim_status(value: object) -> ClaimStatus:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in CLAIM_STATUSES:
            return lowered  # type: ignore[return-value]
    return "unsupported"
