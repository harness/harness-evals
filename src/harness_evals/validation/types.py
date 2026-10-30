"""Versioned, payload-free contracts for configured metric admission."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from harness_evals.core.metric import EvaluationMode, MetricInputs, OnlineScope

if TYPE_CHECKING:
    from harness_evals.adapters.group import GroupDiagnostics

CONTRACT_VERSION = 1


def _check_json(value: Any, depth: int = 0) -> None:
    if depth > 64:
        raise ValueError("Evidence rule parameters exceed the supported nesting depth")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _check_json(item, depth + 1)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _check_json(item, depth + 1)
        return
    raise ValueError("Evidence rule parameters must contain finite JSON values")


class ExecutionKind(str, Enum):
    CASE = "case"
    DATASET = "dataset"
    REPEATED_RUNS = "repeated_runs"


class ReasonCode(str, Enum):
    UNKNOWN_METRIC = "unknown_metric"
    INVALID_CONFIGURATION = "invalid_configuration"
    UNREVIEWED_CONTRACT = "unreviewed_contract"
    UNSUPPORTED_MODE = "unsupported_mode"
    UNSUPPORTED_SCOPE = "unsupported_scope"
    UNSUPPORTED_EXECUTION = "unsupported_execution"
    MISSING_EVIDENCE = "missing_evidence"
    INVALID_EVIDENCE = "invalid_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    PARTIAL_EVIDENCE = "partial_evidence"
    UNKNOWN_CAPTURE = "unknown_capture"
    INCOMPLETE_CAPTURE = "incomplete_capture"
    AMBIGUOUS_ORDER = "ambiguous_order"
    AMBIGUOUS_TOOLS = "ambiguous_tools"
    AMBIGUOUS_USAGE = "ambiguous_usage"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_NOT_CHECKED = "provider_not_checked"


_REASON_MESSAGES = {
    ReasonCode.UNKNOWN_METRIC: "The metric kind is not supported by this SDK.",
    ReasonCode.INVALID_CONFIGURATION: "The selected metric configuration is invalid or unsupported.",
    ReasonCode.UNREVIEWED_CONTRACT: "This metric has no reviewed evidence contract.",
    ReasonCode.UNSUPPORTED_MODE: "The configured metric does not support this evaluation mode.",
    ReasonCode.UNSUPPORTED_SCOPE: "The configured metric does not support this evaluation scope.",
    ReasonCode.UNSUPPORTED_EXECUTION: "The metric requires a different execution kind.",
    ReasonCode.MISSING_EVIDENCE: "Required evidence was not observed.",
    ReasonCode.INVALID_EVIDENCE: "Observed evidence has an unsupported type or value.",
    ReasonCode.INSUFFICIENT_EVIDENCE: "The observed evidence is insufficient for this metric.",
    ReasonCode.PARTIAL_EVIDENCE: "Only part of the observed evidence can be evaluated.",
    ReasonCode.UNKNOWN_CAPTURE: "Capture completeness is unknown.",
    ReasonCode.INCOMPLETE_CAPTURE: "Complete capture is required for this comparison.",
    ReasonCode.AMBIGUOUS_ORDER: "The observed ordering is ambiguous.",
    ReasonCode.AMBIGUOUS_TOOLS: "Tool ownership or association is ambiguous.",
    ReasonCode.AMBIGUOUS_USAGE: "Observed usage contains unresolved ambiguity.",
    ReasonCode.PROVIDER_UNAVAILABLE: "A required scoring resource is unavailable.",
    ReasonCode.PROVIDER_NOT_CHECKED: "Scoring resource availability has not been checked.",
}


@dataclass(frozen=True)
class ValidationReason:
    code: ReasonCode
    severity: str = "error"
    requirement_id: str | None = None
    count: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", ReasonCode(self.code))
        if self.severity not in ("error", "warning"):
            raise ValueError("Unsupported reason severity")
        if self.requirement_id is not None and (
            not self.requirement_id.startswith("r") or any(char not in "r.0123456789" for char in self.requirement_id)
        ):
            raise ValueError("Requirement identifiers must be SDK-generated coordinates")
        if self.count is not None and (
            isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 0
        ):
            raise ValueError("Reason counts must be non-negative integers")

    @property
    def message(self) -> str:
        return _REASON_MESSAGES[self.code]

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "severity": self.severity,
            "message": self.message,
            "requirement_id": self.requirement_id,
            "count": self.count,
        }


@dataclass(frozen=True, init=False)
class EvidenceRequirement:
    """A JSON rule tree; parameters contain configuration, never case evidence."""

    op: str
    path: str | None
    predicate: str
    children: tuple[EvidenceRequirement, ...]
    _parameters_json: str = field(repr=False)

    def __init__(
        self,
        op: str = "field",
        path: str | None = None,
        predicate: str = "present",
        children: tuple[EvidenceRequirement, ...] = (),
        *,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        if type(op) is not str or type(predicate) is not str or (path is not None and type(path) is not str):
            raise ValueError("Unsupported evidence rule structure")
        if parameters is not None and type(parameters) is not dict:
            raise ValueError("Evidence rule parameters must be a JSON object")
        selected_children = tuple(children)
        if any(not isinstance(child, EvidenceRequirement) for child in selected_children):
            raise ValueError("Evidence rule children must be typed requirements")
        _check_json(parameters)
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "predicate", predicate)
        object.__setattr__(self, "children", selected_children)
        object.__setattr__(
            self,
            "_parameters_json",
            json.dumps(parameters or {}, sort_keys=True, allow_nan=False, separators=(",", ":")),
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return json.loads(self._parameters_json)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "path": self.path,
            "predicate": self.predicate,
            "children": [child.to_dict() for child in self.children],
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvidenceRequirement:
        if (
            type(value) is not dict
            or "op" not in value
            or set(value) - {"op", "path", "predicate", "children", "parameters"}
        ):
            raise ValueError("Unsupported evidence rule structure")
        if type(value.get("children", [])) is not list:
            raise ValueError("Evidence rule children must be a JSON array")
        return cls(
            op=value["op"],
            path=value.get("path"),
            predicate=value.get("predicate", "present"),
            children=tuple(cls.from_dict(child) for child in value.get("children", [])),
            parameters=value.get("parameters"),
        )


@dataclass(frozen=True)
class ConfiguredMetricContract:
    """SDK-resolved selected configuration, not a catalog possibility.

    Resolve on the server; deserializing caller-supplied contracts does not
    establish that a configuration was reviewed or authorize execution.
    """

    kind: str
    inputs: MetricInputs
    requirements: EvidenceRequirement
    modes: frozenset[EvaluationMode] = frozenset({EvaluationMode.OFFLINE})
    online_scopes: frozenset[OnlineScope] = frozenset()
    execution: ExecutionKind = ExecutionKind.CASE
    min_runs: int = 0
    resources: tuple[str, ...] = ()
    group_compatible: bool = False
    group_requires_complete: bool = False
    group_requires_order: bool = False
    reviewed: bool = True
    reasons: tuple[ValidationReason, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not str or not isinstance(self.inputs, MetricInputs):
            raise ValueError("Unsupported configured metric contract structure")
        if not isinstance(self.requirements, EvidenceRequirement):
            raise ValueError("Evidence requirements must be typed")
        if type(self.min_runs) is not int or self.min_runs < 0:
            raise ValueError("Repeated-run minimum must be a non-negative integer")
        if any(
            type(value) is not bool
            for value in (self.group_compatible, self.group_requires_complete, self.group_requires_order, self.reviewed)
        ):
            raise ValueError("Contract policy flags must be boolean")
        if any(type(resource) is not str for resource in self.resources):
            raise ValueError("Resource names must be strings")
        if any(not isinstance(reason, ValidationReason) for reason in self.reasons):
            raise ValueError("Contract reasons must be typed")
        object.__setattr__(self, "modes", frozenset(EvaluationMode(mode) for mode in self.modes))
        object.__setattr__(self, "online_scopes", frozenset(OnlineScope(scope) for scope in self.online_scopes))
        object.__setattr__(self, "execution", ExecutionKind(self.execution))
        object.__setattr__(self, "resources", tuple(sorted(set(self.resources))))
        object.__setattr__(self, "reasons", tuple(self.reasons))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONTRACT_VERSION,
            "description_kind": "configured",
            "kind": self.kind,
            "projection": {
                "required": sorted(value.value for value in self.inputs.required),
                "optional": sorted(value.value for value in self.inputs.optional),
                "any_of": [sorted(value.value for value in group) for group in self.inputs.any_of],
            },
            "requirements": self.requirements.to_dict(),
            "modes": sorted(mode.value for mode in self.modes),
            "online_scopes": sorted(scope.value for scope in self.online_scopes),
            "execution": self.execution.value,
            "min_runs": self.min_runs,
            "resources": sorted(self.resources),
            "group_compatible": self.group_compatible,
            "group_requires_complete": self.group_requires_complete,
            "group_requires_order": self.group_requires_order,
            "reviewed": self.reviewed,
            "reasons": [reason.to_dict() for reason in self.reasons],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ConfiguredMetricContract:
        if (
            type(value) is not dict
            or type(value.get("schema_version")) is not int
            or value.get("schema_version") != CONTRACT_VERSION
            or value.get("description_kind") != "configured"
        ):
            raise ValueError("Unsupported configured metric contract version or description kind")
        projection = value["projection"]
        if (
            type(projection) is not dict
            or any(type(projection.get(name)) is not list for name in ("required", "optional", "any_of"))
            or any(type(group) is not list for group in projection["any_of"])
            or any(type(value.get(name)) is not list for name in ("modes", "online_scopes", "resources", "reasons"))
        ):
            raise ValueError("Unsupported configured metric JSON structure")
        return cls(
            kind=value["kind"],
            inputs=MetricInputs(
                required=projection["required"], optional=projection["optional"], any_of=projection.get("any_of", ())
            ),
            requirements=EvidenceRequirement.from_dict(value["requirements"]),
            modes=frozenset(EvaluationMode(mode) for mode in value["modes"]),
            online_scopes=frozenset(OnlineScope(scope) for scope in value["online_scopes"]),
            execution=ExecutionKind(value["execution"]),
            min_runs=value["min_runs"],
            resources=tuple(value["resources"]),
            group_compatible=value["group_compatible"],
            group_requires_complete=value["group_requires_complete"],
            group_requires_order=value["group_requires_order"],
            reviewed=value["reviewed"],
            reasons=tuple(
                ValidationReason(
                    ReasonCode(reason["code"]), reason["severity"], reason.get("requirement_id"), reason.get("count")
                )
                for reason in value["reasons"]
            ),
        )


@dataclass(frozen=True)
class ValidationContext:
    mode: EvaluationMode = EvaluationMode.OFFLINE
    scope: OnlineScope | None = None
    execution: ExecutionKind = ExecutionKind.CASE
    available_resources: frozenset[str] | None = None
    group_diagnostics: GroupDiagnostics | None = None
    outcomes: tuple[bool, ...] | None = None


@dataclass(frozen=True)
class ValidationResult:
    eligible: bool
    evidence_checked: bool
    reasons: tuple[ValidationReason, ...] = ()

    @property
    def ready(self) -> bool:
        return self.eligible and self.evidence_checked

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONTRACT_VERSION,
            "eligible": self.eligible,
            "evidence_checked": self.evidence_checked,
            "ready": self.ready,
            "reasons": [reason.to_dict() for reason in self.reasons],
        }
