"""Stored contracts remain typed, JSON-safe, and independent snapshots."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from harness_evals.validation import (
    ConfiguredMetricContract,
    EvidenceRequirement,
    ReasonCode,
    ValidationReason,
    resolve_metric_contract,
)

pytestmark = pytest.mark.unit


def test_stored_selected_contract_round_trip_preserves_alternatives_and_rules() -> None:
    contract = resolve_metric_contract("hallucination", {"include_messages_as_reference": True})
    assert contract.reviewed and contract.inputs.any_of
    stored = json.loads(json.dumps(contract.to_dict(), allow_nan=False))
    restored = ConfiguredMetricContract.from_dict(stored)
    assert restored == contract
    stored["requirements"]["children"].clear()
    stored["projection"]["any_of"].clear()
    assert restored == contract and restored.requirements.children


@pytest.mark.parametrize("version", [True, False, 1.0, "1", None, 0, 2])
def test_contract_version_is_an_exact_supported_integer(version: Any) -> None:
    stored = resolve_metric_contract("latency").to_dict()
    stored["schema_version"] = version
    with pytest.raises(ValueError, match="version"):
        ConfiguredMetricContract.from_dict(stored)


@pytest.mark.parametrize("name", ["modes", "online_scopes", "resources", "reasons"])
def test_serialized_arrays_do_not_accept_objects_or_strings(name: str) -> None:
    stored = resolve_metric_contract("latency").to_dict()
    stored[name] = {}
    with pytest.raises(ValueError, match="JSON structure"):
        ConfiguredMetricContract.from_dict(stored)


def test_catalog_description_cannot_masquerade_as_a_selected_contract() -> None:
    stored = resolve_metric_contract("latency").to_dict()
    stored["description_kind"] = "catalog"
    with pytest.raises(ValueError, match="description kind"):
        ConfiguredMetricContract.from_dict(stored)


@pytest.mark.parametrize("path", [True, 1, float("nan"), [], {}])
def test_non_string_rule_paths_cannot_escape_in_json(path: Any) -> None:
    with pytest.raises(ValueError, match="rule structure"):
        EvidenceRequirement(path=path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {1: "coerced"}, ("tuple",), object()])
def test_rule_parameters_reject_non_json_nested_values(value: Any) -> None:
    with pytest.raises(ValueError, match="finite JSON"):
        EvidenceRequirement(parameters={"nested": [value]})


def test_parameters_are_snapshotted_on_input_and_each_read() -> None:
    parameters = {"names": ["initial"]}
    requirement = EvidenceRequirement(parameters=parameters)
    parameters["names"].append("source mutation")
    first = requirement.parameters
    first["names"].append("output mutation")
    assert requirement.parameters == {"names": ["initial"]}
    assert json.loads(json.dumps(requirement.to_dict(), allow_nan=False))["parameters"] == {"names": ["initial"]}


@pytest.mark.parametrize("name", ["reviewed", "group_compatible", "group_requires_complete", "group_requires_order"])
def test_policy_flags_do_not_accept_truthy_non_booleans(name: str) -> None:
    with pytest.raises(ValueError, match="boolean"):
        replace(resolve_metric_contract("latency"), **{name: 1})


def test_reason_messages_are_not_taken_from_stored_json() -> None:
    stored = resolve_metric_contract("not_a_kind").to_dict()
    stored["reasons"][0]["message"] = "caller-controlled source payload"
    restored = ConfiguredMetricContract.from_dict(stored)
    assert "caller-controlled" not in json.dumps(restored.to_dict())
    assert restored.reasons[0].code is ReasonCode.UNKNOWN_METRIC


@pytest.mark.parametrize("coordinate", ["metadata.customer", "r.secret", "r[0]", "source-id"])
def test_reason_coordinates_cannot_echo_source_paths_or_identifiers(coordinate: str) -> None:
    with pytest.raises(ValueError, match="coordinates"):
        ValidationReason(ReasonCode.MISSING_EVIDENCE, requirement_id=coordinate)
