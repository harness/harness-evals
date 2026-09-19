"""Tests for the built-in metric catalog."""

from __future__ import annotations

import pytest

from harness_evals.catalog import catalog

_DECISION_KINDS = {"decision_choice", "decision_score", "decision_noul"}


@pytest.mark.unit
def test_catalog_returns_entries():
    entries = catalog()
    assert entries
    kinds = {entry.kind for entry in entries}
    assert len(kinds) == len(entries), "catalog kinds must be unique"


@pytest.mark.unit
def test_decision_kinds_require_provider():
    entries = {entry.kind: entry for entry in catalog()}
    for kind in _DECISION_KINDS:
        assert kind in entries, f"missing catalog kind {kind!r}"
        entry = entries[kind]
        assert entry.requires_provider is True
        assert entry.requires_llm is False
        assert entry.requires_embedding is False


@pytest.mark.unit
def test_decision_kinds_excluded_from_heuristic_registry():
    from harness_evals.metrics.factory import _heuristic_registry

    heuristic = _heuristic_registry()
    assert not (_DECISION_KINDS & set(heuristic))


@pytest.mark.unit
def test_decision_kinds_in_decision_registry():
    from harness_evals.metrics.factory import _decision_registry

    decision = _decision_registry()
    assert set(decision) >= _DECISION_KINDS
