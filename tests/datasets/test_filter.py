"""Tests for golden id filtering."""

from __future__ import annotations

import pytest

from harness_evals.conversation.golden import ConversationGolden
from harness_evals.core.golden import Golden
from harness_evals.datasets.filter import (
    filter_goldens_by_ids,
    filter_goldens_by_tags,
    golden_matches_tag_filter,
    merge_golden_tag_filter,
    parse_golden_ids,
    parse_golden_tags,
    parse_modules,
    resolve_golden_id,
)
from harness_evals.errors import HarnessEvalsError


@pytest.mark.unit
class TestParseGoldenIds:
    def test_comma_separated_string(self) -> None:
        assert parse_golden_ids("a, b,c") == ["a", "b", "c"]

    def test_list(self) -> None:
        assert parse_golden_ids(["x", "y"]) == ["x", "y"]

    def test_none(self) -> None:
        assert parse_golden_ids(None) is None

    def test_empty_string_raises(self) -> None:
        with pytest.raises(HarnessEvalsError, match="at least one id"):
            parse_golden_ids("  ,  ")


@pytest.mark.unit
class TestResolveGoldenId:
    def test_conversation_golden_id(self) -> None:
        golden = ConversationGolden(id="case-1", scenario="s", expected_outcome="o")
        assert resolve_golden_id(golden) == "case-1"

    def test_golden_id_field(self) -> None:
        golden = Golden(input="x", id="row-1")
        assert resolve_golden_id(golden) == "row-1"

    def test_metadata_fallback(self) -> None:
        golden = Golden(input="x", metadata={"golden_id": "meta-1"})
        assert resolve_golden_id(golden) == "meta-1"


@pytest.mark.unit
class TestFilterGoldensByIds:
    def test_preserves_requested_order(self) -> None:
        goldens = [
            ConversationGolden(id="b", scenario="s", expected_outcome="o"),
            ConversationGolden(id="a", scenario="s", expected_outcome="o"),
        ]
        filtered = filter_goldens_by_ids(goldens, ["a", "b"])
        assert [g.id for g in filtered] == ["a", "b"]

    def test_missing_id_raises(self) -> None:
        goldens = [ConversationGolden(id="only-one", scenario="s", expected_outcome="o")]
        with pytest.raises(HarnessEvalsError, match="not found in dataset"):
            filter_goldens_by_ids(goldens, ["missing"])


@pytest.mark.unit
class TestParseModules:
    def test_comma_separated_string(self) -> None:
        assert parse_modules("ci, ce,cd") == ["ci", "ce", "cd"]


@pytest.mark.unit
class TestParseGoldenTags:
    def test_key_value_string(self) -> None:
        assert parse_golden_tags("scenario_type=write,environment=qa") == {
            "scenario_type": "write",
            "environment": "qa",
        }

    def test_dict_with_list_value(self) -> None:
        assert parse_golden_tags({"module": ["ci", "cd"]}) == {"module": ["ci", "cd"]}


@pytest.mark.unit
class TestFilterGoldensByTags:
    def _golden(self, module: str, scenario_type: str) -> ConversationGolden:
        return ConversationGolden(
            id=f"{module}-{scenario_type}",
            scenario="s",
            expected_outcome="o",
            tags={"module": module, "scenario_type": scenario_type},
        )

    def test_filters_by_module(self) -> None:
        goldens = [
            self._golden("ci", "write"),
            self._golden("ce", "write"),
            self._golden("cd", "write"),
        ]
        filtered = filter_goldens_by_tags(goldens, {"module": ["ci", "ce"]})
        assert [g.id for g in filtered] == ["ci-write", "ce-write"]

    def test_and_across_tag_keys(self) -> None:
        goldens = [
            self._golden("ci", "write"),
            self._golden("ci", "read"),
        ]
        filtered = filter_goldens_by_tags(goldens, {"module": "ci", "scenario_type": "write"})
        assert [g.id for g in filtered] == ["ci-write"]

    def test_no_matches_raises(self) -> None:
        goldens = [self._golden("ci", "write")]
        with pytest.raises(HarnessEvalsError, match="No goldens matched tag filter"):
            filter_goldens_by_tags(goldens, {"module": "ssca"})

    def test_merge_modules_into_tag_filter(self) -> None:
        merged = merge_golden_tag_filter(modules=["ci", "cd"], golden_tags={"scenario_type": "write"})
        assert merged == {"scenario_type": "write", "module": ["ci", "cd"]}
        assert golden_matches_tag_filter(self._golden("ci", "write"), merged)
        assert not golden_matches_tag_filter(self._golden("ci", "read"), merged)
