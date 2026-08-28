"""Tests for golden id filtering."""

from __future__ import annotations

import pytest

from harness_evals.conversation.golden import ConversationGolden
from harness_evals.core.golden import Golden
from harness_evals.datasets.filter import (
    filter_goldens_by_ids,
    filter_goldens_by_tags,
    golden_matches_tag_filter,
    intersect_golden_tag_filters,
    intersect_string_filters,
    merge_golden_tag_filter,
    parse_golden_ids,
    parse_golden_tags,
    parse_modules,
    resolve_golden_id,
    resolve_golden_tags,
)
from harness_evals.errors import HarnessEvalsError


@pytest.mark.unit
class TestParseGoldenIds:
    def test_comma_separated_string(self) -> None:
        assert parse_golden_ids("a, b,c") == ["a", "b", "c"]

    def test_list(self) -> None:
        assert parse_golden_ids(["x", "y"]) == ["x", "y"]

    def test_duplicates_are_removed_preserving_order(self) -> None:
        assert parse_golden_ids(["x", "x", "y"]) == ["x", "y"]

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

    def test_langfuse_metadata_fallback(self) -> None:
        golden = Golden(input="x", metadata={"langfuse_dataset_item_id": "item-1"})
        assert resolve_golden_id(golden) == "item-1"


@pytest.mark.unit
class TestFilterGoldensByIds:
    def test_preserves_requested_order(self) -> None:
        goldens = [
            ConversationGolden(id="b", scenario="s", expected_outcome="o"),
            ConversationGolden(id="a", scenario="s", expected_outcome="o"),
        ]
        filtered = filter_goldens_by_ids(goldens, ["a", "b"])
        assert [g.id for g in filtered] == ["a", "b"]

    def test_duplicate_requested_ids_run_once(self) -> None:
        goldens = [ConversationGolden(id="a", scenario="s", expected_outcome="o")]
        filtered = filter_goldens_by_ids(goldens, ["a", "a"])
        assert [g.id for g in filtered] == ["a"]

    def test_missing_id_raises(self) -> None:
        goldens = [ConversationGolden(id="only-one", scenario="s", expected_outcome="o")]
        with pytest.raises(HarnessEvalsError, match="not found in dataset"):
            filter_goldens_by_ids(goldens, ["missing"])

    def test_duplicate_id_raises_when_requested(self) -> None:
        goldens = [
            ConversationGolden(id="dup", scenario="first", expected_outcome="o"),
            ConversationGolden(id="dup", scenario="second", expected_outcome="o"),
        ]
        with pytest.raises(HarnessEvalsError, match="Duplicate golden id 'dup'"):
            filter_goldens_by_ids(goldens, ["dup"])

    def test_unrelated_duplicate_id_does_not_block_requested_filter(self) -> None:
        goldens = [
            ConversationGolden(id="dup", scenario="first", expected_outcome="o"),
            ConversationGolden(id="dup", scenario="second", expected_outcome="o"),
            ConversationGolden(id="row-b", scenario="keep", expected_outcome="o"),
        ]
        filtered = filter_goldens_by_ids(goldens, ["row-b"])
        assert [g.id for g in filtered] == ["row-b"]


@pytest.mark.unit
class TestParseModules:
    def test_comma_separated_string(self) -> None:
        assert parse_modules("ci, ce,cd") == ["ci", "ce", "cd"]

    def test_empty_string_raises_with_modules_label(self) -> None:
        with pytest.raises(HarnessEvalsError, match="'modules' must contain at least one module"):
            parse_modules("  ,  ")

    def test_invalid_type_raises_with_modules_label(self) -> None:
        with pytest.raises(HarnessEvalsError, match="'modules' must be a comma-separated string"):
            parse_modules(123)  # type: ignore[arg-type]

    def test_cli_values_restrict_configured_values(self) -> None:
        assert intersect_string_filters(["ci", "cd"], ["cd", "ce"], field_name="modules") == ["cd"]

    def test_disjoint_cli_values_raise(self) -> None:
        with pytest.raises(HarnessEvalsError, match="no values in common"):
            intersect_string_filters(["ci"], ["cd"], field_name="modules")


@pytest.mark.unit
class TestParseGoldenTags:
    def test_key_value_string(self) -> None:
        assert parse_golden_tags("scenario_type=write,environment=qa") == {
            "scenario_type": "write",
            "environment": "qa",
        }

    def test_dict_with_list_value(self) -> None:
        assert parse_golden_tags({"module": ["ci", "cd"]}) == {"module": ["ci", "cd"]}

    def test_duplicate_keys_aggregate_to_list(self) -> None:
        assert parse_golden_tags("module=ci,module=cd") == {"module": ["ci", "cd"]}

    def test_missing_equals_raises(self) -> None:
        with pytest.raises(HarnessEvalsError, match="key=value"):
            parse_golden_tags("module")

    @pytest.mark.parametrize("value", [{"module": None}, {"module": ""}, {"module": "  "}])
    def test_dict_empty_value_raises(self, value) -> None:
        with pytest.raises(HarnessEvalsError, match=r"golden_tags\.module.*non-empty"):
            parse_golden_tags(value)

    def test_empty_dict_raises(self) -> None:
        with pytest.raises(HarnessEvalsError, match="at least one key/value pair"):
            parse_golden_tags({})

    def test_cli_tags_add_constraints_and_intersect_existing_values(self) -> None:
        assert intersect_golden_tag_filters(
            {"module": ["ci", "cd"], "environment": "qa"},
            {"module": ["cd", "ce"], "scenario_type": "write"},
        ) == {
            "module": "cd",
            "environment": "qa",
            "scenario_type": "write",
        }


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

    def test_resolves_metadata_tags(self) -> None:
        golden = Golden(input="x", metadata={"tags": {"module": "ci"}})
        assert golden_matches_tag_filter(golden, {"module": "ci"})

    def test_merges_golden_tags_with_metadata_module(self) -> None:
        golden = Golden(
            input="x",
            tags={"scenario_type": "write"},
            metadata={"module": "ce"},
        )
        assert resolve_golden_tags(golden) == {"module": "ce", "scenario_type": "write"}
        assert golden_matches_tag_filter(golden, {"module": "ce", "scenario_type": "write"})

    def test_golden_tags_win_over_metadata_on_key_collision(self) -> None:
        golden = Golden(
            input="x",
            tags={"module": "ci"},
            metadata={"module": "ce"},
        )
        assert resolve_golden_tags(golden) == {"module": "ci"}

    def test_merge_modules_into_tag_filter(self) -> None:
        merged = merge_golden_tag_filter(modules=["ci", "cd"], golden_tags={"scenario_type": "write"})
        assert merged == {"scenario_type": "write", "module": ["ci", "cd"]}
        assert golden_matches_tag_filter(self._golden("ci", "write"), merged)
        assert not golden_matches_tag_filter(self._golden("ci", "read"), merged)

    def test_merge_intersects_overlapping_module_filters(self) -> None:
        merged = merge_golden_tag_filter(modules=["ci", "cd"], golden_tags={"module": "cd"})
        assert merged == {"module": "cd"}
        assert not golden_matches_tag_filter(self._golden("ci", "write"), merged)
        assert golden_matches_tag_filter(self._golden("cd", "write"), merged)

    def test_merge_rejects_conflicting_module_filters(self) -> None:
        with pytest.raises(HarnessEvalsError, match="no values in common"):
            merge_golden_tag_filter(modules=["ci"], golden_tags={"module": "cd"})
