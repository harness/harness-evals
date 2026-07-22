"""Tests for ConversationGolden and dataset I/O."""

import json

import pytest

from harness_evals.conversation import (
    ConversationGolden,
    load_conversation_dataset,
    save_conversation_dataset,
)
from harness_evals.core.types import Message


@pytest.mark.unit
class TestConversationGolden:
    def test_basic_creation(self):
        g = ConversationGolden(
            scenario="User asks about refund policy",
            expected_outcome="Agent provides refund timeline and process",
        )
        assert g.scenario == "User asks about refund policy"
        assert g.max_turns == 10
        assert g.max_elicitation_rounds == 10
        assert g.turns is None

    def test_with_turns(self):
        turns = [
            Message(role="user", content="How do I get a refund?"),
            Message(role="assistant", content="You can request a refund within 30 days."),
        ]
        g = ConversationGolden(
            scenario="Refund inquiry",
            expected_outcome="Refund process explained",
            turns=turns,
        )
        assert len(g.turns) == 2
        assert g.turns[0].role == "user"

    def test_to_dict(self):
        g = ConversationGolden(
            scenario="Test scenario",
            expected_outcome="Test outcome",
            max_turns=5,
            user_persona="Frustrated customer",
        )
        d = g.to_dict()
        assert d["scenario"] == "Test scenario"
        assert d["max_turns"] == 5
        assert "turns" not in d  # None fields excluded

    def test_to_dict_with_turns(self):
        g = ConversationGolden(
            scenario="Test",
            expected_outcome="Outcome",
            turns=[Message(role="user", content="hi")],
        )
        d = g.to_dict()
        assert d["turns"][0]["role"] == "user"
        assert d["turns"][0]["content"] == "hi"

    def test_from_dict(self):
        data = {
            "scenario": "Test",
            "expected_outcome": "Done",
            "max_turns": 8,
            "context": ["Background info"],
        }
        g = ConversationGolden.from_dict(data)
        assert g.scenario == "Test"
        assert g.max_turns == 8
        assert g.context == ["Background info"]

    def test_from_dict_with_turns(self):
        data = {
            "scenario": "Test",
            "expected_outcome": "Done",
            "turns": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ],
        }
        g = ConversationGolden.from_dict(data)
        assert len(g.turns) == 2
        assert isinstance(g.turns[0], Message)
        assert g.turns[1].content == "hi there"

    def test_from_dict_ignores_unknown_fields(self):
        data = {
            "scenario": "Test",
            "expected_outcome": "Done",
            "unknown_field": "ignored",
        }
        g = ConversationGolden.from_dict(data)
        assert g.scenario == "Test"

    def test_roundtrip(self):
        original = ConversationGolden(
            id="roundtrip",
            scenario="Roundtrip test",
            expected_outcome="Passes",
            context=["ctx1", "ctx2"],
            max_turns=7,
            max_elicitation_rounds=3,
            initial_prompt="Start here",
            user_persona="Developer",
            elicitation_hints={
                "intents": {"name": "testconnector"},
                "yaml": {"default_action": "accept"},
            },
            metadata={"key": "value"},
            tags={"env": "test"},
        )
        restored = ConversationGolden.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.scenario == original.scenario
        assert restored.max_turns == original.max_turns
        assert restored.max_elicitation_rounds == original.max_elicitation_rounds
        assert restored.initial_prompt == original.initial_prompt
        assert restored.elicitation_hints == original.elicitation_hints
        assert restored.metadata == original.metadata

    def test_rejects_invalid_max_turns(self):
        with pytest.raises(ValueError, match="max_turns"):
            ConversationGolden(scenario="S", expected_outcome="O", max_turns=0)

    def test_rejects_invalid_max_elicitation_rounds(self):
        with pytest.raises(ValueError, match="max_elicitation_rounds"):
            ConversationGolden(scenario="S", expected_outcome="O", max_elicitation_rounds=0)

    def test_rejects_non_dict_elicitation_hints(self):
        with pytest.raises(TypeError, match="elicitation_hints"):
            ConversationGolden(scenario="S", expected_outcome="O", elicitation_hints=[])  # type: ignore[arg-type]

    def test_rejects_invalid_sse_checks(self):
        with pytest.raises(TypeError, match="sse_checks"):
            ConversationGolden(scenario="S", expected_outcome="O", metadata={"sse_checks": "bad"})


@pytest.mark.unit
class TestConversationDatasetIO:
    def test_save_and_load_jsonl(self, tmp_path):
        dataset = [
            ConversationGolden(scenario="S1", expected_outcome="O1"),
            ConversationGolden(scenario="S2", expected_outcome="O2", max_turns=5),
        ]
        path = tmp_path / "test.jsonl"
        save_conversation_dataset(dataset, path)
        loaded = load_conversation_dataset(path)
        assert len(loaded) == 2
        assert loaded[0].scenario == "S1"
        assert loaded[1].max_turns == 5

    def test_save_and_load_json(self, tmp_path):
        dataset = [
            ConversationGolden(scenario="S1", expected_outcome="O1"),
        ]
        path = tmp_path / "test.json"
        save_conversation_dataset(dataset, path, format="json")
        loaded = load_conversation_dataset(path)
        assert len(loaded) == 1
        assert loaded[0].scenario == "S1"

    def test_load_invalid_json_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('{"not": "a list"}')
        with pytest.raises(ValueError, match="list"):
            load_conversation_dataset(path)

    def test_load_resolves_env_values(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PROJECT_NAME", "demo")
        path = tmp_path / "test.jsonl"
        path.write_text(
            json.dumps(
                {
                    "scenario": "Create resource in ${PROJECT_NAME}",
                    "expected_outcome": "Resource created",
                    "context": ["Project: ${PROJECT_NAME}"],
                }
            )
            + "\n"
        )

        loaded = load_conversation_dataset(path)

        assert loaded[0].scenario == "Create resource in demo"
        assert loaded[0].context == ["Project: demo"]

    def test_load_missing_env_value_raises(self, tmp_path):
        path = tmp_path / "test.jsonl"
        path.write_text(
            json.dumps(
                {
                    "scenario": "Create resource in ${MISSING_PROJECT_NAME}",
                    "expected_outcome": "Resource created",
                }
            )
            + "\n"
        )

        with pytest.raises(ValueError, match="MISSING_PROJECT_NAME"):
            load_conversation_dataset(path)

    def test_load_uses_inline_env_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HARNESS_DELEGATE_SELECTOR", raising=False)
        path = tmp_path / "test.jsonl"
        path.write_text(
            json.dumps(
                {
                    "scenario": "Create connector with ${HARNESS_DELEGATE_SELECTOR:-hello}",
                    "expected_outcome": "Connector created",
                }
            )
            + "\n"
        )

        loaded = load_conversation_dataset(path)

        assert loaded[0].scenario == "Create connector with hello"

    def test_load_inline_env_default_overridden_by_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HARNESS_DELEGATE_SELECTOR", "prod-delegate")
        path = tmp_path / "test.jsonl"
        path.write_text(
            json.dumps(
                {
                    "scenario": "Create connector with ${HARNESS_DELEGATE_SELECTOR:-hello}",
                    "expected_outcome": "Connector created",
                }
            )
            + "\n"
        )

        loaded = load_conversation_dataset(path)

        assert loaded[0].scenario == "Create connector with prod-delegate"
