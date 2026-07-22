"""Tests for HarnessSseElicitationAdapter."""

import logging

import pytest
from examples.harness_sse_elicitation_adapter import HarnessSseElicitationAdapter

from harness_evals.conversation import ConversationGolden, ConversationSimulator
from harness_evals.conversation.human_input import HumanInputSimulator, PendingHumanInput
from harness_evals.core.types import Message
from harness_evals.llm.base import BaseLLM

_INTENT_MISS_LOGGER = "harness_evals.conversation.human_input"


def _golden() -> ConversationGolden:
    return ConversationGolden(
        scenario="Create a k8s connector",
        expected_outcome="Connector testconnector created",
        elicitation_hints={
            "intents": {
                "auth_method": "Inherit from Delegate",
                "scope": "Project (Recommended)",
                "project_id": "AICHAT",
                "connector_name": "testconnector",
                "delegate_selector": "hello",
            },
            "matchers": [
                {"intent": "auth_method", "question_contains": ["auth", "authentication"]},
                {"intent": "project_id", "question_contains": ["which project", "project in the", "project should"]},
                {"intent": "scope", "question_contains": ["what scope", "at what scope", "scope should"]},
                {"intent": "delegate_selector", "question_contains": ["delegate", "selector", "tag"]},
                {
                    "intent": "connector_name",
                    "question_contains": ["name would you like", "name for the", "identifier"],
                },
            ],
            "yaml": {"default_action": "accept"},
        },
    )


@pytest.mark.unit
async def test_form_elicitation_uses_live_labels_and_hint_values():
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_form",
            "payload": {
                "review_id": "ask-form",
                "content": {
                    "fields": [
                        {
                            "label": "Which authentication method should be used for the Kubernetes connector?",
                            "options": [
                                {"label": "Inherit from Delegate", "value": "Inherit from Delegate"},
                                {"label": "Service Account Token", "value": "Service Account Token"},
                            ],
                        },
                        {
                            "label": "What scope should this connector be created at?",
                            "options": [
                                {"label": "Project (Recommended)", "value": "Project (Recommended)"},
                                {"label": "Account", "value": "Account"},
                            ],
                        },
                    ]
                },
            },
        }
    )

    result = await HarnessSseElicitationAdapter().respond(pending, _golden(), [])

    assert result["capability_id"] == "ask-form"
    assert result["result"]["action_id"] == "respond"
    assert result["result"]["form_values"] == {
        "Which authentication method should be used for the Kubernetes connector?": "Inherit from Delegate",
        "What scope should this connector be created at?": "Project (Recommended)",
    }


@pytest.mark.unit
async def test_free_text_elicitation_maps_question_to_intent():
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_free_text",
            "payload": {
                "review_id": "ask-name",
                "content": {"question": "What name would you like for the Kubernetes connector?"},
            },
        }
    )

    result = await HarnessSseElicitationAdapter().respond(pending, _golden(), [])

    assert result["capability_id"] == "ask-name"
    assert result["result"] == {"success": True, "action_id": "respond", "free_text": "testconnector"}


@pytest.mark.unit
async def test_select_elicitation_maps_question_to_intent():
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_select",
            "payload": {
                "review_id": "ask-delegate",
                "title": "Select a delegate selector tag",
                "content": {
                    "question": "Which delegate selector tag should be used for this connector?",
                    "options": [
                        {"label": "hello", "value": "hello"},
                        {"label": "prod-delegate", "value": "prod-delegate"},
                    ],
                },
            },
        }
    )

    result = await HarnessSseElicitationAdapter().respond(pending, _golden(), [])

    assert result["capability_id"] == "ask-delegate"
    assert result["result"]["action_id"] == "respond"
    assert result["result"]["selected_value"] == "hello"
    assert result["result"]["form_values"] == {
        "Which delegate selector tag should be used for this connector?": "hello"
    }


@pytest.mark.unit
async def test_project_question_maps_to_project_id_not_scope():
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_select",
            "payload": {
                "review_id": "ask-project",
                "content": {
                    "question": "Which project in the 'AI_Devops' org should the connector be created in?",
                    "options": [
                        {"label": "AICHAT", "value": "AICHAT"},
                        {"label": "Project", "value": "Project"},
                    ],
                },
            },
        }
    )

    result = await HarnessSseElicitationAdapter().respond(pending, _golden(), [])

    assert result["capability_id"] == "ask-project"
    assert result["result"]["selected_value"] == "AICHAT"
    assert result["result"]["form_values"] == {
        "Which project in the 'AI_Devops' org should the connector be created in?": "AICHAT"
    }


@pytest.mark.unit
async def test_adapter_accepts_yaml_without_elicitation_hints():
    yaml_text = "connector:\n  identifier: testconnector\n"
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_yaml",
            "payload": {
                "review_id": "rev-yaml",
                "content": {"yaml": yaml_text, "language": "yaml"},
                "entity_info": {"entity_type": "connector", "request_action": "CREATE_CONNECTOR"},
                "tool_input": {"resource_type": "connector"},
            },
        }
    )
    golden = ConversationGolden(
        scenario="Create a Kubernetes connector named testconnector",
        expected_outcome="Connector testconnector created",
        context=["Org: AI_Devops, Project: AICHAT"],
    )

    result = await HarnessSseElicitationAdapter().respond(pending, golden, [])

    assert result["capability_id"] == "rev-yaml"
    assert result["result"]["action_id"] == "accept"
    assert result["result"]["yaml"] == yaml_text


@pytest.mark.unit
async def test_adapter_uses_llm_for_select_without_elicitation_hints():
    class ScriptedLLM(BaseLLM):
        async def generate(self, prompt: str, **kwargs) -> str:
            return ""

        async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
            return {
                "result": {
                    "success": True,
                    "action_id": "respond",
                    "selected_value": "Account",
                }
            }

    golden = ConversationGolden(
        scenario="Create a Kubernetes service named testservice at account scope",
        expected_outcome="Service testservice created at account scope",
        context=["Create the service at Account scope, not project scope"],
    )
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_select",
            "payload": {
                "review_id": "ask-scope",
                "title": "Service scope",
                "content": {
                    "question": "What scope should this service be created at?",
                    "options": [
                        {"label": "Project (Recommended)", "value": "Project (Recommended)"},
                        {"label": "Account", "value": "Account"},
                    ],
                },
            },
        }
    )

    adapter = HarnessSseElicitationAdapter()
    adapter.llm = ScriptedLLM()
    result = await adapter.respond(pending, golden, [])

    assert result["capability_id"] == "ask-scope"
    assert result["result"]["selected_value"] == "Account"
    assert result["result"]["form_values"] == {
        "What scope should this service be created at?": "Account",
    }


@pytest.mark.unit
async def test_adapter_uses_llm_when_elicitation_hints_are_omitted():
    class ScriptedLLM(BaseLLM):
        async def generate(self, prompt: str, **kwargs) -> str:
            return ""

        async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
            return {
                "result": {
                    "success": True,
                    "action_id": "respond",
                    "form_values": [{"label": "Connector name", "value": "testconnector"}],
                }
            }

    golden = ConversationGolden(
        scenario="Create a Kubernetes connector named testconnector",
        expected_outcome="Connector testconnector created",
        context=["Org: AI_Devops, Project: AICHAT"],
    )
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_form",
            "payload": {
                "review_id": "ask-form",
                "content": {
                    "fields": [
                        {
                            "label": "Connector name",
                            "options": [
                                {"label": "testconnector", "value": "testconnector"},
                            ],
                        }
                    ]
                },
            },
        }
    )

    adapter = HarnessSseElicitationAdapter()
    adapter.llm = ScriptedLLM()
    result = await adapter.respond(pending, golden, [])

    assert result["capability_id"] == "ask-form"
    assert result["result"]["form_values"] == {"Connector name": "testconnector"}


@pytest.mark.unit
async def test_yaml_elicitation_accepts_and_echoes_payload():
    yaml_text = "connector:\n  identifier: testconnector\n"
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_yaml",
            "payload": {
                "review_id": "rev-yaml",
                "content": {"yaml": yaml_text, "language": "yaml"},
                "entity_info": {"entity_type": "connector", "request_action": "CREATE_CONNECTOR"},
                "tool_input": {"resource_type": "connector"},
            },
        }
    )

    result = await HarnessSseElicitationAdapter().respond(pending, _golden(), [])

    assert result["capability_id"] == "rev-yaml"
    assert result["result"]["action_id"] == "accept"
    assert result["result"]["yaml"] == yaml_text
    assert result["result"]["entity_type"] == "connector"
    assert result["result"]["request_action"] == "CREATE_CONNECTOR"
    assert result["result"]["tool_input"] == {"resource_type": "connector"}


@pytest.mark.unit
async def test_scripted_response_takes_precedence():
    golden = ConversationGolden(
        scenario="Create a k8s connector",
        expected_outcome="Connector created",
        metadata={
            "elicitation_script": [
                {
                    "trigger": "elicitation_free_text",
                    "match_question_contains": "delegate",
                    "system_event": {
                        "event_type": "action_completed",
                        "result": {"success": True, "action_id": "respond", "free_text": "scripted"},
                    },
                }
            ]
        },
    )
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_free_text",
            "payload": {"review_id": "ask-delegate", "content": {"question": "Delegate selector?"}},
        }
    )

    from harness_evals.conversation.human_input import HumanInputSimulator

    result = await HumanInputSimulator(adapter=HarnessSseElicitationAdapter()).respond(pending, golden, [])

    assert result["capability_id"] == "ask-delegate"
    assert result["result"]["free_text"] == "scripted"


@pytest.mark.unit
async def test_happy_path_records_no_intent_misses(caplog):
    caplog.set_level(logging.WARNING, logger=_INTENT_MISS_LOGGER)
    adapter = HarnessSseElicitationAdapter()
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_free_text",
            "payload": {
                "review_id": "ask-name",
                "content": {"question": "What name would you like for the Kubernetes connector?"},
            },
        }
    )

    await adapter.respond(pending, _golden(), [])

    assert adapter.intent_misses == []
    assert "Elicitation intent miss" not in caplog.text


@pytest.mark.unit
async def test_llm_fallback_records_no_hints_miss(caplog):
    caplog.set_level(logging.WARNING, logger=_INTENT_MISS_LOGGER)

    class ScriptedLLM(BaseLLM):
        async def generate(self, prompt: str, **kwargs) -> str:
            return ""

        async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
            return {
                "result": {
                    "success": True,
                    "action_id": "respond",
                    "selected_value": "Account",
                }
            }

    golden = ConversationGolden(
        id="k8s-service-create",
        scenario="Create a Kubernetes service at account scope",
        expected_outcome="Service created at account scope",
    )
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_select",
            "payload": {
                "review_id": "ask-scope",
                "content": {
                    "question": "What scope should this service be created at?",
                    "options": [
                        {"label": "Project (Recommended)", "value": "Project (Recommended)"},
                        {"label": "Account", "value": "Account"},
                    ],
                },
            },
        }
    )

    adapter = HarnessSseElicitationAdapter()
    adapter.llm = ScriptedLLM()
    await adapter.respond(pending, golden, [])

    assert len(adapter.intent_misses) == 1
    miss = adapter.intent_misses[0]
    assert miss.reason == "no_hints_llm_fallback"
    assert miss.fallback == "llm"
    assert miss.golden_id == "k8s-service-create"
    assert miss.elicitation_type == "elicitation_select"
    assert "no_hints_llm_fallback" in caplog.text


@pytest.mark.unit
async def test_form_field_intent_miss_records_no_intent_match(caplog):
    caplog.set_level(logging.WARNING, logger=_INTENT_MISS_LOGGER)
    adapter = HarnessSseElicitationAdapter()
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_form",
            "payload": {
                "review_id": "ask-unknown",
                "content": {
                    "fields": [
                        {
                            "label": "Which org should be used?",
                            "options": [
                                {"label": "AI_Devops", "value": "AI_Devops"},
                                {"label": "Other", "value": "Other"},
                            ],
                        }
                    ]
                },
            },
        }
    )

    result = await adapter.respond(pending, _golden(), [])

    assert result["result"]["form_values"]["Which org should be used?"] == "AI_Devops"
    assert len(adapter.intent_misses) == 1
    assert adapter.intent_misses[0].reason == "no_intent_match"
    assert adapter.intent_misses[0].fallback == "first_option"
    assert "no_intent_match" in caplog.text


@pytest.mark.unit
async def test_free_text_intent_miss_records_no_intent_match(caplog):
    caplog.set_level(logging.WARNING, logger=_INTENT_MISS_LOGGER)
    adapter = HarnessSseElicitationAdapter()
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_free_text",
            "payload": {
                "review_id": "ask-org",
                "content": {"question": "Which org should be used?"},
            },
        }
    )

    result = await adapter.respond(pending, _golden(), [])

    assert result["result"]["free_text"] == ""
    assert len(adapter.intent_misses) == 1
    assert adapter.intent_misses[0].reason == "no_intent_match"
    assert adapter.intent_misses[0].fallback == "empty"
    assert "no_intent_match" in caplog.text


@pytest.mark.unit
async def test_select_intent_miss_records_no_intent_match(caplog):
    caplog.set_level(logging.WARNING, logger=_INTENT_MISS_LOGGER)
    adapter = HarnessSseElicitationAdapter()
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_select",
            "payload": {
                "review_id": "ask-org",
                "content": {
                    "question": "Which org should be used?",
                    "options": [
                        {"label": "AI_Devops", "value": "AI_Devops"},
                        {"label": "Other", "value": "Other"},
                    ],
                },
            },
        }
    )

    result = await adapter.respond(pending, _golden(), [])

    assert result["result"]["selected_value"] == "AI_Devops"
    assert len(adapter.intent_misses) == 1
    assert adapter.intent_misses[0].reason == "no_intent_match"
    assert adapter.intent_misses[0].fallback == "first_option"
    assert "no_intent_match" in caplog.text


@pytest.mark.unit
async def test_select_intent_miss_records_missing_intent_value(caplog):
    caplog.set_level(logging.WARNING, logger=_INTENT_MISS_LOGGER)
    golden = ConversationGolden(
        scenario="Create connector",
        expected_outcome="Connector created",
        elicitation_hints={
            "intents": {"auth_method": "Inherit from Delegate"},
            "matchers": [{"intent": "org_id", "question_contains": ["org"]}],
        },
    )
    adapter = HarnessSseElicitationAdapter()
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_select",
            "payload": {
                "review_id": "ask-org",
                "content": {
                    "question": "Which org should be used?",
                    "options": [
                        {"label": "AI_Devops", "value": "AI_Devops"},
                    ],
                },
            },
        }
    )

    await adapter.respond(pending, golden, [])

    assert len(adapter.intent_misses) == 1
    assert adapter.intent_misses[0].reason == "missing_intent_value"
    assert adapter.intent_misses[0].intent == "org_id"
    assert "missing_intent_value" in caplog.text


@pytest.mark.unit
def test_simulator_exports_intent_misses_in_metadata():
    from harness_evals.conversation.human_input import IntentMatchMiss

    adapter = HarnessSseElicitationAdapter()
    adapter.intent_misses.append(
        IntentMatchMiss(
            reason="no_intent_match",
            elicitation_type="elicitation_select",
            question="Which org should be used?",
            golden_id="k8s-connector-create",
            fallback="first_option",
        )
    )
    simulator = ConversationSimulator(human_input_simulator=HumanInputSimulator(adapter=adapter))
    golden = ConversationGolden(scenario="Create connector", expected_outcome="Connector created")

    result = simulator._build_eval_case(golden, [Message(role="assistant", content="done")])

    assert result.metadata["elicitation_intent_misses"] == [
        {
            "reason": "no_intent_match",
            "elicitation_type": "elicitation_select",
            "question": "Which org should be used?",
            "intent": None,
            "golden_id": "k8s-connector-create",
            "fallback": "first_option",
        }
    ]


@pytest.mark.unit
def test_adapter_reset_intent_misses_clears_recorder():
    from harness_evals.conversation.human_input import IntentMatchMiss

    adapter = HarnessSseElicitationAdapter()
    adapter.intent_misses.append(
        IntentMatchMiss(
            reason="no_hints_llm_fallback",
            elicitation_type="elicitation_form",
            question="Connector name",
            fallback="llm",
        )
    )

    adapter.reset_intent_misses()

    assert adapter.intent_misses == []
