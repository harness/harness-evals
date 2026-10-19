"""Tests for HarnessSseElicitationAdapter."""

import json
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
async def test_form_text_field_uses_intent_value():
    golden = ConversationGolden(
        scenario="Create a CCM cost category",
        expected_outcome="Cost category created",
        elicitation_hints={
            "intents": {
                "cost_bucket_name": "Production",
                "aws_filter_by": "By Account",
            },
            "matchers": [
                {
                    "intent": "cost_bucket_name",
                    "question_contains": ["name this cost bucket", "name for this cost bucket"],
                },
                {
                    "intent": "aws_filter_by",
                    "question_contains": ["filter aws costs", "filter AWS costs"],
                },
            ],
        },
    )
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_form",
            "payload": {
                "review_id": "ask-bucket-form",
                "content": {
                    "fields": [
                        {
                            "key": "q0",
                            "label": "What would you like to name this cost bucket?",
                            "type": "text",
                        },
                        {
                            "key": "q1",
                            "label": "How would you like to filter AWS costs for this bucket?",
                            "type": "select",
                            "options": [
                                {"label": "By Account", "value": "By Account"},
                                {"label": "By Service", "value": "By Service"},
                            ],
                        },
                    ]
                },
            },
        }
    )

    result = await HarnessSseElicitationAdapter().respond(pending, golden, [])

    assert result["result"]["form_values"] == {
        "What would you like to name this cost bucket?": "Production",
        "How would you like to filter AWS costs for this bucket?": "By Account",
    }


@pytest.mark.unit
async def test_llm_on_miss_falls_back_when_matcher_misses():
    class ScriptedLLM(BaseLLM):
        async def generate(self, prompt: str, **kwargs) -> str:
            return ""

        async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
            return {
                "result": {
                    "success": True,
                    "action_id": "respond",
                    "free_text": "Create it with just this one bucket",
                }
            }

    golden = ConversationGolden(
        id="ce-cost-category",
        scenario="Create a CCM cost category",
        expected_outcome="Cost category created",
        elicitation_hints={
            "llm_on_miss": True,
            "intents": {"category_name": "eval_cost_category_test"},
            "matchers": [
                {"intent": "category_name", "question_contains": ["name your cost category"]},
            ],
        },
    )
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_free_text",
            "payload": {
                "review_id": "ask-more-buckets",
                "content": {
                    "question": "Would you like to add more cost buckets to this category, or create it with just this one bucket?"
                },
            },
        }
    )

    adapter = HarnessSseElicitationAdapter()
    adapter.llm = ScriptedLLM()
    result = await adapter.respond(pending, golden, [])

    assert result["result"]["free_text"] == "Create it with just this one bucket"
    assert len(adapter.intent_misses) == 1
    assert adapter.intent_misses[0].reason == "hint_miss_llm_fallback"
    assert adapter.intent_misses[0].fallback == "llm"


@pytest.mark.unit
async def test_llm_on_miss_falls_back_when_select_options_empty():
    """Empty option lists must miss so llm_on_miss can answer instead of raising."""

    class ScriptedLLM(BaseLLM):
        async def generate(self, prompt: str, **kwargs) -> str:
            return ""

        async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
            return {
                "result": {
                    "success": True,
                    "action_id": "respond",
                    "selection": "Environment-based",
                }
            }

    golden = ConversationGolden(
        id="ce-empty-select-options",
        scenario="Create a CCM cost category",
        expected_outcome="Cost category created",
        elicitation_hints={
            "llm_on_miss": True,
            "intents": {"bucket_type": "Environment-based"},
            "matchers": [
                {
                    "intent": "bucket_type",
                    "question_contains": ["what buckets do you want", "buckets do you want to create"],
                }
            ],
        },
    )
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_select",
            "payload": {
                "review_id": "ask-buckets",
                "content": {
                    "question": "What buckets do you want to create for this cost category?",
                    "items": [],
                },
            },
        }
    )

    adapter = HarnessSseElicitationAdapter()
    adapter.llm = ScriptedLLM()
    result = await adapter.respond(pending, golden, [])

    assert result["result"]["selection"] == "Environment-based"
    assert len(adapter.intent_misses) == 1
    assert adapter.intent_misses[0].reason == "hint_miss_llm_fallback"
    assert adapter.intent_misses[0].fallback == "llm"


@pytest.mark.unit
async def test_category_name_stays_deterministic_with_llm_on_miss():
    class FailingLLM(BaseLLM):
        async def generate(self, prompt: str, **kwargs) -> str:
            raise AssertionError("LLM should not be called when matcher resolves")

        async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
            raise AssertionError("LLM should not be called when matcher resolves")

    golden = ConversationGolden(
        scenario="Create a CCM cost category",
        expected_outcome="Cost category created",
        elicitation_hints={
            "llm_on_miss": True,
            "intents": {"category_name": "eval_cost_category_test"},
            "matchers": [
                {"intent": "category_name", "question_contains": ["name your cost category"]},
            ],
        },
    )
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_free_text",
            "payload": {
                "review_id": "ask-name",
                "content": {"question": "What would you like to name your Cost Category?"},
            },
        }
    )

    adapter = HarnessSseElicitationAdapter()
    adapter.llm = FailingLLM()
    result = await adapter.respond(pending, golden, [])

    assert result["result"]["free_text"] == "eval_cost_category_test"
    assert adapter.intent_misses == []


@pytest.mark.unit
async def test_select_elicitation_uses_content_items():
    golden = ConversationGolden(
        scenario="Create a CCM cost category",
        expected_outcome="Cost category created",
        elicitation_hints={
            "intents": {"bucket_type": "Environment-based"},
            "matchers": [
                {
                    "intent": "bucket_type",
                    "question_contains": ["what buckets do you want", "buckets do you want to create"],
                }
            ],
        },
    )
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_select",
            "payload": {
                "review_id": "ask-buckets",
                "content": {
                    "question": "What buckets do you want to create for this cost category?",
                    "items": [
                        {"id": "0", "label": "Team-based", "description": "Buckets like Engineering"},
                        {"id": "1", "label": "Environment-based", "description": "Production, Staging, Dev"},
                        {"id": "2", "label": "Region-based", "description": "US-East, EU"},
                    ],
                },
            },
        }
    )

    result = await HarnessSseElicitationAdapter().respond(pending, golden, [])

    assert result["capability_id"] == "ask-buckets"
    assert result["result"]["selection"] == "Environment-based"
    assert result["result"]["selected_value"] == "Environment-based"
    assert result["result"]["form_values"] == {
        "What buckets do you want to create for this cost category?": "Environment-based",
    }


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
    assert result["result"]["selection"] == "hello"
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
    assert result["result"]["selection"] == "AICHAT"
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
async def test_adapter_rejects_yaml_with_cancellation_event():
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_yaml",
            "payload": {
                "review_id": "rev-yaml",
                "content": {"yaml": "pipeline:\n  id: test\n", "language": "yaml"},
                "actions": [
                    {"id": "accept", "sends": "action_completed"},
                    {"id": "deny", "sends": "action_cancelled"},
                ],
            },
        }
    )
    golden = ConversationGolden(
        scenario="Review generated pipeline YAML",
        expected_outcome="Pipeline review is rejected",
        elicitation_hints={"yaml": {"default_action": "deny"}},
    )

    result = await HarnessSseElicitationAdapter().respond(pending, golden, [])

    assert result == {
        "event_type": "action_cancelled",
        "result": {},
        "capability_id": "rev-yaml",
    }


@pytest.mark.unit
async def test_adapter_rejects_yaml_without_declared_actions():
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_yaml",
            "payload": {
                "review_id": "rev-yaml",
                "content": {"yaml": "pipeline:\n  id: test\n", "language": "yaml"},
            },
        }
    )
    golden = ConversationGolden(
        scenario="Review generated pipeline YAML",
        expected_outcome="Pipeline review is rejected",
        elicitation_hints={"yaml": {"default_action": "reject"}},
    )

    result = await HarnessSseElicitationAdapter().respond(pending, golden, [])

    assert result == {
        "event_type": "action_cancelled",
        "result": {},
        "capability_id": "rev-yaml",
    }


@pytest.mark.unit
async def test_adapter_approves_elicitation_confirm_for_cost_category():
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_confirm",
            "payload": {
                "review_id": "rev-ccm",
                "entity_info": {"entity_type": "cost_category", "name": "eval_cost_category_test"},
                "tool_input": {"resource_type": "cost_category", "body": {"name": "eval_cost_category_test"}},
            },
        }
    )
    golden = ConversationGolden(
        scenario="Create a CCM cost category",
        expected_outcome="Cost category created",
        elicitation_hints={"confirm": {"default_action": "approve"}},
    )

    result = await HarnessSseElicitationAdapter().respond(pending, golden, [])

    assert result["capability_id"] == "rev-ccm"
    assert result["result"]["action_id"] == "approve"
    assert result["result"]["entity_info"]["entity_type"] == "cost_category"


@pytest.mark.unit
async def test_adapter_rejects_elicitation_confirm_from_yaml_hint():
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_confirm",
            "payload": {
                "review_id": "rev-ccm",
                "entity_info": {"entity_type": "cost_category", "name": "eval_cost_category_test"},
                "tool_input": {"resource_type": "cost_category", "body": {"name": "eval_cost_category_test"}},
            },
        }
    )
    golden = ConversationGolden(
        scenario="Review a CCM cost category",
        expected_outcome="Cost category review is rejected",
        elicitation_hints={"yaml": {"default_action": "deny"}},
    )

    result = await HarnessSseElicitationAdapter().respond(pending, golden, [])

    assert result == {
        "event_type": "action_cancelled",
        "result": {},
        "capability_id": "rev-ccm",
    }


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
    assert result["result"]["selection"] == "Account"
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
async def test_form_field_intent_miss_raises_unresolved(caplog):
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
                            "type": "select",
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

    with pytest.raises(ValueError, match="Unresolved elicitation_form"):
        await adapter.respond(pending, _golden(), [])

    assert len(adapter.intent_misses) == 1
    assert adapter.intent_misses[0].reason == "no_intent_match"
    assert adapter.intent_misses[0].fallback == "unresolved"
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
async def test_select_falls_back_to_unique_intent_option_when_wording_misleads():
    golden = ConversationGolden(
        scenario="Create CI",
        expected_outcome="Reviewed",
        elicitation_hints={
            "match_strategy": "specific",
            "intents": {"command_mode": "Enter an exact fixed command", "test_command": "make test"},
            "matchers": [{"intent": "test_command", "question_contains": ["test command"]}],
        },
    )
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_select",
            "payload": {
                "review_id": "ask-mode",
                "content": {
                    "question": "What is the exact test command to run?",
                    "options": [
                        {"label": "Enter an exact fixed command", "value": "fixed"},
                        {"label": "Use a required runtime input", "value": "runtime"},
                    ],
                },
            },
        }
    )

    result = await HarnessSseElicitationAdapter().respond(pending, golden, [])

    assert result["result"]["selection"] == "Enter an exact fixed command"


@pytest.mark.unit
async def test_select_uses_first_offered_choice_from_intent_list():
    golden = ConversationGolden(
        scenario="Create CI",
        expected_outcome="Reviewed",
        elicitation_hints={
            "match_strategy": "specific",
            "intents": {"review_action": ["Modify before creating", "Review and edit the YAML"]},
            "matchers": [{"intent": "review_action", "question_contains": ["what would you like to do"]}],
        },
    )
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_select",
            "payload": {
                "review_id": "review",
                "content": {
                    "question": "What would you like to do?",
                    "options": [
                        {"label": "Review and edit the YAML"},
                        {"label": "Change the configuration"},
                        {"label": "Other", "value": "other"},
                    ],
                },
            },
        }
    )

    result = await HarnessSseElicitationAdapter().respond(pending, golden, [])

    assert result["result"]["selection"] == "Review and edit the YAML"


@pytest.mark.unit
async def test_form_field_key_suffix_resolves_before_misleading_label():
    golden = ConversationGolden(
        scenario="Create CI",
        expected_outcome="Reviewed",
        elicitation_hints={
            "match_strategy": "specific",
            "intents": {"bucket": "acme-artifacts", "region": "us-east-1"},
            "matchers": [
                {"intent": "bucket", "question_contains": ["bucket"]},
                {"intent": "region", "question_contains": ["region"]},
            ],
        },
    )
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_form",
            "payload": {
                "review_id": "s3",
                "content": {"fields": [{"key": "s3_region", "label": "S3 bucket region", "type": "text"}]},
            },
        }
    )

    result = await HarnessSseElicitationAdapter().respond(pending, golden, [])

    assert result["result"]["form_values"] == {"S3 bucket region": "us-east-1"}


@pytest.mark.unit
async def test_form_field_key_suffix_defers_to_label_when_a_qualified_intent_exists():
    golden = ConversationGolden(
        scenario="Create CI",
        expected_outcome="Reviewed",
        elicitation_hints={
            "intents": {"repository": "harness/harness", "image_repository": "acme/payments"},
            "matchers": [
                {"intent": "image_repository", "question_contains": ["image repository"]},
                {"intent": "repository", "question_contains": ["repository"]},
            ],
        },
    )
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_form",
            "payload": {
                "review_id": "image",
                "content": {
                    "fields": [{"key": "docker_repository", "label": "Docker image repository", "type": "text"}]
                },
            },
        }
    )

    result = await HarnessSseElicitationAdapter().respond(pending, golden, [])

    assert result["result"]["form_values"] == {"Docker image repository": "acme/payments"}


def _repo_picker(options: list[str]) -> PendingHumanInput:
    return PendingHumanInput.from_metadata(
        {
            "type": "elicitation_form",
            "payload": {
                "review_id": "repo",
                "content": {
                    "fields": [
                        {
                            "key": "q0",
                            "label": "Which repository contains your Go code?",
                            "type": "select",
                            "options": [{"label": o, "value": o} for o in options],
                        }
                    ]
                },
            },
        }
    )


@pytest.mark.unit
async def test_unmatched_select_falls_back_to_exact_intent_value_not_partial_text():
    golden = ConversationGolden(
        scenario="Create CI",
        expected_outcome="Reviewed",
        elicitation_hints={
            "match_strategy": "specific",
            "intents": {
                "repository": "harness/harness",
                "decline_harness_code": "Use a different Git provider",
                "alias_scope": "Pipeline",
            },
            "matchers": [{"intent": "repository", "question_contains": ["repository"]}],
        },
    )
    pending = _repo_picker(["eval_testrepo", "harness_pipelines", "Use a different Git provider"])

    result = await HarnessSseElicitationAdapter().respond(pending, golden, [])

    assert result["result"]["form_values"] == {
        "Which repository contains your Go code?": "Use a different Git provider"
    }


@pytest.mark.unit
async def test_unmatched_menu_accepts_recommended_option_only_when_opted_in():
    hints = {"intents": {"repository": "harness/harness"}, "matchers": []}
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_select",
            "payload": {
                "review_id": "cache",
                "content": {
                    "question": "What cache paths should be used?",
                    "options": [
                        {"id": "0", "label": "~/go/pkg/mod only (Recommended)"},
                        {"id": "1", "label": "~/go/pkg/mod and ~/.cache/go-build"},
                    ],
                },
            },
        }
    )
    adapter = HarnessSseElicitationAdapter()

    with pytest.raises(ValueError, match="Unresolved elicitation_select"):
        await adapter.respond(
            pending, ConversationGolden(scenario="s", expected_outcome="o", elicitation_hints=hints), []
        )

    opted_in = ConversationGolden(
        scenario="s", expected_outcome="o", elicitation_hints={**hints, "on_unmatched": "accept"}
    )
    result = await adapter.respond(pending, opted_in, [])

    assert result["result"]["selection"] == "~/go/pkg/mod only (Recommended)"
    assert adapter.intent_misses[-1].fallback == "accept_default"


@pytest.mark.unit
async def test_unmatched_menu_accepts_single_proceed_option_and_refuses_ambiguity():
    golden = ConversationGolden(
        scenario="s",
        expected_outcome="o",
        elicitation_hints={"intents": {"repository": "harness/harness"}, "on_unmatched": "accept"},
    )

    def menu(labels: list[str]) -> PendingHumanInput:
        return PendingHumanInput.from_metadata(
            {
                "type": "elicitation_select",
                "payload": {
                    "review_id": "next",
                    "content": {
                        "question": "Anything to adjust?",
                        "options": [{"id": str(i), "label": label} for i, label in enumerate(labels)],
                    },
                },
            }
        )

    result = await HarnessSseElicitationAdapter().respond(
        menu(["Change the Go version", "Looks good, continue", "Something else"]), golden, []
    )
    assert result["result"]["selection"] == "Looks good, continue"

    with pytest.raises(ValueError, match="Unresolved elicitation_select"):
        await HarnessSseElicitationAdapter().respond(
            menu(["Change the Go version", "Change the S3 destination", "Something else"]), golden, []
        )


@pytest.mark.unit
async def test_form_select_types_exact_answer_instead_of_ambiguous_partial_option():
    golden = ConversationGolden(
        scenario="Create CI",
        expected_outcome="Reviewed",
        elicitation_hints={
            "match_strategy": "specific",
            "intents": {"test_command": "pytest"},
            "field_keys": {"test_command": "test_command"},
        },
    )
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_form",
            "payload": {
                "review_id": "cmd",
                "content": {
                    "fields": [
                        {
                            "key": "test_command",
                            "label": "Test command",
                            "type": "select",
                            "options": [
                                {"label": "pip install -r requirements.txt && pytest"},
                                {"label": "pip install . && pytest"},
                                {"label": "Custom command", "value": "__custom__"},
                            ],
                        }
                    ]
                },
            },
        }
    )

    result = await HarnessSseElicitationAdapter().respond(pending, golden, [])

    assert result["result"]["form_values"] == {"Test command": "pytest"}


@pytest.mark.unit
async def test_form_field_key_resolves_before_label_text():
    golden = ConversationGolden(
        scenario="Create CI",
        expected_outcome="Reviewed",
        elicitation_hints={
            "match_strategy": "specific",
            "intents": {"pipeline_name": "github_ci", "test_command": "make test"},
            "field_keys": {"identifier": "pipeline_name"},
        },
    )
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_form",
            "payload": {
                "review_id": "ask-form",
                "content": {
                    "fields": [
                        {"key": "identifier", "label": "Identifier", "type": "text"},
                        {"key": "test_command", "label": "Command to run", "type": "text"},
                    ]
                },
            },
        }
    )

    result = await HarnessSseElicitationAdapter().respond(pending, golden, [])

    assert result["result"]["form_values"] == {"Identifier": "github_ci", "Command to run": "make test"}


@pytest.mark.unit
async def test_select_intent_miss_raises_unresolved(caplog):
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

    with pytest.raises(ValueError, match="Unresolved elicitation_select"):
        await adapter.respond(pending, _golden(), [])

    assert len(adapter.intent_misses) == 1
    assert adapter.intent_misses[0].reason == "no_intent_match"
    assert adapter.intent_misses[0].fallback == "unresolved"
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

    with pytest.raises(ValueError, match="Unresolved elicitation_select"):
        await adapter.respond(pending, golden, [])

    assert len(adapter.intent_misses) == 1
    assert adapter.intent_misses[0].reason == "missing_intent_value"
    assert adapter.intent_misses[0].intent == "org_id"
    assert adapter.intent_misses[0].fallback == "unresolved"
    assert "missing_intent_value" in caplog.text


@pytest.mark.unit
async def test_multi_select_returns_selection_labels():
    golden = ConversationGolden(
        scenario="Create cost category",
        expected_outcome="Category created",
        elicitation_hints={
            "intents": {"cloud_providers": "AWS, GCP"},
            "matchers": [{"intent": "cloud_providers", "question_contains": ["cloud provider"]}],
        },
    )
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_multi_select",
            "payload": {
                "review_id": "ask-providers",
                "content": {
                    "question": "Which cloud provider(s) should this bucket match?",
                    "items": [
                        {"id": "0", "label": "AWS"},
                        {"id": "1", "label": "GCP"},
                        {"id": "2", "label": "Azure"},
                        {"id": "3", "label": "All Providers"},
                    ],
                    "allow_multiple": True,
                },
            },
        }
    )

    result = await HarnessSseElicitationAdapter().respond(pending, golden, [])

    assert result["capability_id"] == "ask-providers"
    assert result["result"]["selections"] == ["AWS", "GCP"]
    assert result["result"]["selection"] == "AWS"


@pytest.mark.unit
async def test_form_text_field_keeps_custom_value_even_with_options():
    golden = ConversationGolden(
        scenario="Create cost category",
        expected_outcome="Category created",
        elicitation_hints={
            "intents": {"category_name": "test"},
            "matchers": [{"intent": "category_name", "question_contains": ["name this cost category"]}],
        },
    )
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_form",
            "payload": {
                "review_id": "ask-form",
                "content": {
                    "fields": [
                        {
                            "key": "q0",
                            "label": "What would you like to name this cost category?",
                            "type": "text",
                            "options": [
                                {"label": "Team Costs", "value": "Team Costs"},
                                {"label": "Service Costs", "value": "Service Costs"},
                            ],
                        }
                    ]
                },
            },
        }
    )

    result = await HarnessSseElicitationAdapter().respond(pending, golden, [])

    assert result["result"]["form_values"] == {
        "What would you like to name this cost category?": "test",
    }


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
            fallback="unresolved",
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
            "fallback": "unresolved",
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


@pytest.mark.unit
async def test_form_field_key_suffix_yields_to_label_that_does_not_mention_it():
    golden = ConversationGolden(
        scenario="Create CI",
        expected_outcome="Reviewed",
        elicitation_hints={
            "intents": {"repository": "harness/harness", "image_repository": "acme/payments"},
            "matchers": [{"intent": "image_repository", "question_contains": ["image name"]}],
            "field_keys": {"repo": "repository"},
        },
    )
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_form",
            "payload": {
                "review_id": "docker",
                "content": {
                    "fields": [{"key": "docker_repo", "label": "Docker image name (e.g. myorg/myapp)", "type": "text"}]
                },
            },
        }
    )

    result = await HarnessSseElicitationAdapter().respond(pending, golden, [])

    assert result["result"]["form_values"] == {"Docker image name (e.g. myorg/myapp)": "acme/payments"}


def _select_pending(question: str, options: list[dict]) -> PendingHumanInput:
    return PendingHumanInput.from_metadata(
        {
            "type": "elicitation_select",
            "payload": {"review_id": "s", "content": {"question": question, "options": options}},
        }
    )


@pytest.mark.unit
async def test_reworded_option_resolves_by_keyword_overlap_with_the_intent_answer():
    golden = ConversationGolden(
        scenario="Create CI",
        expected_outcome="Reviewed",
        elicitation_hints={
            "match_strategy": "specific",
            "intents": {"command_mode": "Enter an exact fixed command"},
            "matchers": [{"intent": "command_mode", "question_contains": ["configured"]}],
        },
    )
    pending = _select_pending(
        "How should the test step be configured?",
        [
            {"id": "0", "label": "Enter a fixed command", "description": "You'll type the exact test command"},
            {"id": "1", "label": "Runtime input", "description": "The test command is supplied at run time"},
            {"id": "2", "label": "Omit tests", "description": "Skip the test step"},
        ],
    )

    result = await HarnessSseElicitationAdapter().respond(pending, golden, [])

    assert "Enter a fixed command" in json.dumps(result)


@pytest.mark.unit
async def test_unasked_question_resolves_from_another_known_answer_by_keywords():
    golden = ConversationGolden(
        scenario="Create CI",
        expected_outcome="Reviewed",
        elicitation_hints={
            "match_strategy": "specific",
            "intents": {"test_command": "pip install -r requirements.txt && pytest"},
        },
    )
    pending = _select_pending(
        "How are your Python dependencies managed?",
        [
            {"label": "requirements.txt", "description": "Will install with: pip install -r requirements.txt"},
            {"label": "pyproject.toml / setup.py", "description": "Will install with: pip install ."},
            {"label": "No install needed", "description": "Skip dependency installation"},
        ],
    )

    result = await HarnessSseElicitationAdapter().respond(pending, golden, [])

    assert "requirements.txt" in json.dumps(result)


@pytest.mark.unit
async def test_longer_answer_uses_the_custom_entry_instead_of_a_preset_fragment():
    golden = ConversationGolden(
        scenario="Create CI",
        expected_outcome="Reviewed",
        elicitation_hints={
            "match_strategy": "specific",
            "intents": {"test_command": "pip install -r requirements.txt && pytest"},
            "matchers": [{"intent": "test_command", "question_contains": ["test command"]}],
        },
    )
    options = [
        {"label": "pytest", "value": "pytest"},
        {"label": "python -m unittest discover", "value": "python -m unittest discover"},
        {"label": "Custom command", "value": "custom"},
    ]

    result = await HarnessSseElicitationAdapter().respond(_select_pending("Python test command", options), golden, [])

    assert result["result"]["selection"] == "pip install -r requirements.txt && pytest"


@pytest.mark.unit
async def test_answer_containing_a_preset_still_picks_it_when_no_custom_entry_exists():
    golden = ConversationGolden(
        scenario="Create CI",
        expected_outcome="Reviewed",
        elicitation_hints={
            "intents": {"test_command": "run pytest"},
            "matchers": [{"intent": "test_command", "question_contains": ["test command"]}],
        },
    )
    options = [{"label": "pytest", "value": "pytest"}, {"label": "unittest", "value": "unittest"}]

    result = await HarnessSseElicitationAdapter().respond(_select_pending("Python test command", options), golden, [])

    assert result["result"]["selection"] == "pytest"


@pytest.mark.unit
async def test_keyword_overlap_matches_word_forms_and_ignores_negated_words():
    golden = ConversationGolden(
        scenario="Create CI",
        expected_outcome="Reviewed",
        elicitation_hints={
            "match_strategy": "specific",
            "intents": {"review_action": ["Create the pipeline", "Review and edit the YAML"]},
            "matchers": [{"intent": "review_action", "question_contains": ["what would you like to do"]}],
        },
    )
    pending = _select_pending(
        "The review was cancelled twice now. What would you like to do?",
        [
            {"label": "Try creating again", "description": "Submit the same pipeline YAML for review and creation"},
            {"label": "Change something first", "description": "Modify the pipeline details before trying again"},
            {"label": "Discard", "description": "Abandon this pipeline — don't create it"},
        ],
    )

    result = await HarnessSseElicitationAdapter().respond(pending, golden, [])

    assert "Try creating again" in json.dumps(result)


@pytest.mark.unit
async def test_keyword_overlap_refuses_a_weak_or_tied_match():
    golden = ConversationGolden(
        scenario="Create CI",
        expected_outcome="Reviewed",
        elicitation_hints={"intents": {"repository": "harness/harness"}},
    )
    pending = _select_pending(
        "Pick one",
        [
            {"label": "linux", "description": "Harness Code repo: linux"},
            {"label": "harness_pipelines", "description": "Harness Code repo: harness_pipelines"},
        ],
    )

    with pytest.raises(ValueError):
        await HarnessSseElicitationAdapter().respond(pending, golden, [])


@pytest.mark.unit
async def test_select_does_not_pick_an_identifier_that_is_only_a_prefix_of_the_answer():
    golden = ConversationGolden(
        scenario="Create CI",
        expected_outcome="Reviewed",
        elicitation_hints={
            "match_strategy": "specific",
            "intents": {"git_connector": "github_testgo_account"},
            "matchers": [{"intent": "git_connector", "question_contains": ["github connector"]}],
        },
    )
    pending = _select_pending(
        "Which GitHub connector should the pipeline use?",
        [{"label": "github_testgo"}, {"label": "yogesh_github"}],
    )

    with pytest.raises(ValueError):
        await HarnessSseElicitationAdapter().respond(pending, golden, [])


@pytest.mark.unit
async def test_form_label_naming_a_specialised_intent_beats_the_generic_key_alias():
    golden = ConversationGolden(
        scenario="Create CI",
        expected_outcome="Reviewed",
        elicitation_hints={
            "intents": {"repository": "harness/harness", "image_repository": "acme/payments"},
            "matchers": [
                {"intent": "image_repository", "question_contains": ["image repository"]},
                {"intent": "repository", "question_contains": ["repository"]},
            ],
        },
    )
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_form",
            "payload": {
                "review_id": "f",
                "content": {
                    "fields": [{"key": "repo", "label": "Image repository (e.g. myorg/myapp)", "type": "text"}]
                },
            },
        }
    )

    result = await HarnessSseElicitationAdapter().respond(pending, golden, [])

    assert "acme/payments" in json.dumps(result)
    assert "harness/harness" not in json.dumps(result)


def _select_pending(question: str, options: list[dict]) -> PendingHumanInput:
    return PendingHumanInput.from_metadata(
        {
            "type": "elicitation_select",
            "payload": {"review_id": "ask", "content": {"question": question, "options": options}},
        }
    )


@pytest.mark.unit
async def test_default_mode_skips_exact_and_keyword_fallbacks_for_unmatched_select():
    golden = ConversationGolden(
        scenario="Create CI",
        expected_outcome="Reviewed",
        elicitation_hints={
            "intents": {"command_mode": "Enter an exact fixed command", "test_command": "make test"},
            "matchers": [{"intent": "test_command", "question_contains": ["test command"]}],
        },
    )
    pending = _select_pending(
        "How should the command be provided?",
        [
            {"label": "Enter an exact fixed command", "value": "fixed"},
            {"label": "Use a required runtime input", "value": "runtime"},
        ],
    )

    with pytest.raises(ValueError):
        await HarnessSseElicitationAdapter().respond(pending, golden, [])


@pytest.mark.unit
async def test_default_mode_picks_the_first_partial_option_match():
    golden = ConversationGolden(
        scenario="Create CI",
        expected_outcome="Reviewed",
        elicitation_hints={
            "intents": {"environment": "prod"},
            "matchers": [{"intent": "environment", "question_contains": ["environment"]}],
        },
    )
    pending = _select_pending(
        "Which environment?",
        [{"label": "production-east", "value": "east"}, {"label": "production-west", "value": "west"}],
    )

    result = await HarnessSseElicitationAdapter().respond(pending, golden, [])

    assert result["result"]["selection"] == "production-east"
