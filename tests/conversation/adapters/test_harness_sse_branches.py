"""Branch-coverage tests for HarnessSseElicitationAdapter helpers and edge paths.

test_harness_sse.py covers realistic end-to-end behavior through ``respond()``.
These close the remaining coverage gaps left by the examples/ -> src/ promotion —
deterministic "would miss" gating, dispatch error paths, and the small pure
helpers — by exercising private methods/functions directly where that's the
most direct way to reach a branch.
"""

import pytest

from harness_evals.conversation import ConversationGolden
from harness_evals.conversation.adapters.harness_sse import (
    ElicitationSimulator,
    HarnessSseElicitationAdapter,
    _field_type,
    _llm_schema_for_pending,
    _normalize_llm_result,
    _option_label,
    _option_matches,
    _option_value,
    _select_option,
    _select_options_multi,
)
from harness_evals.conversation.human_input import PendingHumanInput
from harness_evals.llm.base import BaseLLM


def _golden(**hints) -> ConversationGolden:
    return ConversationGolden(scenario="x", expected_outcome="y", elicitation_hints=hints or None)


# -- _llm_schema_for_pending --------------------------------------------------


@pytest.mark.unit
def test_llm_schema_for_multi_select():
    pending = PendingHumanInput.from_metadata({"type": "elicitation_multi_select", "payload": {}})
    schema = _llm_schema_for_pending(pending)
    assert schema["properties"]["result"]["required"] == ["success", "action_id", "selections"]


@pytest.mark.unit
def test_llm_schema_for_select_with_fields_uses_form_schema():
    pending = PendingHumanInput.from_metadata(
        {"type": "elicitation_select", "payload": {"content": {"fields": [{"label": "x"}]}}}
    )
    schema = _llm_schema_for_pending(pending)
    assert schema["properties"]["result"]["required"] == ["success", "action_id", "form_values"]


@pytest.mark.unit
def test_llm_schema_for_unsupported_type_raises():
    pending = PendingHumanInput.from_metadata({"type": "elicitation_mystery", "payload": {}})
    with pytest.raises(ValueError, match="Unsupported LLM elicitation type"):
        _llm_schema_for_pending(pending)


# -- respond(): dispatch errors + llm wiring ----------------------------------


@pytest.mark.unit
async def test_respond_raises_for_unsupported_elicitation_type():
    golden = _golden(intents={})
    pending = PendingHumanInput.from_metadata({"type": "elicitation_mystery", "payload": {"review_id": "r"}})
    with pytest.raises(ValueError, match="Unsupported Harness elicitation type"):
        await HarnessSseElicitationAdapter().respond(pending, golden, [])


@pytest.mark.unit
async def test_llm_system_event_requires_simulator_llm():
    golden = ConversationGolden(scenario="x", expected_outcome="y")
    pending = PendingHumanInput.from_metadata(
        {"type": "elicitation_free_text", "payload": {"review_id": "r", "content": {"question": "Q?"}}}
    )
    with pytest.raises(ValueError, match="configure conversation.simulator_llm"):
        await HarnessSseElicitationAdapter().respond(pending, golden, [])


@pytest.mark.unit
async def test_llm_system_event_rejects_non_mapping_result():
    class _NotADict:
        def get(self, key, default=None):
            return default

    class ScriptedLLM(BaseLLM):
        async def generate(self, prompt: str, **kwargs) -> str:
            return ""

        async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
            return _NotADict()

    golden = ConversationGolden(scenario="x", expected_outcome="y")
    pending = PendingHumanInput.from_metadata(
        {"type": "elicitation_free_text", "payload": {"review_id": "r", "content": {"question": "Q?"}}}
    )
    adapter = HarnessSseElicitationAdapter()
    adapter.llm = ScriptedLLM()
    with pytest.raises(ValueError, match="must include a result object"):
        await adapter.respond(pending, golden, [])


# -- _form_response edge branches ---------------------------------------------


@pytest.mark.unit
async def test_form_response_skips_non_dict_fields():
    golden = _golden(intents={"name": "testval"}, matchers=[{"intent": "name", "question_contains": ["name"]}])
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_form",
            "payload": {
                "review_id": "r",
                "content": {"fields": ["not-a-dict", {"label": "What name?", "type": "text"}]},
            },
        }
    )
    result = await HarnessSseElicitationAdapter().respond(pending, golden, [])
    assert result["result"]["form_values"] == {"What name?": "testval"}


@pytest.mark.unit
async def test_form_multi_select_field_resolves_multiple_values():
    golden = _golden(
        intents={"providers": "AWS, GCP"},
        matchers=[{"intent": "providers", "question_contains": ["cloud providers"]}],
    )
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_form",
            "payload": {
                "review_id": "r",
                "content": {
                    "fields": [
                        {
                            "label": "Which cloud providers?",
                            "type": "multi_select",
                            "options": [
                                {"label": "AWS", "value": "AWS"},
                                {"label": "GCP", "value": "GCP"},
                                {"label": "Azure", "value": "Azure"},
                            ],
                        }
                    ]
                },
            },
        }
    )
    result = await HarnessSseElicitationAdapter().respond(pending, golden, [])
    assert result["result"]["form_values"] == {"Which cloud providers?": "AWS, GCP"}


@pytest.mark.unit
async def test_form_multi_select_field_raises_when_unresolved():
    golden = _golden(
        intents={"providers": "DigitalOcean"},
        matchers=[{"intent": "providers", "question_contains": ["cloud providers"]}],
    )
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_form",
            "payload": {
                "review_id": "r",
                "content": {
                    "fields": [
                        {
                            "label": "Which cloud providers?",
                            "type": "multi_select",
                            "options": [{"label": "AWS", "value": "AWS"}, {"label": "GCP", "value": "GCP"}],
                        }
                    ]
                },
            },
        }
    )
    with pytest.raises(ValueError, match="Unresolved elicitation_form"):
        await HarnessSseElicitationAdapter().respond(pending, golden, [])


@pytest.mark.unit
async def test_form_select_field_raises_when_desired_value_unmatched():
    golden = _golden(intents={"region": "Mars"}, matchers=[{"intent": "region", "question_contains": ["region"]}])
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_form",
            "payload": {
                "review_id": "r",
                "content": {
                    "fields": [
                        {
                            "label": "Which region?",
                            "type": "select",
                            "options": [{"label": "US-East", "value": "US-East"}, {"label": "EU", "value": "EU"}],
                        }
                    ]
                },
            },
        }
    )
    with pytest.raises(ValueError, match="Unresolved elicitation_form"):
        await HarnessSseElicitationAdapter().respond(pending, golden, [])


@pytest.mark.unit
async def test_form_field_without_options_records_empty_miss():
    golden = _golden(intents={})
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_form",
            "payload": {"review_id": "r", "content": {"fields": [{"label": "Any notes?", "type": "text"}]}},
        }
    )
    adapter = HarnessSseElicitationAdapter()
    result = await adapter.respond(pending, golden, [])
    assert result["result"]["form_values"] == {}
    assert adapter.intent_misses[0].fallback == "empty"


# -- _select_response / _multi_select_response edge branches ------------------


@pytest.mark.unit
async def test_select_with_fields_delegates_to_form_response():
    golden = _golden(intents={"region": "EU"}, matchers=[{"intent": "region", "question_contains": ["region"]}])
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_select",
            "payload": {
                "review_id": "r",
                "content": {
                    "fields": [
                        {
                            "label": "Which region?",
                            "type": "select",
                            "options": [{"label": "US", "value": "US"}, {"label": "EU", "value": "EU"}],
                        }
                    ]
                },
            },
        }
    )
    result = await HarnessSseElicitationAdapter().respond(pending, golden, [])
    assert result["result"]["form_values"] == {"Which region?": "EU"}


@pytest.mark.unit
async def test_multi_select_raises_when_unresolved():
    golden = _golden(
        intents={"providers": "DigitalOcean"},
        matchers=[{"intent": "providers", "question_contains": ["cloud provider"]}],
    )
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_multi_select",
            "payload": {
                "review_id": "r",
                "content": {
                    "question": "Which cloud provider(s)?",
                    "items": [{"id": "0", "label": "AWS"}, {"id": "1", "label": "GCP"}],
                },
            },
        }
    )
    with pytest.raises(ValueError, match="Unresolved elicitation_multi_select"):
        await HarnessSseElicitationAdapter().respond(pending, golden, [])


# -- _record_field_miss reason=option_match_miss -------------------------------


@pytest.mark.unit
def test_record_field_miss_reports_option_match_miss_when_desired_set():
    adapter = HarnessSseElicitationAdapter()
    golden = ConversationGolden(scenario="x", expected_outcome="y")
    adapter._record_field_miss(
        golden=golden,
        pending_type="elicitation_select",
        question="Q?",
        intent="region",
        desired="Mars",
        fallback="unresolved",
    )
    assert adapter.intent_misses[0].reason == "option_match_miss"


# -- deterministic "would miss" helpers (llm_on_miss gating) -------------------


@pytest.mark.unit
def test_form_would_miss_true_when_field_unresolved():
    adapter = HarnessSseElicitationAdapter()
    golden = _golden(intents={})
    payload = {"content": {"fields": [{"label": "Any notes?", "type": "text"}]}}
    pending = PendingHumanInput.from_metadata({"type": "elicitation_form", "payload": payload})
    assert adapter._deterministic_would_miss(pending, golden) is True


@pytest.mark.unit
def test_form_would_miss_false_when_field_resolved():
    adapter = HarnessSseElicitationAdapter()
    golden = _golden(intents={"name": "testval"}, matchers=[{"intent": "name", "question_contains": ["name"]}])
    payload = {"content": {"fields": [{"label": "What name?", "type": "text"}]}}
    pending = PendingHumanInput.from_metadata({"type": "elicitation_form", "payload": payload})
    assert adapter._deterministic_would_miss(pending, golden) is False


@pytest.mark.unit
def test_form_would_miss_true_for_unmatched_multi_select_options():
    adapter = HarnessSseElicitationAdapter()
    golden = _golden(
        intents={"providers": "DigitalOcean"}, matchers=[{"intent": "providers", "question_contains": ["providers"]}]
    )
    payload = {
        "content": {"fields": [{"label": "Which providers?", "type": "multi_select", "options": [{"label": "AWS"}]}]}
    }
    assert adapter._form_would_miss(payload, golden) is True


@pytest.mark.unit
def test_form_would_miss_true_for_unmatched_select_options():
    adapter = HarnessSseElicitationAdapter()
    golden = _golden(intents={"region": "Mars"}, matchers=[{"intent": "region", "question_contains": ["region"]}])
    payload = {"content": {"fields": [{"label": "Which region?", "type": "select", "options": [{"label": "US"}]}]}}
    assert adapter._form_would_miss(payload, golden) is True


@pytest.mark.unit
def test_select_would_miss_delegates_to_form_when_fields_present():
    adapter = HarnessSseElicitationAdapter()
    golden = _golden(intents={})
    payload = {"content": {"fields": [{"label": "Any notes?", "type": "text"}]}}
    assert adapter._select_would_miss(payload, golden) is True


@pytest.mark.unit
def test_select_would_miss_true_when_no_intent_resolves():
    adapter = HarnessSseElicitationAdapter()
    golden = _golden(intents={})
    payload = {"content": {"question": "Which region?", "options": [{"label": "US"}]}}
    assert adapter._select_would_miss(payload, golden) is True


@pytest.mark.unit
def test_select_would_miss_false_when_option_matches():
    adapter = HarnessSseElicitationAdapter()
    golden = _golden(intents={"region": "EU"}, matchers=[{"intent": "region", "question_contains": ["region"]}])
    payload = {"content": {"question": "Which region?", "options": [{"label": "US"}, {"label": "EU"}]}}
    assert adapter._select_would_miss(payload, golden) is False


@pytest.mark.unit
def test_multi_select_would_miss_true_when_no_intent_resolves():
    adapter = HarnessSseElicitationAdapter()
    golden = _golden(intents={})
    payload = {"content": {"question": "Which providers?", "items": [{"label": "AWS"}]}}
    assert adapter._multi_select_would_miss(payload, golden) is True


@pytest.mark.unit
def test_multi_select_would_miss_true_when_options_empty():
    adapter = HarnessSseElicitationAdapter()
    golden = _golden(
        intents={"providers": "AWS"}, matchers=[{"intent": "providers", "question_contains": ["providers"]}]
    )
    payload = {"content": {"question": "Which providers?", "items": []}}
    assert adapter._multi_select_would_miss(payload, golden) is True


@pytest.mark.unit
def test_multi_select_would_miss_false_when_options_match():
    adapter = HarnessSseElicitationAdapter()
    golden = _golden(
        intents={"providers": "AWS"}, matchers=[{"intent": "providers", "question_contains": ["providers"]}]
    )
    payload = {"content": {"question": "Which providers?", "items": [{"label": "AWS"}, {"label": "GCP"}]}}
    assert adapter._multi_select_would_miss(payload, golden) is False


@pytest.mark.unit
def test_deterministic_would_miss_dispatches_multi_select():
    adapter = HarnessSseElicitationAdapter()
    golden = _golden(intents={})
    pending = PendingHumanInput.from_metadata(
        {"type": "elicitation_multi_select", "payload": {"content": {"question": "Which?", "items": []}}}
    )
    assert adapter._deterministic_would_miss(pending, golden) is True


@pytest.mark.unit
def test_deterministic_would_miss_defaults_true_for_unknown_type():
    adapter = HarnessSseElicitationAdapter()
    golden = _golden(intents={})
    pending = PendingHumanInput.from_metadata({"type": "elicitation_mystery", "payload": {}})
    assert adapter._deterministic_would_miss(pending, golden) is True


# -- ElicitationSimulator (backward-compatible alias) --------------------------


@pytest.mark.unit
async def test_elicitation_simulator_generate_system_event_delegates_to_respond():
    class ScriptedLLM(BaseLLM):
        async def generate(self, prompt: str, **kwargs) -> str:
            return ""

        async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
            return {"result": {"success": True, "action_id": "respond", "free_text": "ok"}}

    simulator = ElicitationSimulator(ScriptedLLM())
    golden = ConversationGolden(scenario="x", expected_outcome="y")
    elicitation = {"type": "elicitation_free_text", "payload": {"review_id": "r", "content": {"question": "Q?"}}}

    result = await simulator.generate_system_event(elicitation, golden, [])

    assert result["result"]["free_text"] == "ok"
    assert result["capability_id"] == "r"


# -- _normalize_llm_result ------------------------------------------------------


@pytest.mark.unit
def test_normalize_llm_result_derives_selections_from_single_selection():
    pending = PendingHumanInput.from_metadata({"type": "elicitation_multi_select", "payload": {}})
    normalized = _normalize_llm_result({"success": True, "action_id": "respond", "selection": "AWS"}, pending)
    assert normalized["selections"] == ["AWS"]


@pytest.mark.unit
def test_normalize_llm_result_cleans_selections_and_derives_selection():
    pending = PendingHumanInput.from_metadata({"type": "elicitation_multi_select", "payload": {}})
    normalized = _normalize_llm_result(
        {"success": True, "action_id": "respond", "selections": ["AWS", None, "", "GCP"]}, pending
    )
    assert normalized["selections"] == ["AWS", "GCP"]
    assert normalized["selection"] == "AWS"


# -- small pure helpers ----------------------------------------------------------


@pytest.mark.unit
def test_field_type_defaults_to_text_without_options():
    assert _field_type({"label": "Notes"}) == "text"


@pytest.mark.unit
def test_option_label_and_value_handle_bare_strings():
    assert _option_label("AWS") == "AWS"
    assert _option_value("AWS") == "AWS"


@pytest.mark.unit
def test_option_matches_matches_label_or_value_case_insensitively():
    options = [{"label": "AWS", "value": "aws-id"}]
    assert _option_matches(options, "aws") is True
    assert _option_matches(options, "aws-id") is True
    assert _option_matches(options, "gcp") is False


@pytest.mark.unit
def test_select_option_returns_none_for_empty_desired_or_options():
    assert _select_option([{"label": "AWS"}], "") is None
    assert _select_option([], "AWS") is None


@pytest.mark.unit
def test_select_option_matches_by_value_and_substring():
    options = [{"label": "US East", "value": "us-east-1"}]
    assert _select_option(options, "us-east-1") == "US East"
    assert _select_option(options, "East") == "US East"


@pytest.mark.unit
def test_select_option_returns_none_when_nothing_matches():
    assert _select_option([{"label": "AWS"}], "Azure") is None


@pytest.mark.unit
def test_select_options_multi_returns_empty_for_blank_desired():
    assert _select_options_multi([{"label": "AWS"}], "") == []


@pytest.mark.unit
def test_select_options_multi_falls_back_to_single_part_when_split_yields_nothing():
    # desired.strip() is truthy but splitting on separators yields no non-empty parts.
    assert _select_options_multi([{"label": "AWS"}], ",,,") == []


@pytest.mark.unit
def test_select_options_multi_matches_whole_string_against_option_value():
    # Fragments ("all", "none") match neither label nor value, but the raw
    # comma-joined desired string matches the option's value exactly.
    options = [{"label": "Everything Selected", "value": "ALL,NONE"}]
    assert _select_options_multi(options, "ALL,NONE") == ["Everything Selected"]
