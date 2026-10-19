"""Tests for ConversationSimulator elicitation sub-loop."""

import json

import pytest
from examples.harness_sse_elicitation_adapter import ElicitationSimulator

from harness_evals.conversation import ConversationGolden, ConversationMode, ConversationSimulator
from harness_evals.conversation.human_input import PendingHumanInput
from harness_evals.conversation.simulator import _due_scripted_followup, _human_input_preview, _plain_text_followup
from harness_evals.core.types import Message
from harness_evals.llm.base import BaseLLM


class StopLLM(BaseLLM):
    async def generate(self, prompt: str, **kwargs) -> str:
        return ""

    async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
        return {"achieved": True, "reasoning": "done"}


@pytest.mark.unit
def test_cancelled_yaml_preview_reports_rejection():
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_yaml",
            "payload": {
                "review_id": "rev-pipeline",
                "entity_info": {"entity_type": "pipeline_v1"},
            },
        }
    )

    preview = _human_input_preview(
        pending,
        {
            "event_type": "action_cancelled",
            "capability_id": "rev-pipeline",
            "result": {},
        },
    )

    assert preview == "[Simulated YAML review: reject pipeline_v1]"


@pytest.mark.unit
def test_cancelled_confirm_preview_reports_rejection():
    pending = PendingHumanInput.from_metadata(
        {
            "type": "elicitation_confirm",
            "payload": {
                "review_id": "rev-ccm",
                "entity_info": {"entity_type": "cost_category"},
            },
        }
    )

    preview = _human_input_preview(
        pending,
        {
            "event_type": "action_cancelled",
            "capability_id": "rev-ccm",
            "result": {},
        },
    )

    assert preview == "[Simulated elicitation_confirm review: reject cost_category]"


@pytest.mark.unit
async def test_simulator_uses_initial_prompt_and_resolves_elicitation():
    golden = ConversationGolden(
        scenario="Create a k8s connector",
        expected_outcome="Connector created",
        max_turns=1,
        max_elicitation_rounds=3,
        initial_prompt="Create a k8s connector",
        elicitation_hints={
            "intents": {"connector_name": "testconnector"},
            "matchers": [{"intent": "connector_name", "question_contains": ["name", "connector"]}],
        },
    )
    calls: list[dict | None] = []

    async def agent_fn(messages: list[Message], system_event: dict | None = None) -> Message:
        calls.append(system_event)
        if system_event is None:
            assert messages[-1].content == "Create a k8s connector"
        else:
            assert messages[-1].content == "testconnector"
        if system_event is None:
            return Message(
                role="assistant",
                metadata={
                    "pending_elicitation": {
                        "type": "elicitation_free_text",
                        "payload": {
                            "review_id": "ask-name",
                            "content": {"question": "What name would you like for the connector?"},
                        },
                    }
                },
            )
        assert system_event["capability_id"] == "ask-name"
        assert system_event["result"]["free_text"] == "testconnector"
        return Message(role="assistant", content="Connector created.")

    simulator = ConversationSimulator(simulator_llm=StopLLM(), elicitation_simulator=ElicitationSimulator())
    eval_case = await simulator.simulate(golden, agent_fn)

    assert len(calls) == 2
    assert eval_case.output == "Connector created."
    assert eval_case.messages[0].content == "Create a k8s connector"
    assert eval_case.messages[-1].content == "Connector created."
    simulated = [m for m in eval_case.messages if (m.metadata or {}).get("simulated")]
    assert len(simulated) == 1
    assert simulated[0].content == "testconnector"
    assert eval_case.metadata.get("elicitation_trace")


@pytest.mark.unit
async def test_elicitation_resume_delay_before_human_input_call(monkeypatch):
    golden = ConversationGolden(
        scenario="Create a pipeline",
        expected_outcome="Pipeline created",
        max_turns=1,
        max_elicitation_rounds=2,
        initial_prompt="Create a pipeline",
        elicitation_hints={
            "intents": {"pipeline_name": "payments"},
            "matchers": [{"intent": "pipeline_name", "question_contains": ["pipeline", "name"]}],
        },
    )
    sleep_calls: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr("harness_evals.conversation.simulator.asyncio.sleep", fake_sleep)

    calls = 0

    async def agent_fn(messages: list[Message], human_input: dict | None = None) -> Message:
        nonlocal calls
        calls += 1
        if human_input is None:
            return Message(
                role="assistant",
                metadata={
                    "pending_elicitation": {
                        "type": "elicitation_free_text",
                        "payload": {
                            "review_id": "ask-pipeline-name",
                            "content": {"question": "What pipeline name should I use?"},
                        },
                    }
                },
            )
        return Message(role="assistant", content="Pipeline created.")

    simulator = ConversationSimulator(
        simulator_llm=StopLLM(),
        elicitation_simulator=ElicitationSimulator(),
        elicitation_resume_delay_s=2.0,
    )
    eval_case = await simulator.simulate(golden, agent_fn)

    assert sleep_calls == [2.0]
    assert eval_case.output == "Pipeline created."
    assert calls == 2


@pytest.mark.unit
async def test_post_elicitation_settle_delay_before_next_scripted_turn(monkeypatch):
    golden = ConversationGolden(
        scenario="Create then update a pipeline",
        expected_outcome="Pipeline created and updated",
        mode=ConversationMode.SCRIPTED,
        turns=[
            Message(role="user", content="Create a pipeline"),
            Message(role="user", content="Update the pipeline tags"),
        ],
        max_elicitation_rounds=2,
        elicitation_hints={
            "intents": {"pipeline_name": "payments"},
            "matchers": [{"intent": "pipeline_name", "question_contains": ["pipeline", "name"]}],
        },
    )
    sleep_calls: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr("harness_evals.conversation.simulator.asyncio.sleep", fake_sleep)

    calls = 0

    async def agent_fn(messages: list[Message], human_input: dict | None = None) -> Message:
        nonlocal calls
        calls += 1
        if human_input is not None:
            return Message(role="assistant", content="Pipeline created.")
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        if "Update" in (last_user or ""):
            return Message(role="assistant", content="Pipeline updated.")
        return Message(
            role="assistant",
            metadata={
                "pending_elicitation": {
                    "type": "elicitation_free_text",
                    "payload": {
                        "review_id": "ask-pipeline-name",
                        "content": {"question": "What pipeline name should I use?"},
                    },
                }
            },
        )

    simulator = ConversationSimulator(
        simulator_llm=StopLLM(),
        elicitation_simulator=ElicitationSimulator(),
        elicitation_resume_delay_s=2.0,
        post_elicitation_settle_delay_s=3.0,
    )
    eval_case = await simulator.simulate(golden, agent_fn)

    assert sleep_calls == [2.0, 3.0]
    assert eval_case.output == "Pipeline updated."
    assert calls == 3


@pytest.mark.unit
async def test_multi_round_sse_tool_calls_are_not_double_counted():
    golden = ConversationGolden(
        scenario="Create a pipeline",
        expected_outcome="Pipeline created",
        max_turns=1,
        max_elicitation_rounds=2,
        initial_prompt="Create a pipeline",
        elicitation_hints={
            "intents": {"pipeline_name": "payments"},
            "matchers": [{"intent": "pipeline_name", "question_contains": ["pipeline", "name"]}],
        },
    )
    calls = 0

    async def agent_fn(messages: list[Message], system_event: dict | None = None) -> Message:
        nonlocal calls
        calls += 1
        if calls == 1:
            request = {
                "v": [
                    {
                        "name": "mcp__harness_local__validate_pipeline_yaml",
                        "arguments": {"yaml": "pipeline: {}"},
                    }
                ]
            }
            return Message(
                role="assistant",
                metadata={
                    "pending_elicitation": {
                        "type": "elicitation_free_text",
                        "payload": {
                            "review_id": "ask-pipeline-name",
                            "content": {"question": "What pipeline name should I use?"},
                        },
                    },
                    "sse_events": {"assistant_tool_request": [request]},
                    "sse_timeline": [{"event": "assistant_tool_request", "payload": request}],
                },
            )

        request = {
            "v": [
                {
                    "name": "mcp__harness__harness_create",
                    "arguments": {"resource_type": "pipeline_v1"},
                }
            ]
        }
        return Message(
            role="assistant",
            content="Pipeline created.",
            metadata={
                "sse_events": {"assistant_tool_request": [request]},
                "sse_timeline": [{"event": "assistant_tool_request", "payload": request}],
            },
        )

    simulator = ConversationSimulator(simulator_llm=StopLLM(), elicitation_simulator=ElicitationSimulator())
    eval_case = await simulator.simulate(golden, agent_fn)

    expected_names = ["validate_pipeline_yaml", "harness_create"]
    assert [tool_call.name for tool_call in eval_case.tool_calls or []] == expected_names
    assert len(eval_case.metadata["sse_events"]["assistant_tool_request"]) == 2
    message_tool_names = [
        tool_call.name
        for message in eval_case.messages or []
        for tool_call in message.tool_calls or []
        if message.role == "assistant"
    ]
    assert message_tool_names == expected_names


@pytest.mark.unit
async def test_simulator_stops_elicitation_loop_at_round_cap():
    golden = ConversationGolden(
        scenario="Create a k8s connector",
        expected_outcome="Connector created",
        max_turns=1,
        max_elicitation_rounds=2,
        initial_prompt="Create a k8s connector",
        elicitation_hints={
            "intents": {"connector_name": "testconnector"},
            "matchers": [{"intent": "connector_name", "question_contains": ["name", "connector"]}],
        },
    )

    async def agent_fn(messages: list[Message], system_event: dict | None = None) -> Message:
        return Message(
            role="assistant",
            metadata={
                "pending_elicitation": {
                    "type": "elicitation_free_text",
                    "payload": {
                        "review_id": "ask-name",
                        "content": {"question": "What name would you like for the connector?"},
                    },
                }
            },
        )

    simulator = ConversationSimulator(simulator_llm=StopLLM(), elicitation_simulator=ElicitationSimulator())
    eval_case = await simulator.simulate(golden, agent_fn)

    final = eval_case.messages[-1]
    assert final.metadata["elicitation_error"] == "max_elicitation_rounds_exceeded"
    assert final.metadata["elicitation_rounds"] == 2
    simulated = [m for m in eval_case.messages if (m.metadata or {}).get("simulated")]
    assert len(simulated) == 2


@pytest.mark.unit
async def test_simulator_marks_incomplete_when_empty_after_elicitation():
    golden = ConversationGolden(
        scenario="Create a cost category",
        expected_outcome="Category created",
        max_turns=1,
        max_elicitation_rounds=4,
        initial_prompt="Let's create a Cost Category",
        elicitation_hints={
            "intents": {"category_name": "eval_cost_category_test"},
            "matchers": [{"intent": "category_name", "question_contains": ["name"]}],
        },
        metadata={
            "sse_checks": [
                {
                    "event": "assistant_tool_request",
                    "match": [{"path": "$.name", "contains": "harness_create"}],
                }
            ]
        },
    )
    calls = 0

    async def agent_fn(messages: list[Message], system_event: dict | None = None) -> Message:
        nonlocal calls
        calls += 1
        if calls == 1:
            return Message(
                role="assistant",
                content="",
                metadata={
                    "pending_elicitation": {
                        "type": "elicitation_free_text",
                        "payload": {
                            "review_id": "ask-name",
                            "content": {"question": "What name would you like?"},
                        },
                    }
                },
            )
        # Resume returns empty with no pending and no create tool — incomplete.
        return Message(role="assistant", content="", metadata={"sse_events": {}})

    simulator = ConversationSimulator(simulator_llm=StopLLM(), elicitation_simulator=ElicitationSimulator())
    eval_case = await simulator.simulate(golden, agent_fn)

    assert "incomplete_empty_after_elicitation" in (eval_case.metadata.get("elicitation_error") or "")
    assert eval_case.metadata["elicitation_rounds"] == 1


@pytest.mark.unit
async def test_plain_text_followup_uses_elicitation_hints():
    golden = ConversationGolden(
        scenario="Create a cost category",
        expected_outcome="Category created",
        max_turns=1,
        max_elicitation_rounds=4,
        initial_prompt="Let's create a Cost Category",
        elicitation_hints={
            "intents": {"category_name": "eval_cost_category_test"},
            "matchers": [
                {
                    "intent": "category_name",
                    "question_contains": ["name your cost category"],
                }
            ],
        },
    )
    calls = 0

    async def agent_fn(messages: list[Message], system_event: dict | None = None) -> Message:
        nonlocal calls
        calls += 1
        if calls == 1:
            return Message(
                role="assistant",
                content="What would you like to name your Cost Category?",
            )
        return Message(role="assistant", content="Thanks, creating it now.")

    simulator = ConversationSimulator(simulator_llm=StopLLM(), elicitation_simulator=ElicitationSimulator())
    eval_case = await simulator.simulate(golden, agent_fn)

    assert calls == 2
    simulated = [m for m in eval_case.messages if (m.metadata or {}).get("plain_text_followup")]
    assert len(simulated) == 1
    assert simulated[0].content == "eval_cost_category_test"
    assert eval_case.metadata["elicitation_trace"][0]["kind"] == "plain_text_user_reply"


@pytest.mark.unit
async def test_plain_text_followup_ignores_incidental_keywords_in_report_body():
    golden = ConversationGolden(
        scenario="Find account-level templates referenced and actively used in pipelines",
        expected_outcome="Template usage summary",
        max_turns=1,
        max_elicitation_rounds=4,
        initial_prompt="Check templates used in pipelines",
        elicitation_hints={
            "intents": {"search_scope": "account-wide"},
            "matchers": [
                {
                    "intent": "search_scope",
                    "question_contains": ["account level", "account-wide"],
                }
            ],
        },
    )
    calls = 0

    async def agent_fn(messages: list[Message], system_event: dict | None = None) -> Message:
        nonlocal calls
        calls += 1
        return Message(
            role="assistant",
            content=(
                "Summary:\n\n"
                "- testentitysetup (Secret Manager) - Account level\n\n"
                "Would you like me to:\n"
                "- Check a specific project's pipelines for template references?\n"
                "- List all templates in a specific org or project?"
            ),
        )

    simulator = ConversationSimulator(simulator_llm=StopLLM(), elicitation_simulator=ElicitationSimulator())
    eval_case = await simulator.simulate(golden, agent_fn)

    assert calls == 1
    assert not any((m.metadata or {}).get("plain_text_followup") for m in eval_case.messages or [])


@pytest.mark.unit
async def test_eval_case_sse_events_live_at_top_level_not_on_messages():
    golden = ConversationGolden(
        scenario="List projects",
        expected_outcome="Projects listed",
        max_turns=1,
        initial_prompt="List projects",
    )

    async def agent_fn(messages: list[Message], system_event: dict | None = None) -> Message:
        return Message(
            role="assistant",
            content="Here are the projects.",
            metadata={
                "sse_events": {
                    "assistant_tool_request": [{"v": [{"name": "mcp__harness__harness_list"}]}],
                    "assistant_tool_result": [{"v": [{"name": "mcp__harness__harness_list", "result": "{}"}]}],
                    "assistant_message": [{"v": "Here are the projects."}],
                }
            },
        )

    simulator = ConversationSimulator(simulator_llm=StopLLM())
    eval_case = await simulator.simulate(golden, agent_fn)

    assert "assistant_tool_request" in eval_case.metadata["sse_events"]
    for msg in eval_case.messages or []:
        assert "sse_events" not in (msg.metadata or {})


@pytest.mark.unit
async def test_simulator_inserts_tool_calls_and_results_in_message_order():
    golden = ConversationGolden(
        scenario="List projects",
        expected_outcome="Projects listed",
        max_turns=1,
        initial_prompt="List projects",
    )

    async def agent_fn(messages: list[Message], system_event: dict | None = None) -> Message:
        return Message(
            role="assistant",
            content="Here are the projects.",
            metadata={
                "sse_events": {
                    "assistant_tool_request": [
                        {"v": [{"name": "harness_list", "arguments": {"resource_type": "project"}}]}
                    ],
                    "assistant_tool_result": [{"v": [{"name": "harness_list", "result": {"items": [{"id": "p1"}]}}]}],
                },
                "sse_timeline": [
                    {
                        "event": "assistant_tool_request",
                        "payload": {
                            "v": [
                                {
                                    "name": "harness_list",
                                    "arguments": {"resource_type": "project"},
                                }
                            ]
                        },
                    },
                    {
                        "event": "assistant_tool_result",
                        "payload": {
                            "v": [
                                {
                                    "name": "harness_list",
                                    "result": {"items": [{"id": "p1"}]},
                                }
                            ]
                        },
                    },
                    {"event": "assistant_message", "payload": {"v": "Here are the projects."}},
                ],
            },
        )

    simulator = ConversationSimulator(simulator_llm=StopLLM())
    eval_case = await simulator.simulate(golden, agent_fn)

    assert [message.role for message in eval_case.messages or []] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    tool_request = eval_case.messages[1]
    assert tool_request.tool_calls[0].name == "harness_list"
    assert tool_request.tool_calls[0].input == {"resource_type": "project"}
    tool_result = eval_case.messages[2]
    assert tool_result.tool_calls[0].name == "harness_list"
    assert tool_result.tool_calls[0].output == {"items": [{"id": "p1"}]}
    assert json.loads(tool_result.content) == {"items": [{"id": "p1"}]}
    assert eval_case.messages[3].content == "Here are the projects."
    assert "sse_timeline" not in (eval_case.messages[3].metadata or {})
    serialized_messages = eval_case.to_dict()["messages"]
    assert serialized_messages[1]["tool_calls"][0]["name"] == "harness_list"
    assert serialized_messages[1]["tool_calls"][0]["input"] == {"resource_type": "project"}
    assert serialized_messages[2]["tool_calls"][0]["name"] == "harness_list"
    assert serialized_messages[2]["tool_calls"][0]["output"] == {"items": [{"id": "p1"}]}


@pytest.mark.unit
def test_plain_text_followup_answers_a_request_phrased_without_a_question_mark():
    golden = ConversationGolden(
        scenario="Create CI",
        expected_outcome="Reviewed",
        elicitation_hints={
            "match_strategy": "specific",
            "intents": {"repository": "harness/harness"},
            "matchers": [{"intent": "repository", "question_contains": ["repository name"]}],
        },
    )
    message = Message(
        role="assistant",
        content="I still need the GitHub repository name to complete the pipeline. "
        "Please share the full repository path (e.g. `acme/app`).",
    )

    followup = _plain_text_followup(golden, message, used_intents=set())

    assert followup is not None and followup["answer"] == "harness/harness"


def _followup_golden() -> ConversationGolden:
    return ConversationGolden(
        scenario="Create CI then revise",
        expected_outcome="Revised",
        initial_prompt="Set up CI for my Go service.",
        turns=[
            Message(role="user", content="Set up CI for my Go service."),
            Message(role="user", content="Please add `go build ./...` after the test step."),
        ],
        mode=ConversationMode.SIMULATE,
    )


def test_scripted_followup_waits_for_a_yaml_review():
    history = [
        Message(role="user", content="Set up CI for my Go service."),
        Message(role="assistant", content="Which repo?"),
    ]

    assert _due_scripted_followup(_followup_golden(), history) is None


def test_scripted_followup_is_sent_once_after_a_yaml_review():
    golden = _followup_golden()
    history = [
        Message(role="user", content="Set up CI for my Go service."),
        Message(role="user", content="[Simulated YAML review: reject pipeline_v1]"),
        Message(role="assistant", content="Draft cancelled."),
    ]

    followup = _due_scripted_followup(golden, history)
    history.append(Message(role="user", content=followup))

    assert followup == "Please add `go build ./...` after the test step."
    assert _due_scripted_followup(golden, history) is None


def test_plain_text_followup_answers_every_item_of_a_listed_request():
    golden = ConversationGolden(
        scenario="Create CI",
        expected_outcome="Reviewed",
        elicitation_hints={
            "match_strategy": "specific",
            "intents": {"repository": "nginx", "bucket": "acme-artifacts", "region": "us-east-1"},
            "matchers": [
                {"intent": "repository", "question_contains": ["repository name"]},
                {"intent": "bucket", "question_contains": ["bucket"]},
                {"intent": "region", "question_contains": ["region"]},
            ],
        },
    )
    message = Message(
        role="assistant",
        content=(
            "I need several values. Please provide the following:\n\n"
            "1. **GitHub repository name** — the repo that holds your service\n"
            "2. **S3 bucket name** — where the binary is uploaded\n"
            "3. **S3 region** — the AWS region for that bucket\n"
            "4. **Retention days** — how long to keep it"
        ),
    )

    followup = _plain_text_followup(golden, message, used_intents={"region"})

    assert followup is not None
    assert followup["intents"] == ["repository", "bucket"]
    assert followup["answer"] == "Repository: nginx\nBucket: acme-artifacts"


def test_plain_text_followup_answers_every_value_asked_in_one_sentence():
    golden = ConversationGolden(
        scenario="Create CI",
        expected_outcome="Reviewed",
        elicitation_hints={
            "match_strategy": "specific",
            "intents": {"bucket": "acme-artifacts", "region": "us-east-1", "repository": "nginx"},
            "matchers": [
                {"intent": "repository", "question_contains": ["repository name"]},
                {"intent": "bucket", "question_contains": ["bucket"]},
                {"intent": "region", "question_contains": ["region"]},
            ],
        },
    )
    message = Message(role="assistant", content="Got it. Which S3 bucket should the binary go to, and in which region?")

    followup = _plain_text_followup(golden, message, used_intents=set())

    assert followup is not None
    assert followup["answer"] == "Bucket: acme-artifacts\nRegion: us-east-1"


@pytest.mark.unit
async def test_unanswerable_question_after_a_yaml_review_ends_the_conversation():
    golden = ConversationGolden(
        scenario="Create CI",
        expected_outcome="Reviewed",
        mode=ConversationMode.SCRIPTED,
        turns=[Message(role="user", content="Create CI")],
        max_elicitation_rounds=4,
        elicitation_hints={"intents": {"repository": "nginx"}, "yaml": {"default_action": "reject"}},
    )
    review = {"review_id": "rev-1", "content": {"yaml": "pipeline: {}"}, "entity_info": {"entity_type": "pipeline_v1"}}

    async def agent_fn(messages: list[Message], human_input: dict | None = None) -> Message:
        if human_input is None:
            return Message(
                role="assistant",
                metadata={
                    "pending_elicitation": {"type": "elicitation_yaml", "payload": review},
                    "sse_events": {"elicitation_yaml": [review]},
                },
            )
        return Message(
            role="assistant",
            metadata={
                "pending_elicitation": {
                    "type": "elicitation_select",
                    "payload": {
                        "review_id": "ask-change",
                        "content": {
                            "question": "What would you like to change in the pipeline?",
                            "options": [{"id": "0", "label": "Build command"}, {"id": "1", "label": "Pipeline name"}],
                        },
                    },
                }
            },
        )

    simulator = ConversationSimulator(simulator_llm=StopLLM(), elicitation_simulator=ElicitationSimulator())
    eval_case = await simulator.simulate(golden, agent_fn)

    assert eval_case.metadata["sse_events"]["elicitation_yaml"][-1]["content"]["yaml"] == "pipeline: {}"


@pytest.mark.unit
async def test_unrelated_error_after_a_yaml_review_still_propagates():
    golden = ConversationGolden(
        scenario="Create CI",
        expected_outcome="Reviewed",
        mode=ConversationMode.SCRIPTED,
        turns=[Message(role="user", content="Create CI")],
        max_elicitation_rounds=4,
        elicitation_hints={"yaml": {"default_action": "reject"}},
    )
    review = {"review_id": "rev-1", "content": {"yaml": "pipeline: {}"}, "entity_info": {"entity_type": "pipeline_v1"}}

    class BrokenSelectSimulator(ElicitationSimulator):
        async def respond(self, pending, golden, history):
            if pending.type == "elicitation_select":
                raise ValueError("malformed LLM response")
            return await super().respond(pending, golden, history)

    async def agent_fn(messages: list[Message], human_input: dict | None = None) -> Message:
        if human_input is None:
            return Message(
                role="assistant",
                metadata={
                    "pending_elicitation": {"type": "elicitation_yaml", "payload": review},
                    "sse_events": {"elicitation_yaml": [review]},
                },
            )
        return Message(
            role="assistant",
            metadata={
                "pending_elicitation": {
                    "type": "elicitation_select",
                    "payload": {"review_id": "ask", "content": {"question": "Next?", "options": [{"label": "A"}]}},
                }
            },
        )

    simulator = ConversationSimulator(simulator_llm=StopLLM(), elicitation_simulator=BrokenSelectSimulator())

    with pytest.raises(ValueError, match="malformed LLM response"):
        await simulator.simulate(golden, agent_fn)


def test_plain_text_followup_answers_a_please_reply_with_request():
    golden = ConversationGolden(
        scenario="Create CI",
        expected_outcome="Reviewed",
        elicitation_hints={
            "match_strategy": "specific",
            "intents": {"repository": "nginx"},
            "matchers": [{"intent": "repository", "question_contains": ["repository name"]}],
        },
    )
    message = Message(
        role="assistant",
        content="The connector covers multiple repositories. Please reply with the repository name (e.g. `payments`).",
    )

    followup = _plain_text_followup(golden, message, used_intents=set())

    assert followup is not None and followup["answer"] == "nginx"


@pytest.mark.unit
def test_plain_text_followup_default_mode_answers_only_the_first_matched_value():
    golden = ConversationGolden(
        scenario="Create CI",
        expected_outcome="Reviewed",
        elicitation_hints={
            "intents": {"bucket": "acme-artifacts", "region": "us-east-1"},
            "matchers": [
                {"intent": "bucket", "question_contains": ["bucket"]},
                {"intent": "region", "question_contains": ["region"]},
            ],
        },
    )
    message = Message(role="assistant", content="Which S3 bucket should the binary go to, and in which region?")

    followup = _plain_text_followup(golden, message, used_intents=set())

    assert followup == {"intent": "bucket", "answer": "acme-artifacts", "intents": ["bucket"]}


@pytest.mark.unit
def test_plain_text_followup_default_mode_ignores_a_request_without_a_question_mark():
    golden = ConversationGolden(
        scenario="Create CI",
        expected_outcome="Reviewed",
        elicitation_hints={
            "intents": {"repository": "nginx"},
            "matchers": [{"intent": "repository", "question_contains": ["repository name"]}],
        },
    )
    message = Message(
        role="assistant",
        content="The connector covers several repositories, so I need the repository name to continue.",
    )

    assert _plain_text_followup(golden, message, used_intents=set()) is None


def test_identical_scripted_followups_are_each_sent_after_their_own_review():
    repeat = "Run the tests again."
    golden = ConversationGolden(
        scenario="Create CI then revise twice",
        expected_outcome="Revised",
        initial_prompt="Set up CI for my Go service.",
        turns=[
            Message(role="user", content="Set up CI for my Go service."),
            Message(role="user", content=repeat),
            Message(role="user", content=repeat),
        ],
        mode=ConversationMode.SIMULATE,
    )
    history = [
        Message(role="user", content="Set up CI for my Go service."),
        Message(role="user", content="[Simulated YAML review: reject pipeline_v1]"),
        Message(role="user", content=repeat),
        Message(role="user", content="[Simulated YAML review: reject pipeline_v1]"),
    ]

    assert _due_scripted_followup(golden, history) == repeat
    history.append(Message(role="user", content=repeat))
    assert _due_scripted_followup(golden, history) is None
