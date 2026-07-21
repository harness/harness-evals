"""End-to-end config runner tests for conversation evals."""

import json

import pytest

from harness_evals.config.runner import run_config
from harness_evals.config.schema import ConversationSpec, EvalConfig, MetricSpec, TargetSpec, load_config
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.errors import HarnessEvalsError
from harness_evals.llm.base import BaseLLM
from harness_evals.plugins import register_metric
from harness_evals.refs import ResourceRef
from harness_evals.targets.conversational_streaming_http import ConversationalStreamingHttpTarget
from tests.conversation.k8s_connector_sse_fixtures import k8s_connector_turn_responses


class StopLLM(BaseLLM):
    async def generate(self, prompt: str, **kwargs) -> str:
        return "Create a k8s connector"

    async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
        return {"achieved": True, "reasoning": "done"}


@register_metric("conversation_always_pass")
class AlwaysPassMetric(BaseMetric):
    def __init__(self, threshold: float = 1.0, **kwargs):
        super().__init__(
            name="conversation_always_pass", dimension=Dimension.CORRECTNESS, threshold=threshold, **kwargs
        )

    def measure(self, eval_case: EvalCase) -> Score:
        return Score(name=self.name, value=1.0, threshold=self.threshold, metadata=eval_case.metadata)


@pytest.mark.unit
def test_config_runner_replays_k8s_connector_flow_through_conversation_target(tmp_path, monkeypatch):
    responses = iter(k8s_connector_turn_responses())
    requests: list[dict] = []

    def fake_execute(self, body: bytes, headers: dict[str, str]):
        requests.append(json.loads(body.decode("utf-8")))
        return next(responses), "text/event-stream", 1.0, None

    async def fake_execute_async(self, body: bytes, headers: dict[str, str]):
        return fake_execute(self, body, headers)

    monkeypatch.setattr(ConversationalStreamingHttpTarget, "_execute_with_retries", fake_execute)
    monkeypatch.setattr(ConversationalStreamingHttpTarget, "_execute_async", fake_execute_async)
    monkeypatch.setattr("harness_evals.config.runner.build_llm", lambda spec: StopLLM())

    dataset_path = tmp_path / "goldens.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "k8s-connector-create",
                "scenario": "Create a Kubernetes connector in the AICHAT project",
                "expected_outcome": "Connector 'testconnector' created",
                "mode": "simulate",
                "max_turns": 1,
                "max_elicitation_rounds": 6,
                "initial_prompt": "Create a k8s connector",
                "elicitation_hints": {
                    "intents": {
                        "auth_method": "Inherit from Delegate",
                        "scope": "Project (Recommended)",
                        "connector_name": "testconnector",
                        "delegate_selector": "hello",
                    },
                    "matchers": [
                        {"intent": "auth_method", "question_contains": ["auth", "authentication"]},
                        {"intent": "scope", "question_contains": ["scope", "project"]},
                        {"intent": "delegate_selector", "question_contains": ["delegate", "selector", "tag"]},
                        {
                            "intent": "connector_name",
                            "question_contains": ["name would you like", "name for the", "identifier", "connector"],
                        },
                    ],
                    "yaml": {"default_action": "accept"},
                },
                "metadata": {
                    "sse_checks": [
                        {"event": "elicitation_form", "exists": True},
                        {"event": "elicitation_yaml", "exists": True},
                        {"event": "entity_mutation", "path": "$.resource_type", "equals": "connector"},
                        {"event": "entity_mutation", "path": "$.identifier", "equals": "testconnector"},
                        {"event": "assistant_message", "exists": True},
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "conversation.eval.yaml"
    config_path.write_text(
        f"""\
name: conversation-runner-test
plugins:
  - examples.harness_sse_elicitation_adapter
  - examples.sse_events_match_metric
conversation:
  mode: simulate
  max_turns: 1
  max_elicitation_rounds: 6
  simulator_llm: {{provider: openai, name: gpt-4o-mini}}
  elicitation_adapter: harness_sse
dataset: {dataset_path}
target:
  type: conversational_streaming_http
  url: http://example.test/stream
  output_event: assistant_message
  output_path: $.v
  human_input_events:
    - elicitation_form
    - elicitation_free_text
    - elicitation_yaml
  completion_events:
    - assistant_message
  session_metadata_event: stream_metadata
  session_fields:
    conversation_id: conversation_id
    session_id: session_id
  correlation_id_field: review_id
  human_input_body_template:
    system_event: "{{{{human_input}}}}"
    conversation_id: "{{conversation_id}}"
    session_id: "{{session_id}}"
    stream: true
  capture_events:
    - elicitation_form
    - elicitation_free_text
    - elicitation_yaml
    - entity_mutation
    - assistant_message
metrics:
  - conversation_always_pass
  - kind: sse_events_match
    threshold: 0.8
    params:
      row_checks_key: sse_checks
sinks: []
""",
        encoding="utf-8",
    )

    scores = run_config(load_config(str(config_path)), baseline=None)

    assert scores[0][0].passed
    assert len(requests) == 6
    assert requests[0]["prompt"] == "Create a k8s connector"
    assert (
        requests[1]["system_event"]["result"]["form_values"][
            "Which authentication method should be used for the Kubernetes connector?"
        ]
        == "Inherit from Delegate"
    )
    assert requests[2]["system_event"]["result"]["free_text"] == "testconnector"
    assert requests[4]["system_event"]["result"]["free_text"] == "hello"
    assert requests[5]["system_event"]["result"]["action_id"] == "accept"

    sse_score = next(score for score in scores[0] if score.name == "sse_events_match")
    assert sse_score.passed
    assert sse_score.value == 1.0


@pytest.mark.unit
@pytest.mark.parametrize("source", ["http", "langfuse"])
def test_conversation_config_rejects_non_local_dataset_source(source: str) -> None:
    cfg = EvalConfig(
        name="conversation-non-local-dataset",
        dataset=ResourceRef(source=source, id="datasets/conversation-goldens"),
        target=TargetSpec(type="conversational_streaming_http", params={"url": "http://example.test/stream"}),
        metrics=[MetricSpec(kind="exact_match")],
        conversation=ConversationSpec(mode="simulate"),
        sinks=[],
    )

    with pytest.raises(HarnessEvalsError, match="only support local dataset sources"):
        run_config(cfg, baseline=None)
