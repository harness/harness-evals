"""Tests for the production conversation dataset preparation and judge."""

from __future__ import annotations

import asyncio
import csv
import json
import sys
from pathlib import Path

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.llm.base import BaseLLM

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "examples" / "langfuse-prod-datasets"
sys.path.insert(0, str(DATASET_ROOT / "scripts"))
sys.path.insert(0, str(DATASET_ROOT / "metrics"))

sys.path.insert(0, str(REPO_ROOT))

import build_agent_transcripts as transcript_builder  # noqa: E402
import build_conversation_goldens as golden_builder  # noqa: E402
from build_agent_transcripts import build_canonical_conversation  # noqa: E402
from build_review_batches import build_eval_case, eligibility_reasons  # noqa: E402
from conversation_quality import (  # noqa: E402
    HarnessConversationQualityMetric,
    format_conversation,
    normalize_categories,
)
from examples.outcome_goal_metric import OutcomeGoalAccuracyMetric  # noqa: E402

from harness_evals.conversation.golden import ConversationGolden, ConversationMode  # noqa: E402
from harness_evals.conversation.human_input import resolve_intent  # noqa: E402


class FakeJudgeLLM(BaseLLM):
    def __init__(self) -> None:
        self.prompt = ""
        self.system_prompt = ""

    async def generate(self, prompt: str, **kwargs: object) -> str:
        return ""

    async def generate_json(self, prompt: str, schema: dict, **kwargs: object) -> dict:
        self.prompt = prompt
        self.system_prompt = str(kwargs.get("system_prompt") or "")
        return {
            "usefulness": "useful",
            "quality": "good",
            "golden_readiness": "ready",
            "goal_achievement": 0.9,
            "resolution": 0.8,
            "tool_use_quality": 0.7,
            "confidence": 0.85,
            "reasoning": "The tool evidence supports the final answer.",
            "evidence": ["The diagnose call returned the failed step."],
        }


def sample_turns(response: str = "failed step: tests") -> list[dict]:
    return [
        {
            "sequence": 1,
            "trace_id": "trace-1",
            "user_text": "Why did my pipeline fail?",
            "events": [
                {"kind": "assistant_text", "text": "I will inspect the execution."},
                {
                    "kind": "tool",
                    "id": "tool-1",
                    "name": "harness_diagnose",
                    "request": {"execution_id": "exec-1"},
                    "response": response,
                },
            ],
            "final": "The test step failed.",
            "cost_usd": 0.1,
        }
    ]


@pytest.mark.unit
def test_canonical_conversation_preserves_order_and_bounds_tool_output():
    canonical = build_canonical_conversation(
        {
            "conversation_id": "conversation-1",
            "env": "prod1",
            "module": "ci",
            "org_id": "default",
            "project_id": "project",
            "first_timestamp": "2026-07-20T00:00:00Z",
        },
        sample_turns("x" * 3_000),
        sample_type="module-coverage",
        transcript_file="conversation.md",
        tools_file="conversation.tools.json",
    )

    assert [message["role"] for message in canonical["messages"]] == [
        "user",
        "assistant",
        "assistant",
        "assistant",
    ]
    tool_message = canonical["messages"][2]
    assert tool_message["tool_calls"][0]["name"] == "harness_diagnose"
    assert "tool response truncated" in tool_message["tool_calls"][0]["output"]
    assert canonical["metadata"]["truncated_tool_use_ids"] == ["tool-1"]
    assert canonical["input"] == "Why did my pipeline fail?"
    assert canonical["output"] == "The test step failed."


@pytest.mark.unit
def test_eligibility_and_eval_case_conversion(tmp_path):
    canonical = build_canonical_conversation(
        {
            "conversation_id": "conversation-1",
            "env": "prod1",
            "module": "ci",
            "org_id": None,
            "project_id": None,
            "first_timestamp": "2026-07-20T00:00:00Z",
        },
        sample_turns(),
        sample_type="random",
        transcript_file="conversation.md",
        tools_file="conversation.tools.json",
    )

    assert eligibility_reasons(canonical) == []
    eval_case = build_eval_case(canonical, tmp_path / "conversation.conversation.json")
    assert isinstance(eval_case, EvalCase)
    assert eval_case.messages and eval_case.messages[2].tool_calls
    assert eval_case.meta("conversation_id") == "conversation-1"
    assert eval_case.tags == {"environment": "prod1", "module": "ci", "sample_type": "random"}


@pytest.mark.unit
def test_structurally_empty_conversation_is_ineligible():
    reasons = eligibility_reasons(
        {
            "conversation_id": "conversation-1",
            "messages": [],
            "metadata": {"trace_ids": ["trace-1"]},
        }
    )
    assert reasons == ["missing_messages"]


@pytest.mark.unit
def test_normalize_categories_keeps_quality_and_readiness_orthogonal():
    usefulness, quality, readiness, final = normalize_categories("useful", "bad", "needs_rewrite")
    assert usefulness == "useful"
    assert quality == "bad"
    assert readiness == "needs_rewrite"
    assert final == "bad"


@pytest.mark.unit
def test_legacy_needs_improvement_review_maps_to_quality_and_readiness():
    quality, readiness = golden_builder._resolve_review_labels({"final_category": "needs_improvement"})
    assert quality == "good"
    assert readiness == "needs_rewrite"


@pytest.mark.unit
def test_quality_metric_reads_complete_messages_and_tool_calls(tmp_path):
    canonical = build_canonical_conversation(
        {
            "conversation_id": "conversation-1",
            "env": "prod1",
            "module": "ci",
            "org_id": None,
            "project_id": None,
            "first_timestamp": "2026-07-20T00:00:00Z",
        },
        sample_turns(),
        sample_type="random",
        transcript_file="conversation.md",
        tools_file="conversation.tools.json",
    )
    eval_case = build_eval_case(canonical, tmp_path / "conversation.conversation.json")
    llm = FakeJudgeLLM()

    score = asyncio.run(HarnessConversationQualityMetric(llm=llm).a_measure(eval_case))

    assert score.metadata
    assert score.metadata["final_category"] == "good"
    assert score.metadata["quality"] == "good"
    assert score.metadata["golden_readiness"] == "ready"
    assert score.metadata["confidence"] == 0.85
    assert "Why did my pipeline fail?" in llm.prompt
    assert "ASSISTANT_TOOL_CALL:harness_diagnose" in llm.prompt
    assert "failed step: tests" in llm.prompt
    assert "strict evaluator" in llm.system_prompt
    assert score.value == pytest.approx(0.825)


@pytest.mark.unit
def test_formatter_does_not_include_review_notes():
    eval_case = EvalCase.from_dict(
        {
            "input": "Question",
            "output": "Answer",
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ],
            "metadata": {"review_notes": "Human says this is bad"},
        }
    )
    formatted = format_conversation(eval_case)
    assert "Question" in formatted
    assert "Answer" in formatted
    assert "Human says this is bad" not in formatted


@pytest.mark.unit
def test_backfill_uses_cache_and_updates_labels_without_network(tmp_path, monkeypatch):
    sample_dir = tmp_path / "random"
    cache_dir = tmp_path / "cache"
    sample_dir.mkdir()
    cache_dir.mkdir()
    labels_path = sample_dir / "labels.csv"
    fieldnames = [
        "sample_type",
        "conversation_id",
        "env",
        "module",
        "org_id",
        "project_id",
        "num_turns",
        "num_tool_calls",
        "first_timestamp",
        "trace_ids",
        "transcript_file",
        "label",
        "notes",
    ]
    with labels_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "sample_type": "random",
                "conversation_id": "conversation-1",
                "env": "prod1",
                "module": "ci",
                "org_id": "",
                "project_id": "",
                "num_turns": "1",
                "num_tool_calls": "0",
                "first_timestamp": "2026-07-20T00:00:00Z",
                "trace_ids": "trace-1",
                "transcript_file": "01-prod1-ci-conversation.md",
                "label": "good",
                "notes": "keep this note",
            }
        )
    trace = {
        "id": "trace-1",
        "timestamp": "2026-07-20T00:00:00Z",
        "observations": [
            {
                "name": "llm_turn_1",
                "input": {"messages": [{"role": "user", "content": "Why did it fail?"}]},
                "output": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "The test step failed."}],
                },
            }
        ],
    }
    (cache_dir / "trace-1.json").write_text(json.dumps(trace))
    monkeypatch.setattr(transcript_builder, "CACHE_DIR", cache_dir)

    created, missing = transcript_builder.backfill_canonical_conversations(
        sample_dir,
        dataset_name="random",
    )

    assert created == 1
    assert missing == []
    assert (sample_dir / "01-prod1-ci-conversation.conversation.json").is_file()
    with labels_path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["conversation_file"] == "01-prod1-ci-conversation.conversation.json"
    assert row["tools_file"] == "01-prod1-ci-conversation.tools.json"
    assert row["label"] == "good"
    assert row["notes"] == "keep this note"


# --- build_conversation_goldens ------------------------------------------------


def _conv(messages, tool_calls=None, *, module="ci", org="myorg", project="myproj", cid="c-1234-5678"):
    return {
        "conversation_id": cid,
        "messages": messages,
        "tool_calls": tool_calls or [],
        "metadata": {
            "module": module,
            "environment": "prod0",
            "org_id": org,
            "project_id": project,
            "transcript_file": "x.md",
        },
    }


@pytest.mark.unit
def test_review_gate_injections_are_not_real_user_turns():
    conv = _conv(
        [
            {"role": "user", "content": "create a feature flag"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "The user approved the entity. Call harness_create again."},
            {"role": "user", "content": "The user provided the following values:\n- name: foo"},
        ]
    )
    assert golden_builder._user_messages(conv) == ["create a feature flag"]


@pytest.mark.unit
def test_error_analysis_is_detected():
    conv = _conv(
        [{"role": "user", "content": "analyze the error for this pipeline execution"}],
    )
    assert golden_builder.is_error_analysis(conv, "") is True

    conv2 = _conv(
        [{"role": "user", "content": "hello"}],
        tool_calls=[{"name": "mcp__harness__harness_diagnose", "input": {}}],
    )
    assert golden_builder.is_error_analysis(conv2, "") is True


@pytest.mark.unit
def test_write_flow_and_scenario_type():
    write = _conv(
        [{"role": "user", "content": "create a service"}],
        tool_calls=[{"name": "mcp__harness__harness_create", "input": {"resource_type": "service"}}],
    )
    assert golden_builder.is_write_flow(write) is True
    assert golden_builder.scenario_type_of(write) == "write"

    read = _conv([{"role": "user", "content": "list my services"}])
    assert golden_builder.scenario_type_of(read) == "read_only"


@pytest.mark.unit
def test_read_only_named_resource_requires_override():
    conv = _conv([{"role": "user", "content": "refer the existing Foo template and summarize it"}])
    review = {"final_category": "good"}
    golden, record = golden_builder.build_golden(conv, review, None)
    assert golden is None
    assert record.decision == "excluded"
    assert "read-only flow references a production-specific resource" in record.reason


@pytest.mark.unit
def test_write_named_resource_requires_override():
    conv = _conv(
        [{"role": "user", "content": "refer the Bar template and create a new one"}],
        tool_calls=[
            {"name": "mcp__harness__harness_get", "input": {"resource_id": "Bar"}},
            {"name": "mcp__harness__harness_create", "input": {"resource_type": "template"}},
        ],
    )
    golden, record = golden_builder.build_golden(conv, {"final_category": "good"}, None)
    assert golden is None
    assert record.decision == "excluded"
    assert "curated override" in record.reason


@pytest.mark.unit
def test_ce_cost_category_override_is_write_with_elicitation_hints():
    conv = _conv([{"role": "user", "content": "Let's create a Cost Category"}])
    override = {
        "scenario_type": "write",
        "scenario": "Create a CCM cost category at account scope",
        "expected_outcome": "Cost category eval_cost_category_${EVAL_RUN_SUFFIX} is created via harness_create",
        "initial_prompt": "Let's create a Cost Category",
        "max_elicitation_rounds": 24,
        "elicitation_hints": {
            "intents": {"category_name": "eval_cost_category_${EVAL_RUN_SUFFIX}"},
            "matchers": [{"intent": "category_name", "question_contains": ["name this cost category"]}],
            "yaml": {"default_action": "accept"},
        },
        "sse_checks": [
            {
                "event": "assistant_tool_request",
                "path": "$.v[*]",
                "match": [
                    {"path": "$.name", "contains": "harness_create"},
                    {"path": "$.arguments.resource_type", "equals": "cost_category"},
                ],
            },
        ],
    }
    golden, record = golden_builder.build_golden(conv, {"final_category": "unclear"}, override)
    assert record.decision == "emitted"
    assert golden["tags"]["scenario_type"] == "write"
    assert golden["max_elicitation_rounds"] == 24
    assert golden["elicitation_hints"]["llm_on_miss"] is True
    assert golden["elicitation_hints"]["intents"]["category_name"] == "eval_cost_category_${EVAL_RUN_SUFFIX}"
    assert {
        "event": "assistant_tool_request",
        "path": "$.v[*]",
        "match": [
            {"path": "$.name", "contains": "harness_create"},
            {"path": "$.arguments.resource_type", "equals": "cost_category"},
        ],
    } in golden["metadata"]["sse_checks"]


@pytest.mark.unit
def test_ce_golden_matches_observed_aws_bucket_questions():
    dataset_path = REPO_ROOT / "examples" / "prod-conversation.goldens.jsonl"
    ce_data = next(json.loads(line) for line in dataset_path.read_text().splitlines() if '"id":"ce-' in line)
    golden = ConversationGolden.from_dict(ce_data)

    assert (
        resolve_intent(
            "How would you like to group your AWS costs into buckets?",
            golden,
        )
        == "aws_grouping"
    )
    assert (
        resolve_intent(
            "How many cost buckets do you want to create?",
            golden,
        )
        == "bucket_count"
    )
    assert (
        resolve_intent(
            "What name should this bucket have?",
            golden,
        )
        == "cost_bucket_name"
    )


@pytest.mark.unit
def test_read_only_sse_checks_require_tool_result_and_primary_tool_name():
    conv = _conv(
        [{"role": "user", "content": "What projects have ai evals"}],
        tool_calls=[{"name": "mcp__harness__harness_list", "input": {"resource_type": "project"}}],
    )
    golden, record = golden_builder.build_golden(conv, {"final_category": "good"}, None)
    assert record.decision == "emitted"
    assert golden["elicitation_hints"]["llm_on_miss"] is True
    checks = golden["metadata"]["sse_checks"]
    assert {
        "event": "assistant_tool_request",
        "path": "$.v[*]",
        "match": [{"path": "$.name", "contains": "harness_list"}],
    } in checks
    assert {
        "event": "assistant_tool_result",
        "path": "$.v[*]",
        "match": [{"path": "$.name", "contains": "harness_list"}],
    } in checks
    assert {"event": "assistant_message", "exists": True} in checks


@pytest.mark.unit
def test_read_only_pipeline_verify_uses_validate_tool_in_sse_checks():
    conv = _conv(
        [{"role": "user", "content": "verify this pipeline yaml"}],
        tool_calls=[
            {"name": "Read", "input": {"file_path": "/tmp/rules.md"}},
            {"name": "mcp__harness_local__validate_pipeline_v1_yaml", "input": {"yaml": "pipeline:"}},
        ],
        module="code",
    )
    golden, record = golden_builder.build_golden(conv, {"final_category": "good"}, None)
    assert record.decision == "emitted"
    checks = golden["metadata"]["sse_checks"]
    assert any(
        c.get("event") == "assistant_tool_request"
        and c.get("match") == [{"path": "$.name", "contains": "validate_pipeline_v1_yaml"}]
        for c in checks
    )


@pytest.mark.unit
def test_write_override_produces_portable_golden():
    conv = _conv(
        [{"role": "user", "content": "refer the Bar template and create a new one"}],
        tool_calls=[{"name": "mcp__harness__harness_create", "input": {"resource_type": "template"}}],
    )
    override = {
        "scenario_type": "write",
        "scenario": "Create a reusable step template in ${HARNESS_PROJECT}",
        "expected_outcome": "A self-contained step template is created in ${HARNESS_PROJECT}",
        "initial_prompt": "Create a reusable step template",
    }
    golden, record = golden_builder.build_golden(conv, {"final_category": "good"}, override)
    assert record.decision == "emitted"
    assert record.action == "override"
    assert golden["tags"]["scenario_type"] == "write"
    checks = golden["metadata"]["sse_checks"]
    assert {
        "event": "assistant_tool_request",
        "path": "$.v[*]",
        "match": [{"path": "$.name", "contains": "harness_create"}],
    } in checks
    assert {
        "event": "assistant_tool_result",
        "path": "$.v[*]",
        "match": [{"path": "$.name", "contains": "harness_create"}],
    } in checks
    assert {"event": "assistant_message", "exists": True} in checks
    ConversationGolden.from_dict(golden)


@pytest.mark.unit
def test_override_can_exclude():
    conv = _conv([{"role": "user", "content": "do something"}])
    golden, record = golden_builder.build_golden(
        conv, {"final_category": "unclear"}, {"exclude": True, "reason": "oversized"}
    )
    assert golden is None
    assert record.action == "override"
    assert record.reason == "oversized"


@pytest.mark.unit
def test_secret_scan_blocks_emitted_pii():
    conv = _conv([{"role": "user", "content": "email me at alice@corp.com about my services"}])
    golden, record = golden_builder.build_golden(conv, {"final_category": "good"}, None)
    assert golden is None
    assert "secret/PII scan failed" in record.reason


@pytest.mark.unit
def test_useless_category_excluded():
    conv = _conv([{"role": "user", "content": "hi"}])
    golden, record = golden_builder.build_golden(conv, {"final_category": "useless"}, None)
    assert golden is None
    assert "useless" in record.reason


@pytest.mark.unit
def test_convert_end_to_end_all_goldens_validate(tmp_path):
    review = DATASET_ROOT / "results" / "module-coverage-030" / "review.csv"
    conversations = DATASET_ROOT / "module-coverage"
    overrides = DATASET_ROOT / "conversation-golden-overrides.json"
    output = tmp_path / "goldens.jsonl"
    if not review.exists() or not conversations.exists():
        pytest.skip("local production-conversation fixtures are not committed")

    emitted, total, manifest_path = golden_builder.convert(review, conversations, overrides, output)
    assert emitted > 0
    assert total > 0
    assert emitted <= total

    for line in output.read_text().splitlines():
        if not line.strip():
            continue
        golden = ConversationGolden.from_dict(json.loads(line))
        # Faithful replay of prod user turns.
        assert golden.mode == ConversationMode.SCRIPTED
        # No production scope identifiers leaked.
        blob = line
        assert "porgwdev" not in blob

    # Manifest accounts for every source conversation with a decision + reason.
    manifest_lines = [json.loads(x) for x in manifest_path.read_text().splitlines() if x.strip()]
    assert len(manifest_lines) == total
    assert all(rec.get("decision") in {"emitted", "excluded"} for rec in manifest_lines)


# --- outcome_goal_metric -------------------------------------------------------


class _OutcomeJudgeLLM(BaseLLM):
    def __init__(self, score: float = 0.9) -> None:
        self.prompt = ""
        self._score = score

    async def generate(self, prompt: str, **kwargs: object) -> str:
        return ""

    async def generate_json(self, prompt: str, schema: dict, **kwargs: object) -> dict:
        self.prompt = prompt
        return {"reasoning": "achieved", "score": self._score}


@pytest.mark.unit
def test_outcome_metric_injects_expected_outcome():
    eval_case = EvalCase.from_dict(
        {
            "input": "Create a service",
            "output": "done",
            "messages": [
                {"role": "user", "content": "Create a service"},
                {"role": "assistant", "content": "Service created"},
            ],
            "metadata": {"expected_outcome": "A Kubernetes service is created"},
        }
    )
    llm = _OutcomeJudgeLLM(score=0.9)
    score = asyncio.run(OutcomeGoalAccuracyMetric(llm=llm, threshold=0.7).a_measure(eval_case))
    assert "A Kubernetes service is created" in llm.prompt
    assert "Path tolerance" in llm.prompt
    assert "valid alternate workflow" in llm.prompt
    assert score.value == pytest.approx(0.9)
    assert score.passed


@pytest.mark.unit
def test_outcome_metric_includes_pending_and_tools():
    eval_case = EvalCase.from_dict(
        {
            "input": "Create a cost category",
            "output": "",
            "messages": [
                {"role": "user", "content": "Let's create a Cost Category"},
                {
                    "role": "assistant",
                    "content": "",
                    "metadata": {
                        "pending_human_input": {
                            "type": "elicitation_select",
                            "payload": {
                                "content": {
                                    "question": "What cost buckets would you like to create?",
                                    "items": [
                                        {"id": "0", "label": "Single bucket"},
                                        {"id": "1", "label": "Multiple buckets"},
                                    ],
                                }
                            },
                        }
                    },
                },
                {
                    "role": "user",
                    "content": "Single bucket",
                    "metadata": {"simulated": True, "elicitation_type": "elicitation_select"},
                },
                {"role": "assistant", "content": ""},
            ],
            "metadata": {
                "expected_outcome": "Cost category created via harness_create",
                "sse_events": {
                    "assistant_tool_request": [
                        {"v": [{"name": "harness_create", "arguments": {"resource_type": "cost_category"}}]}
                    ]
                },
            },
        }
    )
    llm = _OutcomeJudgeLLM(score=0.8)
    score = asyncio.run(OutcomeGoalAccuracyMetric(llm=llm, threshold=0.7).a_measure(eval_case))
    assert "What cost buckets would you like to create?" in llm.prompt
    assert "Single bucket" in llm.prompt
    assert "harness_create" in llm.prompt
    assert score.value == pytest.approx(0.8)
