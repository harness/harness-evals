"""Tests for the production conversation dataset preparation and judge.

Review (AIPLAT-952): Covers golden builder, Round 3/4 metrics, combined quality+candidate
CSV export, and readonly/write SSE check conventions. write_flow/read_only removed from scoring tests.
"""

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
from examples.outcome_goal_metric import OutcomeGoalAccuracyMetric, _format_conversation  # noqa: E402

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
    usefulness, quality, readiness, final = normalize_categories(
        "useful", "bad", "needs_rewrite"
    )
    assert usefulness == "useful"
    assert quality == "bad"
    assert readiness == "needs_rewrite"
    assert final == "bad"


@pytest.mark.unit
def test_legacy_needs_improvement_review_maps_to_quality_and_readiness():
    quality, readiness = golden_builder._resolve_review_labels(
        {"final_category": "needs_improvement"}
    )
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
    ce_data = next(
        json.loads(line)
        for line in dataset_path.read_text().splitlines()
        if '"id":"ce-' in line
    )
    golden = ConversationGolden.from_dict(ce_data)

    assert resolve_intent(
        "How would you like to group your AWS costs into buckets?",
        golden,
    ) == "aws_grouping"
    assert resolve_intent(
        "How many cost buckets do you want to create?",
        golden,
    ) == "bucket_count"
    assert resolve_intent(
        "What name should this bucket have?",
        golden,
    ) == "cost_bucket_name"


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
def test_read_only_golden_includes_expected_tool_calls_from_sse_checks():
    conv = _conv(
        [{"role": "user", "content": "What projects have ai evals"}],
        tool_calls=[{"name": "mcp__harness__harness_list", "input": {"resource_type": "project"}}],
    )
    golden, record = golden_builder.build_golden(conv, {"final_category": "good"}, None)
    assert record.decision == "emitted"
    assert golden["expected_tool_calls"] == [{"name": "harness_list"}]


def test_read_only_golden_skips_skill_in_expected_tool_calls():
    conv = _conv(
        [{"role": "user", "content": "load a skill"}],
        tool_calls=[{"name": "Skill", "input": {"skill": "debug-pipeline"}}],
    )
    override = {
        "scenario": "Portable scenario",
        "expected_outcome": "Uses harness_list",
        "initial_prompt": "List pipelines",
        "sse_checks": [
            {
                "event": "assistant_tool_request",
                "path": "$.v[*]",
                "match": [{"path": "$.name", "equals": "Skill"}],
            },
            {
                "event": "assistant_tool_request",
                "path": "$.v[*]",
                "match": [
                    {"path": "$.name", "contains": "harness_list"},
                    {"path": "$.arguments.resource_type", "equals": "pipeline"},
                ],
            },
        ],
    }
    golden, record = golden_builder.build_golden(conv, {"final_category": "good"}, override)
    assert record.decision == "emitted"
    assert golden["expected_tool_calls"] == [
        {"name": "harness_list", "input": {"resource_type": "pipeline"}}
    ]


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

    emitted, total, manifest_path = golden_builder.convert(review, [conversations], overrides, output)
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


@pytest.mark.unit
def test_outcome_metric_truncates_large_tool_trace():
    huge = "x" * 50_000
    eval_case = EvalCase.from_dict(
        {
            "input": "Audit templates",
            "output": "done",
            "messages": [
                {"role": "user", "content": "Check templates"},
                {
                    "role": "tool",
                    "content": huge,
                    "tool_calls": [
                        {
                            "name": "harness_list",
                            "input": {"resource_type": "template"},
                            "output": None,
                        }
                    ],
                },
                {"role": "assistant", "content": "Found 3 templates in use."},
            ],
            "metadata": {"expected_outcome": "Template usage summary with evidence"},
        }
    )
    text, meta = _format_conversation(
        eval_case.messages,
        eval_case.metadata or {},
        max_chars=100_000,
    )
    assert len(text) < 100_000
    assert "truncated" in text
    assert huge not in text
    assert meta["judge_conversation_truncated"] is False


@pytest.mark.unit
def test_outcome_metric_prod_readonly_row2_fits_judge_budget():
    results_path = REPO_ROOT / "examples" / "output" / "prod-conversation-readonly-results.jsonl"
    if not results_path.exists():
        pytest.skip("prod readonly results not present locally")
    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    if len(rows) < 2:
        pytest.skip("expected at least two readonly result rows")
    eval_case = EvalCase.from_dict(rows[1]["eval_case"])
    text, meta = _format_conversation(
        eval_case.messages,
        eval_case.metadata or {},
        max_chars=100_000,
    )
    assert len(text) <= 100_000
    assert meta["judge_conversation_chars"] == len(text)


def _minimal_conversation(**overrides: object) -> dict:
    base: dict = {
        "conversation_id": "conv-1",
        "input": "Let's create a Cost Category",
        "output": "Created.",
        "messages": [
            {"role": "user", "content": "Let's create a Cost Category"},
            {
                "role": "assistant",
                "content": "Called tool `Skill`.",
                "tool_calls": [
                    {
                        "name": "Skill",
                        "input": {"skill": "ccm-cost-categories"},
                        "output": "waiting for user to review the yaml",
                    }
                ],
            },
        ],
        "tool_calls": [
            {
                "name": "Skill",
                "input": {"skill": "ccm-cost-categories"},
                "output": "waiting for user to review the yaml",
            },
            {
                "name": "mcp__harness__harness_create",
                "input": {"resource_type": "cost_category"},
                "output": '{"status": "ERROR", "message": "failed"}',
            },
        ],
        "metadata": {
            "module": "ce",
            "environment": "prod1",
            "num_turns": 12,
            "num_tool_calls": 9,
            "total_cost_usd": 0.42,
            "truncated_tool_use_ids": ["tool-big"],
        },
    }
    base.update(overrides)
    return base


class FakeSignalsLLM(BaseLLM):
    def __init__(self, payload: dict | None = None) -> None:
        self.prompt = ""
        self.system_prompt = ""
        self.payload = payload or {
            "high_turns": True,
            "high_cost": True,
            "high_tool_count": True,
            "large_tool_output": True,
            "tool_failure": True,
            "skill_loading": True,
            "hitl_loop": True,
            "multi_turn": True,
            "module_tag": "module:ce",
            "signal_tags": [
                "high_turns",
                "high_cost",
                "high_tool_count",
                "large_tool_output",
                "tool_failure",
                "skill_loading",
                "hitl_loop",
                "multi_turn",
                "module:ce",
            ],
            "confidence": 0.9,
            "reasoning": "High-turn CE write with skill + review gate + tool error.",
            "evidence": ["num_turns=12", "Skill tool", "status ERROR"],
            "criterion_notes": {
                "hitl_loop": "same bucket question asked twice after answer",
            },
        }

    async def generate(self, prompt: str, **kwargs: object) -> str:
        return ""

    async def generate_json(self, prompt: str, schema: dict, **kwargs: object) -> dict:
        self.prompt = prompt
        self.system_prompt = str(kwargs.get("system_prompt") or "")
        return dict(self.payload)


@pytest.mark.unit
def test_conversation_signals_extracts_structural_facts_for_llm_prompt():
    from conversation_signals import extract_structural_facts

    facts = extract_structural_facts(_minimal_conversation())
    assert facts["num_turns"] == 12
    assert facts["total_cost_usd"] == 0.42
    assert facts["num_tool_calls"] == 9
    assert facts["truncated_tool_outputs"] == 1
    assert facts["module"] == "ce"
    assert "Skill" in facts["tool_names_sample"]


@pytest.mark.unit
def test_conversation_signals_quality_gate_only_good_and_bad():
    from conversation_signals import quality_eligible_for_signals, resolve_quality

    assert quality_eligible_for_signals("good")
    assert quality_eligible_for_signals("bad")
    assert not quality_eligible_for_signals("unclear")
    assert not quality_eligible_for_signals("useless")
    assert resolve_quality({"quality": "needs_improvement"}) == "good"


@pytest.mark.unit
def test_conversation_signals_llm_metric_assigns_tags_for_good_row():
    from conversation_signals import HarnessConversationSignalsMetric

    conversation = _minimal_conversation()
    eval_case = EvalCase.from_dict(
        {
            "input": conversation["input"],
            "output": conversation["output"],
            "messages": conversation["messages"],
            "tool_calls": conversation["tool_calls"],
            "metadata": {
                "round1_quality": "good",
                "canonical_conversation": conversation,
                "module": "ce",
            },
        }
    )
    llm = FakeSignalsLLM()
    score = asyncio.run(HarnessConversationSignalsMetric(llm=llm).a_measure(eval_case))
    assert "num_turns" in llm.prompt
    assert "0.42" in llm.prompt
    assert "high_turns" in score.metadata["signal_tags"]
    assert "skill_loading" in score.metadata["signal_tags"]
    assert "module:ce" in score.metadata["signal_tags"]
    assert score.metadata["skipped"] is False


@pytest.mark.unit
def test_conversation_signals_llm_metric_skips_unclear_quality():
    from conversation_signals import HarnessConversationSignalsMetric

    conversation = _minimal_conversation()
    eval_case = EvalCase.from_dict(
        {
            "input": conversation["input"],
            "output": conversation["output"],
            "messages": conversation["messages"],
            "tool_calls": conversation["tool_calls"],
            "metadata": {
                "round1_quality": "unclear",
                "canonical_conversation": conversation,
            },
        }
    )
    llm = FakeSignalsLLM()
    score = asyncio.run(HarnessConversationSignalsMetric(llm=llm).a_measure(eval_case))
    assert score.metadata["skipped"] is True
    assert score.metadata["signals_skipped_reason"] == "quality=unclear"
    assert llm.prompt == ""


@pytest.mark.unit
def test_conversation_candidate_score_structural_criteria():
    from conversation_candidate_score import compute_eval_candidate_score, compute_structural_criteria

    facts = {
        "module": "ce",
        "num_turns": 12,
        "total_cost_usd": 0.42,
        "num_tool_calls": 9,
        "truncated_tool_outputs": 1,
        "max_tool_output_bytes": 9000,
    }
    structural = compute_structural_criteria(facts)
    assert structural["high_turns"] is True
    assert structural["high_cost"] is True
    assert structural["high_tool_count"] is True
    assert structural["large_tool_output"] is True
    assert structural["multi_turn"] is True
    assert compute_eval_candidate_score({**structural, **dict.fromkeys(
        ("tool_failure", "skill_loading", "hitl_loop"), False
    )}) == round((5 / 8) * 5, 2)


@pytest.mark.unit
def test_conversation_candidate_score_llm_metric_scores_good_row():
    from conversation_candidate_score import HarnessConversationCandidateScoreMetric

    conversation = _minimal_conversation()
    eval_case = EvalCase.from_dict(
        {
            "input": conversation["input"],
            "output": conversation["output"],
            "messages": conversation["messages"],
            "tool_calls": conversation["tool_calls"],
            "metadata": {
                "usefulness": "useful",
                "canonical_conversation": conversation,
            },
        }
    )
    llm = FakeSignalsLLM()
    score = asyncio.run(HarnessConversationCandidateScoreMetric(llm=llm).a_measure(eval_case))
    assert score.metadata["eval_candidate_score"] == round((8 / 8) * 5, 2)
    assert score.metadata["criteria"]["tool_failure"] is True
    assert score.metadata["criteria"]["skill_loading"] is True
    assert "harness_create failed" in score.reason
    assert "12 turns" in score.reason or "12 turn" in score.reason
    assert "$0.42" in score.reason
    assert "ccm-cost-categories" in score.reason
    assert "Summary:" not in score.reason
    assert "/tmp/" not in score.reason
    assert llm.prompt


@pytest.mark.unit
def test_build_eval_candidate_reasoning_uses_concrete_values():
    from conversation_candidate_score import (
        build_eval_candidate_reasoning,
        compute_structural_criteria,
        extract_structural_facts,
        merge_criteria,
    )
    from conversation_signals import extract_structural_facts as signals_facts

    conversation = _minimal_conversation()
    facts = signals_facts(conversation)
    structural = compute_structural_criteria(facts)
    llm_result = {
        "tool_failure": True,
        "skill_loading": True,
        "hitl_loop": True,
        "criterion_notes": {
            "hitl_loop": "same bucket question asked twice after answer",
        },
    }
    criteria = merge_criteria(structural, llm_result, facts)
    reasoning = build_eval_candidate_reasoning(
        criteria,
        facts,
        conversation,
        llm_result,
        score_value=4.09,
    )
    assert "12 turn" in reasoning
    assert "harness_create failed" in reasoning
    assert "ccm-cost-categories" in reasoning
    assert "same bucket question" in reasoning
    assert "Summary:" not in reasoning


@pytest.mark.unit
def test_conversation_candidate_score_skips_useless():
    from conversation_candidate_score import HarnessConversationCandidateScoreMetric

    conversation = _minimal_conversation()
    eval_case = EvalCase.from_dict(
        {
            "input": conversation["input"],
            "output": conversation["output"],
            "messages": conversation["messages"],
            "tool_calls": conversation["tool_calls"],
            "metadata": {
                "usefulness": "useless",
                "canonical_conversation": conversation,
            },
        }
    )
    llm = FakeSignalsLLM()
    score = asyncio.run(HarnessConversationCandidateScoreMetric(llm=llm).a_measure(eval_case))
    assert score.metadata["skipped"] is True
    assert score.metadata["eval_candidate_score"] == 0.0
    assert llm.prompt == ""


class CombinedFakeLLM(BaseLLM):
    """Returns quality or candidate-score JSON depending on prompt content."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, prompt: str, **kwargs: object) -> str:
        return ""

    async def generate_json(self, prompt: str, schema: dict, **kwargs: object) -> dict:
        self.prompts.append(prompt)
        if "eval golden selection" in prompt:
            return {
                "high_turns": True,
                "high_cost": True,
                "high_tool_count": True,
                "large_tool_output": True,
                "tool_failure": True,
                "skill_loading": True,
                "hitl_loop": True,
                "multi_turn": True,
                "reasoning": "Stressful read with tool failure.",
                "evidence": ["status ERROR"],
                "criterion_notes": {
                    "hitl_loop": "AskUserQuestion repeated after user answered",
                },
            }
        return {
            "usefulness": "useful",
            "quality": "good",
            "golden_readiness": "ready",
            "goal_achievement": 0.9,
            "resolution": 0.8,
            "tool_use_quality": 0.7,
            "confidence": 0.85,
            "reasoning": "Good outcome.",
            "evidence": ["diagnose returned failed step"],
        }


@pytest.mark.unit
def test_quality_eval_merges_candidate_score_into_review_csv(tmp_path: Path):
    import run_conversation_quality_eval as quality_eval

    conversation = _minimal_conversation()
    case = EvalCase.from_dict(
        {
            "input": conversation["input"],
            "output": conversation["output"],
            "messages": conversation["messages"],
            "tool_calls": conversation["tool_calls"],
            "metadata": {
                "conversation_id": conversation["conversation_id"],
                "module": "ce",
                "environment": "prod1",
                "canonical_file": "sample.conversation.json",
                "num_turns": 12,
                "total_cost_usd": 0.42,
                "canonical_conversation": conversation,
            },
        }
    )
    llm = CombinedFakeLLM()
    quality_metric = HarnessConversationQualityMetric(llm=llm)
    candidate_metric = __import__(
        "conversation_candidate_score", fromlist=["HarnessConversationCandidateScoreMetric"]
    ).HarnessConversationCandidateScoreMetric(llm=llm)

    rows = asyncio.run(
        quality_eval.evaluate_cases(
            [case],
            quality_metric,
            candidate_metric,
            concurrency=1,
        )
    )
    assert rows[0]["quality"] == "good"
    assert float(rows[0]["eval_candidate_score"]) > 0

    low_row = dict(rows[0])
    low_row["conversation_id"] = "low-score"
    low_row["eval_candidate_score"] = 1.0
    high_row = dict(rows[0])
    high_row["conversation_id"] = "high-score"
    high_row["eval_candidate_score"] = 4.5
    assert rows[0]["session_turns"] == 12
    assert rows[0]["session_cost_usd"] == 0.42

    output_dir = tmp_path / "results" / "random-200"
    quality_eval.write_outputs(
        [low_row, high_row],
        output_dir,
        provider="openai",
        model="gpt-4o",
        with_candidate_score=True,
    )

    with (output_dir / "review.csv").open(newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    assert review_rows[0]["conversation_id"] == "high-score"
    assert "eval_candidate_score" in review_rows[0]
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["with_candidate_score"] is True
    assert summary["top_15_by_eval_candidate_score"][0]["conversation_id"] == "high-score"
