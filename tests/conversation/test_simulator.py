"""Tests for ConversationSimulator."""

from typing import Any

import pytest

from harness_evals.conversation import ConversationGolden, ConversationSimulator
from harness_evals.core.types import Message
from tests.conftest import MockLLM


class SimulatorMockLLM(MockLLM):
    """Mock LLM that returns user messages via generate() and stop checks via generate_json()."""

    def __init__(self, user_messages: list[str], stop_after: int = 2):
        self._user_messages = user_messages
        self._user_idx = 0
        self._json_call_count = 0
        self._stop_after = stop_after
        self.generated_prompts: list[str] = []
        super().__init__()

    async def generate(self, prompt: str, **kwargs) -> str:
        self.generated_prompts.append(prompt)
        if self._user_idx < len(self._user_messages):
            msg = self._user_messages[self._user_idx]
            self._user_idx += 1
            return msg
        return "Thank you, that's all."

    async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
        self._json_call_count += 1
        achieved = self._json_call_count >= self._stop_after
        return {"achieved": achieved, "reasoning": "test"}


@pytest.mark.unit
class TestConversationSimulator:
    async def test_basic_simulation(self):
        llm = SimulatorMockLLM(
            user_messages=["What is your refund policy?", "How long does it take?"],
            stop_after=2,
        )

        async def agent_fn(messages: list[Message]) -> Message:
            return Message(role="assistant", content=f"Response to turn {len(messages)}")

        golden = ConversationGolden(
            scenario="Ask about refund policy",
            expected_outcome="Agent explains refund process",
        )

        simulator = ConversationSimulator(simulator_llm=llm)
        result = await simulator.simulate(golden, agent_fn)

        assert result.messages is not None
        assert len(result.messages) >= 4  # At least 2 user + 2 assistant turns
        assert result.input == "Ask about refund policy"
        assert result.metadata["scenario"] == "Ask about refund policy"
        assert result.metadata["expected_outcome"] == "Agent explains refund process"

    async def test_golden_context_steers_simulator_but_is_not_retrieval_context(self):
        llm = SimulatorMockLLM(user_messages=["Please list the pipelines."], stop_after=1)

        async def agent_fn(messages: list[Message]) -> Message:
            return Message(role="assistant", content="Done")

        golden = ConversationGolden(
            scenario="List pipelines",
            expected_outcome="Pipelines listed",
            context=["Prefer Inherit from Delegate auth", "Use project-level scope"],
        )

        result = await ConversationSimulator(simulator_llm=llm).simulate(golden, agent_fn)

        assert (
            "**Background context**: Prefer Inherit from Delegate auth; Use project-level scope"
            in llm.generated_prompts[0]
        )
        # ConversationGolden.context steers simulation. RAG evidence belongs on
        # Message.retrieval_context and must not be fabricated from instructions.
        assert result.context is None

    async def test_max_turns_cap(self):
        llm = SimulatorMockLLM(
            user_messages=["msg"] * 20,
            stop_after=100,  # Never stops naturally
        )

        async def agent_fn(messages: list[Message]) -> Message:
            return Message(role="assistant", content="response")

        golden = ConversationGolden(
            scenario="Test",
            expected_outcome="Outcome",
            max_turns=4,  # Cap at 4 turns total (2 exchanges)
        )

        simulator = ConversationSimulator(simulator_llm=llm)
        result = await simulator.simulate(golden, agent_fn)

        assert len(result.messages) <= 8  # max_turns iterations, each adds 2 messages

    async def test_replay_mode(self):
        llm = MockLLM()  # Should not be called
        turns = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="How are you?"),
            Message(role="assistant", content="I'm doing well, thanks!"),
        ]

        golden = ConversationGolden(
            scenario="Greeting",
            expected_outcome="Polite exchange",
            turns=turns,
        )

        simulator = ConversationSimulator(simulator_llm=llm)
        result = await simulator.simulate(golden, agent_fn=lambda x: None)

        assert result.messages == turns
        assert result.output == "I'm doing well, thanks!"
        assert result.metadata["n_turns"] == 4

    async def test_output_is_last_assistant_message(self):
        llm = SimulatorMockLLM(user_messages=["question"], stop_after=1)

        async def agent_fn(messages: list[Message]) -> Message:
            return Message(role="assistant", content="final answer")

        golden = ConversationGolden(
            scenario="Test",
            expected_outcome="Done",
        )

        simulator = ConversationSimulator(simulator_llm=llm)
        result = await simulator.simulate(golden, agent_fn)

        assert result.output == "final answer"

    async def test_simulate_batch(self):
        llm = SimulatorMockLLM(
            user_messages=["q1", "q2", "q3", "q4"],
            stop_after=1,
        )

        async def agent_fn(messages: list[Message]) -> Message:
            return Message(role="assistant", content="answer")

        goldens = [ConversationGolden(scenario=f"S{i}", expected_outcome=f"O{i}") for i in range(3)]

        simulator = ConversationSimulator(simulator_llm=llm, max_concurrent=2)
        results = await simulator.simulate_batch(goldens, agent_fn)

        assert len(results) == 3
        for r in results:
            assert r.messages is not None

    async def test_metadata_from_golden(self):
        llm = SimulatorMockLLM(user_messages=["hi"], stop_after=1)

        async def agent_fn(messages: list[Message]) -> Message:
            return Message(role="assistant", content="hello")

        golden = ConversationGolden(
            scenario="Test",
            expected_outcome="Done",
            metadata={"custom_key": "custom_value"},
            tags={"env": "test"},
        )

        simulator = ConversationSimulator(simulator_llm=llm)
        result = await simulator.simulate(golden, agent_fn)

        assert result.metadata["custom_key"] == "custom_value"
        assert result.tags == {"env": "test"}


@pytest.mark.unit
def test_build_eval_case_propagates_tool_calls() -> None:
    from harness_evals.conversation.simulator import ConversationSimulator
    from harness_evals.core.types import ToolCall

    golden = ConversationGolden(
        scenario="List pipelines",
        expected_outcome="Lists pipelines",
        expected_tool_calls=[ToolCall(name="harness_list", input={"resource_type": "pipeline"})],
    )
    history = [
        Message(
            role="assistant",
            content="Checking pipelines",
            metadata={
                "sse_events": {
                    "assistant_tool_request": [
                        {
                            "v": [
                                {
                                    "name": "mcp__harness__harness_list",
                                    "arguments": {"resource_type": "pipeline"},
                                }
                            ]
                        }
                    ],
                    "assistant_tool_result": [
                        {"v": [{"name": "mcp__harness__harness_list", "result": '{"items": []}'}]}
                    ],
                }
            },
        ),
        Message(role="assistant", content="No pipelines found."),
    ]

    simulator = ConversationSimulator(simulator_llm=None)
    eval_case = simulator._build_eval_case(golden, history)

    assert eval_case.expected_tool_calls is not None
    assert eval_case.expected_tool_calls[0].name == "harness_list"
    assert eval_case.tool_calls is not None
    assert len(eval_case.tool_calls) == 1
    assert eval_case.tool_calls[0].name == "harness_list"
    assert eval_case.tool_calls[0].input == {"resource_type": "pipeline"}

    from harness_evals.metrics.agent.tool_argument_match import ToolArgumentMatchMetric

    score = ToolArgumentMatchMetric(pair="subsequence").measure(eval_case)
    assert score.value == 1.0


@pytest.mark.unit
def test_build_eval_case_tool_calls_empty_when_assistant_turn_called_no_tools() -> None:
    from harness_evals.conversation.simulator import ConversationSimulator
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_argument_match import ToolArgumentMatchMetric

    golden = ConversationGolden(
        scenario="No tools",
        expected_outcome="Answer",
        expected_tool_calls=[ToolCall(name="harness_list")],
    )
    history = [Message(role="assistant", content="Hello.")]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert eval_case.tool_calls == []

    score = ToolArgumentMatchMetric(skip_when_missing=True).measure(eval_case)
    assert not score.passed
    assert score.value == 0.0


@pytest.mark.unit
def test_build_eval_case_tool_calls_none_when_no_assistant_turn_captured() -> None:
    from harness_evals.conversation.simulator import ConversationSimulator
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_argument_match import ToolArgumentMatchMetric

    golden = ConversationGolden(
        scenario="No tools",
        expected_outcome="Answer",
        expected_tool_calls=[ToolCall(name="harness_list")],
    )
    history = [Message(role="user", content="Hello?")]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert eval_case.tool_calls is None

    score = ToolArgumentMatchMetric(skip_when_missing=True).measure(eval_case)
    assert score.passed
    assert "skip_when_missing" in (score.reason or "")


@pytest.mark.unit
def test_build_eval_case_normalizes_message_embedded_mcp_tool_names() -> None:
    from harness_evals.conversation.simulator import ConversationSimulator
    from harness_evals.core.types import ToolCall

    golden = ConversationGolden(
        scenario="Lookup",
        expected_outcome="Found",
        expected_tool_calls=[ToolCall(name="harness_list", input={"resource_type": "pipeline"})],
    )
    history = [
        Message(
            role="assistant",
            content="Looking up pipelines.",
            tool_calls=[
                ToolCall(name="mcp__harness__harness_list", input={"resource_type": "pipeline"}),
            ],
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert eval_case.tool_calls == [
        ToolCall(name="harness_list", input={"resource_type": "pipeline"}),
    ]


@pytest.mark.unit
def test_build_eval_case_drops_agent_internal_tools_from_message_embedded_calls() -> None:
    from harness_evals.conversation.simulator import ConversationSimulator
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_argument_match import ToolArgumentMatchMetric
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    golden = ConversationGolden(
        scenario="Lookup",
        expected_outcome="Found",
        expected_tool_calls=[ToolCall(name="harness_list")],
    )
    history = [
        Message(
            role="assistant",
            content="Reading then listing.",
            tool_calls=[
                ToolCall(name="Read", input={"path": "/tmp/x"}),
                ToolCall(name="mcp__harness__harness_list", input={}),
            ],
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert eval_case.tool_calls == [ToolCall(name="harness_list", input={})]
    assert ToolArgumentMatchMetric(pair="exact").measure(eval_case).value == 1.0
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 1.0


@pytest.mark.unit
def test_build_eval_case_internal_only_tools_yield_empty_trajectory() -> None:
    """Trace-imported [Read, Bash] must match SSE mcp_only semantics for empty goldens."""
    from harness_evals.conversation.simulator import ConversationSimulator
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    golden = ConversationGolden(
        scenario="No Harness tools",
        expected_outcome="Answered without tools",
        expected_tool_calls=[],
    )
    history = [
        Message(
            role="assistant",
            content="Inspecting locally.",
            tool_calls=[
                ToolCall(name="Read", input={"path": "/tmp/x"}),
                ToolCall(name="Bash", input={"command": "ls"}),
            ],
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert eval_case.tool_calls == []
    assert eval_case.expected_tools == []
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 1.0


_TRAJECTORY_MATRIX: list[tuple[str, list[str], list[str], list[str]]] = [
    # (case id, expected tool names, observed tool names, surviving trajectory)
    ("expected mcp call", ["harness_list"], ["mcp__harness__harness_list"], ["harness_list"]),
    ("unexpected mcp call", [], ["mcp__harness__harness_delete"], ["harness_delete"]),
    ("expected short name", ["lookup_order"], ["lookup_order"], ["lookup_order"]),
    ("unexpected short name", [], ["search_kb"], ["search_kb"]),
    ("expected builtin", ["Write"], ["Write"], ["Write"]),
    ("unexpected builtin", [], ["Read"], []),
    (
        "unexpected short name beside mcp call",
        ["harness_list"],
        ["search_kb", "mcp__harness__harness_list"],
        ["search_kb", "harness_list"],
    ),
    (
        "unexpected builtin beside mcp call",
        ["harness_list"],
        ["Read", "mcp__harness__harness_list"],
        ["harness_list"],
    ),
    (
        "expected builtin beside mcp call",
        ["Write", "harness_create"],
        ["Write", "mcp__harness__harness_create"],
        ["Write", "harness_create"],
    ),
    (
        "expected short name beside mcp call",
        ["lookup_order", "harness_list"],
        ["lookup_order", "mcp__harness__harness_list"],
        ["lookup_order", "harness_list"],
    ),
    ("tool prefixed short name", ["lookup_order"], ["tool:lookup_order"], ["lookup_order"]),
    ("tool prefixed unexpected builtin", [], ["tool:Read"], []),
    ("mcp call colliding with builtin", ["Task"], ["mcp__crm__Task"], ["Task"]),
    ("no tools called", [], [], []),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("case_id", "expected_names", "observed_names", "surviving_names"),
    _TRAJECTORY_MATRIX,
    ids=[case[0] for case in _TRAJECTORY_MATRIX],
)
def test_build_eval_case_trajectory_matrix(
    case_id: str,
    expected_names: list[str],
    observed_names: list[str],
    surviving_names: list[str],
) -> None:
    """Every (call kind x expectation) combination, so no cell is untested.

    Only agent-runtime builtins the golden did not ask for are dropped. A call
    the golden did not expect stays visible under every other shape, since the
    metric has to see a wrong tool to penalize it.
    """
    from harness_evals.core.types import ToolCall

    golden = ConversationGolden(
        scenario="Trajectory matrix",
        expected_outcome="Scored",
        expected_tool_calls=[ToolCall(name=name) for name in expected_names],
    )
    history = [
        Message(
            role="assistant",
            content="Acting.",
            tool_calls=[ToolCall(name=name) for name in observed_names],
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert [call.name for call in eval_case.tool_calls or []] == surviving_names


# SSE applies a closed keep rule (mcp__-routed calls plus names the golden
# expects), so it does not mirror _TRAJECTORY_MATRIX — see
# _tool_calls_for_eval_case for why the two capture paths differ on purpose.
# (case id, expected tool names, observed tool names, surviving trajectory)
_SSE_TRAJECTORY_MATRIX: list[tuple[str, list[str], list[str], list[str]]] = [
    ("expected mcp call", ["harness_list"], ["mcp__harness__harness_list"], ["harness_list"]),
    ("unexpected mcp call", [], ["mcp__harness__harness_delete"], ["harness_delete"]),
    ("expected short name", ["lookup_order"], ["lookup_order"], ["lookup_order"]),
    # Bare + unexpected: runtime machinery on this path, so it is not scored.
    ("unexpected short name", [], ["search_kb"], []),
    ("expected builtin", ["Write"], ["Write"], ["Write"]),
    ("unexpected builtin", [], ["Read"], []),
    # The name this PR's denylist does not enumerate — must not become a wrong tool.
    ("unenumerated runtime builtin", ["harness_list"], ["ToolSearch", "mcp__harness__harness_list"], ["harness_list"]),
    (
        "unexpected bare name beside mcp call",
        ["harness_list"],
        ["search_kb", "mcp__harness__harness_list"],
        ["harness_list"],
    ),
    (
        "expected builtin beside mcp call",
        ["Write", "harness_create"],
        ["Write", "mcp__harness__harness_create"],
        ["Write", "harness_create"],
    ),
    ("mcp call colliding with builtin", ["Task"], ["mcp__crm__Task"], ["Task"]),
    ("no tools called", [], [], []),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("case_id", "expected_names", "observed_names", "surviving_names"),
    _SSE_TRAJECTORY_MATRIX,
    ids=[f"sse-{case[0]}" for case in _SSE_TRAJECTORY_MATRIX],
)
def test_build_eval_case_sse_trajectory_matrix(
    case_id: str,
    expected_names: list[str],
    observed_names: list[str],
    surviving_names: list[str],
) -> None:
    from harness_evals.core.types import ToolCall

    golden = ConversationGolden(
        scenario="Trajectory matrix (SSE)",
        expected_outcome="Scored",
        expected_tool_calls=[ToolCall(name=name) for name in expected_names],
    )
    history = [
        Message(
            role="assistant",
            content="Acting.",
            metadata={
                "sse_events": {
                    "assistant_tool_request": [{"v": [{"name": name, "arguments": {}}]} for name in observed_names]
                }
            },
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert [call.name for call in eval_case.tool_calls or []] == surviving_names


def _capture_shapes(wire_names: list[str]) -> dict[str, tuple[list[Any] | None, dict | None]]:
    """Every way one turn can record the same calls, as (tool_calls, metadata).

    A turn may report its calls as message embeds, as aggregated ``sse_events``,
    as an ordered ``sse_timeline``, or — the shape live capture actually emits,
    see ``StreamingHttpTarget._capture`` — as all three at once. Each is a
    *record of the same calls*, never additional calls, so the trajectory must
    come out identical regardless of which combination arrived.
    """
    from harness_evals.core.types import ToolCall

    args = {"type": "pipeline"}
    embedded = [ToolCall(name=name, input=args) for name in wire_names] or None
    payload = {"v": [{"name": name, "arguments": args} for name in wire_names]}
    events = {"sse_events": {"assistant_tool_request": [payload]}}
    timeline = {"sse_timeline": [{"event": "assistant_tool_request", "payload": payload}]}
    unparseable = {"sse_timeline": [{"event": "assistant_tool_request", "payload": {"v": [{"noname": 1}]}}]}
    return {
        "embedded only": (embedded, None),
        "events only": (None, events),
        "timeline only": (None, timeline),
        "embedded + events": (embedded, events),
        "embedded + timeline": (embedded, timeline),
        "events + timeline": (None, {**events, **timeline}),
        "live capture (all three)": (embedded, {**events, **timeline}),
        "live capture, unparseable timeline": (embedded, {**events, **unparseable}),
        "events + unparseable timeline": (None, {**events, **unparseable}),
    }


_CAPTURE_SHAPE_IDS = list(_capture_shapes(["x"]))

# (case id, wire name observed, short name the golden authors)
_WIRE_NAME_SHAPES = [
    ("mcp call", "mcp__harness__harness_list", "harness_list"),
    ("bare name", "lookup_order", "lookup_order"),
    ("tool prefixed name", "tool:lookup_order", "lookup_order"),
    ("agent builtin", "Write", "Write"),
    ("mcp name colliding with builtin", "mcp__crm__Task", "Task"),
]


@pytest.mark.unit
@pytest.mark.parametrize("shape", _CAPTURE_SHAPE_IDS)
@pytest.mark.parametrize(("name_id", "wire_name", "short_name"), _WIRE_NAME_SHAPES)
def test_build_eval_case_scores_correct_trajectory_under_every_capture_shape(
    shape: str, name_id: str, wire_name: str, short_name: str
) -> None:
    """A trajectory that matches the golden scores 1.0 however it was captured.

    Source selection is per turn, so a turn that recorded its calls two or three
    ways contributes them once. Counting a record twice is not a near miss — an
    exact-match trajectory metric reads the duplicate as a wrong extra call and
    halves a perfect score.
    """
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    golden = ConversationGolden(
        scenario="Capture shape matrix",
        expected_outcome="Scored",
        expected_tool_calls=[ToolCall(name=short_name, input={"type": "pipeline"})],
    )
    tool_calls, metadata = _capture_shapes([wire_name])[shape]
    history = [Message(role="assistant", content="Acting.", tool_calls=tool_calls, metadata=metadata)]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)

    assert [call.name for call in eval_case.tool_calls or []] == [short_name]
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 1.0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("expected_names", "embedded_names", "timeline_name", "surviving_names", "expected_score"),
    [
        (
            [],
            ["search_kb", "search_kb"],
            "mcp__harness__harness_delete",
            ["harness_delete"],
            0.0,
        ),
        (
            ["harness_list"],
            ["search_kb", "search_kb"],
            "mcp__harness__harness_list",
            ["harness_list"],
            1.0,
        ),
    ],
)
def test_build_eval_case_arbitrates_conflicting_timeline_under_the_emit_rule(
    expected_names: list[str],
    embedded_names: list[str],
    timeline_name: str,
    surviving_names: list[str],
    expected_score: float,
) -> None:
    """Completeness and emission must apply the same provenance-specific rule.

    An SSE-backed embedded record takes the closed SSE rule at emission. Counting
    it under the REPLAY denylist instead lets bare runtime names overstate its
    completeness, displace the MCP timeline, and then disappear under the
    different emit rule. That can silently pass a forbidden call or fail a
    correct one.
    """
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    golden = ConversationGolden(
        scenario="Conflicting timeline records",
        expected_outcome="Scored",
        expected_tool_calls=[ToolCall(name=name) for name in expected_names],
    )
    history = [
        Message(
            role="assistant",
            content="Acting.",
            tool_calls=[ToolCall(name=name) for name in embedded_names],
            metadata={
                "sse_timeline": [
                    {
                        "event": "assistant_tool_request",
                        "payload": {"v": [{"name": timeline_name, "arguments": {}}]},
                    }
                ]
            },
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)

    assert [call.name for call in eval_case.tool_calls or []] == surviving_names
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == expected_score


@pytest.mark.unit
@pytest.mark.parametrize(
    ("case_id", "timeline"),
    [
        ("no timeline", None),
        ("empty timeline", []),
        (
            "unparseable timeline",
            [{"event": "assistant_tool_request", "payload": {"v": [{"noname": 1}]}}],
        ),
    ],
)
def test_build_eval_case_prefers_complete_aggregated_events_over_partial_embed(
    case_id: str, timeline: list[dict] | None
) -> None:
    """All timeline states participate in the same three-record arbitration."""
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_argument_match import ToolArgumentMatchMetric
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    args = {"type": "pipeline"}
    golden = ConversationGolden(
        scenario="Partial embedded record",
        expected_outcome="Scored",
        expected_tool_calls=[
            ToolCall(name="harness_list", input=args),
            ToolCall(name="harness_create", input=args),
        ],
    )
    metadata = {
        "sse_events": {
            "assistant_tool_request": [
                {
                    "v": [
                        {"name": "mcp__harness__harness_list", "arguments": args},
                        {"name": "mcp__harness__harness_create", "arguments": args},
                    ]
                }
            ]
        }
    }
    if timeline is not None:
        metadata["sse_timeline"] = timeline
    history = [
        Message(
            role="assistant",
            content="Acting.",
            tool_calls=[ToolCall(name="mcp__harness__harness_create", input=args)],
            metadata=metadata,
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)

    assert [call.name for call in eval_case.tool_calls or []] == ["harness_list", "harness_create"]
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 1.0
    assert ToolArgumentMatchMetric(pair="exact").measure(eval_case).value == 1.0


@pytest.mark.unit
def test_build_eval_case_equal_conflicting_records_prefer_raw_sse_evidence() -> None:
    """Equal-sized contradictory records use deterministic source precedence.

    The raw stream is authoritative over its derived embedded copy. Selecting
    whichever source happens to match the golden would hide an unexpected call.
    """
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    golden = ConversationGolden(
        scenario="Contradictory records",
        expected_outcome="Scored",
        expected_tool_calls=[ToolCall(name="harness_list")],
    )
    history = [
        Message(
            role="assistant",
            content="Acting.",
            tool_calls=[ToolCall(name="mcp__harness__harness_list")],
            metadata={
                "sse_events": {
                    "assistant_tool_request": [{"v": [{"name": "mcp__harness__harness_delete", "arguments": {}}]}]
                }
            },
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)

    assert [call.name for call in eval_case.tool_calls or []] == ["harness_delete"]
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 0.0


@pytest.mark.unit
def test_build_eval_case_unscored_aggregated_events_do_not_erase_local_work() -> None:
    """A zero-relevance aggregate is not a competing record for transcript data."""
    from harness_evals.core.types import ToolCall

    history = [
        Message(
            role="assistant",
            content="Working.",
            tool_calls=[ToolCall(name="Read"), ToolCall(name="Bash")],
            metadata={"sse_events": {"assistant_tool_request": [{"v": [{"name": "ToolSearch", "arguments": {}}]}]}},
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(
        ConversationGolden(scenario="Local work", expected_outcome="Answered"), history
    )

    transcript = [call.name for message in eval_case.messages or [] for call in (message.tool_calls or [])]
    assert transcript == ["Read", "Bash"]
    assert eval_case.tool_calls == []


@pytest.mark.unit
def test_build_eval_case_drops_timeline_results_when_timeline_requests_are_suppressed() -> None:
    """Requests and results stay together when arbitration drops the timeline.

    Emitting a result without its preceding request leaves an LLM judge reading a
    tool output the agent never asked for.
    """
    from harness_evals.core.types import ToolCall

    history = [
        Message(
            role="assistant",
            content="Did local work.",
            tool_calls=[ToolCall(name="Read")],
            metadata={
                "sse_timeline": [
                    {
                        "event": "assistant_tool_request",
                        "payload": {"v": [{"name": "ToolSearch", "arguments": {}}]},
                    },
                    {
                        "event": "assistant_tool_result",
                        "payload": {"v": [{"name": "ToolSearch", "output": "found"}]},
                    },
                    {"event": "assistant_message", "payload": {"v": "Did local work."}},
                ]
            },
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(
        ConversationGolden(scenario="Local work", expected_outcome="Answered"), history
    )

    roles_and_tools = [
        (message.role, [call.name for call in (message.tool_calls or [])]) for message in eval_case.messages or []
    ]
    assert roles_and_tools == [("assistant", ["Read"])]
    assert eval_case.tool_calls == []


@pytest.mark.unit
def test_build_eval_case_places_winning_aggregated_requests_before_assistant_text() -> None:
    """Aggregated requests belong at the tool-request position, not after the turn.

    Appending them after the assistant message inverts request/result chronology
    when a partial timeline still contributes a result — judges that read
    ``eval_case.messages`` then see a result before its request.
    """
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_argument_match import ToolArgumentMatchMetric
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    args = {"type": "pipeline"}
    golden = ConversationGolden(
        scenario="Aggregated chronology",
        expected_outcome="Scored",
        expected_tool_calls=[
            ToolCall(name="harness_list", input=args),
            ToolCall(name="harness_create", input=args),
        ],
    )
    history = [
        Message(
            role="assistant",
            content="Acting.",
            tool_calls=[ToolCall(name="mcp__harness__harness_create", input=args)],
            metadata={
                "sse_events": {
                    "assistant_tool_request": [
                        {
                            "v": [
                                {"name": "mcp__harness__harness_list", "arguments": args},
                                {"name": "mcp__harness__harness_create", "arguments": args},
                            ]
                        }
                    ]
                },
                "sse_timeline": [
                    {
                        "event": "assistant_tool_request",
                        "payload": {"v": [{"name": "mcp__harness__harness_list", "arguments": args}]},
                    },
                    {
                        "event": "assistant_tool_result",
                        "payload": {"v": [{"name": "mcp__harness__harness_list", "output": "ok"}]},
                    },
                    {"event": "assistant_message", "payload": {"v": "Acting."}},
                ],
            },
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)

    assert [call.name for call in eval_case.tool_calls or []] == ["harness_list", "harness_create"]
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 1.0
    assert ToolArgumentMatchMetric(pair="exact").measure(eval_case).value == 1.0
    roles_and_tools = [
        (message.role, [call.name for call in (message.tool_calls or [])], message.content)
        for message in eval_case.messages or []
    ]
    assert roles_and_tools == [
        ("assistant", ["harness_list", "harness_create"], None),
        ("assistant", [], "Acting."),
    ]


@pytest.mark.unit
@pytest.mark.parametrize("second_shape", _CAPTURE_SHAPE_IDS)
@pytest.mark.parametrize("first_shape", _CAPTURE_SHAPE_IDS)
def test_build_eval_case_scores_mixed_capture_conversations(first_shape: str, second_shape: str) -> None:
    """Turns captured different ways in one conversation each contribute their calls.

    ``sse_events`` is flattened across the conversation before scoring, so any
    decision made from that aggregate applies to every turn at once — which is
    how one live turn came to suppress another turn's embedded calls.
    """
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    args = {"type": "pipeline"}
    golden = ConversationGolden(
        scenario="Mixed capture",
        expected_outcome="Scored",
        expected_tool_calls=[ToolCall(name="harness_list", input=args), ToolCall(name="lookup_order", input=args)],
    )
    first_calls, first_metadata = _capture_shapes(["mcp__harness__harness_list"])[first_shape]
    second_calls, second_metadata = _capture_shapes(["lookup_order"])[second_shape]
    history = [
        Message(role="assistant", content="Listing.", tool_calls=first_calls, metadata=first_metadata),
        Message(role="assistant", content="Looking up.", tool_calls=second_calls, metadata=second_metadata),
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)

    assert [call.name for call in eval_case.tool_calls or []] == ["harness_list", "lookup_order"]
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 1.0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("case_id", "metadata"),
    [
        ("no sse keys", None),
        ("assistant text events only", {"sse_events": {"assistant_message": [{"v": "Working."}]}}),
        ("empty timeline", {"sse_timeline": []}),
        (
            "timeline payloads do not parse",
            {"sse_timeline": [{"event": "assistant_tool_request", "payload": {"v": [{"noname": 1}]}}]},
        ),
    ],
)
def test_build_eval_case_scores_wrong_tool_when_turn_captured_no_sse_request(
    case_id: str, metadata: dict | None
) -> None:
    """SSE provenance requires a captured tool request, not merely SSE keys.

    The closed SSE keep rule drops unexpected bare names as runtime machinery.
    Applying it to a turn that never recorded a tool request over SSE would let a
    genuinely wrong tool vanish from scoring while still showing on the
    transcript — a silent pass on an incorrect trajectory.
    """
    from harness_evals.core.types import ToolCall

    history = [
        Message(role="assistant", content="Working.", tool_calls=[ToolCall(name="search_kb")], metadata=metadata)
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(
        ConversationGolden(scenario="Wrong tool", expected_outcome="Answered", expected_tool_calls=[]), history
    )

    assert [call.name for call in eval_case.tool_calls or []] == ["search_kb"]


@pytest.mark.unit
@pytest.mark.parametrize("shape", _CAPTURE_SHAPE_IDS)
def test_build_eval_case_reports_empty_trajectory_under_every_capture_shape(shape: str) -> None:
    """An assistant turn that called nothing is an empty trajectory, not a missing one.

    ``[]`` is scored; ``None`` means capture failed and lets ``skip_when_missing``
    excuse the turn, so collapsing the two hides every missed tool call.
    """
    tool_calls, metadata = _capture_shapes([])[shape]
    history = [Message(role="assistant", content="No tools needed.", tool_calls=tool_calls, metadata=metadata)]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(
        ConversationGolden(scenario="Empty trajectory", expected_outcome="Answered"), history
    )

    assert eval_case.tool_calls == []


@pytest.mark.unit
@pytest.mark.parametrize("shape", _CAPTURE_SHAPE_IDS)
def test_build_eval_case_keeps_local_work_on_transcript_under_every_capture_shape(shape: str) -> None:
    """Agent-internal calls stay on the transcript even when scoring filters them.

    Filtering decides what a trajectory metric grades; it must not decide what a
    reviewer can see happened.
    """
    tool_calls, metadata = _capture_shapes(["Read", "Bash"])[shape]
    history = [Message(role="assistant", content="Working.", tool_calls=tool_calls, metadata=metadata)]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(
        ConversationGolden(scenario="Local work", expected_outcome="Answered"), history
    )

    transcript = [call.name for message in eval_case.messages or [] for call in (message.tool_calls or [])]
    assert transcript == ["Read", "Bash"]
    assert eval_case.tool_calls == []


@pytest.mark.unit
@pytest.mark.parametrize("runtime_tool", ["ToolSearch", "SendMessage", "ScheduleWakeup", "TaskCreate", "Workflow"])
def test_build_eval_case_sse_unenumerated_builtin_does_not_fail_trajectory(runtime_tool: str) -> None:
    """A runtime builtin absent from the denylist must not score as a wrong tool.

    Names are literal, not drawn from _AGENT_INTERNAL_TOOL_NAMES, so this
    catches the case a set-driven test cannot: a builtin nobody enumerated.
    """
    from harness_evals.conversation.simulator import _AGENT_INTERNAL_TOOL_NAMES
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    assert runtime_tool not in _AGENT_INTERNAL_TOOL_NAMES, "pick a name the denylist does not cover"

    golden = ConversationGolden(
        scenario="Only harness_list expected",
        expected_outcome="Listed",
        expected_tool_calls=[ToolCall(name="harness_list")],
    )
    history = [
        Message(
            role="assistant",
            content="Working, then listing.",
            metadata={
                "sse_events": {
                    "assistant_tool_request": [
                        {"v": [{"name": runtime_tool, "arguments": {}}]},
                        {"v": [{"name": "mcp__harness__harness_list", "arguments": {"type": "pipeline"}}]},
                    ]
                }
            },
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert [call.name for call in eval_case.tool_calls or []] == ["harness_list"]
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 1.0


@pytest.mark.unit
def test_build_eval_case_drops_every_known_agent_builtin() -> None:
    """Set-driven so a name added to the denylist can't ship without coverage.

    Passes by construction and so cannot detect a builtin missing from the set;
    that gap is covered on the SSE path by
    test_build_eval_case_sse_unenumerated_builtin_does_not_fail_trajectory.
    """
    from harness_evals.conversation.simulator import _AGENT_INTERNAL_TOOL_NAMES, ConversationSimulator
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    golden = ConversationGolden(
        scenario="No Harness tools",
        expected_outcome="Answered without Harness tools",
        expected_tool_calls=[],
    )
    history = [
        Message(
            role="assistant",
            content="Working locally.",
            tool_calls=[ToolCall(name=name) for name in sorted(_AGENT_INTERNAL_TOOL_NAMES)],
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert eval_case.tool_calls == []
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 1.0


@pytest.mark.unit
def test_build_eval_case_keeps_builtin_calls_on_messages_when_timeline_has_no_tools() -> None:
    """A timeline with no tool requests must not erase the turn's own calls.

    The scored trajectory is empty either way, but the transcript has to keep
    "agent did local work instead of calling the tool" visible for debugging.
    """
    from harness_evals.core.types import ToolCall

    golden = ConversationGolden(
        scenario="Should have called harness_list",
        expected_outcome="Listed",
        expected_tool_calls=[],
    )
    history = [
        Message(
            role="assistant",
            content="Did local work.",
            tool_calls=[ToolCall(name="Read"), ToolCall(name="Bash")],
            metadata={"sse_timeline": [{"event": "assistant_message", "payload": {"v": "Did local work."}}]},
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert [call.name for msg in eval_case.messages or [] for call in msg.tool_calls or []] == ["Read", "Bash"]
    # Builtins are still kept out of the scored trajectory.
    assert eval_case.tool_calls == []


@pytest.mark.unit
def test_build_eval_case_keeps_local_calls_when_timeline_calls_are_filtered() -> None:
    """A parseable but unscored timeline is not a competing tool record.

    Both records score zero, but preserving the turn's calls matters to
    transcript metrics and debugging: they show the agent did local work.
    """
    from harness_evals.core.types import ToolCall

    golden = ConversationGolden(
        scenario="Should have called harness_list",
        expected_outcome="Listed",
        expected_tool_calls=[],
    )
    history = [
        Message(
            role="assistant",
            content="Did local work.",
            tool_calls=[ToolCall(name="Read"), ToolCall(name="Bash")],
            metadata={
                "sse_timeline": [
                    {
                        "event": "assistant_tool_request",
                        "payload": {"v": [{"name": "ToolSearch", "arguments": {}}]},
                    },
                    {"event": "assistant_message", "payload": {"v": "Did local work."}},
                ]
            },
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert [call.name for msg in eval_case.messages or [] for call in msg.tool_calls or []] == ["Read", "Bash"]
    assert eval_case.tool_calls == []


@pytest.mark.unit
def test_build_eval_case_scores_mixed_sse_and_embedded_turns() -> None:
    """One SSE turn must not discard another turn's message-embedded tools."""
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    golden = ConversationGolden(
        scenario="List then look up",
        expected_outcome="Listed and looked up",
        expected_tool_calls=[
            ToolCall(name="harness_list"),
            ToolCall(name="lookup_order"),
        ],
    )
    history = [
        Message(
            role="assistant",
            content="Listing.",
            metadata={
                "sse_events": {
                    "assistant_tool_request": [
                        {"v": [{"name": "mcp__harness__harness_list", "arguments": {}}]},
                    ]
                }
            },
        ),
        Message(
            role="assistant",
            content="Looking up.",
            tool_calls=[ToolCall(name="lookup_order", input={})],
        ),
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert [call.name for call in eval_case.tool_calls or []] == ["harness_list", "lookup_order"]
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 1.0


@pytest.mark.unit
def test_build_eval_case_keeps_embeds_when_sse_events_have_no_tool_requests() -> None:
    """Non-tool sse_events must not blank out message-embedded tool_calls."""
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    golden = ConversationGolden(
        scenario="Look up an order",
        expected_outcome="Looked up",
        expected_tool_calls=[ToolCall(name="lookup_order")],
    )
    history = [
        Message(
            role="assistant",
            content="Looking up.",
            tool_calls=[ToolCall(name="lookup_order", input={})],
            metadata={"sse_events": {"assistant_message": [{"v": "Looking up."}]}},
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert [call.name for call in eval_case.tool_calls or []] == ["lookup_order"]
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 1.0


@pytest.mark.unit
def test_build_eval_case_does_not_double_count_sse_events_and_timeline() -> None:
    """When timeline already materializes live calls, skip aggregated sse_events."""
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    golden = ConversationGolden(
        scenario="List projects",
        expected_outcome="Listed",
        expected_tool_calls=[ToolCall(name="harness_list", input={"type": "project"})],
    )
    history = [
        Message(
            role="assistant",
            content="Here are the projects.",
            metadata={
                "sse_events": {
                    "assistant_tool_request": [
                        {"v": [{"name": "mcp__harness__harness_list", "arguments": {"type": "project"}}]},
                    ]
                },
                "sse_timeline": [
                    {
                        "event": "assistant_tool_request",
                        "payload": {"v": [{"name": "mcp__harness__harness_list", "arguments": {"type": "project"}}]},
                    },
                    {"event": "assistant_message", "payload": {"v": "Here are the projects."}},
                ],
            },
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert [call.name for call in eval_case.tool_calls or []] == ["harness_list"]
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 1.0


@pytest.mark.unit
def test_build_eval_case_keeps_mcp_tool_whose_short_name_collides_with_builtin() -> None:
    """An MCP server tool named Task is under test, not an agent builtin."""
    from harness_evals.conversation.simulator import ConversationSimulator
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    golden = ConversationGolden(
        scenario="Create a CRM task",
        expected_outcome="Task created",
        expected_tool_calls=[ToolCall(name="Task", input={"title": "Follow up"})],
    )
    history = [
        Message(
            role="assistant",
            content="Creating the task.",
            tool_calls=[ToolCall(name="mcp__crm__Task", input={"title": "Follow up"})],
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert eval_case.tool_calls == [ToolCall(name="Task", input={"title": "Follow up"})]
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 1.0


@pytest.mark.unit
def test_build_eval_case_scores_unexpected_mcp_builtin_collision_as_a_call() -> None:
    """A colliding MCP call stays visible so an empty golden scores it as wrong."""
    from harness_evals.conversation.simulator import ConversationSimulator
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    golden = ConversationGolden(
        scenario="No tools should be called",
        expected_outcome="Answered directly",
        expected_tool_calls=[],
    )
    history = [
        Message(
            role="assistant",
            content="Creating a task anyway.",
            tool_calls=[ToolCall(name="mcp__crm__Task", input={"title": "Follow up"})],
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert eval_case.tool_calls == [ToolCall(name="Task", input={"title": "Follow up"})]
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 0.0


@pytest.mark.unit
def test_build_eval_case_keeps_colliding_mcp_call_arriving_via_timeline() -> None:
    """The collision guard must survive the timeline path, which shortens names."""
    from harness_evals.conversation.simulator import ConversationSimulator
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    golden = ConversationGolden(
        scenario="No tools should be called",
        expected_outcome="Answered directly",
        expected_tool_calls=[],
    )
    history = [
        Message(
            role="assistant",
            content="Creating a task anyway.",
            metadata={
                "sse_timeline": [
                    {
                        "event": "assistant_tool_request",
                        "payload": {"v": [{"name": "mcp__crm__Task", "arguments": {"title": "Follow up"}}]},
                    }
                ]
            },
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert eval_case.tool_calls == [ToolCall(name="Task", input={"title": "Follow up"})]
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 0.0
    # The transcript still shows the short name.
    tool_request = next(msg for msg in eval_case.messages if (msg.metadata or {}).get("sse_event"))
    assert tool_request.tool_calls[0].name == "Task"


@pytest.mark.unit
@pytest.mark.parametrize("runtime_tool", ["Read", "ToolSearch"])
def test_build_eval_case_keeps_turn_call_when_timeline_holds_only_builtins(runtime_tool: str) -> None:
    """A builtin-only timeline must not outvote a turn that recorded a real call.

    Parametrized over a denylisted name and one the denylist does not
    enumerate: arbitration counts the timeline with the closed SSE rule, so
    both must lose to the turn's MCP call rather than only the listed one.
    """
    from harness_evals.conversation.simulator import ConversationSimulator
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_argument_match import ToolArgumentMatchMetric
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    golden = ConversationGolden(
        scenario="List pipelines",
        expected_outcome="Listed",
        expected_tool_calls=[ToolCall(name="harness_list", input={"type": "pipeline"})],
    )
    history = [
        Message(
            role="assistant",
            content="Working, then listing.",
            tool_calls=[ToolCall(name="mcp__harness__harness_list", input={"type": "pipeline"})],
            metadata={
                # The timeline captured only runtime machinery, which is filtered
                # out of the trajectory — it must not displace the turn's record.
                "sse_timeline": [
                    {
                        "event": "assistant_tool_request",
                        "payload": {"v": [{"name": runtime_tool, "arguments": {"path": "/tmp/notes"}}]},
                    }
                ]
            },
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert eval_case.tool_calls == [ToolCall(name="harness_list", input={"type": "pipeline"})]
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 1.0
    assert ToolArgumentMatchMetric().measure(eval_case).value == 1.0
    # The turn's real call has to survive on the transcript too — ToolUseMetric
    # and StepEfficiencyMetric read eval_case.messages, not eval_case.tool_calls.
    assert [call.name for msg in eval_case.messages or [] for call in msg.tool_calls or []] == [
        "mcp__harness__harness_list"
    ]


@pytest.mark.unit
def test_build_eval_case_timeline_calls_use_sse_keep_rule() -> None:
    """Timeline-derived calls are SSE-shaped records, so SSE rules filter them.

    Guards the step after arbitration: once the timeline wins, an unenumerated
    runtime tool riding alongside a real MCP call must still not be scored.
    """
    from harness_evals.conversation.simulator import ConversationSimulator
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    golden = ConversationGolden(
        scenario="List pipelines",
        expected_outcome="Listed",
        expected_tool_calls=[ToolCall(name="harness_list", input={"type": "pipeline"})],
    )
    history = [
        Message(
            role="assistant",
            content="Working, then listing.",
            metadata={
                "sse_timeline": [
                    {"event": "assistant_tool_request", "payload": {"v": [{"name": "ToolSearch", "arguments": {}}]}},
                    {
                        "event": "assistant_tool_request",
                        "payload": {"v": [{"name": "mcp__harness__harness_list", "arguments": {"type": "pipeline"}}]},
                    },
                    {"event": "assistant_message", "payload": {"v": "Working, then listing."}},
                ]
            },
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert [call.name for call in eval_case.tool_calls or []] == ["harness_list"]
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 1.0


@pytest.mark.unit
def test_build_eval_case_drops_internal_builtin_arriving_via_timeline() -> None:
    """A genuine builtin on the timeline path is still dropped from the trajectory."""
    from harness_evals.conversation.simulator import ConversationSimulator
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    golden = ConversationGolden(
        scenario="No Harness tools",
        expected_outcome="Answered without tools",
        expected_tool_calls=[],
    )
    history = [
        Message(
            role="assistant",
            content="Reading locally.",
            metadata={
                "sse_timeline": [
                    {
                        "event": "assistant_tool_request",
                        "payload": {"v": [{"name": "Task", "arguments": {"prompt": "investigate"}}]},
                    }
                ]
            },
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert eval_case.tool_calls == []
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 1.0


@pytest.mark.unit
def test_build_eval_case_drops_subagent_and_shell_management_builtins() -> None:
    """Orchestration builtins are dropped alongside an expected Harness call."""
    from harness_evals.conversation.simulator import ConversationSimulator
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    golden = ConversationGolden(
        scenario="Delegate then list pipelines",
        expected_outcome="Listed",
        expected_tool_calls=[ToolCall(name="harness_list", input={"type": "pipeline"})],
    )
    history = [
        Message(
            role="assistant",
            content="Delegating, then listing.",
            tool_calls=[
                ToolCall(name="Task", input={"prompt": "investigate"}),
                ToolCall(name="LS", input={"path": "/repo"}),
                ToolCall(name="BashOutput", input={"id": "1"}),
                ToolCall(name="mcp__harness__harness_list", input={"type": "pipeline"}),
            ],
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert eval_case.tool_calls == [ToolCall(name="harness_list", input={"type": "pipeline"})]
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 1.0


@pytest.mark.unit
def test_build_eval_case_preserves_expected_agent_named_tools() -> None:
    """A golden that expects Write must not have that call erased by the denylist."""
    from harness_evals.conversation.simulator import ConversationSimulator
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_argument_match import ToolArgumentMatchMetric
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    golden = ConversationGolden(
        scenario="Edit a file",
        expected_outcome="Wrote the file",
        expected_tool_calls=[ToolCall(name="Write", input={"path": "/tmp/out"})],
    )
    history = [
        Message(
            role="assistant",
            content="Writing the file.",
            tool_calls=[ToolCall(name="Write", input={"path": "/tmp/out"})],
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert eval_case.tool_calls == [ToolCall(name="Write", input={"path": "/tmp/out"})]
    assert ToolArgumentMatchMetric(pair="exact").measure(eval_case).value == 1.0
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 1.0


@pytest.mark.unit
def test_build_eval_case_matches_prefixed_golden_against_bare_call() -> None:
    """A golden authored with `tool:Write` still matches a bare `Write` call."""
    from harness_evals.conversation.simulator import ConversationSimulator
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    golden = ConversationGolden(
        scenario="Write a file",
        expected_outcome="Written",
        expected_tool_calls=[ToolCall(name="tool:Write", input={"path": "/tmp/out"})],
    )
    history = [
        Message(
            role="assistant",
            content="Writing the file.",
            tool_calls=[ToolCall(name="Write", input={"path": "/tmp/out"})],
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert eval_case.tool_calls == [ToolCall(name="Write", input={"path": "/tmp/out"})]
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 1.0


@pytest.mark.unit
def test_build_eval_case_does_not_double_count_timeline_tool_calls() -> None:
    """A turn carrying both an sse_timeline request and its own tool_calls counts once."""
    from harness_evals.conversation.simulator import ConversationSimulator
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_argument_match import ToolArgumentMatchMetric
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    golden = ConversationGolden(
        scenario="List pipelines",
        expected_outcome="Listed",
        expected_tool_calls=[ToolCall(name="harness_list", input={"type": "pipeline"})],
    )
    history = [
        Message(
            role="assistant",
            content="Listing pipelines.",
            tool_calls=[ToolCall(name="mcp__harness__harness_list", input={"type": "pipeline"})],
            metadata={
                "sse_timeline": [
                    {
                        "event": "assistant_tool_request",
                        "payload": {
                            "v": [
                                {
                                    "name": "mcp__harness__harness_list",
                                    "arguments": {"type": "pipeline"},
                                }
                            ]
                        },
                    }
                ]
            },
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert eval_case.tool_calls == [ToolCall(name="harness_list", input={"type": "pipeline"})]
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 1.0
    assert ToolArgumentMatchMetric(pair="exact").measure(eval_case).value == 1.0


@pytest.mark.unit
def test_build_eval_case_keeps_turn_tool_calls_when_timeline_payload_unparseable() -> None:
    """A timeline that yields no calls must not erase the turn's own trajectory."""
    from harness_evals.conversation.simulator import ConversationSimulator
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    golden = ConversationGolden(
        scenario="Order lookup",
        expected_outcome="Found",
        expected_tool_calls=[ToolCall(name="lookup_order", input={"id": "ORD-1"})],
    )
    history = [
        Message(
            role="assistant",
            content="Looking up the order.",
            tool_calls=[ToolCall(name="lookup_order", input={"id": "ORD-1"})],
            metadata={
                # Entry is well-formed JSON but carries no ``name``, so the SSE
                # payload parser yields nothing for this turn.
                "sse_timeline": [{"event": "assistant_tool_request", "payload": {"v": [{"arguments": {}}]}}]
            },
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert eval_case.tool_calls == [ToolCall(name="lookup_order", input={"id": "ORD-1"})]
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 1.0


@pytest.mark.unit
def test_build_eval_case_prefers_turn_tool_calls_over_partial_timeline() -> None:
    """A timeline that parses only some calls must not cost the turn the rest."""
    from harness_evals.conversation.simulator import ConversationSimulator
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    golden = ConversationGolden(
        scenario="Look up then notify",
        expected_outcome="Both done",
        expected_tool_calls=[
            ToolCall(name="lookup_order", input={"id": "ORD-1"}),
            ToolCall(name="notify_customer", input={"id": "ORD-1"}),
        ],
    )
    history = [
        Message(
            role="assistant",
            content="Looking up then notifying.",
            tool_calls=[
                ToolCall(name="lookup_order", input={"id": "ORD-1"}),
                ToolCall(name="notify_customer", input={"id": "ORD-1"}),
            ],
            metadata={
                # Only the first request parses; the second carries no name.
                "sse_timeline": [
                    {
                        "event": "assistant_tool_request",
                        "payload": {"v": [{"name": "lookup_order", "arguments": {"id": "ORD-1"}}]},
                    },
                    {"event": "assistant_tool_request", "payload": {"v": [{"arguments": {}}]}},
                ]
            },
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert eval_case.tool_calls == [
        ToolCall(name="lookup_order", input={"id": "ORD-1"}),
        ToolCall(name="notify_customer", input={"id": "ORD-1"}),
    ]
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 1.0


@pytest.mark.unit
def test_build_eval_case_keeps_genuine_repeated_tool_calls() -> None:
    """Deduplication must not erase a legitimately repeated identical call."""
    from harness_evals.conversation.simulator import ConversationSimulator
    from harness_evals.core.types import ToolCall

    golden = ConversationGolden(
        scenario="Retry the same lookup",
        expected_outcome="Looked up twice",
        expected_tool_calls=[
            ToolCall(name="lookup_order", input={"id": "ORD-1"}),
            ToolCall(name="lookup_order", input={"id": "ORD-1"}),
        ],
    )
    history = [
        Message(
            role="assistant",
            content="Looking up twice.",
            tool_calls=[
                ToolCall(name="lookup_order", input={"id": "ORD-1"}),
                ToolCall(name="lookup_order", input={"id": "ORD-1"}),
            ],
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert eval_case.tool_calls == [
        ToolCall(name="lookup_order", input={"id": "ORD-1"}),
        ToolCall(name="lookup_order", input={"id": "ORD-1"}),
    ]


@pytest.mark.unit
def test_build_eval_case_strips_langfuse_tool_prefix() -> None:
    """A Langfuse-imported `tool:`-prefixed call normalizes to the golden's short name."""
    from harness_evals.conversation.simulator import ConversationSimulator
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_argument_match import ToolArgumentMatchMetric
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    golden = ConversationGolden(
        scenario="Order lookup",
        expected_outcome="Found",
        expected_tool_calls=[ToolCall(name="lookup_order", input={"id": "ORD-1"})],
    )
    history = [
        Message(
            role="assistant",
            content="Looking up the order.",
            tool_calls=[ToolCall(name="tool:lookup_order", input={"id": "ORD-1"})],
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert eval_case.tool_calls == [ToolCall(name="lookup_order", input={"id": "ORD-1"})]
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 1.0
    assert ToolArgumentMatchMetric(pair="exact").measure(eval_case).value == 1.0


@pytest.mark.unit
def test_build_eval_case_keeps_expected_tool_prefixed_builtin() -> None:
    """`tool:Write` under expected `Write` survives the denylist and matches by name."""
    from harness_evals.conversation.simulator import ConversationSimulator
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    golden = ConversationGolden(
        scenario="Write a file",
        expected_outcome="Written",
        expected_tool_calls=[ToolCall(name="Write", input={"path": "/tmp/out"})],
    )
    history = [
        Message(
            role="assistant",
            content="Writing the file.",
            tool_calls=[ToolCall(name="tool:Write", input={"path": "/tmp/out"})],
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert eval_case.tool_calls == [ToolCall(name="Write", input={"path": "/tmp/out"})]
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 1.0


@pytest.mark.unit
def test_build_eval_case_drops_unexpected_tool_prefixed_builtin() -> None:
    """An unexpected `tool:Read` is still recognized as an agent builtin and dropped."""
    from harness_evals.conversation.simulator import ConversationSimulator
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    golden = ConversationGolden(
        scenario="No Harness tools should be called",
        expected_outcome="Answered from memory",
        expected_tool_calls=[],
    )
    history = [
        Message(
            role="assistant",
            content="Reading my notes.",
            tool_calls=[
                ToolCall(name="tool:Read", input={"path": "/tmp/notes"}),
                ToolCall(name="tool:Skill", input={}),
            ],
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert eval_case.tool_calls == []
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 1.0


@pytest.mark.unit
def test_build_eval_case_keeps_expected_builtin_alongside_mcp_call() -> None:
    """The mcp__ filter must not undo the expected-names exemption."""
    from harness_evals.conversation.simulator import ConversationSimulator
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_argument_match import ToolArgumentMatchMetric
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    golden = ConversationGolden(
        scenario="Write a file then create a pipeline",
        expected_outcome="Both done",
        expected_tool_calls=[
            ToolCall(name="Write", input={"path": "/tmp/out"}),
            ToolCall(name="harness_create", input={"type": "pipeline"}),
        ],
    )
    history = [
        Message(
            role="assistant",
            content="Writing then creating.",
            tool_calls=[
                ToolCall(name="Write", input={"path": "/tmp/out"}),
                ToolCall(name="mcp__harness__harness_create", input={"type": "pipeline"}),
            ],
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert eval_case.tool_calls == [
        ToolCall(name="Write", input={"path": "/tmp/out"}),
        ToolCall(name="harness_create", input={"type": "pipeline"}),
    ]
    assert ToolArgumentMatchMetric(pair="exact").measure(eval_case).value == 1.0
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 1.0


@pytest.mark.unit
def test_build_eval_case_drops_unexpected_builtin_alongside_mcp_call() -> None:
    """An unexpected builtin next to an MCP call is still filtered out."""
    from harness_evals.conversation.simulator import ConversationSimulator
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    golden = ConversationGolden(
        scenario="List pipelines",
        expected_outcome="Listed",
        expected_tool_calls=[ToolCall(name="harness_list")],
    )
    history = [
        Message(
            role="assistant",
            content="Reading then listing.",
            tool_calls=[
                ToolCall(name="Read", input={"path": "/tmp/notes"}),
                ToolCall(name="mcp__harness__harness_list", input={"type": "pipeline"}),
            ],
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert eval_case.tool_calls == [ToolCall(name="harness_list", input={"type": "pipeline"})]
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 1.0


@pytest.mark.unit
def test_build_eval_case_keeps_short_name_alongside_mcp_call() -> None:
    """A hand-authored short name survives when an MCP call is also present."""
    from harness_evals.conversation.simulator import ConversationSimulator
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric

    golden = ConversationGolden(
        scenario="Look up then list",
        expected_outcome="Both done",
        expected_tool_calls=[
            ToolCall(name="lookup_order"),
            ToolCall(name="harness_list"),
        ],
    )
    history = [
        Message(
            role="assistant",
            content="Looking up then listing.",
            tool_calls=[
                ToolCall(name="lookup_order", input={"id": "ORD-1"}),
                ToolCall(name="mcp__harness__harness_list", input={"type": "pipeline"}),
            ],
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert [call.name for call in eval_case.tool_calls or []] == ["lookup_order", "harness_list"]
    assert ToolCorrectnessMetric(mode="exact").measure(eval_case).value == 1.0


@pytest.mark.unit
def test_build_eval_case_preserves_hand_authored_short_tool_names() -> None:
    from harness_evals.conversation.simulator import ConversationSimulator
    from harness_evals.core.types import ToolCall

    golden = ConversationGolden(
        scenario="Order lookup",
        expected_outcome="Emailed",
        expected_tool_calls=[ToolCall(name="lookup_order")],
    )
    history = [
        Message(
            role="assistant",
            content="Looking up the order.",
            tool_calls=[ToolCall(name="lookup_order", input={"id": "ORD-1"})],
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert eval_case.tool_calls == [ToolCall(name="lookup_order", input={"id": "ORD-1"})]


@pytest.mark.unit
def test_build_eval_case_tool_calls_empty_when_sse_without_requests() -> None:
    from harness_evals.conversation.simulator import ConversationSimulator
    from harness_evals.core.types import ToolCall
    from harness_evals.metrics.agent.tool_argument_match import ToolArgumentMatchMetric

    golden = ConversationGolden(
        scenario="No tools",
        expected_outcome="Answer",
        expected_tool_calls=[ToolCall(name="harness_list")],
    )
    history = [
        Message(
            role="assistant",
            content="Hello.",
            metadata={"sse_events": {"assistant_message": [{"v": "Hello."}]}},
        )
    ]

    eval_case = ConversationSimulator(simulator_llm=None)._build_eval_case(golden, history)
    assert eval_case.tool_calls == []

    score = ToolArgumentMatchMetric(skip_when_missing=True).measure(eval_case)
    assert not score.passed
    assert score.value == 0.0


@pytest.mark.unit
class TestShortToolName:
    def test_strips_mcp_server_prefix(self):
        from harness_evals.conversation.simulator import _short_tool_name

        assert _short_tool_name("mcp__harness__harness_create") == "harness_create"
        assert _short_tool_name("mcp__harness_local__validate_pipeline_yaml") == "validate_pipeline_yaml"

    def test_preserves_non_mcp_double_underscore_names(self):
        from harness_evals.conversation.simulator import _short_tool_name

        assert _short_tool_name("my_custom__tool") == "my_custom__tool"
        assert _short_tool_name("Skill") == "Skill"

    def test_sse_payload_uses_short_names(self):
        from harness_evals.conversation.simulator import _tool_calls_from_sse_payload

        calls = _tool_calls_from_sse_payload(
            {
                "v": [
                    {"name": "mcp__harness__harness_create", "arguments": {"resource_type": "pipeline_v1"}},
                    {"name": "my_custom__tool", "arguments": {}},
                ]
            },
            result=False,
        )
        assert [c.name for c in calls] == ["harness_create", "my_custom__tool"]

    def test_raw_tool_calls_from_sse_events_keeps_raw_names_unfiltered(self):
        from harness_evals.conversation.simulator import _raw_tool_calls_from_sse_events

        sse_events = {
            "assistant_tool_request": [
                {"v": [{"name": "Skill", "arguments": {"name": "pipeline-generation"}}]},
                {"v": [{"name": "search_kb", "arguments": {"q": "x"}}]},
                {"v": [{"name": "mcp__harness__harness_create", "arguments": {"resource_type": "pipeline_v1"}}]},
            ]
        }
        calls = _raw_tool_calls_from_sse_events(sse_events)
        # Raw + unfiltered: filtering/normalization happen in _tool_calls_for_eval_case.
        assert [c.name for c in calls] == ["Skill", "search_kb", "mcp__harness__harness_create"]
