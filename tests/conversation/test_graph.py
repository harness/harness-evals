"""Tests for SimulationGraph and graph-driven conversation simulation."""

import pytest

from harness_evals.conversation import (
    BranchNode,
    ConversationGolden,
    ConversationMode,
    ConversationSimulator,
    Edge,
    LLMNode,
    ScriptedNode,
    SimulationGraph,
    StopNode,
)
from harness_evals.core.types import Message
from tests.conftest import MockLLM


@pytest.mark.unit
class TestSimulationGraphValidation:
    def test_valid_graph_constructs(self):
        graph = SimulationGraph(
            nodes={"a": ScriptedNode(message="hi"), "b": StopNode()},
            edges=[Edge(source="a", target="b")],
            predicates={},
            start="a",
        )
        assert graph.start == "a"

    def test_invalid_start_node(self):
        with pytest.raises(ValueError, match="start node 'missing'"):
            SimulationGraph(
                nodes={"a": ScriptedNode(message="hi")},
                edges=[],
                predicates={},
                start="missing",
            )

    def test_invalid_edge_source(self):
        with pytest.raises(ValueError, match="edge source 'missing'"):
            SimulationGraph(
                nodes={"a": ScriptedNode(message="hi"), "b": StopNode()},
                edges=[Edge(source="missing", target="b")],
                predicates={},
                start="a",
            )

    def test_invalid_edge_target(self):
        with pytest.raises(ValueError, match="edge target 'missing'"):
            SimulationGraph(
                nodes={"a": ScriptedNode(message="hi"), "b": StopNode()},
                edges=[Edge(source="a", target="missing")],
                predicates={},
                start="a",
            )

    def test_invalid_predicate_reference(self):
        with pytest.raises(ValueError, match="predicate 'unknown'"):
            SimulationGraph(
                nodes={"a": ScriptedNode(message="hi"), "b": StopNode()},
                edges=[Edge(source="a", target="b", predicate="unknown")],
                predicates={},
                start="a",
            )

    def test_cycle_detection(self):
        with pytest.raises(ValueError, match="cycle detected"):
            SimulationGraph(
                nodes={"a": ScriptedNode(message="hi"), "b": ScriptedNode(message="bye")},
                edges=[Edge(source="a", target="b"), Edge(source="b", target="a")],
                predicates={},
                start="a",
            )

    def test_self_loop_detection(self):
        with pytest.raises(ValueError, match="cycle detected"):
            SimulationGraph(
                nodes={"a": ScriptedNode(message="hi")},
                edges=[Edge(source="a", target="a")],
                predicates={},
                start="a",
            )

    def test_branch_node_cannot_be_start(self):
        with pytest.raises(ValueError, match="start node cannot be a BranchNode"):
            SimulationGraph(
                nodes={"br": BranchNode(), "a": StopNode()},
                edges=[Edge(source="br", target="a")],
                predicates={},
                start="br",
            )

    def test_stop_node_cannot_have_outgoing_edges(self):
        with pytest.raises(ValueError, match="StopNode 'end' cannot have outgoing edges"):
            SimulationGraph(
                nodes={"a": ScriptedNode(message="hi"), "end": StopNode()},
                edges=[Edge(source="a", target="end"), Edge(source="end", target="a")],
                predicates={},
                start="a",
            )


@pytest.mark.unit
class TestSimulationGraphResolveNext:
    def test_unconditional_edge(self):
        graph = SimulationGraph(
            nodes={"a": ScriptedNode(message="hi"), "b": StopNode()},
            edges=[Edge(source="a", target="b")],
            predicates={},
            start="a",
        )
        result = graph.resolve_next("a", Message(role="assistant", content="anything"))
        assert result == "b"

    def test_predicate_match(self):
        graph = SimulationGraph(
            nodes={
                "a": ScriptedNode(message="hi"),
                "b": StopNode(),
                "c": ScriptedNode(message="more"),
            },
            edges=[
                Edge(source="a", target="b", predicate="has_answer"),
                Edge(source="a", target="c", predicate="needs_more"),
            ],
            predicates={
                "has_answer": lambda m: "done" in (m.content or ""),
                "needs_more": lambda m: "done" not in (m.content or ""),
            },
            start="a",
        )
        assert graph.resolve_next("a", Message(role="assistant", content="done")) == "b"
        assert graph.resolve_next("a", Message(role="assistant", content="tell me more")) == "c"

    def test_no_match_returns_none(self):
        graph = SimulationGraph(
            nodes={"a": ScriptedNode(message="hi"), "b": StopNode()},
            edges=[Edge(source="a", target="b", predicate="never")],
            predicates={"never": lambda m: False},
            start="a",
        )
        assert graph.resolve_next("a", Message(role="assistant", content="x")) is None

    def test_default_edge_as_fallback(self):
        graph = SimulationGraph(
            nodes={
                "a": ScriptedNode(message="hi"),
                "b": StopNode(),
                "c": ScriptedNode(message="more"),
            },
            edges=[
                Edge(source="a", target="c", predicate="never"),
                Edge(source="a", target="b"),  # default
            ],
            predicates={"never": lambda m: False},
            start="a",
        )
        assert graph.resolve_next("a", Message(role="assistant", content="x")) == "b"


@pytest.mark.unit
class TestSimulationGraphSerialization:
    def test_round_trip(self):
        predicates = {
            "is_question": lambda m: "?" in (m.content or ""),
            "is_done": lambda m: "done" in (m.content or ""),
        }
        graph = SimulationGraph(
            nodes={
                "greet": ScriptedNode(message="Hello!"),
                "ask": LLMNode(goal="Ask about pricing"),
                "route": BranchNode(),
                "end": StopNode(),
            },
            edges=[
                Edge(source="greet", target="route"),
                Edge(source="route", target="ask", predicate="is_question"),
                Edge(source="route", target="end", predicate="is_done"),
            ],
            predicates=predicates,
            start="greet",
        )

        data = graph.to_dict()
        restored = SimulationGraph.from_dict(data, predicates)

        assert restored.start == graph.start
        assert len(restored.nodes) == len(graph.nodes)
        assert len(restored.edges) == len(graph.edges)
        assert restored.to_dict() == data

    def test_to_dict_format(self):
        graph = SimulationGraph(
            nodes={"a": ScriptedNode(message="hi"), "b": StopNode()},
            edges=[Edge(source="a", target="b")],
            predicates={},
            start="a",
        )
        data = graph.to_dict()
        assert data == {
            "start": "a",
            "nodes": [
                {"id": "a", "type": "scripted", "message": "hi"},
                {"id": "b", "type": "stop"},
            ],
            "edges": [{"source": "a", "target": "b"}],
        }

    def test_from_dict_without_predicates(self):
        data = {
            "start": "a",
            "nodes": [
                {"id": "a", "type": "scripted", "message": "hi"},
                {"id": "b", "type": "stop"},
            ],
            "edges": [{"source": "a", "target": "b"}],
        }
        graph = SimulationGraph.from_dict(data)
        assert graph.start == "a"
        assert isinstance(graph.nodes["a"], ScriptedNode)
        assert isinstance(graph.nodes["b"], StopNode)

    def test_from_dict_unknown_node_type_raises(self):
        data = {
            "start": "a",
            "nodes": [{"id": "a", "type": "unknown_type"}],
            "edges": [],
        }
        with pytest.raises(ValueError, match="unknown node type 'unknown_type'"):
            SimulationGraph.from_dict(data)

    def test_from_dict_missing_required_keys_raises(self):
        with pytest.raises(ValueError, match="missing required key 'start'"):
            SimulationGraph.from_dict({"nodes": [], "edges": []})
        with pytest.raises(ValueError, match="missing required key 'nodes'"):
            SimulationGraph.from_dict({"start": "a", "edges": []})

    def test_from_dict_unexpected_node_keys_raises(self):
        data = {
            "start": "a",
            "nodes": [{"id": "a", "type": "scripted", "message": "hi", "extra": "bad"}],
            "edges": [],
        }
        with pytest.raises(ValueError, match="unexpected keys"):
            SimulationGraph.from_dict(data)

    def test_from_dict_missing_required_node_key_raises(self):
        data = {
            "start": "a",
            "nodes": [{"id": "a", "type": "scripted"}],
            "edges": [],
        }
        with pytest.raises(ValueError, match="missing required keys"):
            SimulationGraph.from_dict(data)

    def test_from_dict_with_predicates_but_none_supplied_raises(self):
        data = {
            "start": "a",
            "nodes": [
                {"id": "a", "type": "scripted", "message": "hi"},
                {"id": "b", "type": "stop"},
            ],
            "edges": [{"source": "a", "target": "b", "predicate": "check"}],
        }
        with pytest.raises(ValueError, match="no predicates dict was supplied"):
            SimulationGraph.from_dict(data)

    def test_multiple_unconditional_edges_from_same_source_raises(self):
        with pytest.raises(ValueError, match="multiple unconditional edges"):
            SimulationGraph(
                nodes={
                    "a": ScriptedNode(message="hi"),
                    "b": StopNode(),
                    "c": StopNode(),
                },
                edges=[Edge(source="a", target="b"), Edge(source="a", target="c")],
                predicates={},
                start="a",
            )


@pytest.mark.unit
class TestEdgeConditionCompilation:
    def test_contains_matches_case_insensitively_by_default(self):
        edge = Edge(source="a", target="b", condition={"type": "contains", "value": "HELLO"})
        assert edge._matcher(Message(role="assistant", content="well hello there")) is True
        assert edge._matcher(Message(role="assistant", content="goodbye")) is False

    def test_contains_case_sensitive_opt_in(self):
        edge = Edge(
            source="a",
            target="b",
            condition={"type": "contains", "value": "HELLO", "case_sensitive": True},
        )
        assert edge._matcher(Message(role="assistant", content="HELLO there")) is True
        assert edge._matcher(Message(role="assistant", content="hello there")) is False

    def test_not_contains(self):
        edge = Edge(source="a", target="b", condition={"type": "not_contains", "value": "error"})
        assert edge._matcher(Message(role="assistant", content="all good")) is True
        assert edge._matcher(Message(role="assistant", content="an ERROR occurred")) is False

    def test_equals_case_insensitive_by_default(self):
        edge = Edge(source="a", target="b", condition={"type": "equals", "value": "Yes"})
        assert edge._matcher(Message(role="assistant", content="yes")) is True
        assert edge._matcher(Message(role="assistant", content="yes please")) is False

    def test_equals_case_sensitive_opt_in(self):
        edge = Edge(
            source="a",
            target="b",
            condition={"type": "equals", "value": "Yes", "case_sensitive": True},
        )
        assert edge._matcher(Message(role="assistant", content="Yes")) is True
        assert edge._matcher(Message(role="assistant", content="yes")) is False

    def test_regex_uses_search_and_is_case_insensitive_by_default(self):
        edge = Edge(source="a", target="b", condition={"type": "regex", "value": r"order #\d+"})
        assert edge._matcher(Message(role="assistant", content="Your ORDER #12345 shipped")) is True
        assert edge._matcher(Message(role="assistant", content="no order here")) is False

    def test_regex_case_sensitive_opt_in(self):
        edge = Edge(
            source="a",
            target="b",
            condition={"type": "regex", "value": "ERROR", "case_sensitive": True},
        )
        assert edge._matcher(Message(role="assistant", content="an ERROR occurred")) is True
        assert edge._matcher(Message(role="assistant", content="an error occurred")) is False

    def test_none_content_treated_as_empty_string(self):
        edge = Edge(source="a", target="b", condition={"type": "contains", "value": "x"})
        assert edge._matcher(Message(role="assistant", content=None)) is False
        not_contains_edge = Edge(source="a", target="b", condition={"type": "not_contains", "value": "x"})
        assert not_contains_edge._matcher(Message(role="assistant", content=None)) is True

    def test_unknown_condition_type_raises(self):
        with pytest.raises(ValueError, match="unknown condition type"):
            Edge(source="a", target="b", condition={"type": "startswith", "value": "x"})

    def test_missing_value_raises(self):
        with pytest.raises(ValueError, match="requires a string 'value'"):
            Edge(source="a", target="b", condition={"type": "contains"})

    def test_non_string_value_raises(self):
        with pytest.raises(ValueError, match="requires a string 'value'"):
            Edge(source="a", target="b", condition={"type": "contains", "value": 123})

    def test_invalid_regex_raises_at_construction(self):
        with pytest.raises(ValueError, match="invalid regex pattern"):
            Edge(source="a", target="b", condition={"type": "regex", "value": "("})

    def test_both_predicate_and_condition_raises(self):
        with pytest.raises(ValueError, match="set only one of 'predicate' or 'condition'"):
            Edge(
                source="a",
                target="b",
                predicate="some_predicate",
                condition={"type": "contains", "value": "x"},
            )


@pytest.mark.unit
class TestSimulationGraphConditionRouting:
    def test_condition_edge_routes_on_match(self):
        graph = SimulationGraph(
            nodes={
                "a": ScriptedNode(message="hi"),
                "b": StopNode(),
                "c": ScriptedNode(message="more"),
            },
            edges=[
                Edge(source="a", target="b", condition={"type": "contains", "value": "done"}),
                Edge(source="a", target="c"),  # default
            ],
            predicates={},
            start="a",
        )
        assert graph.resolve_next("a", Message(role="assistant", content="all done")) == "b"
        assert graph.resolve_next("a", Message(role="assistant", content="keep going")) == "c"

    def test_condition_edge_first_match_wins_mixed_with_predicate(self):
        graph = SimulationGraph(
            nodes={
                "a": ScriptedNode(message="hi"),
                "b": StopNode(),
                "c": ScriptedNode(message="more"),
                "d": StopNode(),
            },
            edges=[
                Edge(source="a", target="b", condition={"type": "contains", "value": "done"}),
                Edge(source="a", target="c", predicate="is_question"),
                Edge(source="a", target="d"),  # default
            ],
            predicates={"is_question": lambda m: "?" in (m.content or "")},
            start="a",
        )
        assert graph.resolve_next("a", Message(role="assistant", content="all done")) == "b"
        assert graph.resolve_next("a", Message(role="assistant", content="need more?")) == "c"
        assert graph.resolve_next("a", Message(role="assistant", content="nothing matches")) == "d"

    def test_condition_edge_plus_default_is_valid(self):
        graph = SimulationGraph(
            nodes={"a": ScriptedNode(message="hi"), "b": StopNode(), "c": StopNode()},
            edges=[
                Edge(source="a", target="b", condition={"type": "contains", "value": "x"}),
                Edge(source="a", target="c"),
            ],
            predicates={},
            start="a",
        )
        assert graph.resolve_next("a", Message(role="assistant", content="no match")) == "c"

    def test_two_unconditional_edges_still_raises_with_condition_present(self):
        with pytest.raises(ValueError, match="multiple unconditional edges"):
            SimulationGraph(
                nodes={"a": ScriptedNode(message="hi"), "b": StopNode(), "c": StopNode()},
                edges=[Edge(source="a", target="b"), Edge(source="a", target="c")],
                predicates={},
                start="a",
            )


@pytest.mark.unit
class TestSimulationGraphConditionSerialization:
    def test_from_dict_condition_edges_need_no_predicates_dict(self):
        data = {
            "start": "a",
            "nodes": [
                {"id": "a", "type": "scripted", "message": "hi"},
                {"id": "b", "type": "stop"},
                {"id": "c", "type": "stop"},
            ],
            "edges": [
                {"source": "a", "target": "b", "condition": {"type": "contains", "value": "yes"}},
                {"source": "a", "target": "c"},
            ],
        }
        graph = SimulationGraph.from_dict(data)  # no predicates dict supplied
        assert graph.resolve_next("a", Message(role="assistant", content="yes please")) == "b"
        assert graph.resolve_next("a", Message(role="assistant", content="nope")) == "c"

    def test_condition_round_trip(self):
        data = {
            "start": "a",
            "nodes": [
                {"id": "a", "type": "scripted", "message": "hi"},
                {"id": "b", "type": "stop"},
                {"id": "c", "type": "stop"},
            ],
            "edges": [
                {
                    "source": "a",
                    "target": "b",
                    "condition": {"type": "regex", "value": "order #\\d+", "case_sensitive": True},
                },
                {"source": "a", "target": "c"},
            ],
        }
        graph = SimulationGraph.from_dict(data)
        assert graph.to_dict() == data

    def test_from_dict_still_raises_for_predicate_edges_without_predicates_dict(self):
        data = {
            "start": "a",
            "nodes": [
                {"id": "a", "type": "scripted", "message": "hi"},
                {"id": "b", "type": "stop"},
            ],
            "edges": [{"source": "a", "target": "b", "predicate": "check"}],
        }
        with pytest.raises(ValueError, match="no predicates dict was supplied"):
            SimulationGraph.from_dict(data)


@pytest.mark.unit
class TestGraphSimulation:
    async def test_scripted_linear_path(self):
        graph = SimulationGraph(
            nodes={
                "turn1": ScriptedNode(message="How do I reset my password?"),
                "turn2": ScriptedNode(message="I don't have that email anymore"),
                "done": StopNode(),
            },
            edges=[
                Edge(source="turn1", target="turn2"),
                Edge(source="turn2", target="done"),
            ],
            predicates={},
            start="turn1",
        )

        async def agent_fn(messages: list[Message]) -> Message:
            return Message(role="assistant", content=f"Response {len(messages)}")

        golden = ConversationGolden(
            scenario="Password reset",
            expected_outcome="Agent helps with reset",
            graph_config=graph.to_dict(),
        )

        simulator = ConversationSimulator(simulator_llm=MockLLM(), graph=graph)
        result = await simulator.simulate(golden, agent_fn)

        assert result.messages is not None
        assert len(result.messages) == 4  # 2 user + 2 assistant
        assert result.messages[0].content == "How do I reset my password?"
        assert result.messages[2].content == "I don't have that email anymore"

    async def test_llm_node_generates_message(self):
        class GoalMockLLM(MockLLM):
            async def generate(self, prompt: str, **kwargs) -> str:
                return "Generated question about pricing"

        graph = SimulationGraph(
            nodes={
                "ask": LLMNode(goal="Ask about pricing"),
                "done": StopNode(),
            },
            edges=[Edge(source="ask", target="done")],
            predicates={},
            start="ask",
        )

        async def agent_fn(messages: list[Message]) -> Message:
            return Message(role="assistant", content="Our pricing starts at $10/mo")

        golden = ConversationGolden(
            scenario="Pricing inquiry",
            expected_outcome="Agent provides pricing info",
            graph_config=graph.to_dict(),
        )

        simulator = ConversationSimulator(simulator_llm=GoalMockLLM(), graph=graph)
        result = await simulator.simulate(golden, agent_fn)

        assert result.messages is not None
        assert result.messages[0].content == "Generated question about pricing"
        assert len(result.messages) == 2

    async def test_conditional_branch(self):
        graph = SimulationGraph(
            nodes={
                "ask": ScriptedNode(message="What's the status of my order?"),
                "route": BranchNode(),
                "clarify": ScriptedNode(message="Order #12345"),
                "done": StopNode(),
            },
            edges=[
                Edge(source="ask", target="route"),
                Edge(source="route", target="clarify", predicate="asks_for_id"),
                Edge(source="route", target="done", predicate="gives_status"),
                Edge(source="clarify", target="done"),
            ],
            predicates={
                "asks_for_id": lambda m: "order number" in (m.content or "").lower(),
                "gives_status": lambda m: "shipped" in (m.content or "").lower(),
            },
            start="ask",
        )

        async def agent_fn(messages: list[Message]) -> Message:
            if len(messages) == 1:
                return Message(role="assistant", content="Can you provide your order number?")
            return Message(role="assistant", content="Order #12345 has shipped!")

        golden = ConversationGolden(
            scenario="Order status check",
            expected_outcome="Get order status",
            graph_config=graph.to_dict(),
        )

        simulator = ConversationSimulator(simulator_llm=MockLLM(), graph=graph)
        result = await simulator.simulate(golden, agent_fn)

        assert result.messages is not None
        assert len(result.messages) == 4
        assert result.messages[0].content == "What's the status of my order?"
        assert result.messages[2].content == "Order #12345"

    async def test_stop_terminates_early(self):
        graph = SimulationGraph(
            nodes={
                "ask": ScriptedNode(message="Hello"),
                "done": StopNode(),
            },
            edges=[Edge(source="ask", target="done")],
            predicates={},
            start="ask",
        )

        async def agent_fn(messages: list[Message]) -> Message:
            return Message(role="assistant", content="Hi!")

        golden = ConversationGolden(
            scenario="Greeting",
            expected_outcome="Greet",
            max_turns=10,
            graph_config=graph.to_dict(),
        )

        simulator = ConversationSimulator(simulator_llm=MockLLM(), graph=graph)
        result = await simulator.simulate(golden, agent_fn)

        assert result.messages is not None
        assert len(result.messages) == 2

    async def test_no_match_stays_on_node(self):
        call_count = 0

        graph = SimulationGraph(
            nodes={
                "ask": ScriptedNode(message="Tell me a joke"),
                "done": StopNode(),
            },
            edges=[Edge(source="ask", target="done", predicate="is_funny")],
            predicates={"is_funny": lambda m: "haha" in (m.content or "")},
            start="ask",
        )

        async def agent_fn(messages: list[Message]) -> Message:
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                return Message(role="assistant", content="haha here's a joke!")
            return Message(role="assistant", content="I'm serious.")

        golden = ConversationGolden(
            scenario="Get a joke",
            expected_outcome="Agent tells joke",
            max_turns=5,
            graph_config=graph.to_dict(),
        )

        simulator = ConversationSimulator(simulator_llm=MockLLM(), graph=graph)
        result = await simulator.simulate(golden, agent_fn)

        assert result.messages is not None
        assert call_count == 3

    async def test_max_turns_caps_execution(self):
        graph = SimulationGraph(
            nodes={
                "ask": ScriptedNode(message="Again"),
                "done": StopNode(),
            },
            edges=[Edge(source="ask", target="done", predicate="never")],
            predicates={"never": lambda m: False},
            start="ask",
        )

        async def agent_fn(messages: list[Message]) -> Message:
            return Message(role="assistant", content="response")

        golden = ConversationGolden(
            scenario="Loop test",
            expected_outcome="Should cap",
            max_turns=3,
            graph_config=graph.to_dict(),
        )

        simulator = ConversationSimulator(simulator_llm=MockLLM(), graph=graph)
        result = await simulator.simulate(golden, agent_fn)

        assert result.messages is not None
        assert len(result.messages) == 6  # 3 turns * 2 messages each

    async def test_graph_from_golden_config(self):
        """Graph can be loaded from golden.graph_config when not passed to simulator."""
        graph = SimulationGraph(
            nodes={"a": ScriptedNode(message="hi"), "b": StopNode()},
            edges=[Edge(source="a", target="b")],
            predicates={},
            start="a",
        )

        async def agent_fn(messages: list[Message]) -> Message:
            return Message(role="assistant", content="hello")

        golden = ConversationGolden(
            scenario="Test",
            expected_outcome="Done",
            graph_config=graph.to_dict(),
        )

        simulator = ConversationSimulator(simulator_llm=MockLLM())
        result = await simulator.simulate(golden, agent_fn)

        assert result.messages is not None
        assert len(result.messages) == 2

    async def test_condition_branch_from_pure_data_with_no_predicates_dict(self):
        """A condition-guarded graph loaded from golden.graph_config drives simulation
        with zero caller-supplied predicates and no simulator LLM."""
        graph_data = {
            "start": "ask",
            "nodes": [
                {"id": "ask", "type": "scripted", "message": "What's the status of my order?"},
                {"id": "route", "type": "branch"},
                {"id": "clarify", "type": "scripted", "message": "Order #12345"},
                {"id": "done", "type": "stop"},
            ],
            "edges": [
                {"source": "ask", "target": "route"},
                {
                    "source": "route",
                    "target": "clarify",
                    "condition": {"type": "contains", "value": "order number"},
                },
                {"source": "route", "target": "done"},
                {"source": "clarify", "target": "done"},
            ],
        }
        graph = SimulationGraph.from_dict(graph_data)  # no predicates dict

        async def agent_fn(messages: list[Message]) -> Message:
            if len(messages) == 1:
                return Message(role="assistant", content="Can you provide your order number?")
            return Message(role="assistant", content="Order #12345 has shipped!")

        golden = ConversationGolden(
            scenario="Order status check",
            expected_outcome="Get order status",
            graph_config=graph.to_dict(),
        )

        simulator = ConversationSimulator(graph=graph)  # no simulator_llm
        result = await simulator.simulate(golden, agent_fn)

        assert result.messages is not None
        assert len(result.messages) == 4
        assert result.messages[0].content == "What's the status of my order?"
        assert result.messages[2].content == "Order #12345"


@pytest.mark.unit
class TestGraphSimulationErrors:
    async def test_branch_node_before_first_agent_response_raises(self):
        """BranchNode reached with only user messages in history raises RuntimeError."""
        graph = SimulationGraph(
            nodes={
                "greet": ScriptedNode(message="Hi"),
                "route": BranchNode(),
                "done": StopNode(),
            },
            edges=[
                Edge(source="greet", target="route"),
                Edge(source="route", target="done"),
            ],
            predicates={},
            start="greet",
        )

        async def agent_fn(messages: list[Message]) -> Message:
            return Message(role="user", content="not assistant")

        golden = ConversationGolden(
            scenario="Test branch before response",
            expected_outcome="Should raise",
            graph_config=graph.to_dict(),
        )

        simulator = ConversationSimulator(simulator_llm=MockLLM(), graph=graph)
        with pytest.raises(RuntimeError, match="no prior agent response"):
            await simulator.simulate(golden, agent_fn)


@pytest.mark.unit
class TestExistingModesUnchanged:
    async def test_replay_still_works(self):
        turns = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
        ]
        golden = ConversationGolden(scenario="Greeting", expected_outcome="Greet", turns=turns)

        async def noop_agent(messages: list[Message]) -> Message:
            return Message(role="assistant", content="unused")

        simulator = ConversationSimulator(simulator_llm=MockLLM())
        result = await simulator.simulate(golden, agent_fn=noop_agent)
        assert result.messages == turns

    async def test_scripted_still_works(self):
        turns = [
            Message(role="user", content="Hello"),
            Message(role="user", content="How are you?"),
        ]
        golden = ConversationGolden(
            scenario="Greeting",
            expected_outcome="Greet",
            turns=turns,
            mode=ConversationMode.SCRIPTED,
        )

        async def agent_fn(messages: list[Message]) -> Message:
            return Message(role="assistant", content="response")

        simulator = ConversationSimulator(simulator_llm=MockLLM())
        result = await simulator.simulate(golden, agent_fn)
        assert result.messages is not None
        assert len(result.messages) == 4
