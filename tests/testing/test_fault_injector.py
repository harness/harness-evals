"""Tests for Fault dataclass and FaultInjector harness."""

import pytest

from harness_evals.testing.fault_injector import Fault, FaultInjector

# ---------------------------------------------------------------------------
# Fault dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFaultDataclass:
    def test_construction(self):
        f = Fault(type="timeout", probability=0.5)
        assert f.type == "timeout"
        assert f.probability == 0.5
        assert f.config == {}

    def test_with_config(self):
        f = Fault(type="malformed_response", probability=0.3, config={"response": "bad"})
        assert f.config["response"] == "bad"

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Fault type must be"):
            Fault(type="crash", probability=0.5)

    def test_invalid_probability(self):
        with pytest.raises(ValueError, match="probability"):
            Fault(type="timeout", probability=1.5)

    def test_all_valid_types(self):
        for ft in ("timeout", "malformed_response", "rate_limit", "empty_response"):
            f = Fault(type=ft, probability=0.1)
            assert f.type == ft


# ---------------------------------------------------------------------------
# FaultInjector
# ---------------------------------------------------------------------------


async def _echo_agent(agent_input: str) -> str:
    return f"echo:{agent_input}"


@pytest.mark.unit
class TestFaultInjector:
    async def test_no_faults_passthrough(self):
        injector = FaultInjector(agent_fn=_echo_agent, faults=[])
        result = await injector.run("hello")
        assert result == "echo:hello"
        assert len(injector.history) == 1
        assert not injector.history[0]["injected"]

    async def test_timeout_always(self):
        injector = FaultInjector(
            agent_fn=_echo_agent,
            faults=[Fault(type="timeout", probability=1.0)],
            seed=42,
        )
        with pytest.raises(TimeoutError, match="Injected timeout"):
            await injector.run("test")
        assert injector.history[0]["injected"]
        assert injector.history[0]["fault_type"] == "timeout"

    async def test_zero_probability_never_injects(self):
        injector = FaultInjector(
            agent_fn=_echo_agent,
            faults=[Fault(type="timeout", probability=0.0)],
            seed=42,
        )
        for _ in range(10):
            result = await injector.run("test")
            assert result == "echo:test"
        assert all(not h["injected"] for h in injector.history)

    async def test_rate_limit(self):
        injector = FaultInjector(
            agent_fn=_echo_agent,
            faults=[Fault(type="rate_limit", probability=1.0)],
        )
        with pytest.raises(RuntimeError, match="rate limited"):
            await injector.run("test")

    async def test_malformed_response(self):
        injector = FaultInjector(
            agent_fn=_echo_agent,
            faults=[Fault(type="malformed_response", probability=1.0, config={"response": "BAD"})],
        )
        result = await injector.run("test")
        assert result == "BAD"

    async def test_malformed_default(self):
        injector = FaultInjector(
            agent_fn=_echo_agent,
            faults=[Fault(type="malformed_response", probability=1.0)],
        )
        result = await injector.run("test")
        assert result == "{malformed}"

    async def test_empty_response(self):
        injector = FaultInjector(
            agent_fn=_echo_agent,
            faults=[Fault(type="empty_response", probability=1.0)],
        )
        result = await injector.run("test")
        assert result == ""

    async def test_seed_reproducibility(self):
        faults = [Fault(type="timeout", probability=0.5)]

        injected_a = []
        for _ in range(20):
            inj = FaultInjector(agent_fn=_echo_agent, faults=faults, seed=123)
            try:
                await inj.run("test")
                injected_a.append(False)
            except TimeoutError:
                injected_a.append(True)

        injected_b = []
        for _ in range(20):
            inj = FaultInjector(agent_fn=_echo_agent, faults=faults, seed=123)
            try:
                await inj.run("test")
                injected_b.append(False)
            except TimeoutError:
                injected_b.append(True)

        assert injected_a == injected_b

    async def test_history_tracking(self):
        injector = FaultInjector(
            agent_fn=_echo_agent,
            faults=[Fault(type="empty_response", probability=0.0)],
        )
        await injector.run("a")
        await injector.run("b")
        assert len(injector.history) == 2
        assert injector.history[0]["input"] == "a"
        assert injector.history[1]["input"] == "b"

    async def test_multiple_fault_types(self):
        """First fault that triggers wins."""
        injector = FaultInjector(
            agent_fn=_echo_agent,
            faults=[
                Fault(type="timeout", probability=0.0),
                Fault(type="empty_response", probability=1.0),
            ],
            seed=42,
        )
        result = await injector.run("test")
        assert result == ""
        assert injector.history[0]["fault_type"] == "empty_response"
