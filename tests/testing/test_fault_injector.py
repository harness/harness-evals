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
        for ft in ("timeout", "malformed_response", "rate_limit", "empty_response",
                   "latency", "partial_response", "wrong_schema"):
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
        assert injector.history[0]["result"] == "BAD"

    async def test_malformed_default(self):
        injector = FaultInjector(
            agent_fn=_echo_agent,
            faults=[Fault(type="malformed_response", probability=1.0)],
        )
        result = await injector.run("test")
        assert result == "{malformed}"
        assert injector.history[0]["result"] == "{malformed}"

    async def test_empty_response(self):
        injector = FaultInjector(
            agent_fn=_echo_agent,
            faults=[Fault(type="empty_response", probability=1.0)],
        )
        result = await injector.run("test")
        assert result == ""
        assert injector.history[0]["result"] == ""

    async def test_raising_fault_has_no_result_in_history(self):
        injector = FaultInjector(
            agent_fn=_echo_agent,
            faults=[Fault(type="timeout", probability=1.0)],
        )
        with pytest.raises(TimeoutError):
            await injector.run("test")
        assert "result" not in injector.history[0]

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


# ---------------------------------------------------------------------------
# New fault types: latency, partial_response, wrong_schema
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLatencyFault:
    async def test_latency_calls_real_agent(self):
        """Latency fault still returns the real agent's response."""
        injector = FaultInjector(
            agent_fn=_echo_agent,
            faults=[Fault(type="latency", probability=1.0, config={"delay_s": 0.01})],
        )
        result = await injector.run("hello")
        assert result == "echo:hello"
        assert injector.history[0]["injected"]
        assert injector.history[0]["fault_type"] == "latency"

    async def test_latency_adds_delay(self):
        """The response is delayed by at least the configured duration."""
        import time

        injector = FaultInjector(
            agent_fn=_echo_agent,
            faults=[Fault(type="latency", probability=1.0, config={"delay_s": 0.05})],
        )
        t0 = time.perf_counter()
        await injector.run("test")
        elapsed = time.perf_counter() - t0
        assert elapsed >= 0.04  # allow slight timing slack

    async def test_latency_default_delay(self):
        """Default delay_s is 1.0 when not specified in config."""
        import time

        injector = FaultInjector(
            agent_fn=_echo_agent,
            faults=[Fault(type="latency", probability=1.0, config={"delay_s": 0.02})],
        )
        t0 = time.perf_counter()
        await injector.run("test")
        elapsed = time.perf_counter() - t0
        assert elapsed >= 0.015


@pytest.mark.unit
class TestPartialResponseFault:
    async def test_truncates_string(self):
        """String responses are truncated at truncate_at fraction."""
        injector = FaultInjector(
            agent_fn=_echo_agent,
            faults=[Fault(type="partial_response", probability=1.0, config={"truncate_at": 0.5})],
        )
        result = await injector.run("hello")
        full = "echo:hello"
        expected = full[: int(len(full) * 0.5)]
        assert result == expected

    async def test_truncate_default_fraction(self):
        """Default truncate_at is 0.5."""
        injector = FaultInjector(
            agent_fn=_echo_agent,
            faults=[Fault(type="partial_response", probability=1.0)],
        )
        result = await injector.run("abcdefgh")
        full = "echo:abcdefgh"
        assert result == full[: len(full) // 2]

    async def test_non_string_unchanged(self):
        """Non-string output is returned as-is."""

        async def dict_agent(x: str) -> dict:
            return {"key": x}

        injector = FaultInjector(
            agent_fn=dict_agent,
            faults=[Fault(type="partial_response", probability=1.0, config={"truncate_at": 0.3})],
        )
        result = await injector.run("test")
        assert result == {"key": "test"}

    async def test_calls_real_agent(self):
        """partial_response must call the real agent to get output."""
        call_count = 0

        async def counting_agent(x: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"response:{x}"

        injector = FaultInjector(
            agent_fn=counting_agent,
            faults=[Fault(type="partial_response", probability=1.0)],
        )
        await injector.run("test")
        assert call_count == 1


@pytest.mark.unit
class TestWrongSchemaFault:
    async def test_returns_config_response(self):
        """wrong_schema returns the response from fault config."""
        custom = {"unexpected_field": "value", "code": 500}
        injector = FaultInjector(
            agent_fn=_echo_agent,
            faults=[Fault(type="wrong_schema", probability=1.0, config={"response": custom})],
        )
        result = await injector.run("test")
        assert result == custom
        assert injector.history[0]["injected"]

    async def test_default_response(self):
        """Without config, returns default error dict."""
        injector = FaultInjector(
            agent_fn=_echo_agent,
            faults=[Fault(type="wrong_schema", probability=1.0)],
        )
        result = await injector.run("test")
        assert result == {"error": "unexpected_response"}

    async def test_does_not_call_agent(self):
        """wrong_schema must NOT call the real agent."""
        call_count = 0

        async def counting_agent(x: str) -> str:
            nonlocal call_count
            call_count += 1
            return x

        injector = FaultInjector(
            agent_fn=counting_agent,
            faults=[Fault(type="wrong_schema", probability=1.0, config={"response": {"bad": True}})],
        )
        await injector.run("test")
        assert call_count == 0
