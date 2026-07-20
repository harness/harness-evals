"""Tests for token-usage capture."""

import asyncio

from harness_evals.llm.usage import (
    TokenUsage,
    collect_token_usage,
    record_token_usage,
)


class TestTokenUsage:
    def test_add_accumulates(self):
        u = TokenUsage()
        u.add(input_tokens=10, output_tokens=2)
        u.add(input_tokens=5, output_tokens=3)
        assert u.input_tokens == 15
        assert u.output_tokens == 5

    def test_none_stays_none_until_reported(self):
        u = TokenUsage()
        assert u.input_tokens is None
        assert u.output_tokens is None
        u.add(input_tokens=1)
        assert u.input_tokens == 1
        assert u.output_tokens is None

    def test_zero_is_recorded_as_zero_not_none(self):
        # "None means unknown, never coerce to 0" — the converse must also hold:
        # a genuine 0 is preserved as 0, distinct from an unreported None.
        u = TokenUsage()
        u.add(input_tokens=0, output_tokens=0)
        assert u.input_tokens == 0
        assert u.output_tokens == 0

    def test_zero_added_to_existing_count_is_a_noop_not_reset(self):
        u = TokenUsage()
        u.add(input_tokens=10, output_tokens=4)
        u.add(input_tokens=0, output_tokens=0)
        assert u.input_tokens == 10
        assert u.output_tokens == 4


class TestCollectTokenUsage:
    def test_records_within_block(self):
        with collect_token_usage() as usage:
            record_token_usage(input_tokens=100, output_tokens=20)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 20

    def test_no_op_outside_block(self):
        # Must not raise when no collector is active.
        record_token_usage(input_tokens=5, output_tokens=1)

    def test_nested_blocks_isolated(self):
        with collect_token_usage() as outer:
            record_token_usage(input_tokens=1)
            with collect_token_usage() as inner:
                record_token_usage(input_tokens=100)
            record_token_usage(input_tokens=1)
        assert inner.input_tokens == 100
        assert outer.input_tokens == 2

    def test_concurrent_tasks_isolated(self):
        async def scoped(i: int, o: int) -> TokenUsage:
            with collect_token_usage() as u:
                record_token_usage(input_tokens=i, output_tokens=o)
                await asyncio.sleep(0.01)
                record_token_usage(input_tokens=1, output_tokens=1)
            return u

        async def main() -> tuple[TokenUsage, TokenUsage]:
            return await asyncio.gather(scoped(100, 20), scoped(300, 50))

        r1, r2 = asyncio.run(main())
        assert (r1.input_tokens, r1.output_tokens) == (101, 21)
        assert (r2.input_tokens, r2.output_tokens) == (301, 51)
