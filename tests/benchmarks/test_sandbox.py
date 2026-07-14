"""Tests for sandboxed code execution."""

from __future__ import annotations

import pytest

from harness_evals.benchmarks.sandbox import execute_python


@pytest.mark.unit
class TestSandbox:
    def test_passing_code(self):
        code = "assert 1 + 1 == 2"
        result = execute_python(code)
        assert result.passed
        assert not result.timed_out

    def test_failing_code(self):
        code = "assert 1 + 1 == 3"
        result = execute_python(code)
        assert not result.passed
        assert "AssertionError" in result.stderr

    def test_syntax_error(self):
        code = "def foo(\n"
        result = execute_python(code)
        assert not result.passed
        assert "SyntaxError" in result.stderr

    def test_stdout_capture(self):
        code = "print('hello world')"
        result = execute_python(code)
        assert result.passed
        assert "hello world" in result.stdout

    def test_timeout(self):
        code = "import time; time.sleep(10)"
        result = execute_python(code, timeout=0.5)
        assert not result.passed
        assert result.timed_out

    def test_runtime_error(self):
        code = "x = 1 / 0"
        result = execute_python(code)
        assert not result.passed
        assert "ZeroDivisionError" in result.stderr

    def test_multiline_function(self):
        code = """
def add(a, b):
    return a + b

assert add(2, 3) == 5
assert add(-1, 1) == 0
"""
        result = execute_python(code)
        assert result.passed
