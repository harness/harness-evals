"""Process-isolated code execution for code generation benchmarks.

WARNING: This is NOT a security sandbox. Generated code runs as the current user
with full filesystem and network access. The isolation is limited to:
- Separate subprocess (no access to parent process state)
- Timeout enforcement (prevents infinite loops)
- Minimal environment variables (no secrets from parent env)
- Temp working directory

For safe execution of untrusted code, wrap with Docker, nsjail, gVisor,
or seccomp. See: https://github.com/openai/human-eval#execution
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecutionResult:
    """Result of sandboxed code execution."""

    passed: bool
    stdout: str
    stderr: str
    timed_out: bool = False


def execute_python(code: str, *, timeout: float = 10.0) -> ExecutionResult:
    """Execute Python code in a subprocess with timeout and restricted environment.

    The code is written to a temp file and run in a fresh Python process with:
    - Timeout enforcement
    - Minimal environment variables (no secrets leakage)
    - Working directory set to a temp directory

    No in-process exec() or eval() is used.

    Args:
        code: Python source code to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        ExecutionResult with pass/fail, stdout, stderr.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = Path(f.name)

    restricted_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": tempfile.gettempdir(),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
    }

    try:
        result = subprocess.run(
            [sys.executable, str(temp_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=restricted_env,
            cwd=tempfile.gettempdir(),
        )
        return ExecutionResult(
            passed=result.returncode == 0,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            passed=False,
            stdout="",
            stderr=f"Execution timed out after {timeout}s",
            timed_out=True,
        )
    finally:
        temp_path.unlink(missing_ok=True)
