"""HumanEval benchmark: Python code generation with process-isolated execution."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from harness_evals.benchmarks.base import BaseBenchmark
from harness_evals.benchmarks.dataset_cache import fetch_hf_dataset
from harness_evals.benchmarks.sandbox import execute_python


class HumanEval(BaseBenchmark):
    """HumanEval: Python code generation benchmark.

    Given a function signature and docstring, generate the function body.
    Evaluated via process-isolated execution of test cases (subprocess with
    restricted env and timeout — NOT a container sandbox).
    """

    def __init__(self, *, timeout: float = 10.0) -> None:
        """Initialize HumanEval benchmark.

        Args:
            timeout: Max seconds for code execution per problem.
        """
        super().__init__(name="humaneval")
        self.timeout = timeout

    async def load_dataset(self, *, offline: bool = False) -> list[dict]:
        return await fetch_hf_dataset("openai/openai_humaneval", split="test", offline=offline)

    def format_prompt(self, item: dict, *, shots: int = 0) -> str:
        prompt = item["prompt"]
        return (
            "Complete the following Python function. "
            "Only output the function body (the code that goes after the function signature). "
            "Do not include the function signature or any explanation.\n\n"
            f"{prompt}"
        )

    async def a_score_response(self, item: dict, response: str) -> tuple[float, str | None]:
        return await asyncio.to_thread(self.score_response, item, response)

    def score_response(self, item: dict, response: str) -> tuple[float, str | None]:
        prompt = item["prompt"]
        test = item["test"]
        entry_point = item["entry_point"]

        completion = _extract_code(response, entry_point)
        full_code = prompt + completion + "\n\n" + test + f"\ncheck({entry_point})\n"

        result = execute_python(full_code, timeout=self.timeout)

        if result.passed:
            return 1.0, None
        if result.timed_out:
            return 0.0, "Execution timed out"
        return 0.0, f"Tests failed: {result.stderr[:200]}"

    def _get_expected(self, item: dict) -> str:
        return item.get("canonical_solution", "")

    def _get_item_metadata(self, item: dict) -> dict[str, Any]:
        return {"task_id": item.get("task_id", ""), "entry_point": item.get("entry_point", "")}


def _extract_code(response: str, entry_point: str) -> str:
    """Extract Python code from model response.

    Handles cases where the model wraps code in markdown blocks or
    re-includes the function signature.
    """
    code_match = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    code = code_match.group(1) if code_match else response

    lines = code.split("\n")
    result_lines = []
    skip_signature = True

    for line in lines:
        if skip_signature and (
            line.strip().startswith(f"def {entry_point}")
            or line.strip().startswith("def ")
            or line.strip().startswith("```")
        ):
            continue
        skip_signature = False
        result_lines.append(line)

    result = "\n".join(result_lines)

    if not result.strip():
        return response

    if not result.startswith(("    ", "\t")):
        result = "    " + result.replace("\n", "\n    ")

    return result
