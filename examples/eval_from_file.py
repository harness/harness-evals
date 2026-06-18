"""run_eval() one-liner loading goldens from a JSONL file.

Run: python examples/eval_from_file.py

No LLM key required — uses only deterministic metrics.
"""

from harness_evals import Golden, run_eval
from harness_evals.core.eval_case import EvalCase
from harness_evals.metrics import ExactMatchMetric


def my_agent(golden: Golden) -> EvalCase:
    return EvalCase.from_golden(golden, output=str(golden.expected or ""))


scores = run_eval(
    "file-demo",
    data="./examples/goldens.jsonl",
    target=my_agent,
    metrics=[ExactMatchMetric()],
)

print(f"\nAll passed: {all(s[0].passed for s in scores)}")
