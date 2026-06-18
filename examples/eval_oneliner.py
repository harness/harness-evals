"""Code-first run_eval() one-liner example.

Run: python examples/eval_oneliner.py

No LLM key required — uses only deterministic metrics.
"""

from harness_evals import Golden, run_eval
from harness_evals.core.eval_case import EvalCase
from harness_evals.metrics import ContainsMetric, ExactMatchMetric

goldens = [
    Golden(input="What is 2+2?", expected="4"),
    Golden(input="Capital of France?", expected="Paris"),
]


def my_agent(golden: Golden) -> EvalCase:
    answers = {"What is 2+2?": "4", "Capital of France?": "Paris"}
    return EvalCase.from_golden(golden, output=answers.get(str(golden.input), "I don't know"))


scores = run_eval(
    "oneliner-demo",
    data=goldens,
    target=my_agent,
    metrics=[ExactMatchMetric(), ContainsMetric()],
)

print(f"\n{len(scores)} cases evaluated")
for i, case_scores in enumerate(scores):
    status = "PASS" if all(s.passed for s in case_scores) else "FAIL"
    print(f"  Case {i + 1}: {status}")
