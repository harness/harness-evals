"""Format recommendation dict into EvalConfig YAML and goldens JSONL."""

from __future__ import annotations
import json
from pathlib import Path


def write_outputs(recommendation: dict, output_dir: str = ".") -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    goldens_path = out / "recommended.goldens.jsonl"
    config_path = out / "recommended.eval.yaml"

    with goldens_path.open("w") as f:
        for case in recommendation.get("recommended_dataset", []):
            golden = {
                "input": case.get("input", ""),
                "expected": case.get("expected", ""),
                "context": case.get("context"),
                "expected_tools": case.get("expected_tools"),
                "expected_tool_calls": case.get("expected_tool_calls"),
                "metadata": case.get("metadata", {}),
                "tags": case.get("tags", {}),
            }
            f.write(json.dumps(golden) + "\n")

    metrics_yaml = ""
    for m in recommendation.get("recommended_metrics", []):
        metrics_yaml += f"  - kind: {m['name']}\n"
        metrics_yaml += f"    threshold: {m['threshold']}\n"

    config_yaml = f"""name: recommended-eval

dataset: {goldens_path.resolve()}

target:
  type: prompt
  # Replace with your actual target
  prompt: ./your-prompt.txt
  model:
    provider: openai
    name: gpt-4o

metrics:
{metrics_yaml}
sinks:
  - stdout
  - type: json
    path: ./results.jsonl

baseline:
  store: json
  path: .evals/baseline.json
"""

    config_path.write_text(config_yaml)
    return config_path, goldens_path


def print_recommendation(recommendation: dict) -> None:
    print("\n=== DIMENSIONS COVERED ===\n")
    for d in recommendation.get("dimensions_covered", []):
        applies = "YES" if d["applies"] else "no"
        print(f"  {d['dimension']:<15} {applies:<5}  {d['rationale']}")

    print("\n=== RECOMMENDED METRICS ===\n")
    for m in recommendation.get("recommended_metrics", []):
        print(f"  {m['name']:<35} threshold={m['threshold']}  ({m['dimension']})")
        print(f"    → {m['rationale']}")

    print("\n=== RECOMMENDED DATASET ===\n")
    for i, case in enumerate(recommendation.get("recommended_dataset", []), 1):
        print(f"  Test Case {i} (tests: {case.get('metric_tested', 'n/a')})")
        inp = case['input'][:80]
        exp = case['expected'][:80]
        print(f"    Input:    {inp}")
        print(f"    Expected: {exp}")

    print("\n=== RECOMMENDED ACTIONS ===\n")
    print(f"  {recommendation.get('recommended_actions', '')}")
    print()
