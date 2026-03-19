"""Generate an evaluation dataset from source documents using an LLM.

Requires: pip install harness-evals[llm]
Set OPENAI_API_KEY (or ANTHROPIC_API_KEY) before running.

Run: python examples/synthesize_dataset.py
"""

import asyncio

from harness_evals.datasets import save_dataset
from harness_evals.llm.openai import OpenAILLM
from harness_evals.synthesizer import Synthesizer

SAMPLE_DOCUMENT = """\
# Kubernetes Delegates

A Kubernetes delegate is a service that connects your Kubernetes cluster to \
the Harness platform. It runs as a pod inside the cluster and communicates \
outbound to the Harness SaaS.

## Installation

Install the delegate using Helm:

```bash
helm repo add harness https://app.harness.io/storage/harness-download/delegate-helm-chart/
helm install my-delegate harness/harness-delegate \\
  --set delegateToken=<TOKEN> \\
  --set accountId=<ACCOUNT_ID>
```

## Architecture

The delegate polls the Harness Manager for tasks. When a pipeline stage \
targets the cluster, the manager routes the task to any healthy delegate \
registered for that account and cluster.

Delegates support auto-scaling: if task queue depth exceeds a threshold, \
additional delegate replicas are started. Minimum replicas default to 1, \
maximum to 10.

## Troubleshooting

If the delegate shows "Disconnected" in the UI:
1. Check network connectivity (port 443 outbound to app.harness.io)
2. Verify the delegate token has not expired
3. Inspect pod logs: `kubectl logs -l app=harness-delegate`
"""


async def main():
    llm = OpenAILLM(model="gpt-4o")
    synth = Synthesizer(llm=llm)

    # 1. Generate QA dataset
    print("Generating QA dataset...")
    qa_goldens = await synth.generate(
        documents=[SAMPLE_DOCUMENT],
        n=10,
        task_type="qa",
        difficulty="mixed",
    )
    save_dataset(qa_goldens, "examples/output/generated_qa.jsonl")
    print(f"  Saved {len(qa_goldens)} QA goldens to examples/output/generated_qa.jsonl")

    for g in qa_goldens[:3]:
        print(f"  Q: {g.input}")
        print(f"  A: {g.expected}")
        print()

    # 2. Generate structured output dataset
    print("Generating structured output dataset...")
    struct_goldens = await synth.generate(
        documents=[SAMPLE_DOCUMENT],
        n=5,
        task_type="structured_output",
        difficulty="medium",
    )
    save_dataset(struct_goldens, "examples/output/generated_structured.jsonl")
    print(f"  Saved {len(struct_goldens)} structured goldens")

    # 3. Generate summarization dataset
    print("Generating summarization dataset...")
    summ_goldens = await synth.generate(
        documents=[SAMPLE_DOCUMENT],
        n=5,
        task_type="summarization",
        difficulty="easy",
    )
    save_dataset(summ_goldens, "examples/output/generated_summarization.jsonl")
    print(f"  Saved {len(summ_goldens)} summarization goldens")

    print("\nDone! All datasets saved to examples/output/")


if __name__ == "__main__":
    asyncio.run(main())
