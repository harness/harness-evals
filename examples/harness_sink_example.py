"""Push evaluation results to Harness AI Evals using HarnessSink.

Before running:
  1. Create a dataset + dataset items in Harness AI Evals UI or API.
  2. Create an eval config (links dataset + target + metric-set).
  3. Create a run under that eval — note the run_id and dataset item IDs.
  4. Set environment variables:

     export HARNESS_ACCOUNT_ID=your_account_id
     export HARNESS_API_KEY=your_api_key
     export HARNESS_PROJECT_ID=your_project_id

Run: python examples/harness_sink_example.py
"""

from harness_evals import EvalCase, evaluate
from harness_evals.metrics import ContainsMetric, ExactMatchMetric
from harness_evals.sinks.harness_sink import HarnessSink

# IDs come from the Harness AI Evals platform.
# In practice, load these from your dataset export or CI environment variables.
RUN_ID = "run-abc123"
DATASET_ITEMS = [
    {"item_id": "item-001", "input": "What is the capital of France?", "expected": "Paris"},
    {"item_id": "item-002", "input": "What is 2 + 2?", "expected": "4"},
]

# Simulate agent responses
AGENT_RESPONSES = {
    "item-001": "Paris",
    "item-002": "4",
}

sink = HarnessSink()  # credentials from env vars

cases = [
    EvalCase(
        input=item["input"],
        output=AGENT_RESPONSES[item["item_id"]],
        expected=item["expected"],
        metadata={
            "harness_run_id": RUN_ID,
            "harness_dataset_item_id": item["item_id"],
        },
    )
    for item in DATASET_ITEMS
]

for case in cases:
    evaluate(case, metrics=[ExactMatchMetric(), ContainsMetric()], sinks=[sink])

# Marks the run as completed in Harness
sink.finalize()
sink.shutdown()

print(f"Results uploaded to run: {RUN_ID}")
