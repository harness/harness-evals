"""Evaluate an AWS Bedrock AgentCore agent with harness-evals using HttpTarget.

Bedrock AgentCore is a model-agnostic agent runtime. You bring your own agent
code; AWS runs it as a managed container. The invoke API is transport-only —
the response schema depends on what your agent returns.

Run:  python examples/integrations/bedrock_agentcore/example.py
Requires: AWS credentials with bedrock-agentcore:InvokeAgentRuntime (or USE_MOCK=1)

Auth options:
  - AWS SigV4 (default, via boto3)
  - Bearer JWT (if runtime configured with OAuth/Cognito inbound auth)
"""

import asyncio
import os
from time import perf_counter

from harness_evals import EvalCase, Golden, evaluate_dataset
from harness_evals.metrics import ContainsMetric, LatencyMetric
from harness_evals.sinks import StdoutSink
from harness_evals.targets import BearerAuth, HttpTarget

MOCK_RESPONSES = {
    "What is the capital of France?": "Paris",
    "What is 2 + 2?": "4",
    "Explain containerization in one sentence": "Containerization packages an application with its dependencies into an isolated unit that runs consistently across environments.",
}


async def mock_agent_invoke(golden: Golden) -> EvalCase:
    """Stub that simulates an AgentCore invocation."""
    input_str = golden.input if isinstance(golden.input, str) else str(golden.input)
    t0 = perf_counter()
    output = MOCK_RESPONSES.get(input_str, "I don't know")
    latency_ms = (perf_counter() - t0) * 1000
    return EvalCase.from_golden(golden, output=output, latency_ms=latency_ms)


async def real_agent_invoke_boto3(golden: Golden) -> EvalCase:
    """Invoke via boto3 (SigV4 auth handled automatically).

    Requires: pip install boto3
    The response body is agent-specific — adjust json.loads parsing to match
    your agent's output format.
    """
    import json
    import uuid

    import boto3

    client = boto3.client("bedrock-agentcore", region_name=os.environ.get("AWS_REGION", "us-east-1"))

    input_str = golden.input if isinstance(golden.input, str) else str(golden.input)
    t0 = perf_counter()
    resp = client.invoke_agent_runtime(
        agentRuntimeArn=os.environ["AGENTCORE_RUNTIME_ARN"],
        runtimeSessionId=str(uuid.uuid4()),
        payload=json.dumps({"prompt": input_str}).encode(),
    )
    body = b"".join(chunk for chunk in resp["response"]).decode()
    latency_ms = (perf_counter() - t0) * 1000

    parsed = json.loads(body)
    # Adjust output_key to match your agent's response schema
    output_key = os.environ.get("AGENTCORE_OUTPUT_KEY", "result")
    output = parsed.get(output_key, body)

    return EvalCase.from_golden(golden, output=output, latency_ms=latency_ms)


def build_http_target() -> HttpTarget:
    """Build HttpTarget for OAuth/JWT-authenticated AgentCore endpoint.

    Use this when your AgentCore runtime has inbound OAuth configured
    (Cognito, Okta, etc.) — you skip SigV4 and use a Bearer token directly.
    """
    return HttpTarget(
        url=os.environ.get(
            "AGENTCORE_ENDPOINT",
            "https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/my-agent/invocations",
        ),
        method="POST",
        auth=BearerAuth(os.environ["AGENTCORE_JWT"]),
        body_template={"prompt": "{{input}}"},
        output_path=f"$.{os.environ.get('AGENTCORE_OUTPUT_KEY', 'result')}",
        timeout_s=120.0,
    )


goldens = [
    Golden(input="What is the capital of France?", expected="Paris"),
    Golden(input="What is 2 + 2?", expected="4"),
    Golden(
        input="Explain containerization in one sentence",
        expected="Containerization packages an application with its dependencies into an isolated unit that runs consistently across environments.",
    ),
]


async def main() -> None:
    use_mock = os.environ.get("USE_MOCK", "1") == "1"

    if use_mock:
        agent_fn = mock_agent_invoke
    elif os.environ.get("AGENTCORE_JWT"):
        # OAuth path — use HttpTarget
        target = build_http_target()
        agent_fn = target.ainvoke
    else:
        # SigV4 path — use boto3 directly
        agent_fn = real_agent_invoke_boto3

    results = await evaluate_dataset(
        goldens,
        agent_fn,
        metrics=[
            ContainsMetric(),
            LatencyMetric(max_ms=60000, threshold=0.5),
        ],
        sinks=[StdoutSink()],
    )
    passed = sum(all(s.passed for s in r) for r in results)
    print(f"\nPass rate: {passed}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
