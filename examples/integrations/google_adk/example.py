"""Evaluate a Google ADK agent with harness-evals using HttpTarget.

Google ADK (pip install google-adk) agents can be deployed via
`adk api_server` or Cloud Run. This example targets the /run endpoint.

Run:  python examples/integrations/google_adk/example.py
Requires: A running ADK endpoint (or set USE_MOCK=1 for local testing)

Note: ADK's /run endpoint requires a session to be created first via
  POST /apps/{app_name}/users/{user_id}/sessions/{session_id}
The response is a JSON array of events; the final answer is the last
event with content.role == "model" containing a text part.
"""

import asyncio
import os
from time import perf_counter

from harness_evals import EvalCase, Golden, evaluate_dataset
from harness_evals.metrics import ContainsMetric, LatencyMetric
from harness_evals.sinks import StdoutSink

MOCK_RESPONSES = {
    "What is the capital of France?": "Paris",
    "What is the weather in New York?": "The weather in New York is currently sunny with a high of 75°F.",
    "How many planets are in the solar system?": "There are 8 planets in the solar system.",
}


async def mock_agent_invoke(golden: Golden) -> EvalCase:
    """Stub that simulates an ADK /run response."""
    input_str = golden.input if isinstance(golden.input, str) else str(golden.input)
    t0 = perf_counter()
    output = MOCK_RESPONSES.get(input_str, "I don't know")
    latency_ms = (perf_counter() - t0) * 1000
    return EvalCase.from_golden(golden, output=output, latency_ms=latency_ms)


async def real_agent_invoke(golden: Golden) -> EvalCase:
    """Calls a deployed ADK agent's /run endpoint and extracts the final text.

    ADK /run request (camelCase):
        {"appName": "...", "userId": "...", "sessionId": "...",
         "newMessage": {"role": "user", "parts": [{"text": "..."}]}}

    ADK /run response: JSON array of event objects. Final answer is the last
    event where content.role == "model" with a "text" part.
    """
    import uuid

    try:
        import httpx
    except ImportError as e:
        raise ImportError("httpx required: pip install httpx") from e

    base_url = os.environ.get("ADK_ENDPOINT", "http://localhost:8000")
    app_name = os.environ.get("ADK_APP_NAME", "my_agent")
    user_id = "eval_user"
    session_id = str(uuid.uuid4())

    token = os.environ.get("ADK_AUTH_TOKEN")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    input_str = golden.input if isinstance(golden.input, str) else str(golden.input)

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Step 1: Create session
        await client.post(
            f"{base_url}/apps/{app_name}/users/{user_id}/sessions/{session_id}",
            headers=headers,
            json={},
        )

        # Step 2: Run
        t0 = perf_counter()
        resp = await client.post(
            f"{base_url}/run",
            headers=headers,
            json={
                "appName": app_name,
                "userId": user_id,
                "sessionId": session_id,
                "newMessage": {"role": "user", "parts": [{"text": input_str}]},
            },
        )
        latency_ms = (perf_counter() - t0) * 1000
        resp.raise_for_status()

    events = resp.json()
    final_text = next(
        (
            p["text"]
            for ev in reversed(events)
            if ev.get("content", {}).get("role") == "model"
            for p in ev["content"]["parts"]
            if "text" in p
        ),
        "",
    )

    return EvalCase.from_golden(golden, output=final_text, latency_ms=latency_ms)


goldens = [
    Golden(input="What is the capital of France?", expected="Paris"),
    Golden(
        input="What is the weather in New York?",
        expected="The weather in New York is currently sunny with a high of 75°F.",
    ),
    Golden(input="How many planets are in the solar system?", expected="There are 8 planets in the solar system."),
]


async def main() -> None:
    use_mock = os.environ.get("USE_MOCK", "1") == "1"
    agent_fn = mock_agent_invoke if use_mock else real_agent_invoke

    results = await evaluate_dataset(
        goldens,
        agent_fn,
        metrics=[
            ContainsMetric(),
            LatencyMetric(max_ms=30000, threshold=0.5),
        ],
        sinks=[StdoutSink()],
    )
    passed = sum(all(s.passed for s in r) for r in results)
    print(f"\nPass rate: {passed}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
