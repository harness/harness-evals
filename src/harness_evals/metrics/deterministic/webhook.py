from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score

if TYPE_CHECKING:
    import httpx


class WebhookMetric(BaseMetric):
    """Score LLM outputs by calling an external HTTP endpoint.

    User must explicitly configure how to interpret the response via one of:
    - response_fn: full control, takes httpx.Response, returns Score
    - response_key: name the boolean field in JSON response (optionally pair with score_key)
    - use_status_code: 2xx = pass (1.0), anything else = fail (0.0)

    Default payload (when no payload_fn): {"input": ..., "output": ..., "expected": ..., "context": ...}
    """

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        payload_fn: Callable[[EvalCase], dict] | None = None,
        response_fn: Callable[[httpx.Response], Score] | None = None,
        response_key: str | None = None,
        score_key: str | None = None,
        use_status_code: bool = False,
        method: str = "POST",
        timeout: float = 30.0,
        threshold: float = 1.0,
        name: str = "webhook",
        dimension: Dimension = Dimension.CORRECTNESS,
        **kwargs: object,
    ) -> None:
        super().__init__(name=name, dimension=dimension, threshold=threshold, **kwargs)

        modes_set = sum([response_fn is not None, response_key is not None, use_status_code])
        if modes_set == 0:
            raise ValueError(
                "Must specify exactly one response handling mode: response_fn, response_key, or use_status_code=True"
            )
        if modes_set > 1:
            raise ValueError("Cannot specify multiple response handling modes simultaneously")

        if score_key is not None and response_key is None:
            raise ValueError("score_key requires response_key to be set")

        self.url = url
        self.headers = headers
        self.payload_fn = payload_fn
        self.response_fn = response_fn
        self.response_key = response_key
        self.score_key = score_key
        self.use_status_code = use_status_code
        self.method = method.upper()
        self.timeout = timeout

    def _build_default_payload(self, eval_case: EvalCase) -> dict:
        payload = {"input": eval_case.input, "output": eval_case.output}
        if eval_case.expected is not None:
            payload["expected"] = eval_case.expected
        if eval_case.context is not None:
            payload["context"] = eval_case.context
        return payload

    def _parse_response(self, response: httpx.Response) -> Score:
        if self.response_fn is not None:
            return self.response_fn(response)

        if self.use_status_code:
            value = 1.0 if 200 <= response.status_code < 300 else 0.0
            reason = None if value == 1.0 else f"HTTP {response.status_code}"
            return Score(name=self.name, value=value, threshold=self.threshold, reason=reason)

        try:
            json_data = response.json()
        except Exception as e:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"Invalid response format: {e}",
            )

        if self.response_key is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="No response_key configured",
            )

        if self.response_key not in json_data:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"Response missing key: {self.response_key}",
            )

        passed = json_data[self.response_key]
        if not isinstance(passed, bool):
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"Response key {self.response_key} is not boolean",
            )

        if self.score_key is not None and self.score_key in json_data:
            try:
                value = float(json_data[self.score_key])
                value = max(0.0, min(1.0, value))
            except (TypeError, ValueError):
                return Score(
                    name=self.name,
                    value=0.0,
                    threshold=self.threshold,
                    reason=f"Invalid score value for {self.score_key}",
                )
        else:
            value = 1.0 if passed else 0.0

        return Score(name=self.name, value=value, threshold=self.threshold)

    def measure(self, eval_case: EvalCase) -> Score:
        return _run_async(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        try:
            import httpx
        except ImportError as e:
            raise ImportError("WebhookMetric requires httpx. Install with: pip install 'harness-evals[harness]'") from e

        payload = self.payload_fn(eval_case) if self.payload_fn else self._build_default_payload(eval_case)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(
                    method=self.method,
                    url=self.url,
                    json=payload,
                    headers=self.headers,
                )

                if not self.use_status_code and not (200 <= response.status_code < 300):
                    return Score(
                        name=self.name,
                        value=0.0,
                        threshold=self.threshold,
                        reason=f"HTTP error {response.status_code}",
                    )

                return self._parse_response(response)

        except httpx.TimeoutException:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="Request timed out",
            )
        except Exception as e:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"Request failed: {type(e).__name__}: {e}",
            )
