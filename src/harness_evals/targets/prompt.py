"""PromptTarget — renders a template through an LLM and grades the prompt+model pair."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from time import perf_counter

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.golden import Golden
from harness_evals.llm.base import BaseLLM
from harness_evals.plugins import register_target
from harness_evals.prompts.template import PromptTemplate
from harness_evals.targets.base import BaseTarget


@register_target("prompt")
@dataclass
class PromptTarget(BaseTarget):
    """Render a PromptTemplate and call a BaseLLM directly.

    Grades a *prompt + model pair* in isolation. The golden's input is
    serialised to a string and injected as the ``input`` template variable.
    Additional template variables are drawn from ``golden.metadata``.
    """

    prompt: PromptTemplate
    model: BaseLLM

    async def ainvoke(self, golden: Golden) -> EvalCase:
        input_str = golden.input if isinstance(golden.input, str) else json.dumps(golden.input, ensure_ascii=False)
        extra_vars = golden.metadata or {}
        rendered = self.prompt.render(input=input_str, **extra_vars)

        t0 = perf_counter()
        output = await self.model.generate(rendered)
        latency_ms = (perf_counter() - t0) * 1000

        return EvalCase.from_golden(golden, output=output, latency_ms=latency_ms)

    async def close(self) -> None:
        close = getattr(self.model, "close", None)
        if close is None:
            return None
        result = close()
        if inspect.isawaitable(result):
            await result
        return None
