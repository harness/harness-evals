"""Turn-level (conversational) RAG metrics.

A single-turn RAG metric scores one ``(query, retrieved_chunks, answer)``
triple. In a multi-turn conversation each assistant turn has its own such
triple: the chunks retrieved to answer that turn live on the assistant
``Message.retrieval_context`` and the per-turn expected answer on
``Message.expected``.

``_ConversationalRAGMetric`` walks ``eval_case.messages``, reconstructs the
per-turn triple for every assistant turn, and delegates scoring to the
corresponding single-turn RAG metric (``FaithfulnessMetric``,
``ContextPrecisionMetric``, etc.) by building a throwaway single-turn
``EvalCase`` per turn. The aggregate score is the mean of the per-turn scores;
the breakdown is stored in ``Score.metadata["turn_scores"]``.

The four public metrics — :class:`TurnFaithfulnessMetric`,
:class:`TurnContextualPrecisionMetric`, :class:`TurnContextualRecallMetric`,
and :class:`TurnContextualRelevancyMetric` — differ only in the delegate they
reuse and the per-turn inputs they require.
"""

from __future__ import annotations

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM
from harness_evals.metrics.rag.context_precision import ContextPrecisionMetric
from harness_evals.metrics.rag.context_recall import ContextRecallMetric
from harness_evals.metrics.rag.context_relevancy import ContextRelevancyMetric
from harness_evals.metrics.rag.faithfulness import FaithfulnessMetric


class _ConversationalRAGMetric(BaseMetric):
    """Base for per-turn RAG metrics that read ``eval_case.messages``.

    Subclasses set:

    - ``_delegate_cls``: the single-turn RAG metric class to reuse per turn.
    - ``_requires_query``: turns without a preceding user query cannot be scored.
    - ``_requires_expected``: turns without a per-turn expected answer cannot be scored.

    An assistant turn is scorable only if it carries a ``retrieval_context``
    (plus a query/expected when the metric requires them). By default a turn
    that is *missing* those inputs is treated as a **localized failure** and
    scored ``0.0`` — a retriever/trace failure on one turn must drag the
    conversation score down, not vanish from the average. Set
    ``allow_skips=True`` to instead exclude such turns from the aggregate
    (they are still recorded in ``turn_scores`` with ``"skipped": True``).

    If there are no assistant turns at all, or ``allow_skips`` leaves nothing
    scorable, the metric returns 0.0 with an explanatory reason.
    """

    _delegate_cls: type[BaseMetric]
    _requires_query: bool = False
    _requires_expected: bool = False

    def __init__(
        self,
        llm: BaseLLM,
        threshold: float = 0.7,
        *,
        name: str,
        dimension: Dimension = Dimension.GROUNDEDNESS,
        allow_skips: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(name=name, dimension=dimension, threshold=threshold, **kwargs)
        self.llm = llm
        self.allow_skips = allow_skips
        self._delegate = self._delegate_cls(llm, threshold=threshold)  # type: ignore[call-arg]

    def _missing_input_reason(self, msg: object, last_user: str | None) -> str | None:
        """Why this assistant turn cannot be scored, or ``None`` if it can.

        Checks run in the same order the single-turn triple is built:
        retrieved chunks, then the query, then the expected answer.
        """
        if not msg.retrieval_context:  # type: ignore[attr-defined]
            return "No retrieval_context on this turn"
        if self._requires_query and not last_user:
            return "No preceding user query for this turn"
        if self._requires_expected and msg.expected is None:  # type: ignore[attr-defined]
            return "No per-turn expected answer for this turn"
        return None

    def measure(self, eval_case: EvalCase) -> Score:
        return _run_async(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        messages = eval_case.messages
        if not messages:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="No messages — turn-level RAG requires eval_case.messages",
            )

        turn_scores: list[dict] = []
        last_user: str | None = None
        assistant_idx = -1

        for i, msg in enumerate(messages):
            if msg.role == "user":
                last_user = msg.content or ""
                continue
            if msg.role != "assistant":
                continue

            assistant_idx += 1

            missing = self._missing_input_reason(msg, last_user)
            if missing is not None:
                if self.allow_skips:
                    turn_scores.append(
                        {
                            "turn": assistant_idx,
                            "message_index": i,
                            "score": None,
                            "skipped": True,
                            "reasoning": missing,
                        }
                    )
                    continue
                # A turn whose retrieval we cannot evaluate is a localized
                # failure, not a free pass: score it 0.0 so it counts against
                # the conversation aggregate.
                turn_scores.append(
                    {
                        "turn": assistant_idx,
                        "message_index": i,
                        "score": 0.0,
                        "skipped": False,
                        "reasoning": missing,
                    }
                )
                continue

            turn_case = EvalCase(
                input=last_user or "",
                output=msg.content or "",
                expected=msg.expected,
                context=msg.retrieval_context,
            )
            sub = await self._delegate.a_measure(turn_case)

            turn_scores.append(
                {
                    "turn": assistant_idx,
                    "message_index": i,
                    "score": sub.value,
                    "skipped": False,
                    "reasoning": sub.reason,
                }
            )

        scored = [t for t in turn_scores if not t["skipped"]]
        n_skipped = len(turn_scores) - len(scored)

        if not scored:
            reason = "No turns with retrieval context to evaluate"
            if n_skipped:
                reason += f" ({n_skipped} turn(s) skipped)"
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=reason,
                metadata={
                    "turn_scores": turn_scores,
                    "n_turns": len(messages),
                    "n_scored_turns": 0,
                    "n_skipped_turns": n_skipped,
                },
            )

        aggregate = sum(t["score"] for t in scored) / len(scored)
        reason = f"Mean of {len(scored)} turn score(s)"
        if n_skipped:
            reason += f" ({n_skipped} skipped)"
        return Score(
            name=self.name,
            value=aggregate,
            threshold=self.threshold,
            reason=reason,
            metadata={
                "turn_scores": turn_scores,
                "n_turns": len(messages),
                "n_scored_turns": len(scored),
                "n_skipped_turns": n_skipped,
            },
        )


class TurnFaithfulnessMetric(_ConversationalRAGMetric):
    """Per-turn faithfulness across a multi-turn RAG conversation.

    For each assistant turn, checks that the answer's claims are grounded in
    that turn's ``retrieval_context``. Aggregate score is the mean of per-turn
    faithfulness; per-turn breakdown in ``score.metadata["turn_scores"]``.
    Turns whose retrieval context is missing score 0.0 (a localized failure)
    unless ``allow_skips=True``.
    """

    _delegate_cls = FaithfulnessMetric

    def __init__(self, llm: BaseLLM, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(
            llm,
            threshold=threshold,
            name="turn_faithfulness",
            dimension=Dimension.GROUNDEDNESS,
            **kwargs,
        )


class TurnContextualPrecisionMetric(_ConversationalRAGMetric):
    """Per-turn context precision across a multi-turn RAG conversation.

    For each assistant turn, checks what fraction of that turn's
    ``retrieval_context`` chunks are relevant to the turn's user query.
    Aggregate score is the mean of per-turn precision; per-turn breakdown in
    ``score.metadata["turn_scores"]``. Turns missing retrieval context or a
    preceding user query score 0.0 (a localized failure) unless ``allow_skips=True``.
    """

    _delegate_cls = ContextPrecisionMetric
    _requires_query = True

    def __init__(self, llm: BaseLLM, threshold: float = 0.5, **kwargs: object) -> None:
        super().__init__(
            llm,
            threshold=threshold,
            name="turn_contextual_precision",
            dimension=Dimension.GROUNDEDNESS,
            **kwargs,
        )


class TurnContextualRecallMetric(_ConversationalRAGMetric):
    """Per-turn context recall across a multi-turn RAG conversation.

    For each assistant turn, checks what fraction of the turn's per-turn
    ``expected`` answer can be attributed to that turn's ``retrieval_context``.
    Aggregate score is the mean of per-turn recall; per-turn breakdown in
    ``score.metadata["turn_scores"]``. Turns missing retrieval context or a
    per-turn expected answer score 0.0 (a localized failure) unless ``allow_skips=True``.
    """

    _delegate_cls = ContextRecallMetric
    _requires_expected = True

    def __init__(self, llm: BaseLLM, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(
            llm,
            threshold=threshold,
            name="turn_contextual_recall",
            dimension=Dimension.GROUNDEDNESS,
            **kwargs,
        )


class TurnContextualRelevancyMetric(_ConversationalRAGMetric):
    """Per-turn context relevancy across a multi-turn RAG conversation.

    For each assistant turn, checks what fraction of that turn's
    ``retrieval_context`` chunks are topically relevant to the turn's user
    query. Aggregate score is the mean of per-turn relevancy; per-turn
    breakdown in ``score.metadata["turn_scores"]``. Turns missing retrieval
    context or a preceding user query score 0.0 (a localized failure) unless
    ``allow_skips=True``.
    """

    _delegate_cls = ContextRelevancyMetric
    _requires_query = True

    def __init__(self, llm: BaseLLM, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(
            llm,
            threshold=threshold,
            name="turn_contextual_relevancy",
            dimension=Dimension.GROUNDEDNESS,
            **kwargs,
        )
