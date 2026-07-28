# Lessons

- When a review dataset must represent broad product behavior, do not treat one
  random sample as sufficient coverage. Produce independent random and
  stratified samples, report their distributions, and keep them non-overlapping.
- Conversation goldens must set `elicitation_hints.llm_on_miss: true`
  consistently. Deterministic matchers stabilize known questions, but unmatched
  wording must fall back to the simulator LLM instead of failing the eval flow.
- Filter golden inventory candidates only by `golden_readiness=ready`, never by
  agent `quality`. Quality and portability are orthogonal: `bad` or `unclear`
  conversations can be ready negative/regression goldens.
