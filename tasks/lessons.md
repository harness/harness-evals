# Lessons

- Validate reviewed configuration at its input/reconstruction boundary instead
  of repeating the same checks in contract assembly. Before removing a check,
  verify every entry path and keep malformed-mutation and no-execution regressions.

- Release readiness must match the workflow's Python matrix, all-extras install,
  pinned tools, test exclusions, and isolated build, not just a convenient newer
  interpreter. Keep user-facing guidance in README and agent guidance in AGENTS;
  do not add repository design files when the user excludes `docs/` additions.
  Run workflow helpers with the selected environment's interpreter as well.

- Admission tests must compare the exact selected configuration and actual scorer
  reads, not only declared roots. Preserve nested runtime types, fallback
  precedence, finite structural values, and at least one usable composite child.
  A passing provider-free validation suite still needs adversarial parity probes.
- Composite readiness must retain usable weights after skips, not just child
  counts. Check current aggregation/weight shapes and local schema reference
  targets without resolving anything externally.
- Exercise real adapter-to-admission-to-scorer paths: deterministic identifier
  ordering is not causal evidence, derived results need execution-scoped
  identity, and unknown arguments are not observed empty objects. Mutable
  options must update both scoring behavior and evidence projection. Test
  repeated-name subset pairing before declaring an algorithm order-independent.
- Restrict malformed-value rejection to the scorer reads being protected;
  preserve supported missing/partial policies for unrelated metric families.
- Do not recognize factory-owned classes through interpreter bytecode layout.
  Use explicit weak identity registration and verify supported Python versions.
- Include newly staged documents in whitespace checks; an unstaged diff omits
  untracked files. Use fail-fast commit gates so a failed check stops the push.

- When a review dataset must represent broad product behavior, do not treat one
  random sample as sufficient coverage. Produce independent random and
  stratified samples, report their distributions, and keep them non-overlapping.
- Conversation goldens must set `elicitation_hints.llm_on_miss: true`
  consistently. Deterministic matchers stabilize known questions, but unmatched
  wording must fall back to the simulator LLM instead of failing the eval flow.
- Filter golden inventory candidates only by `golden_readiness=ready`, never by
  agent `quality`. Quality and portability are orthogonal: `bad` or `unclear`
  conversations can be ready negative/regression goldens.
