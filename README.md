# harness-evals

Open-source AI evaluation framework for LLM agents, prompts, and structured outputs.

Every metric produces a normalized `Score` (0.0–1.0). Pass/fail is determined by a configurable threshold. No magic, no hidden state.

## Install

```bash
pip install harness-evals            # core only
pip install harness-evals[llm]       # + LLM-judged metrics (OpenAI, Anthropic)
pip install harness-evals[otlp]      # + OTLP metrics & traces export
pip install harness-evals[langfuse]  # + Langfuse source/sink
pip install harness-evals[similarity]# + BLEU metric (nltk)
pip install harness-evals[harness]   # + Harness AI Service LLM provider
pip install harness-evals[all]       # everything
```

## Five Dimensions

Every metric belongs to one of five evaluation dimensions. Together they answer: *where is my agent strong, and where is it weak?*

| Dimension | Question | Example Metrics |
|-----------|----------|----------------|
| **Correctness** | Is it right? | ExactMatch, TaskCompletion, GEval, GoalAccuracy |
| **Groundedness** | Is it supported by evidence? | Faithfulness, ContextPrecision, AnswerRelevancy |
| **Safety** | Did it violate policy? | PII, Toxicity, PromptInjection, Hallucination |
| **Trajectory** | Did it take a good path? | PlanAdherence, StepEfficiency, ToolCorrectness |
| **Performance** | Was it fast and cheap? | Latency, TokenCost, CostEfficiency |

Dimensions are set by the metric author — not user-configured. Any combination of metrics in a suite automatically produces a radar chart grouped by dimension.

## Core Concepts

**`Golden`** — what you author: an input, the expected output, and optional context. Lives in your dataset files.

**`EvalCase`** — what metrics receive: a Golden enriched with the agent's actual output and runtime metadata (latency, tokens, cost).

**`BaseMetric`** — a scoring function. Takes an `EvalCase`, returns a `Score`. Each metric is a single class with a `measure()` method. Specialized base classes: `ReliabilityMetric` for multi-run metrics, `SafetyMetric` for safety metrics (reported separately, never averaged).

**`Message`** — a conversation turn: role, content, and optional tool calls. Maps to OpenAI chat messages, Langfuse generations, OTEL LLM spans.

**`ToolCall`** — a tool/function invocation: name, input, output. Maps to OpenAI function calls, Anthropic tool_use blocks, MCP invocations.

**`Score`** — the result: a value between 0.0 and 1.0, a threshold, and an auto-computed `passed` boolean.

**`evaluate()`** — runs multiple metrics on an eval case. Never raises — returns all scores including failures.

**`assert_test()`** — same as `evaluate()`, but raises `AssertionError` if any metric fails. Drop it into pytest.

**Datasets** — `load_dataset()` / `save_dataset()` read and write JSONL or JSON golden files. Source adapters (`LocalDatasetSource`, `HttpDatasetSource`) fetch goldens from local files or HTTP endpoints using the `ResourceRef` URI system.

**Prompt templates** — `PromptTemplate` uses `{{var}}` placeholders (not `str.format`). Prompt source adapters fetch templates from local files or HTTP endpoints.

**Targets** — the system under test. `PromptTarget` renders a prompt template through an LLM. `HttpTarget` POSTs to a deployed agent endpoint. Both implement `BaseTarget.ainvoke(golden) -> EvalCase`.

**Importers** — adapters that hydrate `EvalCase` from production trace data. `LangfuseEvalCaseSource` and `OTELEvalCaseSource` map traces to typed fields automatically.

**Baselines** — `JsonBaselineStore` saves score snapshots per run. `compare_to_baseline()` detects regressions, improvements, and unchanged metrics with a configurable tolerance.

### Data Flow

```
Golden (authored) + agent output → EvalCase → Score (result)
Production traces (Langfuse/OTEL) → Importer → EvalCase → Score (result)
YAML config → run_config() → evaluate_dataset() → Score (result)
run_eval() one-liner → evaluate_dataset() → Score (result)
```

## Quick Start — Local Eval Runner

The fastest way to run an eval is the CLI with a YAML config:

```yaml
# my-eval.eval.yaml
name: support-bot
dataset: ./goldens.jsonl
target:
  type: prompt
  prompt: ./prompts/support-bot.txt
  model: {provider: openai, name: gpt-4o}
metrics:
  - exact_match
  - {kind: geval, threshold: 0.7, params: {criteria: "Correct and helpful?"}}
judge_llm: {provider: openai, name: gpt-4o}
sinks: [stdout]
baseline: {store: json, path: .evals/baseline.json}
```

```bash
harness-evals run my-eval.eval.yaml                # run the eval
harness-evals run my-eval.eval.yaml --baseline     # compare against stored baseline
harness-evals run my-eval.eval.yaml --fail-under 0.8  # CI gate on absolute score
harness-evals list-metrics                         # see all available metrics
harness-evals discover                             # find all *.eval.yaml in cwd
```

Or use the code-first `run_eval()` one-liner:

```python
from harness_evals import Golden, run_eval
from harness_evals.metrics import ExactMatchMetric

run_eval(
    "my-eval",
    data="./goldens.jsonl",             # ref string, ResourceRef, or list[Golden]
    target=my_agent_function,           # BaseTarget, async callable, or sync callable
    metrics=[ExactMatchMetric()],
)
```

Both the YAML runner and `run_eval()` funnel into `evaluate_dataset()` — the same engine used by the programmatic API below.

Model params support `${VAR}` env-var interpolation, so the target and judge can use separate API keys:

```yaml
target:
  type: prompt
  prompt: ./prompt.txt
  model: {provider: openai, name: gpt-4o, api_key: "${TARGET_KEY}"}
judge_llm: {provider: openai, name: gpt-4o, api_key: "${JUDGE_KEY}"}
```

If no `api_key` is specified, the provider falls back to its default env var (e.g. `OPENAI_API_KEY`).

## Usage

### Evaluate a response

```python
from harness_evals import EvalCase, evaluate
from harness_evals.metrics import ExactMatchMetric, LatencyMetric

ec = EvalCase(
    input="What is the capital of France?",
    output="Paris",
    expected="Paris",
    latency_ms=320,
)

scores = evaluate(ec, metrics=[
    ExactMatchMetric(),
    LatencyMetric(max_ms=2000, threshold=0.5),
])

for s in scores:
    print(f"{'PASS' if s.passed else 'FAIL'} {s.name}: {s.value:.2f}")
# PASS exact_match: 1.00
# PASS latency: 0.84
```

### Evaluate structured output (JSON/YAML)

```python
from harness_evals import EvalCase, assert_test
from harness_evals.metrics import JsonDiffMetric, SchemaValidationMetric

expected = {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": "nginx"}}
schema = {"type": "object", "required": ["apiVersion", "kind"], "properties": {
    "apiVersion": {"type": "string"}, "kind": {"type": "string"}
}}

ec = EvalCase(
    input="Create a K8s deployment for nginx",
    output={"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": "nginx"}},
    expected=expected,
)

assert_test(ec, metrics=[
    JsonDiffMetric(threshold=0.9),
    SchemaValidationMetric(schema=schema),
])
```

### Use with pytest

```python
def test_agent_accuracy():
    ec = EvalCase(
        input="What is 2+2?",
        output=agent.run("What is 2+2?"),
        expected="4",
    )
    assert_test(ec, metrics=[ExactMatchMetric()])
```

`assert_test()` raises `AssertionError` on failure — works natively with pytest, unittest, and any CI system.

### Evaluate a dataset with an agent

```python
import asyncio
from harness_evals import Golden, EvalCase, evaluate_dataset
from harness_evals.metrics import ExactMatchMetric

goldens = [
    Golden(input="What is 2+2?", expected="4"),
    Golden(input="Capital of France?", expected="Paris"),
]

async def run_agent(golden: Golden) -> EvalCase:
    result = await agent.arun(golden.input)
    return EvalCase.from_golden(golden, output=result)

results = asyncio.run(evaluate_dataset(goldens, run_agent, metrics=[ExactMatchMetric()]))
```

### Load and save datasets

```python
from harness_evals import load_dataset, save_dataset, Golden

# Load goldens from a JSONL file
goldens = load_dataset("goldens.jsonl")           # JSONL (one JSON object per line)
goldens = load_dataset("goldens.json", format="json")  # JSON array

# Create and save a dataset
goldens = [
    Golden(input="What is 2+2?", expected="4"),
    Golden(input="Capital of France?", expected="Paris"),
]
save_dataset(goldens, "goldens.jsonl")
```

### Evaluate with an HttpTarget

```python
import asyncio
from harness_evals import Golden, evaluate_dataset
from harness_evals.targets import HttpTarget, BearerAuth
from harness_evals.metrics import ExactMatchMetric

target = HttpTarget(
    url="http://localhost:8080/run",
    auth=BearerAuth(token="${API_TOKEN}"),  # ${VAR} resolved from env
    output_path="$.answer",
    timeout_s=30,
)

goldens = [Golden(input="What is 2+2?", expected="4")]
scores = asyncio.run(evaluate_dataset(goldens, target.ainvoke, metrics=[ExactMatchMetric()]))
```

### Compare scores against a baseline

```python
from harness_evals.baseline import JsonBaselineStore, compare_to_baseline
from harness_evals.core.score import Score

store = JsonBaselineStore(baseline_dir=".evals/baselines")

# After an eval run, save scores as a baseline
store.save("run-001", {"exact_match": scores_list})

# Later, compare a new run against the saved baseline
baseline = store.load()  # loads latest
result = compare_to_baseline(current_scores, baseline, tolerance=0.05)
if result.has_regressions:
    print(result.summary())
```

### Measure reliability across multiple runs

```python
from harness_evals import EvalCase, evaluate
from harness_evals.metrics import OutcomeConsistencyMetric

runs = [
    EvalCase(input="task", output=agent.run("task"))
    for _ in range(5)
]

ec = EvalCase(input="task", output=runs[0].output, runs=runs)
scores = evaluate(ec, metrics=[OutcomeConsistencyMetric(threshold=0.8)])
```

### Evaluate with typed tool calls

```python
from harness_evals import EvalCase, ToolCall, evaluate
from harness_evals.metrics import ToolCorrectnessMetric

ec = EvalCase(
    input="Check weather in Paris",
    output="It's 18C and sunny",
    tool_calls=[ToolCall(name="get_weather", input={"city": "Paris"})],
    expected_tools=["get_weather"],
)
scores = evaluate(ec, metrics=[ToolCorrectnessMetric()])
```

For deterministic argument checks, pair `ToolCorrectnessMetric` with `ToolArgumentMatchMetric`:

```python
from harness_evals import EvalCase, ToolCall, evaluate
from harness_evals.metrics import ToolArgumentMatchMetric, ToolCorrectnessMetric

ec = EvalCase(
    input="Check weather in Paris",
    output="It's 18C and sunny",
    tool_calls=[ToolCall(name="get_weather", input={"city": "Paris", "units": "C"})],
    expected_tools=["get_weather"],
    expected_tool_calls=[ToolCall(name="get_weather", input={"city": "Paris"})],
)
scores = evaluate(
    ec,
    metrics=[
        ToolCorrectnessMetric(mode="exact"),
        ToolArgumentMatchMetric(arg_match="subset", ignore_keys={"trace_id"}),
    ],
)
```

### Evaluate conversation messages

```python
from harness_evals import EvalCase, Message

ec = EvalCase(
    input="Help me debug this error",
    output="The issue is a null pointer...",
    messages=[
        Message(role="user", content="Help me debug this error"),
        Message(role="assistant", content="Can you share the stack trace?"),
        Message(role="user", content="Here it is: NullPointerException at..."),
        Message(role="assistant", content="The issue is a null pointer..."),
    ],
)
```

### Multi-turn evaluation with ConversationGolden

`ConversationGolden` supports three modes for multi-turn evaluation:

| Mode | Description | Use case |
|------|-------------|----------|
| `SIMULATE` | LLM generates user turns, agent responds | Testing agent behavior on dynamic scenarios |
| `REPLAY` | Full transcript scored as-is, no agent call | Evaluating historical conversations |
| `SCRIPTED` | Pre-scripted user turns, agent called each turn | Testing agent against a curated dataset |
| `GRAPH` | A declarative DAG drives user turns with conditional branching | Testing agent across branching decision paths (see [SimulationGraph](#branching-scenarios-with-simulationgraph)) |

#### Simulate — LLM-driven user turns

```python
import asyncio
from harness_evals import evaluate_dataset, Message
from harness_evals.conversation import ConversationGolden, ConversationMode
from harness_evals.metrics.conversation import GoalAccuracyMetric, TurnRelevancyMetric
from harness_evals.llm import OpenAILLM

llm = OpenAILLM(model="gpt-4o-mini")

goldens = [
    ConversationGolden(
        scenario="User requests a refund for a damaged item",
        expected_outcome="Agent processes refund and confirms timeline",
        max_turns=8,
        user_persona="Frustrated customer, short replies",
        mode=ConversationMode.SIMULATE,
    )
]

async def my_agent(messages: list[Message]) -> Message:
    # Your agent logic here — receives full history, returns next response
    ...

results = asyncio.run(evaluate_dataset(
    goldens, my_agent,
    metrics=[GoalAccuracyMetric(llm=llm), TurnRelevancyMetric(llm=llm)],
    simulator_llm=llm,
))
```

#### Scripted — bring your own user turns, call agent live

```python
goldens = [
    ConversationGolden(
        scenario="Customer asks about order status",
        expected_outcome="Agent provides tracking info",
        turns=[
            Message(role="user", content="Where is my order #12345?"),
            Message(role="user", content="When will it arrive?"),
            Message(role="user", content="Can I change the delivery address?"),
        ],
        mode=ConversationMode.SCRIPTED,
    )
]

results = asyncio.run(evaluate_dataset(
    goldens, my_agent,
    metrics=[GoalAccuracyMetric(llm=llm), TurnRelevancyMetric(llm=llm)],
    simulator_llm=llm,
))
```

#### Replay — score an existing transcript

```python
goldens = [
    ConversationGolden(
        scenario="Support interaction",
        expected_outcome="Issue resolved",
        turns=[
            Message(role="user", content="My account is locked"),
            Message(role="assistant", content="I can help. What's your email?"),
            Message(role="user", content="alice@example.com"),
            Message(role="assistant", content="Done — your account is unlocked."),
        ],
        mode=ConversationMode.REPLAY,
    )
]

results = asyncio.run(evaluate_dataset(
    goldens, my_agent,  # agent_fn not called in REPLAY mode; simulator_llm still required by API
    metrics=[GoalAccuracyMetric(llm=llm), TurnRelevancyMetric(llm=llm)],
    simulator_llm=llm,
))
```

#### Branching scenarios with SimulationGraph

`SimulationGraph` drives simulation through a declarative DAG (`ConversationMode.GRAPH`). Nodes produce user turns — `ScriptedNode` (fixed message), `LLMNode` (LLM generates a turn toward a goal), `BranchNode` (routes on the agent's last response), `StopNode` (ends the conversation) — and edges connect them, optionally guarded by named predicates. Cycles are structurally rejected; to loop, use a node with no matching outgoing edge (it re-executes until a predicate matches or `max_turns` is reached).

```python
from harness_evals.conversation import ConversationGolden, ConversationMode, ConversationSimulator
from harness_evals.conversation.graph import (
    SimulationGraph, ScriptedNode, LLMNode, BranchNode, StopNode, Edge,
)

graph = SimulationGraph(
    start="greeting",
    nodes={
        "greeting": ScriptedNode(message="I need help with my order"),
        "probe": LLMNode(goal="Ask increasingly specific questions about the delay"),
        "route": BranchNode(),
        "escalate": ScriptedNode(message="I want to speak to a manager"),
        "done": StopNode(),
    },
    edges=[
        Edge(source="greeting", target="probe"),
        Edge(source="probe", target="route"),
        Edge(source="route", target="escalate", predicate="unresolved"),
        Edge(source="route", target="done", predicate="resolved"),
    ],
    predicates={
        "unresolved": lambda msg: "sorry" in msg.content.lower(),
        "resolved": lambda msg: "refund" in msg.content.lower(),
    },
)

simulator = ConversationSimulator(simulator_llm=llm, graph=graph)
```

A graph can also be serialized to/from a plain dict (`SimulationGraph.from_dict(...)`). When loaded from `ConversationGolden.graph_config`, only unconditional edges are supported — predicate functions aren't serializable, so pass them explicitly to `from_dict()` when needed.

### Evaluate production traces from Langfuse

```python
from harness_evals.importers.langfuse import LangfuseEvalCaseSource
from harness_evals import evaluate
from harness_evals.metrics import FaithfulnessMetric, LatencyMetric, PIIMetric
from harness_evals.sinks.langfuse_sink import LangfuseSink

source = LangfuseEvalCaseSource(langfuse_client)
ec = source.from_trace("trace-id-123")

scores = evaluate(ec, metrics=[
    FaithfulnessMetric(llm=llm),
    LatencyMetric(max_ms=3000),
    PIIMetric(),
], sinks=[LangfuseSink()])  # scores written back to the same trace
```

### Batch-evaluate Langfuse traces by filter

```python
from harness_evals.importers.langfuse import LangfuseEvalCaseSource
from harness_evals import evaluate_cases
from harness_evals.metrics import LatencyMetric, PIIMetric
from harness_evals.sinks.langfuse_sink import LangfuseSink

source = LangfuseEvalCaseSource(langfuse_client)
cases = source.from_traces(tags=["production"], user_id="user_123", limit=50)

all_scores = evaluate_cases(cases, metrics=[
    LatencyMetric(max_ms=3000),
    PIIMetric(),
], sinks=[LangfuseSink()])
```

### Evaluate production traces from OpenTelemetry

```python
from harness_evals.importers.otel import OTELEvalCaseSource

ec = OTELEvalCaseSource.from_spans(collected_spans)
scores = evaluate(ec, metrics=[...])
```

> **Note:** The previous import paths (`from harness_evals.sources.langfuse import LangfuseSource`
> and `from harness_evals.sources.otel import OTELSource`) still work but emit a
> `DeprecationWarning`. Migrate to the `importers` paths shown above.

### Write results to a file

```python
from harness_evals.sinks import StdoutSink, JsonSink
from harness_evals.sinks.langfuse_sink import LangfuseSink

scores = evaluate(ec, metrics=[...], sinks=[
    StdoutSink(),
    JsonSink("results/scores.jsonl"),
    LangfuseSink(),  # requires pip install harness-evals[langfuse]
])
```

### Export to an OTLP-compatible backend

For `protocol="http"` (the default in some integrations is `grpc`), set `endpoint` to the OTLP HTTP **base** URL; `OtlpSink` appends `/v1/traces` and `/v1/metrics` for the exporters. Use `protocol="grpc"` with a `host:port` endpoint for gRPC OTLP.

```python
from harness_evals.sinks.otlp_sink import OtlpSink  # requires pip install harness-evals[otlp]

sink = OtlpSink(
    endpoint="http://collector:4317",
    run_id="my-eval-run-001",
    resource_attributes={"deployment.environment": "ci"},
    extra_attributes={"eval.suite_id": "nightly-regression"},
)

scores = evaluate(ec, metrics=[...], sinks=[sink])
```

#### Attach eval spans to an existing trace

If your eval engine already creates OTel spans, pass a `parent_context` so the eval-run span becomes a child (same trace ID, unified view in Jaeger/Tempo):

```python
from opentelemetry import trace
from harness_evals.sinks.otlp_sink import OtlpSink

tracer = trace.get_tracer("my-engine")
with tracer.start_as_current_span("orchestration") as parent:
    ctx = trace.set_span_in_context(parent)
    sink = OtlpSink(endpoint="http://collector:4317", parent_context=ctx)
    evaluate_cases(cases, metrics=[...], sinks=[sink])
```

#### Share a TracerProvider (single export pipeline)

For full control, pass your own `TracerProvider` and/or `MeterProvider`. The sink won't flush or shutdown providers it doesn't own — you retain lifecycle control:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from harness_evals.sinks.otlp_sink import OtlpSink

provider = TracerProvider(resource=my_resource)
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://collector:4317")))
tracer = provider.get_tracer("my-engine")

with tracer.start_as_current_span("orchestration") as parent:
    ctx = trace.set_span_in_context(parent)
    sink = OtlpSink(tracer_provider=provider, parent_context=ctx, run_id="run-123")
    evaluate_cases(cases, metrics=[...], sinks=[sink])

provider.shutdown()  # caller owns lifecycle
```

### Evaluate security remediations

```python
from harness_evals import EvalCase, evaluate
from harness_evals.llm.openai import OpenAILLM  # or AnthropicLLM, HarnessAILLM
from harness_evals.metrics.security import (
    VulnerabilityCorrectnessMetric,
    SecurityCompletenessMetric,
    CodeSafetyMetric,
    CodeQualityMetric,
    ExplanationQualityMetric,
    RootCauseAnalysisMetric,
    ActionabilityMetric,
    remediation_quality_index,
)

llm = OpenAILLM()  # uses OPENAI_API_KEY env var

ec = EvalCase(
    input="CWE-79: Reflected XSS in user_profile.py line 42. User input rendered without escaping.",
    output="## Fix\n```python\nfrom markupsafe import escape\nname = escape(request.args.get('name', ''))\n```",
)

scores = evaluate(ec, metrics=[
    VulnerabilityCorrectnessMetric(llm=llm, threshold=0.5),
    SecurityCompletenessMetric(llm=llm, threshold=0.5),
    CodeSafetyMetric(llm=llm, threshold=0.5),
    CodeQualityMetric(llm=llm, threshold=0.5),
    ExplanationQualityMetric(llm=llm, threshold=0.5),
    RootCauseAnalysisMetric(llm=llm, threshold=0.5),
    ActionabilityMetric(llm=llm, threshold=0.5),
])

rqi = remediation_quality_index(scores)
print(f"RQI: {rqi.value:.3f} ({'PASS' if rqi.passed else 'FAIL'})")
```

### Summarize results across a dataset

```python
from harness_evals import evaluate_cases, summarize

all_scores = evaluate_cases(eval_cases, metrics=[...])
summary = summarize(all_scores)

for name, m in summary.by_metric.items():
    print(f"{name}: mean={m.mean:.2f} pass_rate={m.pass_rate:.0%} ({m.count} cases)")
```

### Synthesize a dataset from documents

`Synthesizer` (and task-specific variants like `ConversationSynthesizer`, `ScriptedConversationSynthesizer`) generate `Golden` datasets from source documents using an LLM. `InputGenerator` produces input variations (rephrasings, adversarial rewrites, complexity ladders) for robustness testing.

```python
import asyncio
from harness_evals import Synthesizer, save_dataset
from harness_evals.llm import OpenAILLM

synth = Synthesizer(llm=OpenAILLM(model="gpt-4o-mini"))
goldens = asyncio.run(synth.generate(documents=[doc_text], n=20, difficulty="mixed"))
save_dataset(goldens, "goldens.jsonl")
```

See `examples/synthesize_dataset.py` for a runnable end-to-end example.

### Optimize a prompt automatically

`PromptOptimizer` runs an iterative diagnose → rewrite → re-evaluate loop against a golden set until it hits a target score or runs out of patience. The judge LLM must be a **different** instance than the model being optimized (self-evaluation is rejected).

```python
import asyncio
from harness_evals import PromptOptimizer
from harness_evals.llm import OpenAILLM
from harness_evals.prompts import PromptTemplate
from harness_evals.metrics import GEvalMetric

optimizer = PromptOptimizer(
    model=OpenAILLM(model="gpt-4o-mini"),
    judge=OpenAILLM(model="gpt-4o"),       # must differ from model
    metrics=[GEvalMetric(criteria="Correct and helpful?", llm=OpenAILLM(model="gpt-4o"))],
    target_score=0.85,
    max_iterations=10,
)

result = asyncio.run(optimizer.optimize(
    PromptTemplate("Answer the question: {{input}}"),  # placeholders must be in input_variables (defaults to ["input"])
    goldens,
))
print(f"{result.initial_score:.2f} → {result.best_score:.2f} in {result.iterations} iters")
print(result.best_prompt.template)
result.save("optimized-prompt.json")
```

## Available Metrics

| Category | Metrics | What They Measure |
|----------|---------|------------------|
| **Deterministic** | ExactMatch, Contains, Regex, NumericDiff, ListContains, Webhook | Exact comparison against expected output, or delegated to an external webhook |
| **Structural** | JsonDiff, SchemaValidation, StructuralSimilarity | Structural similarity and schema conformance for JSON/YAML |
| **Operational** | Latency, TokenCost, CostEfficiency, RetryCount, TurnLatency, TurnTokenCost | Performance and cost from typed fields, including per-turn conversation cost |
| **Reliability** | OutcomeConsistency, ResourceConsistency, TrajectoryConsistency, PromptRobustness, EnvironmentRobustness, FaultRobustness, BrierScore | Consistency across repeated runs, trajectory similarity, robustness to prompt/environment/fault perturbations |
| **Predictability** | Calibration, Discrimination | Expected calibration error and AUC-ROC over confidence scores |
| **MCP** | ToolSelectionAccuracy, MCPTraceCompleteness | MCP tool selection accuracy and trace completeness |
| **Similarity** | Levenshtein, BLEU, EmbeddingSimilarity | String distance, n-gram overlap, and semantic vector similarity |
| **LLM-Judged** | GEval, RubricJudge, Pairwise, DAG, PromptAlignment, Summarization | LLM scores output against criteria, rubric, or A/B comparison. `GEval` supports free-form criteria, numbered `evaluation_steps`, and integer score-band rubrics via `list[RubricLevel]`; `RubricJudge` uses a flat level → description rubric; `DAG` composes judgement nodes into a decision graph; `PromptAlignment` checks instruction-following; `Summarization` scores summary faithfulness and coverage. (requires `[llm]`) |
| **RAG** | Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, AnswerCorrectness, AnswerSimilarity, ContextEntityRecall, ContextRelevancy | Retrieval-augmented generation quality (requires `[llm]`) |
| **Safety** | PII, Toxicity, PromptInjection, Hallucination, Bias, Compliance, HarmSeverity, HarmfulAdvice, MisuseDetection, RoleViolation | PII leaks, toxic content, prompt injection, hallucination, bias, policy compliance, harm severity, harmful advice, misuse, and role/system-prompt violations (reported separately, never averaged) |
| **Agent** | ToolCorrectness, ToolArgumentMatch, TaskCompletion, ArgumentCorrectness, PlanQuality, PlanAdherence, StepEfficiency | Tool call correctness, deterministic argument match, task completion, LLM-judged argument validation, plan quality/adherence, step efficiency (some require `[llm]`) |
| **Conversation** | ConversationCoherence, ConversationResolution, ConversationCompleteness, TurnEfficiency, TurnRelevancy, KnowledgeRetention, RoleAdherence, TopicAdherence, GoalAccuracy, ToolUse | Multi-turn coherence, resolution, completeness, efficiency, relevancy, memory, role/topic adherence, goal accuracy, tool usage (requires `[llm]`) |
| **Security** | VulnerabilityCorrectness, SecurityCompleteness, CodeSafety, CodeQuality, ExplanationQuality, RootCauseAnalysis, Actionability | LLM-as-Judge metrics for AI-generated security vulnerability remediations, with composite Remediation Quality Index (requires LLM provider: `[llm]` or `[harness]`) |

## EvalCase Fields

```python
EvalCase(
    input="the prompt or task",                    # required
    output="what the agent produced",              # required
    expected="ground truth",                       # optional (not needed for LLM-judged metrics)
    context=["retrieved doc 1", "retrieved doc 2"],# optional (for RAG metrics)
    messages=[Message(role="user", content="...")], # optional (for conversation metrics)
    tool_calls=[ToolCall(name="fn", input={...})], # optional (for agent/tool metrics)
    expected_tools=["fn1", "fn2"],                 # optional (expected tool names)
    latency_ms=320,                                # optional (typed, for LatencyMetric)
    token_count=150,                               # optional (typed, for TokenCostMetric)
    cost_usd=0.003,                                # optional (typed, for CostEfficiencyMetric)
    retry_count=0,                                 # optional (typed, for RetryCountMetric)
    confidence=0.95,                               # optional (typed)
    tags={"env": "ci", "model": "gpt-4o"},         # optional (for filtering)
    metadata={"custom_key": "value"},              # optional (extensible)
    runs=[...],                                    # optional (for reliability metrics)
)
```

`Golden` also supports `expected_tools` for defining expected tool names in datasets.

## CLI Reference

```
harness-evals run <config.yaml> [--baseline] [--update-baseline] [--fail-under <float>]
harness-evals import <ref> [-o out.eval.yaml]
harness-evals list-metrics
harness-evals discover [path] [--glob pattern]
```

| Command | Purpose |
|---------|---------|
| `run` | Execute a YAML eval config. Exit 0 = pass, 1 = failure/regression, 2 = config error |
| `import` | Translate a platform eval definition to a local YAML config |
| `list-metrics` | Print a table of all registered metrics with dimensions and thresholds |
| `discover` | Find `**/*.eval.yaml` and `**/eval_*.py` files in a directory |

`--fail-under` and `--baseline` are independent checks. `--fail-under` is an absolute quality floor; baseline gating is a relative regression check. Both can fire in the same run.

## Extending

### Custom metric

```python
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.core.eval_case import EvalCase

class MyMetric(BaseMetric):
    def __init__(self, threshold: float = 0.8):
        super().__init__(name="my_metric", threshold=threshold)

    def measure(self, eval_case: EvalCase) -> Score | None:
        value = compute_something(eval_case)  # return 0.0–1.0
        return Score(name=self.name, value=value, threshold=self.threshold)
        # return None to skip this case (excluded from aggregation)
```

### Custom sink

```python
from harness_evals.core.sink import BaseSink

class MySink(BaseSink):
    def write(self, scores, eval_case):
        for s in scores:
            send_to_my_system(s.name, s.value, s.passed)
```

### Plugin registration

Third-party packages can register metrics, sinks, targets, and source adapters via decorators or entry points.

**Decorator (in-process):**

```python
from harness_evals.plugins import register_metric

@register_metric("my_custom_metric")
class MyCustomMetric(BaseMetric):
    ...
```

**Entry points (distributable packages):**

```toml
# In your package's pyproject.toml
[project.entry-points."harness_evals.metrics"]
my_custom = "my_package.metrics:MyCustomMetric"

[project.entry-points."harness_evals.dataset_sources"]
my_source = "my_package.sources:MyDatasetSource"
```

Eight plugin families are supported: `dataset_sources`, `prompt_sources`, `eval_case_sources`, `eval_config_sources`, `targets`, `metrics`, `baseline_stores`, `sinks`.

Registered metrics appear in `catalog()` and are referenceable by `kind:` in YAML configs. Registered targets are declarable by `type:` in YAML target blocks.

## Documentation

- [Architecture](docs/architecture.md) — system design, data flow, extension points
- [Metrics Guide](docs/metrics-guide.md) — how to write a new metric, templates for every category
- [Integration Guide](docs/integration-guide.md) — pytest, GitHub Actions, Harness CI, GitLab CI
- [Contributing](docs/CONTRIBUTING.md) — development workflow, code style, PR process
- [Framework Integrations](examples/integrations/) — runnable examples for LangChain, CrewAI, DSPy, LlamaIndex, OpenAI Agents, Pydantic AI, Strands, Bedrock AgentCore, Google ADK, LiteLLM, and more
- [Architecture Decision Records](docs/adr/) — why we made key design choices
- [Changelog](CHANGELOG.md) — version history

## Development

```bash
git clone git@github.com:harness/harness-evals.git
cd harness-evals
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
ruff check src/ tests/          # lint
ruff format --check src/ tests/ # format
pytest tests/ -v                # test
```

## References

- Rabanser, Kapoor, Kirgis, Liu, Utpala, Narayanan. ["Towards a Science of AI Agent Reliability"](https://arxiv.org/abs/2602.16666v2). Princeton, 2026. — Defines the 12 reliability metrics across 4 dimensions (consistency, robustness, predictability, safety) that inform this framework's reliability metric design.
- [DeepEval](https://github.com/confident-ai/deepeval) — LLM evaluation framework. Influenced the `measure()` / `a_measure()` metric interface pattern.
- [RAGAS](https://docs.ragas.io) — RAG evaluation toolkit. Informed the RAG metric decomposition (faithfulness, context precision/recall).
- [promptfoo](https://www.promptfoo.dev/docs/intro/) — CLI-first eval tool. Validated the CI/CD-native approach (JUnit output, exit codes, baseline regression).

## License

Apache 2.0 — see [LICENSE](LICENSE).
