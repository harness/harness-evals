# Framework Integration Examples

One pattern, any framework:

```python
target = HttpTarget(url="https://my-agent/run", auth=BearerAuth(os.environ["API_KEY"]))
goldens = [Golden(input="What is 2+2?", expected="4")]
results = await evaluate_dataset(goldens, target.ainvoke, metrics=[ContainsMetric()])
```

For in-process models/agents, use `PromptTarget` instead:

```python
target = PromptTarget(prompt=PromptTemplate(template="{{input}}"), model=OpenAILLM())
results = await evaluate_dataset(goldens, target.ainvoke, metrics=[ContainsMetric()])
```

## Frameworks

| Framework | Pattern | Key Metrics | Example |
|---|---|---|---|
| [OpenAI (direct)](openai/) | `PromptTarget` + `OpenAILLM` | Contains, Latency | Direct API evaluation |
| [Anthropic (direct)](anthropic/) | `PromptTarget` + `AnthropicLLM` | Contains, Latency | Direct API evaluation |
| [LiteLLM](litellm/) | `PromptTarget` + custom `BaseLLM` | Contains, Latency | 100+ models, one adapter |
| [OpenAI Agents SDK](openai_agents/) | Custom `ainvoke` fn | Contains, Latency, TaskCompletion | Agent framework wrapper |
| [PydanticAI](pydantic_ai/) | Custom `ainvoke` fn | Contains, GEval | Structured output agent |
| [DSPy](dspy/) | Custom `ainvoke` fn | Contains, GEval | DSPy module wrapper |
| [Strands](strands/) | Custom `ainvoke` fn | Contains, TaskCompletion | AWS agent framework |
| [LangChain](langchain/) | `HttpTarget` | Contains, TaskCompletion, ToolCorrectness | Deployed chain/agent |
| [LlamaIndex](llama_index/) | `HttpTarget` | Contains, Faithfulness, ContextPrecision | RAG pipeline |
| [CrewAI](crewai/) | `HttpTarget` | Contains, TaskCompletion, ToolCorrectness | Multi-agent crew |
| [Google ADK](google_adk/) | `HttpTarget` | Contains, Latency | Google agent endpoint |
| [Bedrock AgentCore](bedrock_agentcore/) | `HttpTarget` + `BearerAuth` | Contains, Latency | AWS managed agent |

## Running Examples

Each example runs standalone:

```bash
# With mocks (no API key needed — default)
python examples/integrations/openai/example.py

# With real API
USE_MOCK=0 OPENAI_API_KEY=sk-... python examples/integrations/openai/example.py
```

Each test file works with pytest:

```bash
pytest examples/integrations/openai/test_eval.py
```

## Integration Patterns

### PromptTarget (in-process)

Use when you have direct access to the model/agent in Python:

- **Direct LLM providers** (OpenAI, Anthropic): Use the built-in `OpenAILLM` / `AnthropicLLM`
- **LiteLLM**: Subclass `BaseLLM` to wrap `litellm.acompletion()`
- **Agent frameworks** (OpenAI Agents, PydanticAI, DSPy, Strands): Write a custom `ainvoke(golden) -> EvalCase` function that calls the agent and wraps the result

### HttpTarget (deployed)

Use when the agent is deployed as an HTTP service:

- **LangChain**: Deployed via LangServe or LangGraph Platform
- **LlamaIndex**: FastAPI wrapper around query engine
- **CrewAI**: Deployed via CrewAI AMP
- **Google ADK**: `adk api_server` or Cloud Run
- **Bedrock AgentCore**: AWS managed runtime with SigV4/OAuth auth

Configure `output_path` to extract the agent's text from the JSON response, and `body_template` with `{{input}}` / `{{input.field}}` placeholders to format the request.

## Production Sinks

Examples use `StdoutSink` for local dev. Swap for production:

```python
from harness_evals.sinks import StdoutSink
# from harness_evals.sinks import OtlpSink  # OpenTelemetry
# from harness_evals.sinks import LangfuseSink  # Langfuse observability
```
