# Design Spec: Eval Recommender Agent
**Status:** Draft
**Date:** 2026-06-19


# Motivation
Building a new agent on Harness today requires the agent's author to independently figure out which evaluations should be run against it. This includes determining which metrics from the harness-evals catalog apply, what a representative golden dataset should look like, and what thresholds are reasonable. Most agent authors do not have deep familiarity with the Five Dimensions framework or the metric catalog. As a result, agents frequently ship with no evaluations at all, or with evaluations that do not exercise the dimensions that actually matter for that agent's failure modes. This gap has been identified as a priority: any agent that is built and certified by Harness should come with a set of evaluations that have been run and that continue to run regularly whenever the agent is updated.
This spec describes an Eval Recommender Agent. Given a description of any agent, including its purpose, tools, datasets, expected outputs, and constraints, the agent returns a concrete and runnable evaluation suite, defined here as an EvalConfig YAML file paired with a goldens JSONL file (see Section 3 for the full definition). This suite includes recommended metrics mapped to the Five Dimensions framework, a golden dataset, suggested thresholds, and the harness-evals code or command line steps needed to wire the evaluation suite into a pipeline. The longer-term goal is for this recommendation to run automatically whenever an agent is created or updated, and for the resulting evaluation suite to become a permanent part of that agent's certification record.


# Design Principles
The output produced by this agent must be runnable rather than merely descriptive. A recommendation that simply states "use ExactMatch and Latency" in prose is not useful on its own. The recommender must emit real Golden-shaped JSON and real harness-evals command line or SDK invocations that exist today, not commands that have not yet been built. The output should be something an engineer can paste and run directly, rather than a description of what they would need to write themselves, and rather than a command that will fail because it does not yet exist in the CLI.


The recommender must be independently invocable. Its interface, meaning the relationship between its inputs and its outputs, must not be hard-coded to only work inside a worker-agent pipeline step. The first consumer of this agent is worker agents, but the same interface should be callable from other contexts later, such as a command line tool, a continuous integration bot, or an agent-authoring user interface, without requiring the interface itself to be redesigned.


The recommender should map onto the existing metric catalog rather than inventing a parallel one. Recommendations should reference real catalog metric names from harness-evals, such as exact_match, geval, and pii, so that the output is directly usable inside an EvalConfig YAML file or a metrics list, rather than requiring translation from generic metric names into real ones. Because the catalog contains more than eighty metrics, recommendations should not be limited to a small static set of "key examples" — see Section 3 for how the full catalog should be made available to the agent.


For now, the recommender should recommend evaluations rather than gate them. This agent proposes a suite of evaluations, but it does not decide what is mandatory in order for an agent to be certified. The question of certification policy, meaning which dimensions and thresholds are required, is treated as an explicit non-goal of this spec and is described further in the Non-goals section below.


The system should fail visibly if platform write access does not work as intended. If the agent is connected to the harness MCP server but is not actually invoking that server to register the evaluation suite, this must show up as a visible gap during testing. It should never be silently assumed to be working simply because the server is listed as available. This is a known current gap and is described further in the Current Status and Known Gaps section below.


# 1. Architecture
The intended flow begins when an agent is created or updated on Harness. The Eval Recommender is invoked, taking as input the agent's purpose, the tools it uses, the datasets it relies on, its expected outputs, and any constraints or permissions associated with it. The recommender reasons against the Five Dimensions framework, which consists of Correctness, Groundedness, Safety, Trajectory, and Performance, as well as against the full harness-evals metric catalog. It then produces output covering which dimensions apply to the agent and why, recommended metrics and thresholds for each applicable dimension, a recommended dataset consisting of Golden-shaped JSON test cases, and recommended actions in the form of runnable harness-evals command line or SDK steps that exist today.


In the target state, the resulting evaluation suite would then be attached to the agent's definition through the harness MCP server, and the suite would rerun automatically whenever the agent's prompt, tools, model, or datasets change.


As of today, this process runs as a single Harness AI agent step called the eval-recommender-agent, embedded inside a pipeline called the eval-recommender-pipeline. The pipeline takes an agent specification as a runtime input and returns the four output sections described above as the agent's output. The steps involving attaching the suite to the agent's definition and automatically rerunning the suite on change have not yet been implemented. These gaps are described further in the Current Status and Known Gaps section below.


# 2. Inputs
Currently, there is a single required input, referred to as the agent specification. This is a free-text string that describes the agent to be evaluated, including its purpose, its inputs, its expected output format, and any constraints or permissions it operates under.


In the target state, once this input moves from a single free-text string to a more structured call, the same underlying information, namely purpose, tools, datasets, and expected outputs, would be passed as discrete fields rather than as one paragraph of text. This would allow the recommender to eventually be invoked programmatically by an agent-authoring flow, rather than only through a manually written specification.


# 3. Output Format
The agent returns four sections, always in the same order.


The first section, **Dimensions Covered**, provides a breakdown of each of the five dimensions, stating whether that dimension applies to the agent being evaluated and offering a brief rationale for why.


The second section, **Recommended Metrics**, lists, for each applicable dimension, the specific metric name from the catalog, an explanation of why that metric applies to this particular agent, and a suggested threshold value. Because the catalog contains more than eighty metrics, the agent's prompt should not rely on a static list of "key example" metrics per dimension. Instead, the output of the harness-evals list-metrics command should be injected dynamically into the agent's context at runtime, so that recommendations can draw from the full, current catalog rather than a fixed subset hard-coded into the instructions.


The third section, **Recommended Dataset**, provides three to five concrete golden test cases. Each test case includes a realistic input rather than a placeholder, an expected output, and a note on which metric the test case is designed to exercise. Each test case must conform to the actual Golden schema used by the SDK, defined as a dataclass with the following fields: input, expected, context, expected_tools, expected_tool_calls, metadata, and tags. This schema is reproduced in full in the Golden Schema appendix at the end of this document, and should be included directly in the agent's instructions so the model has a reliable basis for producing valid instances, rather than being asked to produce "Golden-shaped JSON" without ever being shown the shape.


The fourth section, **Recommended Actions**, lists runnable steps for wiring the above recommendations into a pipeline. This section must distinguish clearly between command line invocations that exist today and SDK functions that exist today, versus capabilities that would require new work. The CLI as it exists today has exactly four subcommands: run, import, list-metrics, and discover. The subcommands assert, publish, gate, and trend do not exist in the CLI today; gating behavior that does exist today is available as flags on run, specifically harness-evals run <config.yaml> --fail-under <threshold> --baseline <file>. Recommended Actions should therefore use harness-evals run with these flags for today's gating needs, rather than referencing subcommands that do not yet exist. On the SDK side, the most relevant entry point for what this agent recommends is the higher-level evaluate_dataset() function, which takes a list of goldens, an agent function, and a list of metrics, and runs evaluation end to end. This should be included in Recommended Actions alongside or instead of the lower-level evaluate() and assert_test() calls. If new CLI subcommands such as assert, publish, gate, or trend are judged worth adding to the OSS library to support richer recommended actions in the future, that work is tracked explicitly in the Build Order section below, not assumed to already exist.


# 4. Implementation
This agent runs as a Harness AI Agent within a quality assurance environment. The container image used is the standard harness-ai-agent image. The underlying model is an Anthropic model, routed through a configured connector using a Bedrock inference profile. The harness MCP server has been attached to this agent through a configured connector. The intention behind this connection is to allow the agent to register the recommended evaluation suite directly on the platform, rather than only printing command line instructions as plain text. As described in the Current Status and Known Gaps section below, this connection has been established, but it is not yet being actively used by the model. The pipeline that hosts this agent consists of a single stage containing a single agent step. The pipeline takes the agent specification as a pipeline-level input and runs on cloud infrastructure provided by the platform.


# 5. Target Architecture Once Attached to Agent Lifecycle
In the target state, whenever a worker agent is created or updated through the platform's user interface, the Eval Recommender would be invoked automatically. Its inputs would be derived directly from the agent's own definition, namely its purpose, its tools, its datasets, its expected output schema, and its permissions. The recommended suite would be generated in the same way it is generated today. The resulting suite would then be written back through the harness MCP server, meaning the golden dataset would be persisted, the evaluation configuration would be persisted, and the suite would be attached to the agent's certification record. The suite would then rerun automatically whenever the agent's prompt, tools, model, or dataset changes. Achieving this target state requires the MCP integration described above to actually function as intended, meaning the model must invoke write-capable tools rather than simply describing command line steps in text. It also requires the existence of a certification record or schema to write to, which does not currently exist. This open question is described further as the fifth item in the Open Questions section below.


# 6. Independent Invocability
Consistent with the second design principle, the recommender's input and output contract should not assume that it will always be called from inside a worker-agent pipeline step. Concretely, this means that its inputs, whether that is the current free-text agent specification or its structured equivalent in the future, should be passable from any calling context, including a pipeline step, a command line tool, or a hook inside an agent-authoring user interface. Its outputs are always the same four sections regardless of how it was invoked, which is already true today, since the agent's output format does not depend on the calling context. The recommender also does not assume access to any pipeline-specific runtime context beyond the agent specification itself. What has not yet been built is any entry point other than an agent step inside a pipeline. The interface itself is invocation-agnostic in principle, but there is currently exactly one way to actually invoke it.


# 7. Relationship to Certification
This spec intentionally does not define what is mandatory in order for an agent to be considered certified by Harness. It defines only what the recommender proposes. A follow-up spec should define which dimensions are mandatory versus optional for each category of agent, what threshold tightness is required in order to pass certification, where the certification record itself lives and what schema it follows, and how a failing or regressing evaluation suite should block or flag a certified agent. This work is intentionally deferred and is described further in the Non-goals section below and in the second and fifth items of the Open Questions section.


# Non-Goals
The certification policy, meaning which dimensions and thresholds are mandatory, is intentionally treated as a separate follow-up spec and is not defined in this document. A standalone user interface or command line entry point is also out of scope for this spec. The current implementation is a pipeline agent step, and a dedicated command line or user interface surface is considered future work. Automatic write-back to a certification record is similarly out of scope, since no such record or schema currently exists to write to. This spec covers only the generation of the recommendation itself. Adding new CLI subcommands (assert, publish, gate, trend) to the harness-evals library is out of scope for this spec to design in detail, though the need for them is tracked in the Build Order section below as a prerequisite for a future, richer version of Recommended Actions.


# 8. Build Order
The **first step** is to decide the CLI surface for richer recommended actions. This means deciding, for each capability the agent currently references in spirit (asserting a result, publishing a result, gating on a baseline, tracking trend over time), whether it should be a new CLI subcommand contributed to harness-evals, or whether it is already covered by existing flags on run. Gating, for example, is already covered today by harness-evals run --fail-under <threshold> --baseline <file> and does not require a new subcommand.

The **second step** is to implement any new subcommands decided on in step one. This is real SDK work, not a prompt change, and should be scoped and contributed to the OSS library before the agent's instructions are allowed to reference it.

The **third step** is to harden the output contract, now that the prompt can accurately reflect what exists. This means updating the agent's instructions so that the Recommended Dataset section includes the actual Golden schema inline, so that Recommended Metrics is grounded in the dynamically injected output of harness-evals list-metrics rather than a static example list, and so that Recommended Actions explicitly requires an executable code block using commands and SDK functions that exist today, including evaluate_dataset(), rather than relying on the model to default to the correct shape or to guess at commands that may not exist.

The **fourth step** is to debug the MCP tool invocation gap. The harness MCP server is currently attached to the agent, but the model is not yet calling its tools. This step involves confirming whether the issue is a prompting gap, meaning the instructions never explicitly tell the model to use the available tools, or a platform-side limitation, and then fixing the issue accordingly.

The **fifth step** is to define the target inputs as structured fields rather than as a single free-text agent specification string, so that the recommender can eventually be invoked programmatically from an agent-authoring flow.

The **sixth step** is to define exactly where the output gets written once MCP write access is functioning correctly. This requires deciding what object the resulting suite should attach to, which in turn requires a certification record schema that does not yet exist. As part of this step, a decision should be made on whether "eval suite" remains defined purely in terms of existing SDK primitives, namely an EvalConfig YAML file paired with a goldens JSONL file, or whether a dedicated EvalSuite construct is worth contributing to the OSS library as a first-class abstraction. See Section 3 and the Eval Suite Definition appendix for the current working definition.

The **seventh step** is to wire automatic invocation of the recommender into agent creation and update events, once the previous six steps have been completed successfully.

The first two steps require deciding on and potentially implementing new SDK functionality, and should happen before the prompt is hardened in step three, since the prompt cannot accurately require commands that do not yet exist. Steps three and four require no new platform infrastructure beyond that decision and can be completed promptly once steps one and two are resolved. The remaining three steps depend on decisions that are described in the Open Questions section below.


# 9. Resolved Design Decisions
Regarding output format, the decision is to use four fixed sections, namely Dimensions, Metrics, Dataset, and Actions, always presented in that order. The rationale is that this produces a predictable structure that downstream tooling, or a human reader, can parse reliably. Regarding where the recommender currently runs, the decision is to implement it as a pipeline agent step rather than as a standalone application. The rationale is that this represents the fastest path to a working end-to-end loop. Standalone invocability is treated as a target property of the interface itself, rather than as a requirement that the recommender ship as a fully separate application on day one. Regarding metric naming, the decision is that all recommendations must use real harness-evals catalog metric names, sourced dynamically from harness-evals list-metrics rather than a static list. The rationale is that the resulting output should be directly usable inside an EvalConfig file, and should reflect the full catalog rather than an arbitrarily narrowed subset. Regarding the initial trigger in the target state, the decision is that the recommender should be triggered by agent creation and agent updates only. The rationale is that this matches the stated requirement that certified agents come with regularly run evaluations. Broader triggers, such as an on-demand chat-based invocation, are not precluded by this decision, but they are not the initial trigger for this version. Regarding what "eval suite" means, the decision is to define it, for now, as an EvalConfig YAML file paired with a goldens JSONL file, since no EvalSuite class or suite-level abstraction currently exists in the SDK. Whether to contribute a dedicated EvalSuite construct to the OSS library is tracked as part of Build Order step six rather than decided here.


# 10. Open Questions
The first open question is whether Recommended Metrics should be restricted to metrics that already exist in the catalog, or whether the agent should be allowed to propose entirely new metrics that do not yet exist within harness-evals.


The second open question concerns the boundary between recommended and mandatory requirements for certification. Specifically, it asks which dimensions or metrics, if any, should be considered non-negotiable in order for an agent to be considered certified by Harness.


The third open question is whether the evaluation suite should rerun on every single pipeline execution, or only when the agent's underlying definition changes, meaning its prompt, tools, model, or dataset.


The fourth open question is where the recommender should ultimately be invokable from. Possible answers include the current pipeline step, an agent-authoring user interface, or both.


The fifth open question concerns what it technically means to attach a suite to an agent's certification record. This includes determining whether a certification record or schema already exists to write to, or whether one needs to be designed as part of this work.


The sixth open question is whether a dedicated EvalSuite construct should be contributed to the harness-evals OSS library as a first-class abstraction, rather than continuing to define "suite" informally as an EvalConfig plus a goldens file.


# 11. Verified Output
A test was conducted using an agent specification describing an Incident Root Cause Agent. This agent reads raw deployment failure logs and returns a structured diagnosis covering what broke, why it broke, and what should be done to fix it, with the output always organized into three sections. The exact specification given as input was:
I am building an Incident Root Cause Agent that reads raw deployment failure logs and returns a structured diagnosis covering what broke, why it broke, and what to fix. The output is always in three sections: WHAT BROKE, WHY IT BROKE, and WHAT TO FIX. 


The complete pipeline execution log for this run was retrieved and reviewed directly. The log capture itself is truncated and does not contain the agent's full final output. What follows is only the portion that is verifiably present in that log, reproduced verbatim.
The agent's in-progress reasoning stream included the following recommended pipeline step, shown here exactly as it appears in the log:
- run:
    name: Gate on Eval Suite
    script: |
      harness-evals gate \
        --pipeline incident_rca_agent_evals \
        --ref ${{ pipeline.sequenceId }} \
        --timeout 600
  on-failure:
    errors: all
    action: abort


It also included the following step intended to catch gradual metric degradation over time:
- run:
    name: Drift Alert
    script: |
      harness-evals trend \
        --metric faithfulness,geval,hallucination \
        --window 14d \
        --alert-on-slope -0.03    # alert if any metric drops >3% over 2 weeks


After this, the log records that the agent completed in two turns, with the following usage reported verbatim:
=== Agent complete (2 turns) ===
Tokens: input=5 output=6622 cache_read=24552 cache_creation=33194
Cost: $0.233723


The log then begins to print the agent's final consolidated result. The beginning of that result, reproduced exactly as captured, reads:
Agent result: Here is the full Five Dimensions evaluation recommendation for your **Incident Root Cause Agent**.


---


## DIMENSIONS COVERED


| Dimension | Applies | Rationale |
|---|---|---|
| **Correctness** | Yes, HIGH | The agent must identify the correct broken component, accurate root cause, and actionable fix. Wrong diagnosis in incident response directly extends MTTR. |
| **Groundedness** | Yes, HIGH | Every claim in the three sections must be traceable to the provided deployment logs. The agent must




The log capture ends mid-sentence at this exact point. The Safety, Trajectory, and Performance rows of the dimensions table, the entire Recommended Metrics section, the entire Recommended Dataset section, and the entire Recommended Actions section of the final result are not present in the available log and cannot be verified from it. This means the only artifacts confirmed verbatim from this test run are the two YAML steps shown above, which were part of the agent's in-progress reasoning rather than its final consolidated answer, and the token and cost usage line.


This test still confirms that the recommendation loop runs to completion, produces structured markdown and YAML rather than only plain prose, and reasons about gating and drift detection using a Five Dimensions framing. It also confirms directly and concretely the gap described in the Design Principles and Section 3 above: the harness-evals gate and harness-evals trend commands shown above do not exist in the CLI today, which is why Build Order steps one and two now precede the prompt-hardening step. A complete, untruncated capture of a future test run should be obtained and substituted here once available, so that the full Recommended Metrics, Recommended Dataset, and Recommended Actions sections can be verified rather than only partially observed.




# 12. Current Status and Known Gaps
The agent's ability to take an agent specification as a runtime pipeline input is currently working as intended. Its ability to return a structured four-section output is also working as intended.


The current instructions given to the model ask it to provide "golden test cases" without explicitly requiring that those test cases take the form of valid Golden-shaped JSON, and they ask it to "list the steps" rather than explicitly requiring an executable code block. They also reference example metrics statically rather than injecting the full catalog dynamically, and they reference CLI subcommands such as assert, publish, gate, and trend that do not exist today. In practice, the model has nonetheless been producing real command line invocations and well-structured test cases, as shown in the Verified Output section above, but several of those invocations reference commands that would fail if a user tried to run them. The instructions need to be tightened to require the correct output shape and to reference only commands and SDK functions that actually exist, rather than relying on the model to default to the correct format and the correct command set on its own. This is tracked as Build Order steps one through three above.


The harness MCP server has been successfully attached to the agent. However, the agent actually invoking that MCP server's tools in order to write changes to the platform is not yet happening. The server is available to the agent, but it remains unused by the model in tested runs.


The capability to attach the resulting suite to an agent's certification record has not been implemented, since no certification record schema currently exists. Automatic triggering of the recommender upon agent creation or update has also not been implemented. The current trigger for this agent remains a manually initiated pipeline run.


# Failure Modes
Several failure modes have not yet been addressed and should be accounted for in the implementation. If the agent specification provided as input is too vague to support a confident recommendation, the agent's current behavior in that situation is undefined and should be tested explicitly, with a defined fallback such as asking a clarifying question or stating explicitly that the specification is insufficient rather than guessing. If the model hallucinates a metric name that does not exist in the catalog, this would currently go undetected, since there is no validation step between the agent's output and the user pasting that output into a real config. A validation step, such as a harness-evals run --validate flag or an equivalent dry-run mode that checks metric names and Golden field names against the real schema and catalog before execution, should be added to the architecture and called out explicitly here once designed. If the generated Golden JSON is structurally invalid, for example missing a required field or using a field name that does not exist on the dataclass, this should also be caught by the same validation step described above rather than discovered only when a user actually tries to run the suite.




# Deferred
Several pieces of work are intentionally deferred beyond the scope of this spec. These include defining a structured, non-free-text input schema for the agent specification; writing a separate certification policy spec covering mandatory dimensions and thresholds; building a standalone command line or user interface entry point outside of the existing pipeline step; implementing automatic reruns of the evaluation suite whenever the underlying agent, prompt, tool, model, or dataset changes; and the detailed design of any new CLI subcommands identified in Build Order steps one and two.


# Appendix: Golden Schema
Every entry in the Recommended Dataset section must conform to the actual Golden dataclass used by the SDK. The fields are:
* input — the input given to the agent under evaluation.
* expected — the expected or ground-truth output for that input.
* context — optional. Additional context the agent had access to, relevant for groundedness-style metrics.
* expected_tools — optional. The names of tools the agent was expected to call, relevant for trajectory-style metrics.
* expected_tool_calls — optional. The specific tool calls, including arguments, the agent was expected to make.
* metadata — optional. A free-form field for any additional information relevant to the test case.
* tags — optional. Labels usable for filtering or grouping test cases.

This schema should be included directly in the agent's instructions so that the model has a concrete basis for producing valid Golden instances, rather than being asked to produce "Golden-shaped JSON" without ever being shown the actual shape.

# Appendix: Eval Suite Definition
For the purposes of this spec, an "evaluation suite" is defined as the combination of an EvalConfig YAML file, which specifies the dataset reference, the target, the metrics, and the sinks, paired with a goldens JSONL file, which contains the actual test cases referenced by that config. There is currently no dedicated EvalSuite class or suite-level abstraction in the SDK. Whether to contribute one as a first-class construct is tracked as Build Order step six and Open Question six above, and is not decided within this spec. 


