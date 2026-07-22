# Extended Framework Mappers
We need to add session mappers for Google ADK, Claude (Anthropic SDK), and OpenAI Agents SDK. Together, these three SDKs account for over 190 million monthly PyPI downloads (`anthropic` 154M, `openai-agents` 30M, `google-adk` 16M) and represent a significant portion of production agents today. Supporting these agent SDKs will move strands-evals towards becoming an agent-agnostic framework, enabling more customers to evaluate their agents without migrating their existing infrastructure.

# Session Mappers
Three session mappers are introduced that convert their respective SDK agent spans to strands-eval sessions. They handle the instrumentation-specific quirks outlined below.

## Google ADK

Google ADK has native OpenTelemetry instrumentation built into the framework itself. When tracing is enabled, ADK automatically emits spans following the OTel GenAI semantic conventions (`gen_ai.*`) and Google-specific attributes (`gcp.vertex.agent.*`). The full LLM request and response payloads are serialized as JSON strings on `call_llm` spans.

## Claude Agents SDK

The Anthropic SDK is instrumented via `opentelemetry-instrumentation-anthropic` from Traceloop's OpenLLMetry project (~5M monthly downloads). It emits OTel spans following GenAI semantic conventions (`gen_ai.*`). Full conversation data is stored as JSON-encoded span attributes: `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions`, and `gen_ai.tool.definitions`.

The Claude Agent SDK also ships a built-in first-party tracing system that emits OTel spans from its CLI subprocess using proprietary `claude_code.*` span names (`interaction`, `llm_request`, `tool`). A mapper is not built for the built-in tracing because:
- Message content, system prompts, tool definitions, and available tools are all absent from trace spans. The only path to this data is `OTEL_LOG_RAW_API_BODIES`, which emits full API request/response JSON as OTel **log events** (not span attributes), requiring cross-signal correlation between the traces and logs pipelines.
- The built-in tracing is designed for operational monitoring (latency, token spend, error rate). Users who need content for evaluation use Traceloop, which captures everything by default with no opt-in flags.

An alternative third-party instrumentor exists: `openinference-instrumentation-anthropic` from Arize (~558K monthly downloads). A mapper is not built for OpenInference because:
- OpenLLMetry has ~9× more downloads (~5M/month), indicating it is the dominant instrumentation choice for Anthropic SDK users.
- OpenInference is primarily adopted by teams using the Arize Phoenix backend. Teams using general-purpose OTel backends (CloudWatch, Langfuse, OpenSearch, Datadog, Honeycomb) use OpenLLMetry instead.

## OpenAI Agents SDK

The OpenAI Agents SDK is instrumented via the official OTel package `opentelemetry-instrumentation-openai-agents-v2` from `opentelemetry-python-contrib`. It hooks into the SDK's `TracingProcessor` interface and emits OTel spans following GenAI semantic conventions (`gen_ai.*`). Tool definitions (schemas/descriptions), system prompts, `tool_call_id` values, and finish reasons are absent from the trace. The instrumentor emits structural noise spans (`Agent workflow` and `unknown`) that carry no useful data and must be filtered out during mapping.

The SDK also ships a built-in first-party tracing system with proprietary span types (`agent`, `response`, `function`, `turn`, `task`). A mapper is not built for the built-in tracing because:
- The built-in exporter sends data to the OpenAI Traces dashboard in a proprietary format. Users routing traces to external backends (CloudWatch, Langfuse, OpenSearch) use the OTel instrumentor, which is what they would feed into strands-evals.
- System prompts, tool definitions, and finish reasons remain unavailable through either path, so the built-in tracing offers no data advantage over the OTel instrumentor.

# Integration Testing

Integration tests are written to verify the new mappers work as expected. They test the following cases:
- Invocation with tool use
- Invocation with failed tool
- Invocation with multiple agents

The mappers are tested with live agents. This requires a Google ADK, Anthopic, and OpenAI API key to be added to the secrets manager. While Bedrock models can be used to test the Anthropic SDK and Traceloop instrumentor, it cannot be used for the Claude Agent SDK and its built-in tracing. The use of live agents catches schema drift (when the instrumentor updates its format) and validates the pipeline end-to-end.

If live agents cannot be used, we can alternatively create fixtures that respond with static traces. This is fast, free, and does not require storing API keys. However, this does not test end-to-end and must be periodically maintained.

# Appendix: ADK Field Mapping Chart

All example values come directly from the `to_json()` output of captured spans.

`→` in the ADK JSON path column means: parse the parent field (a JSON string), then access the nested path inside it.

## `invoke_agent` → `AgentInvocationSpan`

| ADK JSON path | Evals field | Example value |
|---|---|---|
| `context.trace_id` | `span_info.trace_id` | `"0x10049373ec731ba0d7ead39b274cfdb9"` |
| `context.span_id` | `span_info.span_id` | `"0xf888743b46f02a74"` |
| `parent_id` | `span_info.parent_span_id` | `"0x6fe59ff3399c5633"` |
| `start_time` | `span_info.start_time` | `"2026-07-21T17:32:33.388957Z"` |
| `end_time` | `span_info.end_time` | `"2026-07-21T17:32:35.979276Z"` |
| `attributes.gen_ai.agent.name` | `metadata["agent_name"]` | `"math_agent"` |
| `attributes.gen_ai.agent.description` | `metadata["agent_description"]` | `"A helpful math assistant"` |
| `attributes.gen_ai.conversation.id` | `span_info.session_id` | `"dcfd93b3-f2fb-410c-8817-57c78ba1effb"` |
| child `call_llm` → `attributes.gcp.vertex.agent.llm_request` → `.contents[0].parts[0].text` | `AgentInvocationSpan.user_prompt` | `"What is 15 multiplied by 37?"` |
| last child `call_llm` → `attributes.gcp.vertex.agent.llm_response` → `.content.parts[*].text` | `AgentInvocationSpan.agent_response` | `"15 multiplied by 37 is 555."` |
| child `call_llm` → `attributes.gcp.vertex.agent.llm_request` → `.config.tools[*].function_declarations[*].name` | `AgentInvocationSpan.available_tools[*].name` | `["calculator"]` |
| child `call_llm` → `.function_declarations[*].description` | `AgentInvocationSpan.available_tools[*].description` | `"Evaluate a mathematical expression..."` |
| child `call_llm` → `.function_declarations[*].parameters_json_schema` | `AgentInvocationSpan.available_tools[*].parameters` | `{"properties": {"expression": {"title": "Expression", "type": "string"}}, "required": ["expression"]}` |
| child `call_llm` → `attributes.gcp.vertex.agent.llm_request` → `.config.system_instruction` | `AgentInvocationSpan.system_prompt` | `"You are a math assistant. Use the calculator tool..."` |

## `generate_content` → `InferenceSpan`

| ADK JSON path | Evals field | Example value |
|---|---|---|
| `context.trace_id` | `span_info.trace_id` | `"0x10049373ec731ba0d7ead39b274cfdb9"` |
| `context.span_id` | `span_info.span_id` | `"0x65869d21ba181515"` |
| `parent_id` | `span_info.parent_span_id` | `"0x98875dc14b85fd33"` |
| `start_time` | `span_info.start_time` | `"2026-07-21T17:32:34.462870Z"` |
| `end_time` | `span_info.end_time` | `"2026-07-21T17:32:35.415994Z"` |
| `attributes.gen_ai.system` | `InferenceSpan.metadata["gen_ai.system"]` | `"gemini"` |
| `attributes.gen_ai.request.model` | `InferenceSpan.metadata["model"]` | `"gemini-2.5-flash"` |
| `attributes.gen_ai.agent.name` | `InferenceSpan.metadata["agent_name"]` | `"math_agent"` |
| `attributes.gen_ai.conversation.id` | `span_info.session_id` | `"dcfd93b3-f2fb-410c-8817-57c78ba1effb"` |
| `attributes.gen_ai.usage.input_tokens` | `InferenceSpan.metadata["input_tokens"]` | `140` |
| `attributes.gen_ai.usage.output_tokens` | `InferenceSpan.metadata["output_tokens"]` | `98` |
| `attributes.gen_ai.usage.reasoning.output_tokens` | `InferenceSpan.metadata["reasoning_tokens"]` | `80` |
| `attributes.gen_ai.response.finish_reasons` | `InferenceSpan.metadata["finish_reasons"]` | `["stop"]` |
| `attributes.gcp.vertex.agent.invocation_id` | `InferenceSpan.metadata["invocation_id"]` | `"e-1b9afdcd-6204-482d-b567-80ad3f48ac2e"` |
| `attributes.gcp.vertex.agent.event_id` | `InferenceSpan.metadata["event_id"]` | `"52be2dc9-2611-433a-b628-f6919b57ed6b"` |
| parent `call_llm` → `attributes.gcp.vertex.agent.llm_request` → `.contents` | `InferenceSpan.messages` | user: `"What is 15 multiplied by 37?"` |

## `call_llm` → `InferenceSpan` + `AgentInvocationSpan` (data source)

`call_llm` is the parent of `generate_content` and carries the full serialized request and response. It is the primary source for reconstructing `InferenceSpan.messages` and `AgentInvocationSpan` content.

| ADK JSON path | Evals field | Example value |
|---|---|---|
| `attributes.gen_ai.system` | `InferenceSpan.metadata["gen_ai.system"]` | `"gcp.vertex.agent"` |
| `attributes.gen_ai.request.model` | `InferenceSpan.metadata["model"]` | `"gemini-2.5-flash"` |
| `attributes.gcp.vertex.agent.session_id` | `span_info.session_id` | `"dcfd93b3-f2fb-410c-8817-57c78ba1effb"` |
| `attributes.gcp.vertex.agent.invocation_id` | `InferenceSpan.metadata["invocation_id"]` | `"e-1b9afdcd-6204-482d-b567-80ad3f48ac2e"` |
| `attributes.gcp.vertex.agent.event_id` | `InferenceSpan.metadata["event_id"]` | `"52be2dc9-2611-433a-b628-f6919b57ed6b"` |
| `attributes.gen_ai.usage.input_tokens` | `InferenceSpan.metadata["input_tokens"]` | `140` |
| `attributes.gen_ai.usage.output_tokens` | `InferenceSpan.metadata["output_tokens"]` | `98` |
| `attributes.gen_ai.usage.reasoning.output_tokens` | `InferenceSpan.metadata["reasoning_tokens"]` | `80` |
| `attributes.gen_ai.response.finish_reasons` | `InferenceSpan.metadata["finish_reasons"]` | `["stop"]` |
| `attributes.gcp.vertex.agent.llm_request` (JSON) → `.config.system_instruction` | `AgentInvocationSpan.system_prompt` | `"You are a math assistant. Use the calculator tool..."` |
| `attributes.gcp.vertex.agent.llm_request` → `.config.tools[*].function_declarations[*]` | `AgentInvocationSpan.available_tools` | `[{name: "calculator", description: "...", parameters_json_schema: {...}}]` |
| `attributes.gcp.vertex.agent.llm_request` → `.contents[*]` | `InferenceSpan.messages` | `[{parts: [{text: "What is 15 multiplied by 37?"}], role: "user"}]` |
| `attributes.gcp.vertex.agent.llm_response` (JSON) → `.content.parts[*].text` | `AssistantMessage.content[*]` (text) | `"15 multiplied by 37 is 555."` |
| `attributes.gcp.vertex.agent.llm_response` → `.content.parts[*].function_call` | `AssistantMessage.content[*]` (tool call) | `{name: "calculator", args: {expression: "15 * 37"}}` |
| `attributes.gcp.vertex.agent.llm_response` → `.finish_reason` | `InferenceSpan.metadata["finish_reasons"]` | `"STOP"` |
| `attributes.gcp.vertex.agent.llm_response` → `.usage_metadata.prompt_token_count` | `InferenceSpan.metadata["input_tokens"]` | `140` |
| `attributes.gcp.vertex.agent.llm_response` → `.usage_metadata.candidates_token_count` | `InferenceSpan.metadata["output_tokens"]` | `18` |
| `attributes.gcp.vertex.agent.llm_response` → `.usage_metadata.thoughts_token_count` | `InferenceSpan.metadata["reasoning_tokens"]` | `80` |

## `execute_tool` → `ToolExecutionSpan`

| ADK JSON path | Evals field | Example value |
|---|---|---|
| `context.trace_id` | `span_info.trace_id` | `"0x10049373ec731ba0d7ead39b274cfdb9"` |
| `context.span_id` | `span_info.span_id` | `"0x3215d0bf2389a7a2"` |
| `parent_id` | `span_info.parent_span_id` | `"0x65869d21ba181515"` |
| `start_time` | `span_info.start_time` | `"2026-07-21T17:32:35.414398Z"` |
| `end_time` | `span_info.end_time` | `"2026-07-21T17:32:35.415190Z"` |
| `attributes.gen_ai.tool.name` | `ToolExecutionSpan.tool_call.name` | `"calculator"` |
| `attributes.gen_ai.tool.call.id` | `ToolExecutionSpan.tool_call.tool_call_id` | `"adk-4d2a5d40-cd48-46db-98b6-caa4acedd18c"` |
| `attributes.gcp.vertex.agent.tool_call_args` (JSON string) | `ToolExecutionSpan.tool_call.arguments` | `{"expression": "15 * 37"}` |
| `attributes.gcp.vertex.agent.tool_response` (JSON string) | `ToolExecutionSpan.tool_result.content` | `{"result": "555"}` |
| `attributes.gen_ai.tool.call.id` | `ToolExecutionSpan.tool_result.tool_call_id` | `"adk-4d2a5d40-cd48-46db-98b6-caa4acedd18c"` (same as `tool_call.tool_call_id`) |
| _(not present in ADK trace)_ | `ToolExecutionSpan.tool_result.error` | `null` (infer from `status.status_code`; no error string available) |
| `attributes.gen_ai.tool.description` | `ToolExecutionSpan.metadata["description"]` | `"Evaluate a mathematical expression and return the result..."` |
| `attributes.gen_ai.tool.type` | `ToolExecutionSpan.metadata["tool_type"]` | `"FunctionTool"` |
| `attributes.gcp.vertex.agent.event_id` | `ToolExecutionSpan.metadata["event_id"]` | `"9f53f941-cc36-4a54-816d-445b462405c4"` |


---

# Appendix: Claude Agents SDK (OpenLLMetry / Traceloop) Field Mapping Chart

All example values come directly from the JSON output of captured spans.

Instrumentation: `opentelemetry-instrumentation-anthropic` from Traceloop's OpenLLMetry project. Monkey-patches `anthropic.Anthropic().messages.create()` to emit OTel spans following GenAI semantic conventions. Agent-level and tool-execution spans are manual (added by the application). The instrumentor auto-instruments LLM API calls only.

Span hierarchy (single tool-use interaction):
```
invoke_agent (manual, root)
  anthropic.chat (auto-instrumented, tool_use response)
  execute_tool calculator (manual)
  anthropic.chat (auto-instrumented, final response)
```

## `invoke_agent` (manual) → `AgentInvocationSpan`

| OpenLLMetry JSON path | Evals field | Example value |
|---|---|---|
| `context.trace_id` | `span_info.trace_id` | `"0x696c0bb70daf00bf84cc0a21560ba2f9"` |
| `context.span_id` | `span_info.span_id` | `"0x8ecae48bda910b63"` |
| `parent_id` | `span_info.parent_span_id` | `null` (root) |
| `start_time` | `span_info.start_time` | `"2026-07-21T18:55:09.906079Z"` |
| `end_time` | `span_info.end_time` | `"2026-07-21T18:55:13.826913Z"` |
| `attributes.gen_ai.agent.name` | `metadata["agent_name"]` | `"math_agent"` |
| `attributes.gen_ai.agent.description` | `metadata["agent_description"]` | `"A helpful math assistant"` |
| `attributes.gen_ai.request.model` | `metadata["model"]` | `"claude-sonnet-4-6"` |
| first child `anthropic.chat` → `attributes.gen_ai.input.messages` → `[0].parts[0].content` | `AgentInvocationSpan.user_prompt` | `"What is 15 multiplied by 37?"` |
| last child `anthropic.chat` → `attributes.gen_ai.output.messages` → `[0].parts[0].content` | `AgentInvocationSpan.agent_response` | `"15 multiplied by 37 is **555**."` |
| first child `anthropic.chat` → `attributes.gen_ai.tool.definitions` (JSON) → `[*].name` | `AgentInvocationSpan.available_tools[*].name` | `["calculator"]` |
| first child `anthropic.chat` → `attributes.gen_ai.tool.definitions` → `[*].description` | `AgentInvocationSpan.available_tools[*].description` | `"Evaluate a mathematical expression and return the result."` |
| first child `anthropic.chat` → `attributes.gen_ai.tool.definitions` → `[*].input_schema` | `AgentInvocationSpan.available_tools[*].parameters` | `{"type": "object", "properties": {"expression": {"type": "string", ...}}, "required": ["expression"]}` |
| first child `anthropic.chat` → `attributes.gen_ai.system_instructions` → `[0].content` | `AgentInvocationSpan.system_prompt` | `"You are a math assistant. Use the calculator tool to evaluate mathematical expressions."` |

## `anthropic.chat` → `InferenceSpan`

| OpenLLMetry JSON path | Evals field | Example value |
|---|---|---|
| `context.trace_id` | `span_info.trace_id` | `"0x696c0bb70daf00bf84cc0a21560ba2f9"` |
| `context.span_id` | `span_info.span_id` | `"0xfa606e1d3f9b6618"` |
| `parent_id` | `span_info.parent_span_id` | `"0x8ecae48bda910b63"` |
| `start_time` | `span_info.start_time` | `"2026-07-21T18:55:09.906124Z"` |
| `end_time` | `span_info.end_time` | `"2026-07-21T18:55:11.908613Z"` |
| `attributes.gen_ai.request.model` | `InferenceSpan.metadata["model"]` | `"claude-sonnet-4-6"` |
| `attributes.gen_ai.response.model` | `InferenceSpan.metadata["response_model"]` | `"claude-sonnet-4-6"` |
| `attributes.gen_ai.provider.name` | `InferenceSpan.metadata["gen_ai.system"]` | `"anthropic"` |
| `attributes.gen_ai.response.id` | `InferenceSpan.metadata["response_id"]` | `"msg_011CdFiCvbvVDiupfBwjVGC5"` |
| `attributes.gen_ai.usage.input_tokens` | `InferenceSpan.metadata["input_tokens"]` | `616` |
| `attributes.gen_ai.usage.output_tokens` | `InferenceSpan.metadata["output_tokens"]` | `56` |
| `attributes.gen_ai.usage.total_tokens` | `InferenceSpan.metadata["total_tokens"]` | `672` |
| `attributes.gen_ai.usage.cache_read.input_tokens` | `InferenceSpan.metadata["cache_read_tokens"]` | `0` |
| `attributes.gen_ai.usage.cache_creation.input_tokens` | `InferenceSpan.metadata["cache_creation_tokens"]` | `0` |
| `attributes.gen_ai.response.finish_reasons` | `InferenceSpan.metadata["finish_reasons"]` | `["tool_call"]` or `["stop"]` |
| `attributes.gen_ai.request.max_tokens` | `InferenceSpan.metadata["max_tokens"]` | `1024` |
| `attributes.gen_ai.input.messages` (JSON string) | `InferenceSpan.messages` | user: `"What is 15 multiplied by 37?"` |
| `attributes.gen_ai.output.messages` (JSON string) | `InferenceSpan.messages` (assistant turn) | tool_call or text content |
| `attributes.gen_ai.system_instructions` (JSON string) | `InferenceSpan.metadata["system_instructions"]` | `"You are a math assistant..."` |
| `attributes.gen_ai.tool.definitions` (JSON string) | `InferenceSpan.metadata["tool_definitions"]` | `[{name: "calculator", description: "...", input_schema: {...}}]` |

## `execute_tool` (manual) → `ToolExecutionSpan`

| OpenLLMetry JSON path | Evals field | Example value |
|---|---|---|
| `context.trace_id` | `span_info.trace_id` | `"0x696c0bb70daf00bf84cc0a21560ba2f9"` |
| `context.span_id` | `span_info.span_id` | `"0x17aa8779b5dfe87f"` |
| `parent_id` | `span_info.parent_span_id` | `"0x8ecae48bda910b63"` |
| `start_time` | `span_info.start_time` | `"2026-07-21T18:55:11.908776Z"` |
| `end_time` | `span_info.end_time` | `"2026-07-21T18:55:11.908866Z"` |
| `attributes.gen_ai.tool.name` | `ToolExecutionSpan.tool_call.name` | `"calculator"` |
| `attributes.gen_ai.tool.call.id` | `ToolExecutionSpan.tool_call.tool_call_id` | `"toolu_01Ub5oV7rReMq7uiUpodztbh"` |
| `events[0]` (`gen_ai.tool.message`) → `attributes.content` | `ToolExecutionSpan.tool_call.arguments` | `{"expression": "15 * 37"}` |
| `events[1]` (`gen_ai.choice`) → `attributes.message` | `ToolExecutionSpan.tool_result.content` | `[{"text": "555"}]` |
| `attributes.gen_ai.tool.call.id` | `ToolExecutionSpan.tool_result.tool_call_id` | `"toolu_01Ub5oV7rReMq7uiUpodztbh"` |
| `attributes.gen_ai.tool.status` | `ToolExecutionSpan.tool_result.error` | `"success"` → `error = null` |

---

# Appendix: OpenAI Agents SDK (OTel Instrumentor) Field Mapping Chart

All example values come directly from the JSON output of captured spans.

Instrumentation: `opentelemetry-instrumentation-openai-agents-v2` from `opentelemetry-python-contrib`. Hooks into the SDK's `TracingProcessor` interface and emits OTel spans following GenAI semantic conventions. The instrumentor also emits structural noise spans (`Agent workflow` with `SpanKind.SERVER` and `unknown` spans) that carry no useful data and must be filtered during mapping.

Span hierarchy (single tool-use interaction):
```
Agent workflow (noise, root, SpanKind.SERVER)
  unknown (noise)
    invoke_agent math_agent (SpanKind.CLIENT)
      unknown (noise)
        chat gpt-4o-mini-2024-07-18 (LLM call → tool_use)
        execute_tool calculator
      unknown (noise)
        chat gpt-4o-mini-2024-07-18 (LLM call → final output)
```

**Noise spans to filter:** Spans named `"Agent workflow"` (root) and `"unknown"` carry only provider metadata (`gen_ai.system`, `server.address`) with no evaluation-relevant data. Filter by `attributes.gen_ai.operation.name == "unknown"` or `name == "Agent workflow"`.

## `invoke_agent` → `AgentInvocationSpan`

| OTel JSON path | Evals field | Example value |
|---|---|---|
| `context.trace_id` | `span_info.trace_id` | `"0x87d999039e503b18371c24654b77bada"` |
| `context.span_id` | `span_info.span_id` | `"0x95454481da047692"` |
| `parent_id` | `span_info.parent_span_id` | `"0x43f58fcbde22c9ec"` |
| `start_time` | `span_info.start_time` | `"2026-07-21T23:41:30.662404Z"` |
| `end_time` | `span_info.end_time` | `"2026-07-21T23:41:32.941825Z"` |
| `attributes.gen_ai.agent.name` | `metadata["agent_name"]` | `"math_agent"` |
| `attributes.gen_ai.agent.description` | `metadata["agent_description"]` | `"OpenAI Agents instrumentation"` |
| `attributes.gen_ai.request.model` | `metadata["model"]` | `"gpt-4o-mini-2024-07-18"` |
| `attributes.gen_ai.input.messages` (JSON) → `[0].parts[0].content` | `AgentInvocationSpan.user_prompt` | `"What is 15 multiplied by 37?"` |
| `attributes.gen_ai.output.messages` (JSON) → `[0].parts[0].content` | `AgentInvocationSpan.agent_response` | `"15 multiplied by 37 is 555."` |
| _(not present)_ | `AgentInvocationSpan.available_tools` | _(unavailable; only tool names on child spans)_ |
| _(not present)_ | `AgentInvocationSpan.system_prompt` | _(unavailable)_ |

## `chat` → `InferenceSpan`

| OTel JSON path | Evals field | Example value |
|---|---|---|
| `context.trace_id` | `span_info.trace_id` | `"0x87d999039e503b18371c24654b77bada"` |
| `context.span_id` | `span_info.span_id` | `"0x4f94640d38ca4060"` |
| `parent_id` | `span_info.parent_span_id` | `"0x767e7df577fbe94d"` |
| `start_time` | `span_info.start_time` | `"2026-07-21T23:41:30.666160Z"` |
| `end_time` | `span_info.end_time` | `"2026-07-21T23:41:32.288724Z"` |
| `attributes.gen_ai.request.model` | `InferenceSpan.metadata["model"]` | `"gpt-4o-mini-2024-07-18"` |
| `attributes.gen_ai.response.model` | `InferenceSpan.metadata["response_model"]` | `"gpt-4o-mini-2024-07-18"` |
| `attributes.gen_ai.provider.name` | `InferenceSpan.metadata["gen_ai.system"]` | `"openai"` |
| `attributes.gen_ai.response.id` | `InferenceSpan.metadata["response_id"]` | `"resp_01f16038ea4bb958006a6003ab443c819c993e2accd44eaf6e"` |
| `attributes.gen_ai.usage.input_tokens` | `InferenceSpan.metadata["input_tokens"]` | `97` |
| `attributes.gen_ai.usage.output_tokens` | `InferenceSpan.metadata["output_tokens"]` | `18` |
| `attributes.gen_ai.input.messages` (JSON string) | `InferenceSpan.messages` | user: `"What is 15 multiplied by 37?"` |
| `attributes.gen_ai.output.messages` (JSON string) | `InferenceSpan.messages` (assistant turn) | tool_call or text content |
| _(not present)_ | `InferenceSpan.metadata["finish_reasons"]` | _(unavailable)_ |
| _(not present)_ | `InferenceSpan.metadata["system_instructions"]` | _(unavailable)_ |
| _(not present)_ | `InferenceSpan.metadata["tool_definitions"]` | _(unavailable)_ |

## `execute_tool` → `ToolExecutionSpan`

| OTel JSON path | Evals field | Example value |
|---|---|---|
| `context.trace_id` | `span_info.trace_id` | `"0x87d999039e503b18371c24654b77bada"` |
| `context.span_id` | `span_info.span_id` | `"0x9ef70436ae808eb2"` |
| `parent_id` | `span_info.parent_span_id` | `"0x767e7df577fbe94d"` |
| `start_time` | `span_info.start_time` | `"2026-07-21T23:41:32.289276Z"` |
| `end_time` | `span_info.end_time` | `"2026-07-21T23:41:32.289787Z"` |
| `attributes.gen_ai.tool.name` | `ToolExecutionSpan.tool_call.name` | `"calculator"` |
| `attributes.gen_ai.tool.call.arguments` (JSON string) | `ToolExecutionSpan.tool_call.arguments` | `{"expression": "15 * 37"}` |
| `attributes.gen_ai.tool.call.result` | `ToolExecutionSpan.tool_result.content` | `"555"` |
| `attributes.gen_ai.tool.type` | `ToolExecutionSpan.metadata["tool_type"]` | `"function"` |
| _(not present)_ | `ToolExecutionSpan.tool_call.tool_call_id` | _(unavailable)_ |
| _(not present)_ | `ToolExecutionSpan.tool_result.tool_call_id` | _(unavailable)_ |
| _(not present)_ | `ToolExecutionSpan.tool_result.error` | _(infer from `status.status_code`)_ |

## `unknown` and `Agent workflow` (noise, filter during mapping)

Structural spans emitted by the instrumentor that carry no evaluation-relevant data. Identified by `attributes.gen_ai.operation.name == "unknown"` or span name `"Agent workflow"`. These wrap the `invoke_agent` and `chat`/`execute_tool` spans but contain only provider metadata (`gen_ai.system`, `server.address`, `server.port`). Discard during mapping.
