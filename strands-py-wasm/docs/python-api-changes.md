# Python API Changes

Tracks all Python SDK API changes that result from the WASM bridge architecture. Each feature section documents the TypeScript SDK design, the WASM bridge implementation, and the resulting Python API change with code evidence.

---

## Conversation Manager

The Python conversation manager classes are config holders. The actual implementation runs inside the TypeScript SDK WASM guest.

### 1. Conversation manager is not accessible after construction

**TS design:** The agent stores the conversation manager as a private field.

```typescript
// strands-ts/src/agent/agent.ts:191
private readonly _conversationManager: ConversationManager
```

There is no public getter. Users configure it at construction and never access it again.

**WASM bridge:** The config is serialized through the WIT contract during agent construction. No handle to the TS conversation manager instance is retained on the Python side.

**Python API change:**

```python
# Standalone Python SDK (1.x) — worked
agent = Agent(conversation_manager=SlidingWindowConversationManager())
agent.conversation_manager  # accessible

# WASM bridged Python SDK (2.x) — not available
agent = Agent(conversation_manager=SlidingWindowConversationManager())
agent.conversation_manager  # AttributeError
```

Not needed. The conversation manager operates automatically via hooks registered during `initAgent()`.

### 2. No manual `reduce_context()` or `apply_management()`

**TS design:** Context reduction is hook driven. The base class registers an `AfterModelCallEvent` callback that catches overflow errors and calls `reduce()` automatically.

```typescript
// strands-ts/src/conversation-manager/conversation-manager.ts:100-108
initAgent(agent: LocalAgent): void {
    agent.addHook(AfterModelCallEvent, async (event) => {
      if (event.error instanceof ContextWindowOverflowError) {
        if (await this.reduce({ agent: event.agent, model: event.model, error: event.error })) {
          event.retry = true
        }
      }
    })
  }
```

`SlidingWindowConversationManager` adds proactive trimming via a second hook:

```typescript
// strands-ts/src/conversation-manager/sliding-window-conversation-manager.ts:72-78
public override initAgent(agent: LocalAgent): void {
    super.initAgent(agent)
    agent.addHook(AfterInvocationEvent, (event) => {
      this._applyManagement(event.agent.messages)
    })
  }
```

There are no public methods to trigger these manually. The hooks system is the invocation mechanism.

**WASM bridge:** `createConversationManager()` in `strands-wasm/entry.ts` instantiates the real TS class. The TS `Agent` constructor adds it to `PluginRegistry`, which calls `initAgent()`. Both hooks are registered inside the WASM guest.

**Python API change:**

```python
# Standalone Python SDK (1.x) — worked
cm = SlidingWindowConversationManager()
agent = Agent(conversation_manager=cm)
cm.reduce_context(agent)     # manually trigger reduction
cm.apply_management(agent)   # manually trigger window trimming

# WASM bridged Python SDK (2.x) — not available
cm = SlidingWindowConversationManager()
agent = Agent(conversation_manager=cm)
cm.reduce_context(agent)     # AttributeError — no such method
cm.apply_management(agent)   # AttributeError — no such method
```

Not needed. Overflow recovery fires automatically on `ContextWindowOverflowError`. Proactive trimming fires automatically after every invocation when messages exceed `windowSize`.

### 3. Summarization accepts a model config, not an agent

**TS design:** `SummarizingConversationManager` accepts a `model`, not an agent. Summarization calls the model directly.

```typescript
// strands-ts/src/conversation-manager/summarizing-conversation-manager.ts:46-51
export type SummarizingConversationManagerConfig = {
  model?: Model
  summaryRatio?: number
  preserveRecentMessages?: number
  summarizationSystemPrompt?: string
}
```

```typescript
// strands-ts/src/conversation-manager/summarizing-conversation-manager.ts:157-160
private async _generateSummary(messagesToSummarize: Message[], model: Model): Promise<Message> {
    // ...
    const stream = model.streamAggregated(summarizationMessages, {
      systemPrompt: this._summarizationSystemPrompt,
    })
```

**WASM bridge:** The Python user provides a model config dict. `createConversationManager()` in `strands-wasm/entry.ts` parses the JSON and calls `createModel()` to instantiate a TS model:

```typescript
// strands-wasm/entry.ts:427-430
if (cmConfig.summarizationModelConfig) {
    const parsed = JSON.parse(cmConfig.summarizationModelConfig)
    summaryModel = createModel(parsed)
}
```

**Python API change:**

```python
# Standalone Python SDK (1.x) — accepted a full Agent instance
summarizer = Agent(model=some_model, system_prompt="Summarize.")
agent = Agent(conversation_manager=SummarizingConversationManager(
    summarization_agent=summarizer,
))

# WASM bridged Python SDK (2.x) — accepts a model config dict
agent = Agent(conversation_manager=SummarizingConversationManager(
    summarization_model_config={
        "provider": "bedrock",
        "model_id": "us.anthropic.claude-3-haiku-20240307-v1:0",
    },
))
```

The WASM boundary cannot serialize a live `Agent` instance. The model config dict is instantiated as a TS model inside the guest, which matches the TS SDK's design of calling the model directly rather than re-entering the agent loop.

### 4. `per_turn` parameter not supported

**TS design:** `SlidingWindowConversationManager` does not implement `per_turn`. Proactive trimming runs unconditionally after every invocation via the `AfterInvocationEvent` hook when messages exceed `windowSize`.

**Python API change:**

```python
# Standalone Python SDK (1.x) — worked
agent = Agent(conversation_manager=SlidingWindowConversationManager(per_turn=3))

# WASM bridged Python SDK (2.x) — not supported
agent = Agent(conversation_manager=SlidingWindowConversationManager(per_turn=3))
# per_turn is silently ignored (caught by **_kwargs)
```

The TS SDK trims after every invocation when the window is exceeded, which is equivalent to `per_turn=True`.

### 5. Session state methods not available

**TS design:** The TS SDK has its own session management system. Conversation manager state persistence (`_summary_message`, `removed_message_count`) is not part of the `ConversationManager` interface.

**Python API change:**

```python
# Standalone Python SDK (1.x) — worked
state = cm.get_state()
cm.restore_from_session(state)
cm.removed_message_count

# WASM bridged Python SDK (2.x) — not available
```

---

## WIT Contract

The `conversation-manager-config` uses a flat record with a string `strategy` discriminator (`"none"`, `"sliding-window"`, `"summarizing"`) rather than a WIT variant. This works around a wasmtime-py limitation where `option<variant>` types are not properly supported.

```wit
record conversation-manager-config {
    strategy: string,
    window-size: s32,
    should-truncate-results: bool,
    summary-ratio: option<f64>,
    preserve-recent-messages: option<s32>,
    summarization-system-prompt: option<string>,
    summarization-model-config: option<string>,
}
```

Fields irrelevant to the selected strategy are set to zero values or `None`.

---

## Python Config Reference

### `NullConversationManager`

No parameters. Disables conversation management. Overflow errors propagate uncaught.

### `SlidingWindowConversationManager`

| Parameter | Type | Default | TS equivalent |
|---|---|---|---|
| `window_size` | `int` | `40` | `windowSize` |
| `should_truncate_results` | `bool` | `True` | `shouldTruncateResults` |

### `SummarizingConversationManager`

| Parameter | Type | Default | TS equivalent |
|---|---|---|---|
| `summary_ratio` | `float` | `0.3` | `summaryRatio` (clamped 0.1 to 0.8) |
| `preserve_recent_messages` | `int` | `10` | `preserveRecentMessages` |
| `summarization_system_prompt` | `str \| None` | `None` | `summarizationSystemPrompt` |
| `summarization_model_config` | `dict \| None` | `None` | Serialized to JSON, parsed by TS guest, passed to `createModel()` to produce a `Model` for `config.model` |

Model config dict format:

```python
{
    "provider": "bedrock",       # "bedrock", "anthropic", "openai", or "gemini"
    "model_id": "us.anthropic.claude-3-haiku-20240307-v1:0",
    "region": "us-west-2",       # bedrock only
    "api_key": "...",            # anthropic, openai, gemini only
}
```

---

## Structured Output

### Overview

Structured output validation runs inside the TypeScript SDK WASM guest using `StructuredOutputTool`. Python sends a flattened JSON schema through the WIT contract. The TS agent loop registers the tool, handles force-retry when the model doesn't call it, validates with Zod, and returns the validated JSON on the stop event. Python instantiates the Pydantic model from the validated JSON.

### 1. Parameter name changed

**TS design:** The TS Agent accepts `structuredOutputSchema` (a Zod schema) on `AgentConfig` and `InvokeOptions`.

```typescript
// strands-ts/src/types/agent.ts:165
structuredOutputSchema?: z.ZodSchema
```

**WASM bridge:** Python sends the JSON schema as a string through `structured-output-schema` on `agent-config` and `stream-args`. `entry.ts` reconstructs a Zod schema via `z.fromJSONSchema()`.

**Python API change:**

```python
# Standalone Python SDK (1.x)
agent = Agent(structured_output_model=MyModel)
result = agent("prompt", structured_output_model=MyModel)

# WASM bridged Python SDK (2.x)
agent = Agent(structured_output=MyModel)
result = agent("prompt", structured_output=MyModel)
```

### 2. Tool name is fixed

**TS design:** `StructuredOutputTool` uses a fixed name defined in `strands-ts/src/tools/structured-output-tool.ts:9`:

```typescript
export const STRUCTURED_OUTPUT_TOOL_NAME = 'strands_structured_output'
```

**Python API change:**

```python
# Standalone Python SDK (1.x) — tool name was the Pydantic model class name
# Conversation history shows: toolUse name="MyModel"

# WASM bridged Python SDK (2.x) — fixed tool name
# Conversation history shows: toolUse name="strands_structured_output"
```

This does not affect user code. The tool name appears in conversation history but users do not reference it directly.

### 3. Force-retry uses toolChoice, not a user message

**TS design:** When the model stops without calling the structured output tool, the TS agent loop sets `toolChoice: { tool: { name: 'strands_structured_output' } }` and continues the loop (`strands-ts/src/agent/agent.ts:771`). No user message is appended.

**Python API change:**

```python
# Standalone Python SDK (1.x) — customizable force prompt
agent = Agent()
result = agent("prompt", structured_output_model=MyModel, structured_output_prompt="Format as MyModel now.")

# WASM bridged Python SDK (2.x) — structured_output_prompt not available
agent = Agent()
result = agent("prompt", structured_output=MyModel)
# Force-retry handled automatically by TS SDK via toolChoice
```

The TS mechanism is more reliable because it forces the model to call the specific tool rather than relying on a text prompt.

### 4. Validation runs in Zod, instantiation in Pydantic

**TS design:** `StructuredOutputTool.stream()` validates via `this._schema.parse(toolUse.input)` (Zod) at `strands-ts/src/tools/structured-output-tool.ts:59`. On success, returns a `JsonBlock` with the validated data. On error, returns the error message for LLM retry.

**WASM bridge:** The validated JSON returns through `stop-data.structured-output` as a JSON string. Python instantiates the Pydantic model from it: `so_model(**json.loads(json_str))`.

**Python API change:**

```python
# Both versions — same user experience
result = agent("Describe a person", structured_output=Person)
result.structured_output  # Person(name="Alice", age=30) — Pydantic instance
```

Validation errors from Zod trigger model retry inside the TS guest (the model sees the error message and corrects its output). Custom Pydantic `@field_validator` logic that goes beyond JSON schema constraints cannot be enforced by Zod. Zod validates schema structure; Pydantic adds business logic on instantiation.

### 5. Deprecated method removed

**TS design:** No equivalent to `agent.structured_output(model, prompt)`. Structured output is configured via the constructor or per-invocation options.

**Python API change:**

```python
# Standalone Python SDK (1.x) — deprecated method
result = agent.structured_output(MyModel, "prompt")

# WASM bridged Python SDK (2.x) — use standard invocation
result = agent("prompt", structured_output=MyModel)
# Or via helper method:
result = agent.structured_output(MyModel, "prompt")  # still available, delegates to above
```

### 6. Error on force failure

**TS design:** Throws `StructuredOutputError` (`strands-ts/src/errors.ts:203`) with message `"The model failed to invoke the structured output tool even after it was forced."`.

**Python API change:**

```python
# Standalone Python SDK (1.x)
from strands.types.exceptions import StructuredOutputException

# WASM bridged Python SDK (2.x)
from strands.types.exceptions import StructuredOutputError
```

The exception is raised when the TS SDK's error message is detected in the stream.
