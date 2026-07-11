Defined in: [src/agent/agent.ts:410](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L410)

Orchestrates the interaction between a model, a set of tools, and MCP clients. The Agent is responsible for managing the lifecycle of tools and clients and invoking the core decision-making loop.

## Implements

-   `InvokableAgent`

## Constructors

### Constructor

```ts
new Agent(config?): Agent;
```

Defined in: [src/agent/agent.ts:506](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L506)

Creates an instance of the Agent.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `config?` | [`AgentConfig`](/docs/api/typescript/AgentConfig/index.md) | The configuration for the agent. |

#### Returns

`Agent`

## Properties

### messages

```ts
messages: Message[];
```

Defined in: [src/agent/agent.ts:417](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L417)

The conversation history of messages between user and assistant.

#### Implementation of

```ts
LocalAgent.messages
```

---

### appState

```ts
readonly appState: StateStore;
```

Defined in: [src/agent/agent.ts:422](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L422)

App state storage accessible to tools and application logic. State is not passed to the model during inference.

#### Implementation of

```ts
LocalAgent.appState
```

---

### modelState

```ts
readonly modelState: StateStore;
```

Defined in: [src/agent/agent.ts:428](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L428)

Runtime state for the model provider. Used by stateful models to persist provider-specific data (e.g., response IDs for conversation chaining) across invocations.

#### Implementation of

```ts
LocalAgent.modelState
```

---

### model

```ts
model: Model;
```

Defined in: [src/agent/agent.ts:434](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L434)

The model provider used by the agent for inference.

#### Implementation of

```ts
LocalAgent.model
```

---

### systemPrompt?

```ts
optional systemPrompt?: SystemPrompt;
```

Defined in: [src/agent/agent.ts:439](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L439)

The system prompt to pass to the model provider.

#### Implementation of

```ts
LocalAgent.systemPrompt
```

---

### name

```ts
readonly name: string;
```

Defined in: [src/agent/agent.ts:444](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L444)

The name of the agent.

#### Implementation of

```ts
InvokableAgent.name
```

---

### id

```ts
readonly id: string;
```

Defined in: [src/agent/agent.ts:449](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L449)

The unique identifier of the agent instance.

#### Implementation of

```ts
LocalAgent.id
```

---

### description?

```ts
readonly optional description?: string;
```

Defined in: [src/agent/agent.ts:454](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L454)

Optional description of what the agent does.

#### Implementation of

```ts
InvokableAgent.description
```

---

### sessionManager?

```ts
readonly optional sessionManager?: SessionManager;
```

Defined in: [src/agent/agent.ts:459](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L459)

The session manager for saving and restoring agent sessions, if configured.

---

### memoryManager?

```ts
readonly optional memoryManager?: MemoryManager;
```

Defined in: [src/agent/agent.ts:463](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L463)

The memory manager for cross-session memory retrieval and storage, if configured.

---

### \_interruptState

```ts
_interruptState: InterruptState;
```

Defined in: [src/agent/agent.ts:494](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L494)

Interrupt state for human-in-the-loop workflows.

## Accessors

### sandbox

#### Get Signature

```ts
get sandbox(): Sandbox;
```

Defined in: [src/agent/agent.ts:473](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L473)

Execution environment for running commands, code, and file operations.

##### Throws

DefaultNotConfiguredError if no sandbox is configured for this environment (e.g. browsers, where no host default is registered).

##### Returns

[`Sandbox`](/docs/api/typescript/Sandbox/index.md)

#### Implementation of

```ts
LocalAgent.sandbox
```

---

### tools

#### Get Signature

```ts
get tools(): Tool[];
```

Defined in: [src/agent/agent.ts:900](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L900)

The tools this agent can use.

##### Returns

[`Tool`](/docs/api/typescript/Tool/index.md)\[\]

---

### toolRegistry

#### Get Signature

```ts
get toolRegistry(): ToolRegistry;
```

Defined in: [src/agent/agent.ts:907](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L907)

The tool registry for managing the agent’s tools.

##### Returns

`ToolRegistry`

#### Implementation of

```ts
LocalAgent.toolRegistry
```

---

### metrics

#### Get Signature

```ts
get metrics(): AgentMetrics;
```

Defined in: [src/agent/agent.ts:914](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L914)

Read-only snapshot of accumulated agent metrics (cycles, token usage, tool stats).

##### Returns

[`AgentMetrics`](/docs/api/typescript/AgentMetrics/index.md)

#### Implementation of

```ts
LocalAgent.metrics
```

---

### isInvoking

#### Get Signature

```ts
get isInvoking(): boolean;
```

Defined in: [src/agent/agent.ts:921](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L921)

Whether the agent is currently processing an invocation.

##### Returns

`boolean`

---

### tool

#### Get Signature

```ts
get tool(): ToolCallerProxy;
```

Defined in: [src/agent/agent.ts:942](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L942)

Direct tool calling accessor.

Returns a proxy where each property is a [ToolHandle](/docs/api/typescript/ToolHandle/index.md) with `.invoke()` and `.stream()` methods:

```typescript
const result = await agent.tool.calculator!.invoke({ a: 5, b: 3 })

for await (const event of agent.tool.calculator!.stream({ a: 5, b: 3 })) {
  console.log('progress:', event)
}
```

Supports underscore-to-hyphen and case-insensitive name resolution. Results are recorded in message history by default (pass `{ recordDirectToolCall: false }` to skip).

##### Returns

[`ToolCallerProxy`](/docs/api/typescript/ToolCallerProxy/index.md)

---

### cancelSignal

#### Get Signature

```ts
get cancelSignal(): AbortSignal;
```

Defined in: [src/agent/agent.ts:952](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L952)

The cancellation signal for the current invocation.

Tools can pass this to cancellable operations (e.g., `fetch(url, { signal: agent.cancelSignal })`). Hooks can check `event.agent.cancelSignal.aborted` to detect cancellation.

##### Returns

`AbortSignal`

#### Implementation of

```ts
LocalAgent.cancelSignal
```

## Methods

### addHook()

```ts
addHook<T>(
   eventType,
   callback,
   options?): HookCleanup;
```

Defined in: [src/agent/agent.ts:655](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L655)

Register a hook callback for a specific event type.

#### Type Parameters

| Type Parameter |
| --- |
| `T` *extends* [`HookableEvent`](/docs/api/typescript/HookableEvent/index.md) |

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `eventType` | [`HookableEventConstructor`](/docs/api/typescript/HookableEventConstructor/index.md)<`T`\> | The event class constructor to register the callback for |
| `callback` | [`HookCallback`](/docs/api/typescript/HookCallback/index.md)<`T`\> | The callback function to invoke when the event occurs |
| `options?` | [`HookCallbackOptions`](/docs/api/typescript/HookCallbackOptions/index.md) | Optional configuration including execution order |

#### Returns

`HookCleanup`

Cleanup function that removes the callback when invoked

#### Example

```typescript
const agent = new Agent({ model })

const cleanup = agent.addHook(BeforeInvocationEvent, (event) => {
  console.log('Invocation started')
})

// Later, to remove the hook:
cleanup()
```

#### Implementation of

```ts
LocalAgent.addHook
```

---

### addMiddleware()

#### Call Signature

```ts
addMiddleware<TContext, TResult, TEvent>(phase, handler): () => void;
```

Defined in: [src/agent/agent.ts:675](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L675)

Register an Input phase handler that transforms context before execution. Input handlers run before Wrap and Output handlers.

##### Type Parameters

| Type Parameter |
| --- |
| `TContext` |
| `TResult` |
| `TEvent` |

##### Parameters

| Parameter | Type |
| --- | --- |
| `phase` | `MiddlewareInputPhase`<`TContext`, `TResult`, `TEvent`\> |
| `handler` | [`MiddlewareInputHandler`](/docs/api/typescript/MiddlewareInputHandler/index.md)<`TContext`\> |

##### Returns

() => `void`

##### Example

```typescript
agent.addMiddleware(InvokeModelStage.Input, async (context) => ({
  ...context,
  systemPrompt: injectToSystemPrompt(context),
}))
```

##### Implementation of

```ts
LocalAgent.addMiddleware
```

#### Call Signature

```ts
addMiddleware<TContext, TResult, TEvent>(phase, handler): () => void;
```

Defined in: [src/agent/agent.ts:683](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L683)

Register a Wrap phase handler via the explicit `.Wrap` sub-token. Equivalent to passing the stage token directly.

##### Type Parameters

| Type Parameter |
| --- |
| `TContext` |
| `TResult` |
| `TEvent` |

##### Parameters

| Parameter | Type |
| --- | --- |
| `phase` | `MiddlewareWrapPhase`<`TContext`, `TResult`, `TEvent`\> |
| `handler` | [`MiddlewareHandler`](/docs/api/typescript/MiddlewareHandler/index.md)<`TContext`, `TResult`, `TEvent`\> |

##### Returns

() => `void`

##### Implementation of

```ts
LocalAgent.addMiddleware
```

#### Call Signature

```ts
addMiddleware<TContext, TResult, TEvent>(phase, handler): () => void;
```

Defined in: [src/agent/agent.ts:700](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L700)

Register an Output phase handler that transforms the result after execution. Output handlers see the result after Wrap handlers complete. Execution order: Input → Wrap → Output.

##### Type Parameters

| Type Parameter |
| --- |
| `TContext` |
| `TResult` |
| `TEvent` |

##### Parameters

| Parameter | Type |
| --- | --- |
| `phase` | `MiddlewareOutputPhase`<`TContext`, `TResult`, `TEvent`\> |
| `handler` | [`MiddlewareOutputHandler`](/docs/api/typescript/MiddlewareOutputHandler/index.md)<`TResult`\> |

##### Returns

() => `void`

##### Example

```typescript
agent.addMiddleware(InvokeModelStage.Output, async (result) => {
  log(`Model returned stopReason=${result.result.stopReason}`)
  return result
})
```

##### Implementation of

```ts
LocalAgent.addMiddleware
```

#### Call Signature

```ts
addMiddleware<TContext, TResult, TEvent>(stage, handler): () => void;
```

Defined in: [src/agent/agent.ts:725](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L725)

Register a middleware handler for a given stage (Wrap phase). Middleware wraps stage execution and can intercept, transform, or short-circuit operations.

##### Type Parameters

| Type Parameter |
| --- |
| `TContext` |
| `TResult` |
| `TEvent` |

##### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `stage` | [`MiddlewareStage`](/docs/api/typescript/MiddlewareStage/index.md)<`TContext`, `TResult`, `TEvent`\> | The stage token identifying the interception point |
| `handler` | [`MiddlewareHandler`](/docs/api/typescript/MiddlewareHandler/index.md)<`TContext`, `TResult`, `TEvent`\> | The middleware handler function (async generator) |

##### Returns

A cleanup function that removes the middleware when called

() => `void`

##### Example

```typescript
const cleanup = agent.addMiddleware(InvokeModelStage, async function* (context, next) {
  const start = Date.now()
  const result = yield* next(context)
  console.log(`Model call took ${Date.now() - start}ms`)
  return result
})

// Later, remove the middleware:
cleanup()
```

##### Implementation of

```ts
LocalAgent.addMiddleware
```

---

### initialize()

```ts
initialize(): Promise<void>;
```

Defined in: [src/agent/agent.ts:771](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L771)

#### Returns

`Promise`<`void`\>

---

### cancel()

```ts
cancel(): void;
```

Defined in: [src/agent/agent.ts:984](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L984)

Cancels the current agent invocation cooperatively.

The agent will stop at the next cancellation-safe point:

-   During model response streaming
-   Before tool execution
-   Between sequential tool executions
-   At the top of each agent loop cycle

If a tool is already executing, it will run to completion unless the tool checks LocalAgent.cancelSignal | cancelSignal internally.

Hook callbacks can check `event.agent.cancelSignal.aborted` to detect cancellation and adjust their behavior accordingly.

The stream/invoke call will return an AgentResult with `stopReason: 'cancelled'`. If the agent is not currently invoking, this is a no-op.

#### Returns

`void`

#### Example

```typescript
const agent = new Agent({ model, tools })

// Cancel after 5 seconds
setTimeout(() => agent.cancel(), 5000)
const result = await agent.invoke('Do something')
console.log(result.stopReason) // 'cancelled'
```

---

### invoke()

```ts
invoke(args, options?): Promise<AgentResult>;
```

Defined in: [src/agent/agent.ts:1016](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L1016)

Invokes the agent and returns the final result.

This is a convenience method that consumes the stream() method and returns only the final AgentResult. Use stream() if you need access to intermediate streaming events.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `args` | [`InvokeArgs`](/docs/api/typescript/InvokeArgs/index.md) | Arguments for invoking the agent |
| `options?` | [`InvokeOptions`](/docs/api/typescript/InvokeOptions/index.md) | Optional per-invocation options |

#### Returns

`Promise`<[`AgentResult`](/docs/api/typescript/AgentResult/index.md)\>

Promise that resolves to the final AgentResult

#### Example

```typescript
const agent = new Agent({ model, tools })
const result = await agent.invoke('What is 2 + 2?')
console.log(result.lastMessage) // Agent's response
```

#### Implementation of

```ts
InvokableAgent.invoke
```

---

### stream()

```ts
stream(args, options?): AsyncGenerator<AgentStreamEvent, AgentResult, undefined>;
```

Defined in: [src/agent/agent.ts:1055](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L1055)

Streams the agent execution, yielding events and returning the final result.

The agent loop manages the conversation flow by:

1.  Streaming model responses and yielding all events
2.  Executing tools when the model requests them
3.  Continuing the loop until the model completes without tool use

Use this method when you need access to intermediate streaming events. For simple request/response without streaming, use invoke() instead.

An explicit goal of this method is to always leave the message array in a way that the agent can be reinvoked with a user prompt after this method completes. To that end assistant messages containing tool uses are only added after tool execution succeeds with valid toolResponses

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `args` | [`InvokeArgs`](/docs/api/typescript/InvokeArgs/index.md) | Arguments for invoking the agent |
| `options?` | [`InvokeOptions`](/docs/api/typescript/InvokeOptions/index.md) | Optional per-invocation options |

#### Returns

`AsyncGenerator`<[`AgentStreamEvent`](/docs/api/typescript/AgentStreamEvent/index.md), [`AgentResult`](/docs/api/typescript/AgentResult/index.md), `undefined`\>

Async generator that yields AgentStreamEvent objects and returns AgentResult

#### Example

```typescript
const agent = new Agent({ model, tools })

for await (const event of agent.stream('Hello')) {
  console.log('Event:', event.type)
}
// Messages array is mutated in place and contains the full conversation
```

#### Implementation of

```ts
InvokableAgent.stream
```

---

### asTool()

```ts
asTool(options?): Tool;
```

Defined in: [src/agent/agent.ts:1290](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L1290)

Returns a [Tool](/docs/api/typescript/Tool/index.md) that wraps this agent, allowing it to be used as a tool by another agent.

The returned tool accepts a single `input` string parameter, invokes this agent, and returns the text response as a tool result.

**Note:** You can also pass an Agent directly in another agent’s [tools](/docs/api/typescript/AgentConfig/index.md#tools) array — it will be wrapped automatically via this method.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `options?` | [`AgentAsToolOptions`](/docs/api/typescript/AgentAsToolOptions/index.md) | Optional configuration for the tool name, description, and context preservation |

#### Returns

[`Tool`](/docs/api/typescript/Tool/index.md)

A Tool wrapping this agent

#### Example

```typescript
const researcher = new Agent({ name: 'researcher', description: 'Finds info', printer: false })

// Explicit wrapping
const writer = new Agent({ tools: [researcher.asTool()] })

// Automatic wrapping (equivalent)
const writer = new Agent({ tools: [researcher] })
```

---

### takeSnapshot()

```ts
takeSnapshot(options): Snapshot;
```

Defined in: [src/agent/agent.ts:1324](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L1324)

Captures a point-in-time snapshot of the agent’s current state.

Use snapshots to checkpoint agent state for later restoration, enabling use cases like undo/redo, branching conversations, and session persistence.

Fields are selected via a preset/include/exclude model:

1.  Start with preset fields (e.g. `'session'` captures all fields)
2.  Add any `include` fields
3.  Remove any `exclude` fields

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `options` | [`TakeSnapshotOptions`](/docs/api/typescript/TakeSnapshotOptions/index.md) | Controls which fields to capture and optional app data to store |

#### Returns

[`Snapshot`](/docs/api/typescript/Snapshot/index.md)

A [Snapshot](/docs/api/typescript/Snapshot/index.md) containing the captured agent state

#### Throws

Error if no fields would be included after applying options

#### Example

```typescript
// Capture all session-relevant state
const snapshot = agent.takeSnapshot({ preset: 'session' })

// Capture only messages and state
const partial = agent.takeSnapshot({ include: ['messages', 'state'] })

// Capture session state but exclude interrupts
const noInterrupts = agent.takeSnapshot({ preset: 'session', exclude: ['interrupts'] })

// Attach application-owned metadata
const withMeta = agent.takeSnapshot({ preset: 'session', appData: { userId: 'u-123' } })
```

#### Implementation of

```ts
LocalAgent.takeSnapshot
```

---

### loadSnapshot()

```ts
loadSnapshot(snapshot): void;
```

Defined in: [src/agent/agent.ts:1353](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/agent/agent.ts#L1353)

Restores agent state from a previously captured snapshot.

Only fields present in `snapshot.data` are restored; absent fields are left unchanged. This allows partial snapshots to update specific aspects of state without affecting others.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `snapshot` | [`Snapshot`](/docs/api/typescript/Snapshot/index.md) | The snapshot to restore from |

#### Returns

`void`

#### Throws

Error if `snapshot.schemaVersion` is incompatible or scope is wrong

#### Example

```typescript
// Save and restore a conversation checkpoint
const checkpoint = agent.takeSnapshot({ preset: 'session' })

// ... agent continues processing ...

// Restore to the checkpoint
agent.loadSnapshot(checkpoint)

// Restore from a JSON-serialized snapshot (e.g. from storage)
const stored = JSON.parse(savedSnapshotJson)
agent.loadSnapshot(stored)
```

#### Implementation of

```ts
LocalAgent.loadSnapshot
```