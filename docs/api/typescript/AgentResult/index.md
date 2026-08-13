Defined in: [src/types/agent.ts:413](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/types/agent.ts#L413)

Result returned by the agent loop.

## Constructors

### Constructor

```ts
new AgentResult(data): AgentResult;
```

Defined in: [src/types/agent.ts:467](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/types/agent.ts#L467)

#### Parameters

| Parameter | Type |
| --- | --- |
| `data` | { `stopReason`: [`StopReason`](/docs/api/typescript/StopReason/index.md); `lastMessage`: [`Message`](/docs/api/typescript/Message/index.md); `invocationState`: [`InvocationState`](/docs/api/typescript/InvocationState/index.md); `traces?`: [`AgentTrace`](/docs/api/typescript/AgentTrace/index.md)\[\]; `metrics?`: [`AgentMetrics`](/docs/api/typescript/AgentMetrics/index.md); `structuredOutput?`: `output`<`ZodType`\>; `interrupts?`: [`Interrupt`](/docs/api/typescript/Interrupt/index.md)\[\]; `checkpoint?`: `Checkpoint`; } |
| `data.stopReason` | [`StopReason`](/docs/api/typescript/StopReason/index.md) |
| `data.lastMessage` | [`Message`](/docs/api/typescript/Message/index.md) |
| `data.invocationState` | [`InvocationState`](/docs/api/typescript/InvocationState/index.md) |
| `data.traces?` | [`AgentTrace`](/docs/api/typescript/AgentTrace/index.md)\[\] |
| `data.metrics?` | [`AgentMetrics`](/docs/api/typescript/AgentMetrics/index.md) |
| `data.structuredOutput?` | `output`<`ZodType`\> |
| `data.interrupts?` | [`Interrupt`](/docs/api/typescript/Interrupt/index.md)\[\] |
| `data.checkpoint?` | `Checkpoint` |

#### Returns

`AgentResult`

## Properties

### type

```ts
readonly type: "agentResult";
```

Defined in: [src/types/agent.ts:414](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/types/agent.ts#L414)

---

### stopReason

```ts
readonly stopReason: StopReason;
```

Defined in: [src/types/agent.ts:419](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/types/agent.ts#L419)

The stop reason from the final model response.

---

### lastMessage

```ts
readonly lastMessage: Message;
```

Defined in: [src/types/agent.ts:424](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/types/agent.ts#L424)

The last message added to the messages array.

---

### traces?

```ts
readonly optional traces?: AgentTrace[];
```

Defined in: [src/types/agent.ts:430](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/types/agent.ts#L430)

Local execution traces collected during the agent invocation. Contains timing and hierarchy of operations within the agent loop.

---

### structuredOutput?

```ts
readonly optional structuredOutput?: output<ZodType>;
```

Defined in: [src/types/agent.ts:436](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/types/agent.ts#L436)

The validated structured output from the LLM, if a schema was provided. Type represents any validated Zod schema output.

---

### metrics?

```ts
readonly optional metrics?: AgentMetrics;
```

Defined in: [src/types/agent.ts:442](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/types/agent.ts#L442)

Aggregated metrics for the agent’s loop execution. Tracks cycle counts, token usage, tool execution stats, and model latency.

---

### invocationState

```ts
readonly invocationState: InvocationState;
```

Defined in: [src/types/agent.ts:450](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/types/agent.ts#L450)

Per-invocation state passed into the agent, threaded through hooks and tools, and surfaced here at the end of the invocation. See [InvocationState](/docs/api/typescript/InvocationState/index.md) for details. Always defined — defaults to `{}` when no `invocationState` was provided in [InvokeOptions](/docs/api/typescript/InvokeOptions/index.md).

---

### interrupts?

```ts
readonly optional interrupts?: Interrupt[];
```

Defined in: [src/types/agent.ts:456](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/types/agent.ts#L456)

Interrupts that caused the agent to stop, when `stopReason` is `'interrupt'`. Contains the unanswered interrupts that require human input to resume.

---

### checkpoint?

```ts
readonly optional checkpoint?: Checkpoint;
```

Defined in: [src/types/agent.ts:465](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/types/agent.ts#L465)

**`Experimental`**

Checkpoint captured when the agent paused for durable execution. Populated only when `stopReason` is `'checkpoint'`. See the experimental checkpoint module for usage.

## Accessors

### contextSize

#### Get Signature

```ts
get contextSize(): number;
```

Defined in: [src/types/agent.ts:502](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/types/agent.ts#L502)

The most recent input token count from the last model invocation. Convenience accessor that delegates to `metrics.latestContextSize`. Returns `undefined` when no metrics or invocations are available.

##### Returns

`number`

---

### projectedContextSize

#### Get Signature

```ts
get projectedContextSize(): number;
```

Defined in: [src/types/agent.ts:511](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/types/agent.ts#L511)

Projected context size for the next model call (inputTokens + outputTokens from the last call). Convenience accessor that delegates to `metrics.projectedContextSize`. Returns `undefined` when no metrics or invocations are available.

##### Returns

`number`

## Methods

### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/types/agent.ts:525](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/types/agent.ts#L525)

Custom JSON serialization that excludes traces, metrics, and invocationState. Traces and metrics are excluded to avoid sending large payloads over the wire in API responses; `invocationState` is excluded because its values are caller-owned and may not be serializable (see [InvocationState](/docs/api/typescript/InvocationState/index.md)).

All three remain accessible via their properties for debugging.

#### Returns

`object`

Object representation for safe serialization

---

### toString()

```ts
toString(): string;
```

Defined in: [src/types/agent.ts:545](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/types/agent.ts#L545)

Extracts a string representation of the result.

Priority order:

1.  `interrupts` serialized as JSON, if any are present
2.  `structuredOutput` serialized as JSON
3.  Text from `textBlock`, `reasoningBlock`, and `citationsBlock` content blocks

#### Returns

`string`

String representation of the result: JSON for interrupts/structuredOutput, or text content joined by newlines.