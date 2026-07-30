Defined in: [src/hooks/events.ts:319](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L319)

Event triggered after a tool execution completes. Fired after tool execution finishes, whether successful or failed. Uses reverse callback ordering for proper cleanup semantics.

Hook callbacks can mutate [result](#result) to rewrite the tool result before it propagates to the model (e.g. to redact or truncate output).

## Extends

-   [`HookableEvent`](/docs/api/typescript/HookableEvent/index.md)

## Constructors

### Constructor

```ts
new AfterToolCallEvent(data): AfterToolCallEvent;
```

Defined in: [src/hooks/events.ts:340](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L340)

#### Parameters

| Parameter | Type |
| --- | --- |
| `data` | { `agent`: `LocalAgent`; `toolUse`: [`ToolUseData`](/docs/api/typescript/ToolUseData/index.md); `tool`: [`Tool`](/docs/api/typescript/Tool/index.md); `result`: [`ToolResultBlock`](/docs/api/typescript/ToolResultBlock/index.md); `invocationState`: [`InvocationState`](/docs/api/typescript/InvocationState/index.md); `error?`: `Error`; } |
| `data.agent` | `LocalAgent` |
| `data.toolUse` | [`ToolUseData`](/docs/api/typescript/ToolUseData/index.md) |
| `data.tool` | [`Tool`](/docs/api/typescript/Tool/index.md) |
| `data.result` | [`ToolResultBlock`](/docs/api/typescript/ToolResultBlock/index.md) |
| `data.invocationState` | [`InvocationState`](/docs/api/typescript/InvocationState/index.md) |
| `data.error?` | `Error` |

#### Returns

`AfterToolCallEvent`

#### Overrides

[`HookableEvent`](/docs/api/typescript/HookableEvent/index.md).[`constructor`](/docs/api/typescript/HookableEvent/index.md#constructor)

## Properties

### type

```ts
readonly type: "afterToolCallEvent";
```

Defined in: [src/hooks/events.ts:320](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L320)

---

### agent

```ts
readonly agent: LocalAgent;
```

Defined in: [src/hooks/events.ts:321](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L321)

---

### toolUse

```ts
readonly toolUse: ToolUseData;
```

Defined in: [src/hooks/events.ts:322](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L322)

---

### tool

```ts
readonly tool: Tool;
```

Defined in: [src/hooks/events.ts:323](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L323)

---

### result

```ts
result: ToolResultBlock;
```

Defined in: [src/hooks/events.ts:329](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L329)

The tool result. Can be replaced by hook callbacks to transform the result before it enters the conversation history.

---

### error?

```ts
readonly optional error?: Error;
```

Defined in: [src/hooks/events.ts:331](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L331)

---

### invocationState

```ts
readonly invocationState: InvocationState;
```

Defined in: [src/hooks/events.ts:332](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L332)

---

### retry?

```ts
optional retry?: boolean;
```

Defined in: [src/hooks/events.ts:338](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L338)

Optional flag that can be set by hook callbacks to request a retry of the tool call. When set to true, the agent will re-execute the tool.

## Methods

### toJSON()

```ts
toJSON(): Pick<AfterToolCallEvent, "toolUse" | "type" | "result"> & {
  error?: {
     message?: string;
  };
};
```

Defined in: [src/hooks/events.ts:368](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L368)

Serializes for wire transport, excluding the agent reference, tool instance, invocationState, and mutable retry flag. Converts Error to an extensible object for safe wire serialization. Called automatically by JSON.stringify().

#### Returns

`Pick`<`AfterToolCallEvent`, `"toolUse"` | `"type"` | `"result"`\> & { `error?`: { `message?`: `string`; }; }