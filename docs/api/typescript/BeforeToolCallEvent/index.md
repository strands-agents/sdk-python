Defined in: [src/hooks/events.ts:246](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L246)

Event triggered just before a tool is executed. Fired after tool lookup but before execution begins.

Hook callbacks can:

-   Set [cancel](#cancel) to prevent the tool from executing.
-   Set [selectedTool](#selectedtool) to execute a different tool in place of the registry’s match.
-   Mutate [toolUse](#tooluse) in place, or replace it wholesale, to rewrite the tool input or name before execution. If `name` is changed and `selectedTool` is not set, the tool is re-resolved from the registry under the new name.

## Extends

-   [`HookableEvent`](/docs/api/typescript/HookableEvent/index.md)

## Implements

-   `Interruptible`

## Constructors

### Constructor

```ts
new BeforeToolCallEvent(data): BeforeToolCallEvent;
```

Defined in: [src/hooks/events.ts:271](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L271)

#### Parameters

| Parameter | Type |
| --- | --- |
| `data` | { `agent`: `LocalAgent`; `toolUse`: [`ToolUseData`](/docs/api/typescript/ToolUseData/index.md); `tool`: [`Tool`](/docs/api/typescript/Tool/index.md); `invocationState`: [`InvocationState`](/docs/api/typescript/InvocationState/index.md); } |
| `data.agent` | `LocalAgent` |
| `data.toolUse` | [`ToolUseData`](/docs/api/typescript/ToolUseData/index.md) |
| `data.tool` | [`Tool`](/docs/api/typescript/Tool/index.md) |
| `data.invocationState` | [`InvocationState`](/docs/api/typescript/InvocationState/index.md) |

#### Returns

`BeforeToolCallEvent`

#### Overrides

[`HookableEvent`](/docs/api/typescript/HookableEvent/index.md).[`constructor`](/docs/api/typescript/HookableEvent/index.md#constructor)

## Properties

### type

```ts
readonly type: "beforeToolCallEvent";
```

Defined in: [src/hooks/events.ts:247](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L247)

---

### agent

```ts
readonly agent: LocalAgent;
```

Defined in: [src/hooks/events.ts:248](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L248)

---

### toolUse

```ts
toolUse: ToolUseData;
```

Defined in: [src/hooks/events.ts:249](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L249)

---

### tool

```ts
readonly tool: Tool;
```

Defined in: [src/hooks/events.ts:250](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L250)

---

### invocationState

```ts
readonly invocationState: InvocationState;
```

Defined in: [src/hooks/events.ts:251](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L251)

---

### cancel

```ts
cancel: string | boolean = false;
```

Defined in: [src/hooks/events.ts:258](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L258)

Set by hook callbacks to cancel this tool call. When set to `true`, a default cancel message is used. When set to a string, that string is used as the tool result error message.

---

### selectedTool

```ts
selectedTool: Tool = undefined;
```

Defined in: [src/hooks/events.ts:269](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L269)

Set by hook callbacks to execute a replacement tool instead of [tool](#tool). When undefined, the tool looked up from the registry (or re-resolved from a mutated `toolUse.name`) is used.

If multiple callbacks set `selectedTool`, the last callback to run wins. Callbacks run in registration order for this event, so the last-registered callback’s value is the one used.

## Methods

### interrupt()

```ts
interrupt<T>(params): T;
```

Defined in: [src/hooks/events.ts:292](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L292)

Raises an interrupt for human-in-the-loop workflows. If a response is available (from a previous resume), returns it immediately. Otherwise, throws an InterruptError to halt agent execution.

#### Type Parameters

| Type Parameter | Default type |
| --- | --- |
| `T` | [`JSONValue`](/docs/api/typescript/JSONValue/index.md) |

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `params` | [`InterruptParams`](/docs/api/typescript/InterruptParams/index.md) | Interrupt parameters including name and optional reason |

#### Returns

`T`

The user’s response when resuming from an interrupt

#### Implementation of

```ts
Interruptible.interrupt
```

---

### toJSON()

```ts
toJSON(): Pick<BeforeToolCallEvent, "type" | "toolUse">;
```

Defined in: [src/hooks/events.ts:306](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L306)

Serializes for wire transport, excluding the agent reference, tool instance, invocationState, and mutable cancel / selectedTool fields. Called automatically by JSON.stringify().

#### Returns

`Pick`<`BeforeToolCallEvent`, `"type"` | `"toolUse"`\>