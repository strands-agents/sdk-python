Defined in: [src/middleware/stages.ts:116](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/middleware/stages.ts#L116)

Context passed to tool-stage middleware. Contains everything needed to understand and potentially modify the tool call.

## Extends

-   [`MiddlewareInterruptible`](/docs/api/typescript/MiddlewareInterruptible/index.md)

## Properties

### agent

```ts
readonly agent: LocalAgent;
```

Defined in: [src/middleware/stages.ts:118](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/middleware/stages.ts#L118)

The agent instance (escape hatch for advanced use cases).

---

### tool

```ts
readonly tool: Tool;
```

Defined in: [src/middleware/stages.ts:120](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/middleware/stages.ts#L120)

The resolved tool implementation, or undefined if not found.

---

### toolUse

```ts
readonly toolUse: ToolUseData;
```

Defined in: [src/middleware/stages.ts:122](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/middleware/stages.ts#L122)

The tool use request (name, toolUseId, input).

---

### invocationState

```ts
readonly invocationState: InvocationState;
```

Defined in: [src/middleware/stages.ts:124](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/middleware/stages.ts#L124)

Per-invocation state. Shared by reference — mutations are visible to hooks, tools, and AgentResult.

---

### cancelSignal

```ts
readonly cancelSignal: AbortSignal;
```

Defined in: [src/middleware/stages.ts:126](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/middleware/stages.ts#L126)

Executor-owned cancellation signal for this tool call; middleware can observe but cannot replace it.

## Methods

### interrupt()

```ts
interrupt<T>(params): MiddlewareInterruptResult<T>;
```

Defined in: [src/middleware/stages.ts:50](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/middleware/stages.ts#L50)

Request a human-in-the-loop interrupt.

On first execution (no prior response), throws `InterruptError` to halt the agent. On resume (after the user provides a response), returns the response wrapped in `MiddlewareInterruptResult`.

#### Type Parameters

| Type Parameter | Default type |
| --- | --- |
| `T` | [`JSONValue`](/docs/api/typescript/JSONValue/index.md) |

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `params` | [`InterruptParams`](/docs/api/typescript/InterruptParams/index.md) | Interrupt parameters (name, optional reason, optional preemptive response) |

#### Returns

[`MiddlewareInterruptResult`](/docs/api/typescript/MiddlewareInterruptResult/index.md)<`T`\>

The user’s response wrapped in `{ response: T }`

#### Throws

InterruptError when no response has been provided yet

#### Inherited from

[`MiddlewareInterruptible`](/docs/api/typescript/MiddlewareInterruptible/index.md).[`interrupt`](/docs/api/typescript/MiddlewareInterruptible/index.md#interrupt)