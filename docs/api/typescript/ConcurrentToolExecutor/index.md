Defined in: [src/tools/executors/concurrent.ts:28](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/tools/executors/concurrent.ts#L28)

Executes tool calls concurrently.

## Example

```typescript
import { Agent, ConcurrentToolExecutor } from '@strands-agents/sdk'

// The string shorthand keeps imports minimal (concurrent is also the default).
const agent = new Agent({ toolExecutor: 'concurrent' })

// Passing an instance is equivalent if you prefer to be explicit.
const explicitAgent = new Agent({ toolExecutor: new ConcurrentToolExecutor() })
```

## Extends

-   `ToolExecutor`

## Constructors

### Constructor

```ts
new ConcurrentToolExecutor(): ConcurrentToolExecutor;
```

#### Returns

`ConcurrentToolExecutor`

#### Inherited from

```ts
ToolExecutor.constructor
```

## Methods

### executeTool()

```ts
protected executeTool(
   options,
   toolUseBlock,
invocationState): AsyncGenerator<AgentStreamEvent, ToolResultBlock, undefined>;
```

Defined in: [src/tools/executors/executor.ts:75](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/tools/executors/executor.ts#L75)

#### Parameters

| Parameter | Type |
| --- | --- |
| `options` | `ToolExecutorOptions` |
| `toolUseBlock` | [`ToolUseBlock`](/docs/api/typescript/ToolUseBlock/index.md) |
| `invocationState` | [`InvocationState`](/docs/api/typescript/InvocationState/index.md) |

#### Returns

`AsyncGenerator`<[`AgentStreamEvent`](/docs/api/typescript/AgentStreamEvent/index.md), [`ToolResultBlock`](/docs/api/typescript/ToolResultBlock/index.md), `undefined`\>

#### Inherited from

```ts
ToolExecutor.executeTool
```

---

### \_storePendingToolExecution()

```ts
protected _storePendingToolExecution(
   options,
   assistantMessage,
   completedToolResults): void;
```

Defined in: [src/tools/executors/executor.ts:175](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/tools/executors/executor.ts#L175)

#### Parameters

| Parameter | Type |
| --- | --- |
| `options` | `ToolExecutorOptions` |
| `assistantMessage` | [`Message`](/docs/api/typescript/Message/index.md) |
| `completedToolResults` | `ReadonlyMap`<`string`, [`ToolResultBlock`](/docs/api/typescript/ToolResultBlock/index.md)\> |

#### Returns

`void`

#### Inherited from

```ts
ToolExecutor._storePendingToolExecution
```