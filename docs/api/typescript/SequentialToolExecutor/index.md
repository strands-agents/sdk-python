Defined in: [src/tools/executors/sequential.ts:23](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/tools/executors/sequential.ts#L23)

Executes tool calls one at a time.

## Example

```typescript
import { Agent, SequentialToolExecutor } from '@strands-agents/sdk'

// The string shorthand keeps imports minimal.
const agent = new Agent({ toolExecutor: 'sequential' })

// Passing an instance is equivalent if you prefer to be explicit.
const explicitAgent = new Agent({ toolExecutor: new SequentialToolExecutor() })
```

## Extends

-   `ToolExecutor`

## Constructors

### Constructor

```ts
new SequentialToolExecutor(): SequentialToolExecutor;
```

#### Returns

`SequentialToolExecutor`

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

Defined in: [src/tools/executors/executor.ts:75](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/tools/executors/executor.ts#L75)

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

Defined in: [src/tools/executors/executor.ts:175](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/tools/executors/executor.ts#L175)

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