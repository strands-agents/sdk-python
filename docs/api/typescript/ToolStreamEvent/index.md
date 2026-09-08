Defined in: [src/tools/tool.ts:73](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/tools/tool.ts#L73)

Event yielded during tool execution to report streaming progress. Tools can yield zero or more of these events before returning the final ToolResult.

## Example

```typescript
const streamEvent = new ToolStreamEvent({
  data: 'Processing step 1...'
})

// Or with structured data
const streamEvent = new ToolStreamEvent({
  data: { progress: 50, message: 'Halfway complete' }
})
```

## Implements

-   [`ToolStreamEventData`](/docs/api/typescript/ToolStreamEventData/index.md)

## Constructors

### Constructor

```ts
new ToolStreamEvent(eventData): ToolStreamEvent;
```

Defined in: [src/tools/tool.ts:85](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/tools/tool.ts#L85)

#### Parameters

| Parameter | Type |
| --- | --- |
| `eventData` | { `data?`: `unknown`; } |
| `eventData.data?` | `unknown` |

#### Returns

`ToolStreamEvent`

## Properties

### type

```ts
readonly type: "toolStreamEvent";
```

Defined in: [src/tools/tool.ts:77](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/tools/tool.ts#L77)

Discriminator for tool stream events.

#### Implementation of

[`ToolStreamEventData`](/docs/api/typescript/ToolStreamEventData/index.md).[`type`](/docs/api/typescript/ToolStreamEventData/index.md#type)

---

### data?

```ts
readonly optional data?: unknown;
```

Defined in: [src/tools/tool.ts:83](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/tools/tool.ts#L83)

Caller-provided data for the progress update. Can be any type of data the tool wants to report.

#### Implementation of

[`ToolStreamEventData`](/docs/api/typescript/ToolStreamEventData/index.md).[`data`](/docs/api/typescript/ToolStreamEventData/index.md#data)