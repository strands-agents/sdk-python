Defined in: [src/tools/tool.ts:70](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/tools/tool.ts#L70)

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

Defined in: [src/tools/tool.ts:82](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/tools/tool.ts#L82)

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

Defined in: [src/tools/tool.ts:74](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/tools/tool.ts#L74)

Discriminator for tool stream events.

#### Implementation of

[`ToolStreamEventData`](/docs/api/typescript/ToolStreamEventData/index.md).[`type`](/docs/api/typescript/ToolStreamEventData/index.md#type)

---

### data?

```ts
readonly optional data?: unknown;
```

Defined in: [src/tools/tool.ts:80](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/tools/tool.ts#L80)

Caller-provided data for the progress update. Can be any type of data the tool wants to report.

#### Implementation of

[`ToolStreamEventData`](/docs/api/typescript/ToolStreamEventData/index.md).[`data`](/docs/api/typescript/ToolStreamEventData/index.md#data)