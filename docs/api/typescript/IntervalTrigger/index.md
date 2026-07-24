Defined in: [src/memory/extraction/triggers.ts:45](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/memory/extraction/triggers.ts#L45)

Runs extraction every N agent invocations.

A controllable middle ground: extraction (and any model call it entails) happens on a cadence rather than every turn, while the high-water mark guarantees the messages from the skipped turns are still processed when the trigger does fire.

## Example

```typescript
extraction: { trigger: [new IntervalTrigger({ turns: 5 })] }
```

## Extends

-   [`ExtractionTrigger`](/docs/api/typescript/ExtractionTrigger/index.md)

## Constructors

### Constructor

```ts
new IntervalTrigger(options): IntervalTrigger;
```

Defined in: [src/memory/extraction/triggers.ts:49](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/memory/extraction/triggers.ts#L49)

#### Parameters

| Parameter | Type |
| --- | --- |
| `options` | [`IntervalTriggerOptions`](/docs/api/typescript/IntervalTriggerOptions/index.md) |

#### Returns

`IntervalTrigger`

#### Overrides

[`ExtractionTrigger`](/docs/api/typescript/ExtractionTrigger/index.md).[`constructor`](/docs/api/typescript/ExtractionTrigger/index.md#constructor)

## Properties

### name

```ts
readonly name: "interval" = 'interval';
```

Defined in: [src/memory/extraction/triggers.ts:46](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/memory/extraction/triggers.ts#L46)

Stable identifier for this trigger kind, used in logging.

#### Overrides

[`ExtractionTrigger`](/docs/api/typescript/ExtractionTrigger/index.md).[`name`](/docs/api/typescript/ExtractionTrigger/index.md#name)

## Methods

### attach()

```ts
attach(context): void;
```

Defined in: [src/memory/extraction/triggers.ts:57](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/memory/extraction/triggers.ts#L57)

Wire this trigger into the agent lifecycle.

Called once per store during [MemoryManager](/docs/api/typescript/MemoryManager/index.md) initialization. Register hooks on `context.agent` and call `context.fire()` when extraction should run.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `context` | [`ExtractionTriggerContext`](/docs/api/typescript/ExtractionTriggerContext/index.md) | The agent to attach to and the fire callback bound to this trigger’s store |

#### Returns

`void`

#### Overrides

[`ExtractionTrigger`](/docs/api/typescript/ExtractionTrigger/index.md).[`attach`](/docs/api/typescript/ExtractionTrigger/index.md#attach)