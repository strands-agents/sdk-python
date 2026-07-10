Defined in: [src/memory/extraction/triggers.ts:17](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/memory/extraction/triggers.ts#L17)

Runs extraction after every agent invocation.

The highest-fidelity option: nothing said in a turn is missed. Also the most expensive when an [Extractor](/docs/api/typescript/Extractor/index.md) is configured (a model call per turn) — for a server-side-extraction backend with no extractor it is just a per-turn write.

## Example

```typescript
extraction: { trigger: [new InvocationTrigger()] }
```

## Extends

-   [`ExtractionTrigger`](/docs/api/typescript/ExtractionTrigger/index.md)

## Constructors

### Constructor

```ts
new InvocationTrigger(): InvocationTrigger;
```

#### Returns

`InvocationTrigger`

#### Inherited from

[`ExtractionTrigger`](/docs/api/typescript/ExtractionTrigger/index.md).[`constructor`](/docs/api/typescript/ExtractionTrigger/index.md#constructor)

## Properties

### name

```ts
readonly name: "invocation" = 'invocation';
```

Defined in: [src/memory/extraction/triggers.ts:18](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/memory/extraction/triggers.ts#L18)

Stable identifier for this trigger kind, used in logging.

#### Overrides

[`ExtractionTrigger`](/docs/api/typescript/ExtractionTrigger/index.md).[`name`](/docs/api/typescript/ExtractionTrigger/index.md#name)

## Methods

### attach()

```ts
attach(context): void;
```

Defined in: [src/memory/extraction/triggers.ts:20](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/memory/extraction/triggers.ts#L20)

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