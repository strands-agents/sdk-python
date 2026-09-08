Defined in: [src/memory/extraction/types.ts:108](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/memory/extraction/types.ts#L108)

Controls when a store’s [ExtractionConfig](/docs/api/typescript/ExtractionConfig/index.md) runs.

A trigger is a self-attaching value object: [attach](#attach) wires whatever agent hooks the trigger needs and calls [ExtractionTriggerContext.fire](/docs/api/typescript/ExtractionTriggerContext/index.md#fire) when extraction should happen. Extend this class for custom triggering logic.

A trigger must eventually fire for its store’s buffered messages to be written: the high-water-mark dedup means skipped turns are still picked up on the *next* fire, but a trigger that never fires never extracts (and its messages stay buffered for the session). For a guaranteed final write at a boundary, the caller uses [MemoryManager.flush](/docs/api/typescript/MemoryManager/index.md#flush), which force-completes regardless of triggers.

## Extended by

-   [`InvocationTrigger`](/docs/api/typescript/InvocationTrigger/index.md)
-   [`IntervalTrigger`](/docs/api/typescript/IntervalTrigger/index.md)

## Constructors

### Constructor

```ts
new ExtractionTrigger(): ExtractionTrigger;
```

#### Returns

`ExtractionTrigger`

## Properties

### name

```ts
abstract readonly name: string;
```

Defined in: [src/memory/extraction/types.ts:110](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/memory/extraction/types.ts#L110)

Stable identifier for this trigger kind, used in logging.

## Methods

### attach()

```ts
abstract attach(context): void;
```

Defined in: [src/memory/extraction/types.ts:120](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/memory/extraction/types.ts#L120)

Wire this trigger into the agent lifecycle.

Called once per store during [MemoryManager](/docs/api/typescript/MemoryManager/index.md) initialization. Register hooks on `context.agent` and call `context.fire()` when extraction should run.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `context` | [`ExtractionTriggerContext`](/docs/api/typescript/ExtractionTriggerContext/index.md) | The agent to attach to and the fire callback bound to this trigger’s store |

#### Returns

`void`