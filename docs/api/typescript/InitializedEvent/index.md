Defined in: [src/hooks/events.ts:117](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/hooks/events.ts#L117)

Event triggered when an agent has finished initialization. Fired after the agent has been fully constructed and all built-in components have been initialized.

## Extends

-   [`HookableEvent`](/docs/api/typescript/HookableEvent/index.md)

## Constructors

### Constructor

```ts
new InitializedEvent(data): InitializedEvent;
```

Defined in: [src/hooks/events.ts:121](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/hooks/events.ts#L121)

#### Parameters

| Parameter | Type |
| --- | --- |
| `data` | { `agent`: `LocalAgent`; } |
| `data.agent` | `LocalAgent` |

#### Returns

`InitializedEvent`

#### Overrides

[`HookableEvent`](/docs/api/typescript/HookableEvent/index.md).[`constructor`](/docs/api/typescript/HookableEvent/index.md#constructor)

## Properties

### type

```ts
readonly type: "initializedEvent";
```

Defined in: [src/hooks/events.ts:118](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/hooks/events.ts#L118)

---

### agent

```ts
readonly agent: LocalAgent;
```

Defined in: [src/hooks/events.ts:119](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/hooks/events.ts#L119)

## Methods

### toJSON()

```ts
toJSON(): Pick<InitializedEvent, "type">;
```

Defined in: [src/hooks/events.ts:130](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/hooks/events.ts#L130)

Serializes for wire transport, excluding the agent reference. Called automatically by JSON.stringify().

#### Returns

`Pick`<`InitializedEvent`, `"type"`\>