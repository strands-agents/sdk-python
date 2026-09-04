Defined in: [src/hooks/events.ts:107](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/hooks/events.ts#L107)

Mutable tool-use descriptor carried on tool-call hook events. Matches the shape of the tool use block the model emitted; hooks on [BeforeToolCallEvent](/docs/api/typescript/BeforeToolCallEvent/index.md) may mutate its fields (or reassign the object) to rewrite the input or tool name before the tool executes. The model-issued tool-use ID remains the authoritative provider correlation key.

## Properties

### name

```ts
name: string;
```

Defined in: [src/hooks/events.ts:108](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/hooks/events.ts#L108)

---

### toolUseId

```ts
toolUseId: string;
```

Defined in: [src/hooks/events.ts:109](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/hooks/events.ts#L109)

---

### input

```ts
input: JSONValue;
```

Defined in: [src/hooks/events.ts:110](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/hooks/events.ts#L110)