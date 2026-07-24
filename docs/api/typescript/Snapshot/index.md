Defined in: [src/types/snapshot.ts:20](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/types/snapshot.ts#L20)

Point-in-time capture of agent or orchestrator state.

## Properties

### scope

```ts
scope: Scope;
```

Defined in: [src/types/snapshot.ts:22](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/types/snapshot.ts#L22)

Scope identifying the snapshot context (agent or multi-agent).

---

### schemaVersion

```ts
schemaVersion: string;
```

Defined in: [src/types/snapshot.ts:24](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/types/snapshot.ts#L24)

Schema version string for forward compatibility.

---

### createdAt

```ts
createdAt: string;
```

Defined in: [src/types/snapshot.ts:26](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/types/snapshot.ts#L26)

ISO 8601 timestamp of when snapshot was created.

---

### data

```ts
data: Record<string, JSONValue>;
```

Defined in: [src/types/snapshot.ts:28](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/types/snapshot.ts#L28)

Framework-owned state data.

---

### appData

```ts
appData: Record<string, JSONValue>;
```

Defined in: [src/types/snapshot.ts:30](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/types/snapshot.ts#L30)

Application-owned data. Strands does not read or modify this.