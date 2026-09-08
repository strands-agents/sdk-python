```ts
type TakeSnapshotOptions = {
  preset?: SnapshotPreset;
  include?: SnapshotField[];
  exclude?: SnapshotField[];
  appData?: Record<string, JSONValue>;
};
```

Defined in: [src/agent/snapshot.ts:54](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/agent/snapshot.ts#L54)

Options for taking a snapshot of agent state.

## Properties

### preset?

```ts
optional preset?: SnapshotPreset;
```

Defined in: [src/agent/snapshot.ts:59](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/agent/snapshot.ts#L59)

Preset to use as the starting set of fields. If not specified, starts with an empty set (unless include is specified).

---

### include?

```ts
optional include?: SnapshotField[];
```

Defined in: [src/agent/snapshot.ts:64](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/agent/snapshot.ts#L64)

Fields to add to the snapshot. These are added to the preset fields (if any).

---

### exclude?

```ts
optional exclude?: SnapshotField[];
```

Defined in: [src/agent/snapshot.ts:69](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/agent/snapshot.ts#L69)

Fields to exclude from the snapshot. Applied after preset and include to filter out specific fields.

---

### appData?

```ts
optional appData?: Record<string, JSONValue>;
```

Defined in: [src/agent/snapshot.ts:74](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/agent/snapshot.ts#L74)

Application-owned data to store in the snapshot. Strands does not read or modify this data.