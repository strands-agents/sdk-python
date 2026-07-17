Defined in: [src/memory/extraction/types.ts:89](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/memory/extraction/types.ts#L89)

Context handed to [ExtractionTrigger.attach](/docs/api/typescript/ExtractionTrigger/index.md#attach) so a trigger can wire itself into the agent lifecycle and signal when extraction should run for its store.

## Properties

### agent

```ts
agent: LocalAgent;
```

Defined in: [src/memory/extraction/types.ts:91](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/memory/extraction/types.ts#L91)

The agent the trigger attaches its hooks to.

---

### fire

```ts
fire: () => void;
```

Defined in: [src/memory/extraction/types.ts:93](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/memory/extraction/types.ts#L93)

Save this store’s unsaved messages now. Runs in the background and returns immediately, so calling it from a hook never blocks the agent. To await completion, see [MemoryManager.flush](/docs/api/typescript/MemoryManager/index.md#flush).

#### Returns

`void`