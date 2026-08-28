Defined in: [src/injection/types.ts:22](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/injection/types.ts#L22)

The context an injection consumer receives on each model call, passed to `renderContent` and to a predicate [InjectionConfig.trigger](/docs/api/typescript/MemoryInjectionConfig/index.md#trigger).

## Properties

### messages

```ts
messages: MessageData[];
```

Defined in: [src/injection/types.ts:24](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/injection/types.ts#L24)

The current conversation, as data.

---

### appState

```ts
appState: StateStore;
```

Defined in: [src/injection/types.ts:26](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/injection/types.ts#L26)

Durable app state shared across calls, hooks, and tools — read what a tool stashed last turn.

---

### agent

```ts
agent: LocalAgent;
```

Defined in: [src/injection/types.ts:28](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/injection/types.ts#L28)

The agent the injection is attached to (escape hatch for advanced consumers).