Defined in: [src/background-tasks/types.ts:6](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/background-tasks/types.ts#L6)

Configures background tool execution.

## Properties

### waitForCompletion?

```ts
readonly optional waitForCompletion?: boolean;
```

Defined in: [src/background-tasks/types.ts:8](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/background-tasks/types.ts#L8)

Wait for background work before an invocation returns. Defaults to `true`.

---

### agentic?

```ts
readonly optional agentic?: readonly (string | Tool)[];
```

Defined in: [src/background-tasks/types.ts:10](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/background-tasks/types.ts#L10)

Tools or registered tool names whose execution mode is selected by the model. Defaults to `['*']`.

---

### always?

```ts
readonly optional always?: readonly (string | Tool)[];
```

Defined in: [src/background-tasks/types.ts:12](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/background-tasks/types.ts#L12)

Tools or registered tool names that always execute in the background.

---

### never?

```ts
readonly optional never?: readonly (string | Tool)[];
```

Defined in: [src/background-tasks/types.ts:14](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/background-tasks/types.ts#L14)

Tools or registered tool names that never execute in the background.

---

### maxConcurrency?

```ts
readonly optional maxConcurrency?: number;
```

Defined in: [src/background-tasks/types.ts:16](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/background-tasks/types.ts#L16)

Maximum number of physically executing background tasks. Defaults to `4`.

---

### timeout?

```ts
readonly optional timeout?: number;
```

Defined in: [src/background-tasks/types.ts:18](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/background-tasks/types.ts#L18)

Per-execution timeout in milliseconds. Defaults to `Infinity`.