Defined in: [src/models/routing/strategy.ts:16](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/models/routing/strategy.ts#L16)

Read-only inputs supplied to a routing strategy.

## Properties

### messages

```ts
readonly messages: Message[];
```

Defined in: [src/models/routing/strategy.ts:18](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/models/routing/strategy.ts#L18)

Fresh copy of the messages for this strategy ask.

---

### systemPrompt?

```ts
readonly optional systemPrompt?: SystemPrompt;
```

Defined in: [src/models/routing/strategy.ts:20](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/models/routing/strategy.ts#L20)

Fresh copy of the system prompt for this strategy ask.

---

### toolSpecs

```ts
readonly toolSpecs: ToolSpec[];
```

Defined in: [src/models/routing/strategy.ts:22](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/models/routing/strategy.ts#L22)

Fresh copy of tool specifications for this strategy ask.

---

### candidates

```ts
readonly candidates: readonly RoutingCandidate[];
```

Defined in: [src/models/routing/strategy.ts:24](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/models/routing/strategy.ts#L24)

Stable configured candidate instances.

---

### invocationState

```ts
readonly invocationState: Readonly<InvocationState>;
```

Defined in: [src/models/routing/strategy.ts:26](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/models/routing/strategy.ts#L26)

Live invocation state, exposed as read-only.

---

### attempts

```ts
readonly attempts: readonly RoutingAttempt[];
```

Defined in: [src/models/routing/strategy.ts:28](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/models/routing/strategy.ts#L28)

Chronological attempts made during this invocation.