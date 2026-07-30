```ts
type InjectionTrigger = "userTurn" | "everyTurn";
```

Defined in: [src/injection/types.ts:16](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/injection/types.ts#L16)

Determines when injection runs before a model call.

-   `'userTurn'`: only when the latest message is a fresh user ask (a `user` message with no tool result) — the common case for chat agents, where it keeps the user’s ask the final message the model sees.
-   `'everyTurn'`: before every model call, including mid-task tool-result turns — for autonomous agents that should consult injected context at each step.

For finer control, pass a predicate as [InjectionConfig.trigger](/docs/api/typescript/MemoryInjectionConfig/index.md#trigger) instead.