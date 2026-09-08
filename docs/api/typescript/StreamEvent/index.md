Defined in: [src/hooks/events.ts:81](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/hooks/events.ts#L81)

Base class for all events yielded by `agent.stream()`. Carries no hookability — subclasses that should be hookable extend [HookableEvent](/docs/api/typescript/HookableEvent/index.md) instead.

## Extended by

-   [`HookableEvent`](/docs/api/typescript/HookableEvent/index.md)

## Constructors

### Constructor

```ts
new StreamEvent(): StreamEvent;
```

#### Returns

`StreamEvent`