Defined in: [src/hooks/events.ts:89](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/hooks/events.ts#L89)

Base class for events that can be subscribed to via the hook system. Only events extending this class are dispatched to [HookRegistry](/docs/api/typescript/HookRegistry/index.md) callbacks. All current events extend this class. [StreamEvent](/docs/api/typescript/StreamEvent/index.md) exists as the base for potential future stream-only events that should not be hookable.

## Extends

-   [`StreamEvent`](/docs/api/typescript/StreamEvent/index.md)

## Extended by

-   [`InitializedEvent`](/docs/api/typescript/InitializedEvent/index.md)
-   [`BeforeInvocationEvent`](/docs/api/typescript/BeforeInvocationEvent/index.md)
-   [`AfterInvocationEvent`](/docs/api/typescript/AfterInvocationEvent/index.md)
-   [`MessageAddedEvent`](/docs/api/typescript/MessageAddedEvent/index.md)
-   [`BeforeToolCallEvent`](/docs/api/typescript/BeforeToolCallEvent/index.md)
-   [`AfterToolCallEvent`](/docs/api/typescript/AfterToolCallEvent/index.md)
-   [`BeforeModelCallEvent`](/docs/api/typescript/BeforeModelCallEvent/index.md)
-   [`AfterModelCallEvent`](/docs/api/typescript/AfterModelCallEvent/index.md)
-   [`BeforeToolsEvent`](/docs/api/typescript/BeforeToolsEvent/index.md)
-   [`AfterToolsEvent`](/docs/api/typescript/AfterToolsEvent/index.md)
-   [`ContentBlockEvent`](/docs/api/typescript/ContentBlockEvent/index.md)
-   [`ModelMessageEvent`](/docs/api/typescript/ModelMessageEvent/index.md)
-   [`ToolResultEvent`](/docs/api/typescript/ToolResultEvent/index.md)
-   [`ToolStreamUpdateEvent`](/docs/api/typescript/ToolStreamUpdateEvent/index.md)
-   [`AgentResultEvent`](/docs/api/typescript/AgentResultEvent/index.md)
-   [`InterruptEvent`](/docs/api/typescript/InterruptEvent/index.md)
-   [`ModelStreamUpdateEvent`](/docs/api/typescript/ModelStreamUpdateEvent/index.md)

## Constructors

### Constructor

```ts
new HookableEvent(): HookableEvent;
```

#### Returns

`HookableEvent`

#### Inherited from

[`StreamEvent`](/docs/api/typescript/StreamEvent/index.md).[`constructor`](/docs/api/typescript/StreamEvent/index.md#constructor)