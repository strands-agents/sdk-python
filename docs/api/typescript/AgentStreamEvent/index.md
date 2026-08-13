```ts
type AgentStreamEvent =
  | ModelStreamUpdateEvent
  | ContentBlockEvent
  | ModelMessageEvent
  | ToolStreamUpdateEvent
  | ToolResultEvent
  | BeforeInvocationEvent
  | AfterInvocationEvent
  | BeforeModelCallEvent
  | AfterModelCallEvent
  | BeforeToolsEvent
  | AfterToolsEvent
  | BeforeToolCallEvent
  | AfterToolCallEvent
  | MessageAddedEvent
  | InterruptEvent
  | AgentResultEvent;
```

Defined in: [src/types/agent.ts:597](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/types/agent.ts#L597)

Union type representing all possible streaming events from an agent. This includes model events, tool events, and agent-specific lifecycle events.

This is a discriminated union where each event has a unique type field, allowing for type-safe event handling using switch statements.

Every member extends [HookableEvent](/docs/api/typescript/HookableEvent/index.md) (which extends [StreamEvent](/docs/api/typescript/StreamEvent/index.md)), making all events both streamable and subscribable via hook callbacks. Raw data objects from lower layers (model, tools) should be wrapped in a StreamEvent subclass at the agent boundary rather than added directly.