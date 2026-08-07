```ts
type InterruptSource = "tool" | "hook" | "middleware" | "multiagent-hook";
```

Defined in: [src/interrupt.ts:23](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/interrupt.ts#L23)

Origin of an interrupt:

-   `'tool'` — raised by a tool callback via `ToolContext.interrupt()`.
-   `'hook'` — raised by an agent-level hook (e.g. `BeforeToolCallEvent.interrupt()`).
-   `'multiagent-hook'` — raised by a multi-agent hook (e.g. `BeforeNodeCallEvent.interrupt()`).