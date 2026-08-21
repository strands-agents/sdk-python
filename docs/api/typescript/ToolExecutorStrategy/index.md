```ts
type ToolExecutorStrategy = "sequential" | "concurrent";
```

Defined in: [src/agent/agent.ts:150](https://github.com/strands-agents/harness-sdk/blob/f6765ab42296e23ac7af7c93e0ff3626fd09a60c/strands-ts/src/agent/agent.ts#L150)

Strategy for executing tool calls that the model emits in a single assistant turn.

-   `'concurrent'` (default) — runs all tool calls from a single turn in parallel. Per-tool event order (`BeforeToolCallEvent` → `ToolStreamUpdateEvent*` → `AfterToolCallEvent` → `ToolResultEvent`) is preserved, while cross-tool events may interleave.
-   `'sequential'` — runs tool calls one at a time

Cancellation works identically in both modes: [Agent.cancel](/docs/api/typescript/Agent/index.md#cancel) flips [Agent.cancelSignal](/docs/api/typescript/Agent/index.md#cancelsignal), which SDK-managed tool contexts expose as `context.cancelSignal`. In concurrent mode, prompt batch-wide cancellation requires every in-flight tool to honor the signal.