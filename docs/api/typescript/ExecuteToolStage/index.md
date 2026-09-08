```ts
const ExecuteToolStage: MiddlewareStage<ExecuteToolContext, ExecuteToolResult, AgentStreamEvent>;
```

Defined in: [src/middleware/stages.ts:175](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/middleware/stages.ts#L175)

Built-in stage wrapping individual tool execution. Middleware registered for this stage can add telemetry, validate inputs, or mock responses.