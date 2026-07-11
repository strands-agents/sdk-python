```ts
const ExecuteToolStage: MiddlewareStage<ExecuteToolContext, ExecuteToolResult, AgentStreamEvent>;
```

Defined in: [src/middleware/stages.ts:161](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/middleware/stages.ts#L161)

Built-in stage wrapping individual tool execution. Middleware registered for this stage can add telemetry, validate inputs, or mock responses.