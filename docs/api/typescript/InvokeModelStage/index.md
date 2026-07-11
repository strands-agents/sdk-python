```ts
const InvokeModelStage: MiddlewareStage<InvokeModelContext, InvokeModelResult, AgentStreamEvent>;
```

Defined in: [src/middleware/stages.ts:155](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/middleware/stages.ts#L155)

Built-in stage wrapping core model invocation. Middleware registered for this stage can rate-limit, cache, or transform model inputs.