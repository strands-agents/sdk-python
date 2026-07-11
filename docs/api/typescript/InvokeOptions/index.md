Defined in: [src/types/agent.ts:92](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/types/agent.ts#L92)

Options for a single agent invocation.

## Properties

### structuredOutputSchema?

```ts
optional structuredOutputSchema?: ZodSchema;
```

Defined in: [src/types/agent.ts:96](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/types/agent.ts#L96)

Zod schema for structured output validation, overriding the constructor-provided schema for this invocation only.

---

### invocationState?

```ts
optional invocationState?: InvocationState;
```

Defined in: [src/types/agent.ts:105](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/types/agent.ts#L105)

Per-invocation state. Passed to lifecycle hook events and tools, and returned on [AgentResult.invocationState](/docs/api/typescript/AgentResult/index.md#invocationstate). Mutable — hooks and tools may read and write. See [InvocationState](/docs/api/typescript/InvocationState/index.md) for details.

Defaults to an empty object when omitted.

---

### cancelSignal?

```ts
optional cancelSignal?: AbortSignal;
```

Defined in: [src/types/agent.ts:135](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/types/agent.ts#L135)

External AbortSignal for cancelling the agent invocation.

Use this when cancellation is driven by something outside the agent — for example, a client disconnect, a framework-managed request lifecycle, or a declarative timeout. The agent composes this signal with its own internal controller, so both `agent.cancel()` and this signal can trigger cancellation independently.

When the signal fires, the agent stops at the next cancellation checkpoint and returns an AgentResult with `stopReason: 'cancelled'`. See LocalAgent.cancelSignal for how tools can participate in cancellation.

#### Example

```typescript
// Timeout-based cancellation
const result = await agent.invoke('Hello', {
  cancelSignal: AbortSignal.timeout(5000),
})

// Framework-driven cancellation (e.g., client disconnect)
app.post('/chat', async (req, res) => {
  const result = await agent.invoke(req.body.message, {
    cancelSignal: req.signal,
  })
  res.json(result)
})
```

---

### limits?

```ts
optional limits?: {
  turns?: number;
  outputTokens?: number;
  totalTokens?: number;
};
```

Defined in: [src/types/agent.ts:153](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/types/agent.ts#L153)

Per-invocation budget caps. Each cap, when set, bounds the agent loop for this `invoke()` / `stream()` call only — counters are not cumulative across reuses of the same agent.

Caps are checked at the top of each loop iteration. Tools requested by the previous turn always run to completion before a cap fires, so `agent.messages` remains in a reinvokable state.

Each cap, when set, must be a positive finite number. Omit any field (or `limits` itself) for no limit on that dimension.

Priority on simultaneous trip (highest first): `turns`, `totalTokens`, `outputTokens`. The corresponding `stopReason` is `'limitTurns'`, `'limitTotalTokens'`, or `'limitOutputTokens'`.

#### turns?

```ts
optional turns?: number;
```

Maximum number of agent loop iterations (turns). A turn is one model call plus any tool execution that follows. Counted against `metrics.latestAgentInvocation.cycles.length`.

#### outputTokens?

```ts
optional outputTokens?: number;
```

Maximum cumulative model-generated tokens, summed across every model call in the agent loop (`metrics.latestAgentInvocation.usage.outputTokens`).

Distinct from per-call provider-level `maxTokens` settings (e.g. `GoogleModelConfig.params.maxOutputTokens`), which bound a single model call’s output. This cap bounds the loop’s cumulative output across however many calls it makes.

Soft cap: a single oversized model response can overshoot the budget. The agent stops at the first turn boundary on or after the budget is reached; it does not bound any individual model call.

#### totalTokens?

```ts
optional totalTokens?: number;
```

Maximum cumulative input + output tokens (`metrics.latestAgentInvocation.usage.totalTokens`). Each model call’s input includes prior turns, so this counter compounds across the run — it approximates the total token spend you would be billed for.

Soft cap: a single oversized model response can overshoot the budget. The agent stops at the first turn boundary on or after the budget is reached; it does not bound any individual model call.