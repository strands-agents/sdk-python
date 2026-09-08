Defined in: [src/telemetry/meter.ts:167](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/telemetry/meter.ts#L167)

Read-only snapshot of aggregated agent metrics.

Returned by Meter.metrics and stored on [AgentResult](/docs/api/typescript/AgentResult/index.md). Provides access to cycle counts, tool usage, token consumption, and per-invocation breakdowns. Supports serialization via [toJSON](#tojson).

## Example

```typescript
const result = await agent.invoke('Hello')
console.log(result.metrics?.cycleCount)
console.log(result.metrics?.totalDuration)
console.log(result.metrics?.accumulatedData)
console.log(result.metrics?.toolMetrics)
console.log(JSON.stringify(result.metrics))
```

## Implements

-   `JSONSerializable`<`AgentMetricsData`\>

## Constructors

### Constructor

```ts
new AgentMetrics(data?): AgentMetrics;
```

Defined in: [src/telemetry/meter.ts:215](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/telemetry/meter.ts#L215)

#### Parameters

| Parameter | Type |
| --- | --- |
| `data?` | `Partial`<`AgentMetricsData`\> |

#### Returns

`AgentMetrics`

## Properties

### cycleCount

```ts
readonly cycleCount: number;
```

Defined in: [src/telemetry/meter.ts:171](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/telemetry/meter.ts#L171)

Total number of agent loop cycles executed across all invocations.

---

### accumulatedUsage

```ts
readonly accumulatedUsage: Usage;
```

Defined in: [src/telemetry/meter.ts:176](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/telemetry/meter.ts#L176)

Accumulated token usage across all model invocations.

---

### accumulatedMetrics

```ts
readonly accumulatedMetrics: Metrics;
```

Defined in: [src/telemetry/meter.ts:181](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/telemetry/meter.ts#L181)

Accumulated performance metrics across all model invocations.

---

### agentInvocations

```ts
readonly agentInvocations: InvocationMetricsData[];
```

Defined in: [src/telemetry/meter.ts:188](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/telemetry/meter.ts#L188)

Per-invocation metrics for recent invocations. Only the most recent 50 entries are retained to prevent unbounded memory growth. For full history, collect [latestAgentInvocation](#latestagentinvocation) from each [AgentResult](/docs/api/typescript/AgentResult/index.md).

---

### toolMetrics

```ts
readonly toolMetrics: Record<string, ToolMetricsData>;
```

Defined in: [src/telemetry/meter.ts:193](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/telemetry/meter.ts#L193)

Per-tool execution metrics keyed by tool name.

---

### latestContextSize

```ts
readonly latestContextSize: number;
```

Defined in: [src/telemetry/meter.ts:200](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/telemetry/meter.ts#L200)

The total prompt the model processed on the last invocation, including cached tokens. Represents the current context window utilization. Returns `undefined` when no invocations have occurred.

---

### projectedContextSize

```ts
readonly projectedContextSize: number;
```

Defined in: [src/telemetry/meter.ts:208](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/telemetry/meter.ts#L208)

Projected context size for the next model call (total prompt including cached tokens plus the generated output from the last call). Represents the baseline token count the next invocation will start with. Returns `undefined` when no invocations have occurred.

---

### totalDuration

```ts
readonly totalDuration: number;
```

Defined in: [src/telemetry/meter.ts:213](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/telemetry/meter.ts#L213)

Total duration of all cycles across all invocations in milliseconds.

## Accessors

### latestAgentInvocation

#### Get Signature

```ts
get latestAgentInvocation(): InvocationMetricsData;
```

Defined in: [src/telemetry/meter.ts:229](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/telemetry/meter.ts#L229)

The most recent agent invocation, or undefined if none exist.

##### Returns

`InvocationMetricsData`

---

### accumulatedData

#### Get Signature

```ts
get accumulatedData(): {
  usage: Usage;
  metrics: Metrics;
};
```

Defined in: [src/telemetry/meter.ts:236](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/telemetry/meter.ts#L236)

Accumulated usage and performance metrics across all model invocations.

##### Returns

```ts
{
  usage: Usage;
  metrics: Metrics;
}
```

| Name | Type | Defined in |
| --- | --- | --- |
| `usage` | [`Usage`](/docs/api/typescript/Usage/index.md) | [src/telemetry/meter.ts:236](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/telemetry/meter.ts#L236) |
| `metrics` | [`Metrics`](/docs/api/typescript/Metrics/index.md) | [src/telemetry/meter.ts:236](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/telemetry/meter.ts#L236) |

---

### averageCycleTime

#### Get Signature

```ts
get averageCycleTime(): number;
```

Defined in: [src/telemetry/meter.ts:243](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/telemetry/meter.ts#L243)

Average cycle duration in milliseconds, or 0 if no cycles exist.

##### Returns

`number`

---

### toolUsage

#### Get Signature

```ts
get toolUsage(): Record<string, ToolMetricsData & {
  averageTime: number;
  successRate: number;
}>;
```

Defined in: [src/telemetry/meter.ts:250](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/telemetry/meter.ts#L250)

Per-tool execution statistics with computed averages and rates.

##### Returns

`Record`<`string`, `ToolMetricsData` & { `averageTime`: `number`; `successRate`: `number`; }>

## Methods

### toJSON()

```ts
toJSON(): AgentMetricsData;
```

Defined in: [src/telemetry/meter.ts:268](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/telemetry/meter.ts#L268)

Returns a JSON-serializable representation of all collected metrics. Called automatically by JSON.stringify().

#### Returns

`AgentMetricsData`

A plain object suitable for round-trip serialization

#### Implementation of

```ts
JSONSerializable.toJSON
```