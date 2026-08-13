Defined in: [src/telemetry/meter.ts:166](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/telemetry/meter.ts#L166)

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

Defined in: [src/telemetry/meter.ts:213](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/telemetry/meter.ts#L213)

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

Defined in: [src/telemetry/meter.ts:170](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/telemetry/meter.ts#L170)

Total number of agent loop cycles executed across all invocations.

---

### accumulatedUsage

```ts
readonly accumulatedUsage: Usage;
```

Defined in: [src/telemetry/meter.ts:175](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/telemetry/meter.ts#L175)

Accumulated token usage across all model invocations.

---

### accumulatedMetrics

```ts
readonly accumulatedMetrics: Metrics;
```

Defined in: [src/telemetry/meter.ts:180](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/telemetry/meter.ts#L180)

Accumulated performance metrics across all model invocations.

---

### agentInvocations

```ts
readonly agentInvocations: InvocationMetricsData[];
```

Defined in: [src/telemetry/meter.ts:187](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/telemetry/meter.ts#L187)

Per-invocation metrics for recent invocations. Only the most recent 50 entries are retained to prevent unbounded memory growth. For full history, collect [latestAgentInvocation](#latestagentinvocation) from each [AgentResult](/docs/api/typescript/AgentResult/index.md).

---

### toolMetrics

```ts
readonly toolMetrics: Record<string, ToolMetricsData>;
```

Defined in: [src/telemetry/meter.ts:192](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/telemetry/meter.ts#L192)

Per-tool execution metrics keyed by tool name.

---

### latestContextSize

```ts
readonly latestContextSize: number;
```

Defined in: [src/telemetry/meter.ts:199](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/telemetry/meter.ts#L199)

The most recent input token count from the last model invocation. Represents the current context window utilization. Returns `undefined` when no invocations have occurred.

---

### projectedContextSize

```ts
readonly projectedContextSize: number;
```

Defined in: [src/telemetry/meter.ts:206](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/telemetry/meter.ts#L206)

Projected context size for the next model call (inputTokens + outputTokens from the last call). Represents the baseline token count the next invocation will start with. Returns `undefined` when no invocations have occurred.

---

### totalDuration

```ts
readonly totalDuration: number;
```

Defined in: [src/telemetry/meter.ts:211](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/telemetry/meter.ts#L211)

Total duration of all cycles across all invocations in milliseconds.

## Accessors

### latestAgentInvocation

#### Get Signature

```ts
get latestAgentInvocation(): InvocationMetricsData;
```

Defined in: [src/telemetry/meter.ts:227](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/telemetry/meter.ts#L227)

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

Defined in: [src/telemetry/meter.ts:234](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/telemetry/meter.ts#L234)

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
| `usage` | [`Usage`](/docs/api/typescript/Usage/index.md) | [src/telemetry/meter.ts:234](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/telemetry/meter.ts#L234) |
| `metrics` | [`Metrics`](/docs/api/typescript/Metrics/index.md) | [src/telemetry/meter.ts:234](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/telemetry/meter.ts#L234) |

---

### averageCycleTime

#### Get Signature

```ts
get averageCycleTime(): number;
```

Defined in: [src/telemetry/meter.ts:241](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/telemetry/meter.ts#L241)

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

Defined in: [src/telemetry/meter.ts:248](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/telemetry/meter.ts#L248)

Per-tool execution statistics with computed averages and rates.

##### Returns

`Record`<`string`, `ToolMetricsData` & { `averageTime`: `number`; `successRate`: `number`; }>

## Methods

### toJSON()

```ts
toJSON(): AgentMetricsData;
```

Defined in: [src/telemetry/meter.ts:266](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/telemetry/meter.ts#L266)

Returns a JSON-serializable representation of all collected metrics. Called automatically by JSON.stringify().

#### Returns

`AgentMetricsData`

A plain object suitable for round-trip serialization

#### Implementation of

```ts
JSONSerializable.toJSON
```