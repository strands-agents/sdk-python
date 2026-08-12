Defined in: [src/telemetry/tracer.ts:81](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/telemetry/tracer.ts#L81)

Execution trace for performance analysis. Tracks timing and hierarchy of operations within the agent loop. Fields default to null for JSON serialization compatibility.

## Implements

-   `JSONSerializable`<`AgentTraceData`\>

## Constructors

### Constructor

```ts
new AgentTrace(name, options?): AgentTrace;
```

Defined in: [src/telemetry/tracer.ts:105](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/telemetry/tracer.ts#L105)

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `name` | `string` | Display name for this trace |
| `options?` | { `parent?`: `AgentTrace`; `startTime?`: `number`; } | Optional configuration for parent and startTime |
| `options.parent?` | `AgentTrace` | \- |
| `options.startTime?` | `number` | \- |

#### Returns

`AgentTrace`

## Properties

### id

```ts
readonly id: string;
```

Defined in: [src/telemetry/tracer.ts:83](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/telemetry/tracer.ts#L83)

Unique identifier (UUID) for this trace.

---

### name

```ts
readonly name: string;
```

Defined in: [src/telemetry/tracer.ts:85](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/telemetry/tracer.ts#L85)

Human-readable display name (e.g., “Cycle 1”, “Tool: calc”, “stream\_messages”).

---

### parentId

```ts
readonly parentId: string;
```

Defined in: [src/telemetry/tracer.ts:87](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/telemetry/tracer.ts#L87)

ID of the parent trace, if this trace is nested. Null for root traces.

---

### startTime

```ts
readonly startTime: number;
```

Defined in: [src/telemetry/tracer.ts:89](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/telemetry/tracer.ts#L89)

Start time in milliseconds since epoch.

---

### endTime

```ts
endTime: number = null;
```

Defined in: [src/telemetry/tracer.ts:91](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/telemetry/tracer.ts#L91)

End time in milliseconds since epoch. Null until trace is ended.

---

### duration

```ts
duration: number = 0;
```

Defined in: [src/telemetry/tracer.ts:93](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/telemetry/tracer.ts#L93)

Duration in milliseconds (endTime - startTime).

---

### children

```ts
readonly children: AgentTrace[] = [];
```

Defined in: [src/telemetry/tracer.ts:95](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/telemetry/tracer.ts#L95)

Child traces nested under this trace.

---

### metadata

```ts
readonly metadata: Record<string, string> = {};
```

Defined in: [src/telemetry/tracer.ts:97](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/telemetry/tracer.ts#L97)

Additional metadata for this trace (e.g., cycleId, toolUseId, toolName).

---

### message

```ts
message: Message = null;
```

Defined in: [src/telemetry/tracer.ts:99](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/telemetry/tracer.ts#L99)

Message associated with this trace (e.g., model output). Null if not applicable.

## Methods

### end()

```ts
end(endTime?): void;
```

Defined in: [src/telemetry/tracer.ts:119](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/telemetry/tracer.ts#L119)

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `endTime?` | `number` | Optional end time in milliseconds since epoch |

#### Returns

`void`

---

### toJSON()

```ts
toJSON(): AgentTraceData;
```

Defined in: [src/telemetry/tracer.ts:124](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/telemetry/tracer.ts#L124)

#### Returns

`AgentTraceData`

#### Implementation of

```ts
JSONSerializable.toJSON
```