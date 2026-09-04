Defined in: [src/telemetry/tracer.ts:82](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/telemetry/tracer.ts#L82)

Execution trace for performance analysis. Tracks timing and hierarchy of operations within the agent loop. Fields default to null for JSON serialization compatibility.

## Implements

-   `JSONSerializable`<`AgentTraceData`\>

## Constructors

### Constructor

```ts
new AgentTrace(name, options?): AgentTrace;
```

Defined in: [src/telemetry/tracer.ts:106](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/telemetry/tracer.ts#L106)

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

Defined in: [src/telemetry/tracer.ts:84](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/telemetry/tracer.ts#L84)

Unique identifier (UUID) for this trace.

---

### name

```ts
readonly name: string;
```

Defined in: [src/telemetry/tracer.ts:86](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/telemetry/tracer.ts#L86)

Human-readable display name (e.g., “Cycle 1”, “Tool: calc”, “stream\_messages”).

---

### parentId

```ts
readonly parentId: string;
```

Defined in: [src/telemetry/tracer.ts:88](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/telemetry/tracer.ts#L88)

ID of the parent trace, if this trace is nested. Null for root traces.

---

### startTime

```ts
readonly startTime: number;
```

Defined in: [src/telemetry/tracer.ts:90](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/telemetry/tracer.ts#L90)

Start time in milliseconds since epoch.

---

### endTime

```ts
endTime: number = null;
```

Defined in: [src/telemetry/tracer.ts:92](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/telemetry/tracer.ts#L92)

End time in milliseconds since epoch. Null until trace is ended.

---

### duration

```ts
duration: number = 0;
```

Defined in: [src/telemetry/tracer.ts:94](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/telemetry/tracer.ts#L94)

Duration in milliseconds (endTime - startTime).

---

### children

```ts
readonly children: AgentTrace[] = [];
```

Defined in: [src/telemetry/tracer.ts:96](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/telemetry/tracer.ts#L96)

Child traces nested under this trace.

---

### metadata

```ts
readonly metadata: Record<string, string> = {};
```

Defined in: [src/telemetry/tracer.ts:98](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/telemetry/tracer.ts#L98)

Additional metadata for this trace (e.g., cycleId, toolUseId, toolName).

---

### message

```ts
message: Message = null;
```

Defined in: [src/telemetry/tracer.ts:100](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/telemetry/tracer.ts#L100)

Message associated with this trace (e.g., model output). Null if not applicable.

## Methods

### end()

```ts
end(endTime?): void;
```

Defined in: [src/telemetry/tracer.ts:120](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/telemetry/tracer.ts#L120)

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

Defined in: [src/telemetry/tracer.ts:125](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/telemetry/tracer.ts#L125)

#### Returns

`AgentTraceData`

#### Implementation of

```ts
JSONSerializable.toJSON
```