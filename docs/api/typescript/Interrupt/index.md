Defined in: [src/interrupt.ts:28](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/interrupt.ts#L28)

Represents an interrupt that can pause agent execution for human-in-the-loop workflows.

## Properties

### id

```ts
readonly id: string;
```

Defined in: [src/interrupt.ts:32](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/interrupt.ts#L32)

Unique identifier for this interrupt.

---

### name

```ts
readonly name: string;
```

Defined in: [src/interrupt.ts:37](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/interrupt.ts#L37)

User-defined name for the interrupt.

---

### reason?

```ts
readonly optional reason?: JSONValue;
```

Defined in: [src/interrupt.ts:42](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/interrupt.ts#L42)

User-provided reason for raising the interrupt.

---

### response?

```ts
optional response?: JSONValue;
```

Defined in: [src/interrupt.ts:47](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/interrupt.ts#L47)

Human response provided when resuming the agent after an interrupt.

---

### source

```ts
readonly source: InterruptSource;
```

Defined in: [src/interrupt.ts:54](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/interrupt.ts#L54)

Where this interrupt was raised from — a tool callback, an agent-level hook, or a multi-agent orchestrator hook. Always populated. When deserializing a snapshot produced by an older SDK that did not record this field, defaults to `'hook'`.

## Methods

### toJSON()

```ts
toJSON(): {
  id: string;
  name: string;
  reason?: JSONValue;
  response?: JSONValue;
  source: InterruptSource;
};
```

Defined in: [src/interrupt.ts:73](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/interrupt.ts#L73)

Serializes the interrupt to a JSON-compatible object.

#### Returns

```ts
{
  id: string;
  name: string;
  reason?: JSONValue;
  response?: JSONValue;
  source: InterruptSource;
}
```

| Name | Type | Defined in |
| --- | --- | --- |
| `id` | `string` | [src/interrupt.ts:73](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/interrupt.ts#L73) |
| `name` | `string` | [src/interrupt.ts:73](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/interrupt.ts#L73) |
| `reason?` | [`JSONValue`](/docs/api/typescript/JSONValue/index.md) | [src/interrupt.ts:73](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/interrupt.ts#L73) |
| `response?` | [`JSONValue`](/docs/api/typescript/JSONValue/index.md) | [src/interrupt.ts:73](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/interrupt.ts#L73) |
| `source` | [`InterruptSource`](/docs/api/typescript/InterruptSource/index.md) | [src/interrupt.ts:73](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/interrupt.ts#L73) |