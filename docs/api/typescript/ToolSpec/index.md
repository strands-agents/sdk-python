Defined in: [src/tools/types.ts:13](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/tools/types.ts#L13)

Specification for a tool that can be used by the model. Defines the tool’s name, description, and input schema.

## Properties

### name

```ts
name: string;
```

Defined in: [src/tools/types.ts:17](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/tools/types.ts#L17)

The unique name of the tool.

---

### description

```ts
description: string;
```

Defined in: [src/tools/types.ts:23](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/tools/types.ts#L23)

A description of what the tool does. This helps the model understand when to use the tool.

---

### inputSchema?

```ts
optional inputSchema?: JSONSchema7;
```

Defined in: [src/tools/types.ts:29](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/tools/types.ts#L29)

JSON Schema defining the expected input structure for the tool. If omitted, defaults to an empty object schema allowing no input parameters.

---

### outputSchema?

```ts
optional outputSchema?: JSONSchema7;
```

Defined in: [src/tools/types.ts:34](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/tools/types.ts#L34)

JSON Schema defining the expected output structure for the tool.

---

### annotations?

```ts
optional annotations?: Record<string, JSONValue | undefined>;
```

Defined in: [src/tools/types.ts:43](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/tools/types.ts#L43)

Untrusted tool-behavior hints (e.g. MCP `readOnlyHint`, `destructiveHint`); never a security boundary.

Not sent to model provider APIs. A missing key means unknown, not `false` — per MCP spec `destructiveHint` and `openWorldHint` default to `true` when absent — and the field is absent entirely for non-MCP tools.