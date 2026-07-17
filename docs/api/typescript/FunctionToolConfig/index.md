Defined in: [src/tools/function-tool.ts:60](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/tools/function-tool.ts#L60)

Configuration options for creating a FunctionTool.

## Properties

### name

```ts
name: string;
```

Defined in: [src/tools/function-tool.ts:62](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/tools/function-tool.ts#L62)

The unique name of the tool

---

### description

```ts
description: string;
```

Defined in: [src/tools/function-tool.ts:64](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/tools/function-tool.ts#L64)

Human-readable description of the tool’s purpose

---

### inputSchema?

```ts
optional inputSchema?: JSONSchema7;
```

Defined in: [src/tools/function-tool.ts:66](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/tools/function-tool.ts#L66)

JSON Schema defining the expected input structure. If omitted, defaults to an empty object schema.

---

### callback

```ts
callback: FunctionToolCallback;
```

Defined in: [src/tools/function-tool.ts:68](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/tools/function-tool.ts#L68)

Function that implements the tool logic