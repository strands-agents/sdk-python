Defined in: [src/tools/zod-tool.ts:20](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/tools/zod-tool.ts#L20)

Configuration for creating a Zod-based tool.

## Type Parameters

| Type Parameter | Default type | Description |
| --- | --- | --- |
| `TInput` *extends* `z.ZodType` | `undefined` | \- | Zod schema type for input validation |
| `TReturn` | [`JSONValue`](/docs/api/typescript/JSONValue/index.md) | Return type of the callback function |

## Properties

### name

```ts
name: string;
```

Defined in: [src/tools/zod-tool.ts:22](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/tools/zod-tool.ts#L22)

The name of the tool

---

### description?

```ts
optional description?: string;
```

Defined in: [src/tools/zod-tool.ts:25](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/tools/zod-tool.ts#L25)

A description of what the tool does (optional)

---

### inputSchema?

```ts
optional inputSchema?: TInput;
```

Defined in: [src/tools/zod-tool.ts:31](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/tools/zod-tool.ts#L31)

Zod schema for input validation and JSON schema generation. If omitted or z.void(), the tool takes no input parameters.

---

### callback

```ts
callback: (input, context?) =>
  | TReturn
  | AsyncGenerator<unknown, TReturn, never>
| Promise<TReturn>;
```

Defined in: [src/tools/zod-tool.ts:42](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/tools/zod-tool.ts#L42)

Callback function that implements the tool’s functionality.

The return value must be JSON-serializable.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `input` | `ZodInferred`<`TInput`\> | Validated input matching the Zod schema |
| `context?` | [`ToolContext`](/docs/api/typescript/ToolContext/index.md) | Optional execution context |

#### Returns

| `TReturn` | `AsyncGenerator`<`unknown`, `TReturn`, `never`\> | `Promise`<`TReturn`\>

The result (can be a value, Promise, or AsyncGenerator)