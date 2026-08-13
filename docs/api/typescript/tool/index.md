Creates an InvokableTool from either a Zod schema or JSON schema configuration.

When a Zod schema is provided as `inputSchema`, input is validated at runtime and the callback receives typed input. When a JSON schema (or no schema) is provided, the callback receives `unknown` input with no runtime validation.

## Example

```typescript
import { tool } from '@strands-agents/sdk'
import { z } from 'zod'

// With Zod schema (typed + validated)
const calculator = tool({
  name: 'calculator',
  description: 'Adds two numbers',
  inputSchema: z.object({ a: z.number(), b: z.number() }),
  callback: (input) => input.a + input.b,
})

// With JSON schema (untyped, no validation)
const greeter = tool({
  name: 'greeter',
  description: 'Greets a person',
  inputSchema: {
    type: 'object',
    properties: { name: { type: 'string' } },
    required: ['name'],
  },
  callback: (input) => `Hello, ${(input as { name: string }).name}!`,
})
```

## Param

**config**

Tool configuration

## Call Signature

```ts
function tool<TInput, TReturn>(config): InvokableTool<infer<TInput>, TReturn>;
```

Defined in: [src/tools/tool-factory.ts:26](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/tools/tool-factory.ts#L26)

Creates an InvokableTool from a Zod schema and callback function.

### Type Parameters

| Type Parameter | Default type | Description |
| --- | --- | --- |
| `TInput` *extends* `ZodType` | \- | Zod schema type for input validation |
| `TReturn` | [`JSONValue`](/docs/api/typescript/JSONValue/index.md) | Return type of the callback function |

### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `config` | [`ZodToolConfig`](/docs/api/typescript/ZodToolConfig/index.md)<`TInput`, `TReturn`\> | Tool configuration with Zod schema |

### Returns

[`InvokableTool`](/docs/api/typescript/InvokableTool/index.md)<`infer`<`TInput`\>, `TReturn`\>

An InvokableTool with typed input and output

## Call Signature

```ts
function tool(config): InvokableTool<unknown, JSONValue>;
```

Defined in: [src/tools/tool-factory.ts:36](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/tools/tool-factory.ts#L36)

Creates an InvokableTool from a JSON schema and callback function.

### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `config` | [`FunctionToolConfig`](/docs/api/typescript/FunctionToolConfig/index.md) | Tool configuration with optional JSON schema |

### Returns

[`InvokableTool`](/docs/api/typescript/InvokableTool/index.md)<`unknown`, [`JSONValue`](/docs/api/typescript/JSONValue/index.md)\>

An InvokableTool with unknown input