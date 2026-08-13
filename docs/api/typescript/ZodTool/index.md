Defined in: [src/tools/zod-tool.ts:52](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/tools/zod-tool.ts#L52)

Zod-based tool implementation. Extends Tool abstract class and implements InvokableTool interface.

## Extends

-   [`Tool`](/docs/api/typescript/Tool/index.md)

## Type Parameters

| Type Parameter | Default type |
| --- | --- |
| `TInput` *extends* `z.ZodType` | `undefined` | \- |
| `TReturn` | [`JSONValue`](/docs/api/typescript/JSONValue/index.md) |

## Implements

-   [`InvokableTool`](/docs/api/typescript/InvokableTool/index.md)<`ZodInferred`<`TInput`\>, `TReturn`\>

## Constructors

### Constructor

```ts
new ZodTool<TInput, TReturn>(config): ZodTool<TInput, TReturn>;
```

Defined in: [src/tools/zod-tool.ts:75](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/tools/zod-tool.ts#L75)

#### Parameters

| Parameter | Type |
| --- | --- |
| `config` | [`ZodToolConfig`](/docs/api/typescript/ZodToolConfig/index.md)<`TInput`, `TReturn`\> |

#### Returns

`ZodTool`<`TInput`, `TReturn`\>

#### Overrides

[`Tool`](/docs/api/typescript/Tool/index.md).[`constructor`](/docs/api/typescript/Tool/index.md#constructor)

## Accessors

### name

#### Get Signature

```ts
get name(): string;
```

Defined in: [src/tools/zod-tool.ts:117](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/tools/zod-tool.ts#L117)

The unique name of the tool.

##### Returns

`string`

The unique name of the tool. This MUST match the name in the toolSpec.

#### Implementation of

[`InvokableTool`](/docs/api/typescript/InvokableTool/index.md).[`name`](/docs/api/typescript/InvokableTool/index.md#name)

#### Overrides

[`Tool`](/docs/api/typescript/Tool/index.md).[`name`](/docs/api/typescript/Tool/index.md#name)

---

### description

#### Get Signature

```ts
get description(): string;
```

Defined in: [src/tools/zod-tool.ts:124](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/tools/zod-tool.ts#L124)

Human-readable description of what the tool does.

##### Returns

`string`

Human-readable description of what the tool does. This helps the model understand when to use the tool.

This MUST match the description in the toolSpec.description.

#### Implementation of

[`InvokableTool`](/docs/api/typescript/InvokableTool/index.md).[`description`](/docs/api/typescript/InvokableTool/index.md#description)

#### Overrides

[`Tool`](/docs/api/typescript/Tool/index.md).[`description`](/docs/api/typescript/Tool/index.md#description)

---

### toolSpec

#### Get Signature

```ts
get toolSpec(): ToolSpec;
```

Defined in: [src/tools/zod-tool.ts:131](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/tools/zod-tool.ts#L131)

OpenAPI JSON specification for the tool.

##### Returns

[`ToolSpec`](/docs/api/typescript/ToolSpec/index.md)

OpenAPI JSON specification for the tool. Defines the tool’s name, description, and input schema.

#### Implementation of

[`InvokableTool`](/docs/api/typescript/InvokableTool/index.md).[`toolSpec`](/docs/api/typescript/InvokableTool/index.md#toolspec)

#### Overrides

[`Tool`](/docs/api/typescript/Tool/index.md).[`toolSpec`](/docs/api/typescript/Tool/index.md#toolspec)

## Methods

### stream()

```ts
stream(toolContext): ToolStreamGenerator;
```

Defined in: [src/tools/zod-tool.ts:142](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/tools/zod-tool.ts#L142)

Executes the tool with streaming support. Delegates to internal FunctionTool implementation.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `toolContext` | [`ToolContext`](/docs/api/typescript/ToolContext/index.md) | Context information including the tool use request and invocation state |

#### Returns

[`ToolStreamGenerator`](/docs/api/typescript/ToolStreamGenerator/index.md)

Async generator that yields ToolStreamEvents and returns a ToolResultBlock

#### Implementation of

[`InvokableTool`](/docs/api/typescript/InvokableTool/index.md).[`stream`](/docs/api/typescript/InvokableTool/index.md#stream)

#### Overrides

[`Tool`](/docs/api/typescript/Tool/index.md).[`stream`](/docs/api/typescript/Tool/index.md#stream)

---

### invoke()

```ts
invoke(input, context?): Promise<TReturn>;
```

Defined in: [src/tools/zod-tool.ts:158](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/tools/zod-tool.ts#L158)

Invokes the tool directly with type-safe input and returns the unwrapped result.

Unlike stream(), this method:

-   Returns the raw result (not wrapped in ToolResult)
-   Consumes async generators and returns only the final value
-   Lets errors throw naturally (not wrapped in error ToolResult)

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `input` | `ZodInferred`<`TInput`\> | The input parameters for the tool |
| `context?` | [`ToolContext`](/docs/api/typescript/ToolContext/index.md) | Optional tool execution context |

#### Returns

`Promise`<`TReturn`\>

The unwrapped result

#### Implementation of

[`InvokableTool`](/docs/api/typescript/InvokableTool/index.md).[`invoke`](/docs/api/typescript/InvokableTool/index.md#invoke)