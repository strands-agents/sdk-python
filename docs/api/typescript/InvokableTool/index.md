Defined in: [src/tools/tool.ts:166](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/tools/tool.ts#L166)

Extended tool interface that supports direct invocation with type-safe input and output. This interface is useful for testing and standalone tool execution.

## Extends

-   [`Tool`](/docs/api/typescript/Tool/index.md)

## Type Parameters

| Type Parameter | Description |
| --- | --- |
| `TInput` | Type for the tool’s input parameters |
| `TReturn` | Type for the tool’s return value |

## Properties

### name

```ts
abstract name: string;
```

Defined in: [src/tools/tool.ts:109](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/tools/tool.ts#L109)

The unique name of the tool. This MUST match the name in the toolSpec.

#### Inherited from

[`Tool`](/docs/api/typescript/Tool/index.md).[`name`](/docs/api/typescript/Tool/index.md#name)

---

### description

```ts
abstract description: string;
```

Defined in: [src/tools/tool.ts:116](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/tools/tool.ts#L116)

Human-readable description of what the tool does. This helps the model understand when to use the tool.

This MUST match the description in the toolSpec.description.

#### Inherited from

[`Tool`](/docs/api/typescript/Tool/index.md).[`description`](/docs/api/typescript/Tool/index.md#description)

---

### toolSpec

```ts
abstract toolSpec: ToolSpec;
```

Defined in: [src/tools/tool.ts:121](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/tools/tool.ts#L121)

OpenAPI JSON specification for the tool. Defines the tool’s name, description, and input schema.

#### Inherited from

[`Tool`](/docs/api/typescript/Tool/index.md).[`toolSpec`](/docs/api/typescript/Tool/index.md#toolspec)

## Methods

### stream()

```ts
abstract stream(toolContext): ToolStreamGenerator;
```

Defined in: [src/tools/tool.ts:156](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/tools/tool.ts#L156)

Executes the tool with streaming support. Yields zero or more ToolStreamEvents during execution, then returns exactly one ToolResultBlock as the final value.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `toolContext` | [`ToolContext`](/docs/api/typescript/ToolContext/index.md) | Context information including the tool use request and invocation state |

#### Returns

[`ToolStreamGenerator`](/docs/api/typescript/ToolStreamGenerator/index.md)

Async generator that yields ToolStreamEvents and returns a ToolResultBlock

#### Example

```typescript
const context = {
  toolUse: {
    name: 'calculator',
    toolUseId: 'calc-123',
    input: { operation: 'add', a: 5, b: 3 }
  },
}

// The return value is only accessible via explicit .next() calls
const generator = tool.stream(context)
for await (const event of generator) {
  // Only yields are captured here
  console.log('Progress:', event.data)
}
// Or manually handle the return value:
let result = await generator.next()
while (!result.done) {
  console.log('Progress:', result.value.data)
  result = await generator.next()
}
console.log('Final result:', result.value.status)
```

#### Inherited from

[`Tool`](/docs/api/typescript/Tool/index.md).[`stream`](/docs/api/typescript/Tool/index.md#stream)

---

### invoke()

```ts
invoke(input, context?): Promise<TReturn>;
```

Defined in: [src/tools/tool.ts:179](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/tools/tool.ts#L179)

Invokes the tool directly with type-safe input and returns the unwrapped result.

Unlike stream(), this method:

-   Returns the raw result (not wrapped in ToolResult)
-   Consumes async generators and returns only the final value
-   Lets errors throw naturally (not wrapped in error ToolResult)

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `input` | `TInput` | The input parameters for the tool |
| `context?` | [`ToolContext`](/docs/api/typescript/ToolContext/index.md) | Optional tool execution context |

#### Returns

`Promise`<`TReturn`\>

The unwrapped result