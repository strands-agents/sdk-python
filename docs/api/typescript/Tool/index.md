Defined in: [src/tools/tool.ts:107](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/tools/tool.ts#L107)

Interface for tool implementations. Tools are used by agents to interact with their environment and perform specific actions.

The Tool interface provides a streaming execution model where tools can yield progress events during execution before returning a final result.

Most implementations should use FunctionTool rather than implementing this interface directly.

## Extended by

-   [`InvokableTool`](/docs/api/typescript/InvokableTool/index.md)
-   [`FunctionTool`](/docs/api/typescript/FunctionTool/index.md)
-   [`ZodTool`](/docs/api/typescript/ZodTool/index.md)

## Constructors

### Constructor

```ts
new Tool(): Tool;
```

#### Returns

`Tool`

## Properties

### name

```ts
abstract name: string;
```

Defined in: [src/tools/tool.ts:112](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/tools/tool.ts#L112)

The unique name of the tool. This MUST match the name in the toolSpec.

---

### description

```ts
abstract description: string;
```

Defined in: [src/tools/tool.ts:119](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/tools/tool.ts#L119)

Human-readable description of what the tool does. This helps the model understand when to use the tool.

This MUST match the description in the toolSpec.description.

---

### toolSpec

```ts
abstract toolSpec: ToolSpec;
```

Defined in: [src/tools/tool.ts:124](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/tools/tool.ts#L124)

OpenAPI JSON specification for the tool. Defines the tool’s name, description, and input schema.

## Methods

### stream()

```ts
abstract stream(toolContext): ToolStreamGenerator;
```

Defined in: [src/tools/tool.ts:159](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/tools/tool.ts#L159)

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