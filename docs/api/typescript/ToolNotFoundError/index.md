Defined in: [src/errors.ts:217](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/errors.ts#L217)

Error thrown when a tool cannot be found by name.

Thrown by ToolRegistry.resolve when the requested tool name doesn’t match any registered tool, even after underscore-to-hyphen normalization and case-insensitive matching.

## Extends

-   `Error`

## Constructors

### Constructor

```ts
new ToolNotFoundError(toolName): ToolNotFoundError;
```

Defined in: [src/errors.ts:226](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/errors.ts#L226)

Creates a new ToolNotFoundError.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `toolName` | `string` | The tool name that was not found |

#### Returns

`ToolNotFoundError`

#### Overrides

```ts
Error.constructor
```

## Properties

### toolName

```ts
readonly toolName: string;
```

Defined in: [src/errors.ts:219](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/errors.ts#L219)

The tool name that was requested but not found.