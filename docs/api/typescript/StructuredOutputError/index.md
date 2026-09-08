Defined in: [src/errors.ts:203](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/errors.ts#L203)

Thrown when the model fails to produce structured output. This occurs when the LLM refuses to use the structured output tool even after being forced via toolChoice.

## Extends

-   `Error`

## Constructors

### Constructor

```ts
new StructuredOutputError(message): StructuredOutputError;
```

Defined in: [src/errors.ts:204](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/errors.ts#L204)

#### Parameters

| Parameter | Type |
| --- | --- |
| `message` | `string` |

#### Returns

`StructuredOutputError`

#### Overrides

```ts
Error.constructor
```