Defined in: [src/errors.ts:122](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/errors.ts#L122)

Error thrown when a model provider returns a throttling or rate limit error.

This error indicates that the model API has rate limited the request. Users can handle this error in hooks to implement custom retry strategies using the `AfterModelCallEvent.retry` mechanism.

## Extends

-   [`ModelError`](/docs/api/typescript/ModelError/index.md)

## Constructors

### Constructor

```ts
new ModelThrottledError(message, options?): ModelThrottledError;
```

Defined in: [src/errors.ts:129](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/errors.ts#L129)

Creates a new ModelThrottledError.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `message` | `string` | Error message describing the throttling condition |
| `options?` | `ErrorOptions` | Optional error options including cause for error chaining |

#### Returns

`ModelThrottledError`

#### Overrides

[`ModelError`](/docs/api/typescript/ModelError/index.md).[`constructor`](/docs/api/typescript/ModelError/index.md#constructor)