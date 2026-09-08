```ts
type MiddlewareOutputHandler<TResult> = (result) => TResult | Promise<TResult>;
```

Defined in: [src/middleware/types.ts:78](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/middleware/types.ts#L78)

Handler for Output phase — transforms result after execution.

## Type Parameters

| Type Parameter |
| --- |
| `TResult` |

## Parameters

| Parameter | Type |
| --- | --- |
| `result` | `TResult` |

## Returns

`TResult` | `Promise`<`TResult`\>