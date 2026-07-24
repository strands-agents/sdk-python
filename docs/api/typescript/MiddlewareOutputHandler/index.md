```ts
type MiddlewareOutputHandler<TResult> = (result) => TResult | Promise<TResult>;
```

Defined in: [src/middleware/types.ts:78](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/middleware/types.ts#L78)

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