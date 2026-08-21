```ts
type MiddlewareInputHandler<TContext> = (context) => TContext | Promise<TContext>;
```

Defined in: [src/middleware/types.ts:75](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/middleware/types.ts#L75)

Handler for Input phase — transforms context before execution.

## Type Parameters

| Type Parameter |
| --- |
| `TContext` |

## Parameters

| Parameter | Type |
| --- | --- |
| `context` | `TContext` |

## Returns

`TContext` | `Promise`<`TContext`\>