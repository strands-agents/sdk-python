```ts
type MiddlewareInputHandler<TContext> = (context) => TContext | Promise<TContext>;
```

Defined in: [src/middleware/types.ts:75](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/middleware/types.ts#L75)

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