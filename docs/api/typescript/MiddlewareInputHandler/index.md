```ts
type MiddlewareInputHandler<TContext> = (context) => TContext | Promise<TContext>;
```

Defined in: [src/middleware/types.ts:75](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/middleware/types.ts#L75)

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