```ts
type MiddlewareNext<TContext, TResult, TEvent> = (context) => AsyncGenerator<TEvent, TResult, undefined>;
```

Defined in: [src/middleware/types.ts:59](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/middleware/types.ts#L59)

The `next` function passed to middleware. Returns an async generator that yields events of type TEvent and returns the stage result. Middleware can choose not to call `next` to short-circuit execution.

## Type Parameters

| Type Parameter |
| --- |
| `TContext` |
| `TResult` |
| `TEvent` |

## Parameters

| Parameter | Type |
| --- | --- |
| `context` | `TContext` |

## Returns

`AsyncGenerator`<`TEvent`, `TResult`, `undefined`\>