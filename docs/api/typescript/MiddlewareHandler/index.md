```ts
type MiddlewareHandler<TContext, TResult, TEvent> = (context, next) => AsyncGenerator<TEvent, TResult, undefined>;
```

Defined in: [src/middleware/types.ts:69](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/middleware/types.ts#L69)

A middleware handler function (Wrap phase). Receives the context and a `next` function to call the next layer. Must be an async generator that yields TEvent and returns TResult. Middleware can yield its own events, forward events from next, or suppress them.

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
| `next` | [`MiddlewareNext`](/docs/api/typescript/MiddlewareNext/index.md)<`TContext`, `TResult`, `TEvent`\> |

## Returns

`AsyncGenerator`<`TEvent`, `TResult`, `undefined`\>