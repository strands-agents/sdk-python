```ts
type MiddlewareNextOf<S> = S extends MiddlewareStage<infer C, infer R, infer E> ? MiddlewareNext<C, R, E> : never;
```

Defined in: [src/middleware/types.ts:103](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/middleware/types.ts#L103)

Extracts the `MiddlewareNext` type from a stage token. Use this to type the `next` parameter in standalone middleware methods.

## Type Parameters

| Type Parameter |
| --- |
| `S` |

## Example

```typescript
private async *_handler(context: ..., next: MiddlewareNextOf<typeof AgentStreamStage>) { ... }
```