```ts
type MiddlewareHandlerOf<S> = S extends MiddlewareStage<infer C, infer R, infer E> ? MiddlewareHandler<C, R, E> : never;
```

Defined in: [src/middleware/types.ts:91](https://github.com/strands-agents/harness-sdk/blob/f6765ab42296e23ac7af7c93e0ff3626fd09a60c/strands-ts/src/middleware/types.ts#L91)

Extracts the `MiddlewareHandler` type from a stage token. Use this to type middleware methods or properties without repeating the generic parameters.

## Type Parameters

| Type Parameter |
| --- |
| `S` |

## Example

```typescript
class MyPlugin implements Plugin {
  private _handler: MiddlewareHandlerOf<typeof InvokeModelStage> = async function* (context, next) { ... }
}
```