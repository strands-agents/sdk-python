Defined in: [src/middleware/stages.ts:23](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/middleware/stages.ts#L23)

Result returned by `interrupt()` in middleware contexts. Returns an object to allow future extension (e.g., cached data, metadata) without a breaking change to callers.

## Type Parameters

| Type Parameter | Default type |
| --- | --- |
| `T` | [`JSONValue`](/docs/api/typescript/JSONValue/index.md) |

## Properties

### response

```ts
response: T;
```

Defined in: [src/middleware/stages.ts:30](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/middleware/stages.ts#L30)

The resolved response value from the interrupt. The generic `T` is a caller assertion — the actual value is whatever the user passed in `InterruptResponseContent.response` (a `JSONValue`). No runtime validation is performed; callers are responsible for ensuring the shape matches.