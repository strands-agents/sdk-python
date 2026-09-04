Defined in: [src/middleware/types.ts:8](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/middleware/types.ts#L8)

A stage token that identifies a middleware interception point. Stages are created via `createStage()` and carry their Context/Event/Result types as generics, enabling full type inference at registration sites.

Third parties can create custom stages — the SDK does not maintain a closed set.

## Type Parameters

| Type Parameter |
| --- |
| `TContext` |
| `TResult` |
| `TEvent` |

## Properties

### name

```ts
readonly name: string;
```

Defined in: [src/middleware/types.ts:10](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/middleware/types.ts#L10)

Human-readable name for debugging and logging.

---

### Input

```ts
readonly Input: MiddlewareInputPhase<TContext, TResult, TEvent>;
```

Defined in: [src/middleware/types.ts:14](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/middleware/types.ts#L14)

Phase sub-token: transform context before execution.

---

### Wrap

```ts
readonly Wrap: MiddlewareWrapPhase<TContext, TResult, TEvent>;
```

Defined in: [src/middleware/types.ts:16](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/middleware/types.ts#L16)

Phase sub-token: full async generator wrap (before + call next + after).

---

### Output

```ts
readonly Output: MiddlewareOutputPhase<TContext, TResult, TEvent>;
```

Defined in: [src/middleware/types.ts:18](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/middleware/types.ts#L18)

Phase sub-token: transform result after execution.