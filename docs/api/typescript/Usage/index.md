Defined in: [src/models/streaming.ts:493](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/models/streaming.ts#L493)

Token usage statistics for a model invocation. Tracks input, output, and total tokens, plus cache-related metrics.

## Properties

### inputTokens

```ts
inputTokens: number;
```

Defined in: [src/models/streaming.ts:497](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/models/streaming.ts#L497)

Number of tokens in the input (prompt).

---

### outputTokens

```ts
outputTokens: number;
```

Defined in: [src/models/streaming.ts:502](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/models/streaming.ts#L502)

Number of tokens in the output (completion).

---

### totalTokens

```ts
totalTokens: number;
```

Defined in: [src/models/streaming.ts:507](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/models/streaming.ts#L507)

Total number of tokens (input + output).

---

### cacheReadInputTokens?

```ts
optional cacheReadInputTokens?: number;
```

Defined in: [src/models/streaming.ts:513](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/models/streaming.ts#L513)

Number of input tokens read from cache. This can reduce latency and cost.

---

### cacheWriteInputTokens?

```ts
optional cacheWriteInputTokens?: number;
```

Defined in: [src/models/streaming.ts:519](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/models/streaming.ts#L519)

Number of input tokens written to cache. These tokens can be reused in future requests.