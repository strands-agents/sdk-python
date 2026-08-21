Defined in: [src/retry/backoff-strategy.ts:84](https://github.com/strands-agents/harness-sdk/blob/f6765ab42296e23ac7af7c93e0ff3626fd09a60c/strands-ts/src/retry/backoff-strategy.ts#L84)

Options for [LinearBackoff](/docs/api/typescript/LinearBackoff/index.md).

## Properties

### baseMs?

```ts
optional baseMs?: number;
```

Defined in: [src/retry/backoff-strategy.ts:86](https://github.com/strands-agents/harness-sdk/blob/f6765ab42296e23ac7af7c93e0ff3626fd09a60c/strands-ts/src/retry/backoff-strategy.ts#L86)

Base delay in ms. Delay grows as `baseMs * attempt`. Default 1000.

---

### maxMs?

```ts
optional maxMs?: number;
```

Defined in: [src/retry/backoff-strategy.ts:88](https://github.com/strands-agents/harness-sdk/blob/f6765ab42296e23ac7af7c93e0ff3626fd09a60c/strands-ts/src/retry/backoff-strategy.ts#L88)

Upper bound applied before jitter. Default 30\_000.

---

### jitter?

```ts
optional jitter?: JitterKind;
```

Defined in: [src/retry/backoff-strategy.ts:90](https://github.com/strands-agents/harness-sdk/blob/f6765ab42296e23ac7af7c93e0ff3626fd09a60c/strands-ts/src/retry/backoff-strategy.ts#L90)

Jitter mode. Default ‘full’.