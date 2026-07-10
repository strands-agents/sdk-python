Defined in: [src/sandbox/types.ts:32](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/sandbox/types.ts#L32)

Metadata about a file or directory in a sandbox.

Provides minimal structured information that lets tools distinguish files from directories and report sizes. `isDir` and `size` are `undefined` when the backend cannot determine them accurately.

## Properties

### name

```ts
readonly name: string;
```

Defined in: [src/sandbox/types.ts:33](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/sandbox/types.ts#L33)

---

### isDir?

```ts
readonly optional isDir?: boolean;
```

Defined in: [src/sandbox/types.ts:34](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/sandbox/types.ts#L34)

---

### size?

```ts
readonly optional size?: number;
```

Defined in: [src/sandbox/types.ts:35](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/sandbox/types.ts#L35)