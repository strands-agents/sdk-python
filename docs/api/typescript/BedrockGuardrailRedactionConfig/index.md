Defined in: [src/models/bedrock.ts:162](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/models/bedrock.ts#L162)

Redaction configuration for Bedrock guardrails. Controls whether and how blocked content is replaced.

## Properties

### input?

```ts
optional input?: boolean;
```

Defined in: [src/models/bedrock.ts:164](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/models/bedrock.ts#L164)

Redact input when blocked.

#### Default Value

```ts
true
```

---

### inputMessage?

```ts
optional inputMessage?: string;
```

Defined in: [src/models/bedrock.ts:167](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/models/bedrock.ts#L167)

Replacement message for redacted input.

#### Default Value

```ts
'[User input redacted.]'
```

---

### output?

```ts
optional output?: boolean;
```

Defined in: [src/models/bedrock.ts:170](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/models/bedrock.ts#L170)

Redact output when blocked.

#### Default Value

```ts
false
```

---

### outputMessage?

```ts
optional outputMessage?: string;
```

Defined in: [src/models/bedrock.ts:173](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/models/bedrock.ts#L173)

Replacement message for redacted output.

#### Default Value

```ts
'[Assistant output redacted.]'
```