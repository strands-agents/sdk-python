Defined in: [src/models/bedrock.ts:158](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/models/bedrock.ts#L158)

Redaction configuration for Bedrock guardrails. Controls whether and how blocked content is replaced.

## Properties

### input?

```ts
optional input?: boolean;
```

Defined in: [src/models/bedrock.ts:160](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/models/bedrock.ts#L160)

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

Defined in: [src/models/bedrock.ts:163](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/models/bedrock.ts#L163)

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

Defined in: [src/models/bedrock.ts:166](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/models/bedrock.ts#L166)

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

Defined in: [src/models/bedrock.ts:169](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/models/bedrock.ts#L169)

Replacement message for redacted output.

#### Default Value

```ts
'[Assistant output redacted.]'
```