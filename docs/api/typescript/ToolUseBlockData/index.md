Defined in: [src/types/messages.ts:252](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/types/messages.ts#L252)

Data for a tool use block.

## Properties

### name

```ts
name: string;
```

Defined in: [src/types/messages.ts:256](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/types/messages.ts#L256)

The name of the tool to execute.

---

### toolUseId

```ts
toolUseId: string;
```

Defined in: [src/types/messages.ts:261](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/types/messages.ts#L261)

Unique identifier for this tool use instance.

---

### input

```ts
input: JSONValue;
```

Defined in: [src/types/messages.ts:267](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/types/messages.ts#L267)

The input parameters for the tool. This can be any JSON-serializable value.

---

### reasoningSignature?

```ts
optional reasoningSignature?: string;
```

Defined in: [src/types/messages.ts:273](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/types/messages.ts#L273)

Reasoning signature from thinking models (e.g., Gemini). Must be preserved and sent back to the model for multi-turn tool use.