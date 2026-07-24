Defined in: [src/models/streaming.ts:379](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/models/streaming.ts#L379)

Information about a tool use that is starting.

## Properties

### type

```ts
type: "toolUseStart";
```

Defined in: [src/models/streaming.ts:383](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/models/streaming.ts#L383)

Discriminator for tool use start.

---

### name

```ts
name: string;
```

Defined in: [src/models/streaming.ts:388](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/models/streaming.ts#L388)

The name of the tool being used.

---

### toolUseId

```ts
toolUseId: string;
```

Defined in: [src/models/streaming.ts:393](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/models/streaming.ts#L393)

Unique identifier for this tool use.

---

### reasoningSignature?

```ts
optional reasoningSignature?: string;
```

Defined in: [src/models/streaming.ts:399](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/models/streaming.ts#L399)

Reasoning signature from thinking models (e.g., Gemini). Must be preserved and sent back to the model for multi-turn tool use.