Defined in: [src/models/streaming.ts:430](https://github.com/strands-agents/harness-sdk/blob/f6765ab42296e23ac7af7c93e0ff3626fd09a60c/strands-ts/src/models/streaming.ts#L430)

Tool use input delta within a content block. Represents incremental tool input being generated.

## Properties

### type

```ts
type: "toolUseInputDelta";
```

Defined in: [src/models/streaming.ts:434](https://github.com/strands-agents/harness-sdk/blob/f6765ab42296e23ac7af7c93e0ff3626fd09a60c/strands-ts/src/models/streaming.ts#L434)

Discriminator for tool use input delta.

---

### input

```ts
input: string;
```

Defined in: [src/models/streaming.ts:439](https://github.com/strands-agents/harness-sdk/blob/f6765ab42296e23ac7af7c93e0ff3626fd09a60c/strands-ts/src/models/streaming.ts#L439)

Partial JSON string representing the tool input.