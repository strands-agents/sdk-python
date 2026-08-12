```ts
type MemoryContentBlockType = DistributedKeyof<ContentBlockData>;
```

Defined in: [src/memory/extraction/types.ts:19](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/memory/extraction/types.ts#L19)

Content block kinds that [MemoryMessageFilter](/docs/api/typescript/MemoryMessageFilter/index.md) can exclude before messages reach an [Extractor](/docs/api/typescript/Extractor/index.md) or the no-extractor passthrough.

Derived from the SDK’s [ContentBlockData](/docs/api/typescript/ContentBlockData/index.md) union: every member’s discriminator key is its kind (`{ toolUse: ... }` → `'toolUse'`, `{ text: string }` → `'text'`), so this tracks new block types automatically instead of drifting. Matches the runtime `_blockKind` used to filter.