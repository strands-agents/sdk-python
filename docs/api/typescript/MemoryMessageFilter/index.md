Defined in: [src/memory/extraction/types.ts:28](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/memory/extraction/types.ts#L28)

Filters content blocks out of messages before extraction.

Blocks whose kind is in [exclude](#exclude) are stripped; a message left with no content is dropped entirely. Defaults to excluding tool traffic (`toolUse` / `toolResult`), which is rarely useful as long-term memory and adds noise.

## Properties

### exclude

```ts
exclude: MemoryContentBlockType[];
```

Defined in: [src/memory/extraction/types.ts:30](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/memory/extraction/types.ts#L30)

Content block kinds to strip before extraction.