```ts
type ToolStreamGenerator = AsyncGenerator<ToolStreamEvent, ToolResultBlock, undefined>;
```

Defined in: [src/tools/tool.ts:96](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/tools/tool.ts#L96)

Type alias for the async generator returned by tool stream methods. Yields ToolStreamEvents during execution and returns a ToolResultBlock.