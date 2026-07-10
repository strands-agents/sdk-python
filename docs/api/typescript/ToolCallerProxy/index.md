```ts
type ToolCallerProxy = Record<string, ToolHandle>;
```

Defined in: [src/agent/tool-caller.ts:68](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/agent/tool-caller.ts#L68)

The public type of the tool caller proxy. Provides dynamic property access where each property is a [ToolHandle](/docs/api/typescript/ToolHandle/index.md) with `.invoke()` and `.stream()` methods.