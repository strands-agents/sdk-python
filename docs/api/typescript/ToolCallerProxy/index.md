```ts
type ToolCallerProxy = Record<string, ToolHandle>;
```

Defined in: [src/agent/tool-caller.ts:68](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/agent/tool-caller.ts#L68)

The public type of the tool caller proxy. Provides dynamic property access where each property is a [ToolHandle](/docs/api/typescript/ToolHandle/index.md) with `.invoke()` and `.stream()` methods.