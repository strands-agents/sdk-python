Defined in: [src/agent/tool-caller.ts:40](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/agent/tool-caller.ts#L40)

A handle to a specific tool, providing `.invoke()` and `.stream()` methods.

Returned by the Proxy get trap when accessing `agent.tool.toolName`. This aligns with the agent-level `agent.invoke()` / `agent.stream()` pattern.

## Properties

### invoke

```ts
invoke: (input?, options?) => Promise<ToolResultBlock>;
```

Defined in: [src/agent/tool-caller.ts:48](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/agent/tool-caller.ts#L48)

Invoke the tool and return the final result.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `input?` | [`JSONValue`](/docs/api/typescript/JSONValue/index.md) | The input parameters for the tool |
| `options?` | [`DirectToolCallOptions`](/docs/api/typescript/DirectToolCallOptions/index.md) | Optional configuration for this call |

#### Returns

`Promise`<[`ToolResultBlock`](/docs/api/typescript/ToolResultBlock/index.md)\>

The tool result

---

### stream

```ts
stream: (input?, options?) => AsyncGenerator<ToolStreamEvent, ToolResultBlock, undefined>;
```

Defined in: [src/agent/tool-caller.ts:57](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/agent/tool-caller.ts#L57)

Stream the tool execution, yielding intermediate events and returning the final result.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `input?` | [`JSONValue`](/docs/api/typescript/JSONValue/index.md) | The input parameters for the tool |
| `options?` | [`DirectToolCallOptions`](/docs/api/typescript/DirectToolCallOptions/index.md) | Optional configuration for this call |

#### Returns

`AsyncGenerator`<[`ToolStreamEvent`](/docs/api/typescript/ToolStreamEvent/index.md), [`ToolResultBlock`](/docs/api/typescript/ToolResultBlock/index.md), `undefined`\>

Async generator that yields ToolStreamEvents and returns ToolResultBlock