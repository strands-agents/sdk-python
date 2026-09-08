Defined in: [src/plugins/plugin.ts:51](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/plugins/plugin.ts#L51)

Interface for objects that extend agent functionality.

Plugins provide a composable way to add behavior changes to agents by registering hook callbacks in their `initAgent` method. Each plugin must have a unique name for identification, logging, and duplicate prevention.

## Examples

```typescript
class LoggingPlugin implements Plugin {
  get name(): string {
    return 'logging-plugin'
  }

  initAgent(agent: LocalAgent): void {
    agent.addHook(BeforeInvocationEvent, (event) => {
      console.log('Agent invocation started')
    })
  }
}

const agent = new Agent({
  model,
  plugins: [new LoggingPlugin()],
})
```

**With tools**

```typescript
class MyToolPlugin implements Plugin {
  get name(): string {
    return 'my-tool-plugin'
  }

  getTools(): Tool[] {
    return [myTool]
  }
}
```

## Properties

### name

```ts
readonly name: string;
```

Defined in: [src/plugins/plugin.ts:58](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/plugins/plugin.ts#L58)

A stable string identifier for the plugin. Used for logging, duplicate detection, and plugin management.

For strands-vended plugins, names should be prefixed with `strands:`.

## Methods

### initAgent()

```ts
initAgent(agent): void | Promise<void>;
```

Defined in: [src/plugins/plugin.ts:68](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/plugins/plugin.ts#L68)

Initialize the plugin with the agent instance.

Implement this method to register hooks and perform custom initialization. Tool registration from [getTools](#gettools) is handled automatically by the PluginRegistry.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `agent` | `LocalAgent` | The agent instance this plugin is being attached to |

#### Returns

`void` | `Promise`<`void`\>

---

### getTools()?

```ts
optional getTools(): Tool[];
```

Defined in: [src/plugins/plugin.ts:76](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/plugins/plugin.ts#L76)

Returns tools provided by this plugin for auto-registration. Implement to provide plugin-specific tools.

#### Returns

[`Tool`](/docs/api/typescript/Tool/index.md)\[\]

Array of tools to register with the agent