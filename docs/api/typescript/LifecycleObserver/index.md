Defined in: [src/types/lifecycle-observer.ts:10](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/types/lifecycle-observer.ts#L10)

Implementors are given the agent at registration time so they can subscribe to hook events of their choice via LocalAgent.addHook. This is the extension point for components that need to observe arbitrary lifecycle events. Each observer method is optional — implementors define only the surfaces they care about, and the agent probes for each at registration.

## Properties

### name

```ts
readonly name: string;
```

Defined in: [src/types/lifecycle-observer.ts:12](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/types/lifecycle-observer.ts#L12)

Stable identifier for this observer. Used for logging and duplicate detection.

## Methods

### observeAgent()?

```ts
optional observeAgent(agent): void | Promise<void>;
```

Defined in: [src/types/lifecycle-observer.ts:18](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/types/lifecycle-observer.ts#L18)

Called once when the observer is registered with an agent. Implementations typically subscribe to one or more events via `agent.addHook`.

#### Parameters

| Parameter | Type |
| --- | --- |
| `agent` | `LocalAgent` |

#### Returns

`void` | `Promise`<`void`\>