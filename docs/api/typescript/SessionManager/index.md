Defined in: [src/session/session-manager.ts:103](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/session/session-manager.ts#L103)

Manages session persistence for agents, enabling conversation state to be saved and restored across invocations using pluggable storage backends.

Also supports multi-agent orchestrators (Graph, Swarm) via the MultiAgentPlugin interface. Scope is auto-detected based on whether initAgent or initMultiAgent is called.

## Example

```typescript
import { SessionManager, FileStorage } from '@strands-agents/sdk'

const session = new SessionManager({
  sessionId: 'my-session',
  storage: { snapshot: new FileStorage() },
})
const agent = new Agent({ sessionManager: session })
```

## Implements

-   [`Plugin`](/docs/api/typescript/Plugin/index.md)
-   `MultiAgentPlugin`

## Constructors

### Constructor

```ts
new SessionManager(config): SessionManager;
```

Defined in: [src/session/session-manager.ts:126](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/session/session-manager.ts#L126)

#### Parameters

| Parameter | Type |
| --- | --- |
| `config` | [`SessionManagerConfig`](/docs/api/typescript/SessionManagerConfig/index.md) |

#### Returns

`SessionManager`

## Accessors

### name

#### Get Signature

```ts
get name(): string;
```

Defined in: [src/session/session-manager.ts:115](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/session/session-manager.ts#L115)

Unique identifier for this plugin.

##### Returns

`string`

A stable string identifier for the plugin. Used for logging, duplicate detection, and plugin management.

For strands-vended plugins, names should be prefixed with `strands:`.

#### Implementation of

[`Plugin`](/docs/api/typescript/Plugin/index.md).[`name`](/docs/api/typescript/Plugin/index.md#name)

---

### sessionId

#### Get Signature

```ts
get sessionId(): string;
```

Defined in: [src/session/session-manager.ts:122](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/session/session-manager.ts#L122)

The session identifier for this session manager.

##### Returns

`string`

## Methods

### initAgent()

```ts
initAgent(agent): void;
```

Defined in: [src/session/session-manager.ts:153](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/session/session-manager.ts#L153)

Initializes the plugin by registering lifecycle hook callbacks.

#### Parameters

| Parameter | Type |
| --- | --- |
| `agent` | `LocalAgent` |

#### Returns

`void`

#### Implementation of

[`Plugin`](/docs/api/typescript/Plugin/index.md).[`initAgent`](/docs/api/typescript/Plugin/index.md#initagent)

---

### saveSnapshot()

#### Call Signature

```ts
saveSnapshot(params): Promise<void>;
```

Defined in: [src/session/session-manager.ts:189](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/session/session-manager.ts#L189)

Saves a snapshot of the target’s current state.

##### Parameters

| Parameter | Type |
| --- | --- |
| `params` | { `target`: `LocalAgent`; `isLatest`: `boolean`; } |
| `params.target` | `LocalAgent` |
| `params.isLatest` | `boolean` |

##### Returns

`Promise`<`void`\>

#### Call Signature

```ts
saveSnapshot(params): Promise<void>;
```

Defined in: [src/session/session-manager.ts:190](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/session/session-manager.ts#L190)

Saves a snapshot of the target’s current state.

##### Parameters

| Parameter | Type |
| --- | --- |
| `params` | { `target`: [`Graph`](/docs/api/typescript/Graph/index.md) | [`Swarm`](/docs/api/typescript/Swarm/index.md); `state?`: `MultiAgentState`; `isLatest`: `boolean`; } |
| `params.target` | [`Graph`](/docs/api/typescript/Graph/index.md) | [`Swarm`](/docs/api/typescript/Swarm/index.md) |
| `params.state?` | `MultiAgentState` |
| `params.isLatest` | `boolean` |

##### Returns

`Promise`<`void`\>

---

### deleteSession()

```ts
deleteSession(): Promise<void>;
```

Defined in: [src/session/session-manager.ts:208](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/session/session-manager.ts#L208)

Deletes all snapshots and manifests for this session from storage.

#### Returns

`Promise`<`void`\>

---

### listSnapshotIds()

```ts
listSnapshotIds(params): Promise<string[]>;
```

Defined in: [src/session/session-manager.ts:213](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/session/session-manager.ts#L213)

Lists all available immutable snapshot IDs for the given agent target.

#### Parameters

| Parameter | Type |
| --- | --- |
| `params` | { `target`: `LocalAgent`; `limit?`: `number`; `startAfter?`: `string`; } |
| `params.target` | `LocalAgent` |
| `params.limit?` | `number` |
| `params.startAfter?` | `string` |

#### Returns

`Promise`<`string`\[\]>

---

### restoreSnapshot()

#### Call Signature

```ts
restoreSnapshot(params): Promise<boolean>;
```

Defined in: [src/session/session-manager.ts:222](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/session/session-manager.ts#L222)

Loads a snapshot from storage and restores it into the target. Returns false if no snapshot exists.

##### Parameters

| Parameter | Type |
| --- | --- |
| `params` | { `target`: `LocalAgent`; `snapshotId?`: `string`; } |
| `params.target` | `LocalAgent` |
| `params.snapshotId?` | `string` |

##### Returns

`Promise`<`boolean`\>

#### Call Signature

```ts
restoreSnapshot(params): Promise<boolean>;
```

Defined in: [src/session/session-manager.ts:223](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/session/session-manager.ts#L223)

Loads a snapshot from storage and restores it into the target. Returns false if no snapshot exists.

##### Parameters

| Parameter | Type |
| --- | --- |
| `params` | { `target`: [`Graph`](/docs/api/typescript/Graph/index.md) | [`Swarm`](/docs/api/typescript/Swarm/index.md); `state?`: `MultiAgentState`; `snapshotId?`: `string`; } |
| `params.target` | [`Graph`](/docs/api/typescript/Graph/index.md) | [`Swarm`](/docs/api/typescript/Swarm/index.md) |
| `params.state?` | `MultiAgentState` |
| `params.snapshotId?` | `string` |

##### Returns

`Promise`<`boolean`\>

---

### initMultiAgent()

```ts
initMultiAgent(orchestrator): void;
```

Defined in: [src/session/session-manager.ts:322](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/session/session-manager.ts#L322)

Initializes the multi-agent plugin by registering orchestrator lifecycle hooks.

#### Parameters

| Parameter | Type |
| --- | --- |
| `orchestrator` | `MultiAgent` |

#### Returns

`void`

#### Implementation of

```ts
MultiAgentPlugin.initMultiAgent
```