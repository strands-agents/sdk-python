```ts
type SnapshotTriggerCallback = (params) => boolean;
```

Defined in: [src/session/types.ts:41](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/session/types.ts#L41)

Callback function to determine when to create immutable snapshots. Called after each agent invocation to decide if a snapshot should be saved.

## Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `params` | [`SnapshotTriggerParams`](/docs/api/typescript/SnapshotTriggerParams/index.md) | Snapshot trigger parameters |

## Returns

`boolean`

true to create a snapshot, false to skip

## Example

```ts
// Snapshot every 10 messages
const trigger: SnapshotTriggerCallback = ({ agentData }) => agentData.messages.length % 10 === 0

// Snapshot when conversation exceeds 20 messages
const trigger: SnapshotTriggerCallback = ({ agentData }) => agentData.messages.length > 20
```