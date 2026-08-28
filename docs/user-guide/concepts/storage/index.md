Pass a Storage backend to the Agent and every subsystem that needs persistence resolves from it automatically. The SDK handles namespacing so sessions and offloaded content never collide. You can also pass storage directly to individual plugins when different subsystems need different backends.

The SDK ships three backends. Pick one based on where you need your data to live:

| Backend | Where data lives | Best for |
| --- | --- | --- |
| `InMemoryStorage` | Process memory | Tests, short-lived agents |
| `LocalFileStorage` | Local filesystem | Development, single-machine |
| `S3Storage` | Amazon S3 | Production, multi-instance |

## Agent-level storage

The simplest approach: pass a single storage backend to the Agent and let subsystems resolve from it.

(( tab "Python" ))
```python
storage = S3Storage("my-bucket", prefix="agents/prod/")

agent = Agent(
    storage=storage,
    session_manager=SnapshotSessionManager("my-session"),
    context_manager="auto",
)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { S3Storage } from '@strands-agents/sdk/storage'
import { Agent, SessionManager } from '@strands-agents/sdk'

const storage = new S3Storage('my-bucket', {
  prefix: 'agents/prod/',
})

const agent = new Agent({
  storage,
  sessionManager: new SessionManager({
    sessionId: 'my-session',
  }),
  contextManager: 'auto',
})
```
(( /tab "TypeScript" ))

Both the session manager and context offloader read from the same backend without extra wiring. Each subsystem auto-namespaces its keys (`session/` for sessions, `offloader/` for offloaded content), so data never collides.

## Per-plugin storage

When different subsystems need different backends, pass storage directly to the plugin. This overrides the agent-level default for that plugin only.

(( tab "Python" ))
```python
agent = Agent(
    session_manager=SnapshotSessionManager(
        "my-session", storage=S3Storage("my-bucket")
    ),
    plugins=[ContextOffloader(storage=InMemoryStorage())],
)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { InMemoryStorage, S3Storage } from '@strands-agents/sdk/storage'
import { Agent, SessionManager } from '@strands-agents/sdk'
import { ContextOffloader } from '@strands-agents/sdk/vended-plugins/context-offloader'

const agent = new Agent({
  sessionManager: new SessionManager({
    sessionId: 'my-session',
    storage: new S3Storage('my-bucket'),
  }),
  plugins: [
    new ContextOffloader({
      storage: new InMemoryStorage(),
    }),
  ],
})
```
(( /tab "TypeScript" ))

## Precedence

Storage resolves in this order for each subsystem:

1.  **Explicit**: storage passed directly to the plugin
2.  **Agent-level**: the agent’s `storage``storage` parameter (namespaced automatically)
3.  **Fallback**: `InMemoryStorage` for Context Offloader; `LocalFileStorage` for Session Manager in Python, or an error in TypeScript

Note

Python `SessionManager` is deprecated, and does not use the provided `Storage` parameter. Use `SnapshotSessionManager` in Python or `SessionManager` in TypeScript to take advantage of agent-level storage.

## Built-in backends

### InMemoryStorage

Data lives in process memory. No constructor arguments. Fast, zero-config, gone when the process exits.

(( tab "Python" ))
```python
storage = InMemoryStorage()
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { InMemoryStorage } from '@strands-agents/sdk/storage'

const storage = new InMemoryStorage()
```
(( /tab "TypeScript" ))

### LocalFileStorage

Each key becomes a file under a base directory. Writes are atomic (temp file + rename).

| Parameter | Default | Description |
| --- | --- | --- |
| `base_dir``baseDir` | `"./.strands/"` | Root directory |
| `sandbox``sandbox` | `None`/`undefined` | Optional [Sandbox](/docs/user-guide/concepts/sandbox/index.md) |

(( tab "Python" ))
```python
storage = LocalFileStorage("./my-data/")
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { LocalFileStorage } from '@strands-agents/sdk/storage'

const storage = new LocalFileStorage('./my-data/')
```
(( /tab "TypeScript" ))

You can also bind a sandbox after construction with `for_sandbox(sandbox)``forSandbox(sandbox)`, which returns a new instance routed through the sandbox.

### S3Storage

Stores data as objects in an S3 bucket. The AWS SDK loads lazily, so applications that never construct an `S3Storage` pay nothing.

| Parameter | Default | Description |
| --- | --- | --- |
| `bucket` | *(required)* | S3 bucket name |
| `prefix` | `""` | Key prefix (namespace within the bucket) |
| `region_name``region` | `None`/`undefined` | AWS region override |
| `boto_session``s3Client` | `None`/`undefined` | Pre-configured client |

(( tab "Python" ))
```python
storage = S3Storage("my-bucket", prefix="agents/prod/")
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { S3Storage } from '@strands-agents/sdk/storage'

const storage = new S3Storage('my-bucket', {
  prefix: 'agents/prod/',
})
```
(( /tab "TypeScript" ))

You cannot pass both a region and a pre-configured client; pick one or the other.

## Custom backends

Implement four async methods (`write, read, delete, list``write, read, delete, list`) and pass your class anywhere a `Storage` is accepted. In Python, `Storage` is a [protocol](/docs/api/python/strands.storage.storage#Storage); in TypeScript, implement the [interface](/docs/api/typescript/Storage/index.md).

Community backends can add methods beyond the core four (e.g. `search` for vector similarity, or structured queries for databases like DynamoDB). Plugins that only need basic persistence use the four standard methods; plugins that need richer access can check for and use the extra surface your backend provides.

When using agent-level storage, each subsystem scopes its keys under its own prefix automatically (`session/`, `offloader/`), so you never need to worry about collisions. If you write a custom plugin that consumes agent-level storage, call

`storage.namespace('my-prefix/')``storage.namespace('my-prefix/')`

to claim your own prefix and avoid overlapping with other subsystems.

## Next steps

-   [Context Offloader](/docs/user-guide/concepts/plugins/context-offloader/index.md): offload large tool results
-   [Session Management](/docs/user-guide/concepts/agents/session-management/index.md): persist conversations across restarts
-   [Sandbox](/docs/user-guide/concepts/sandbox/index.md): route Storage I/O through a sandboxed environment

## Related pages

- [Conversation Management](/docs/user-guide/concepts/agents/conversation-management/index.md) (1 shared tag)
- [Bidirectional Streaming Session Management](/docs/user-guide/concepts/bidirectional-streaming/session-management/index.md) (1 shared tag)
- [Session Management](/docs/user-guide/concepts/agents/session-management/index.md) (1 shared tag)
- [State Management](/docs/user-guide/concepts/agents/state/index.md) (1 shared tag)


## Implementation

### Python

- [harness-sdk/strands-py/src/strands/storage/storage.py](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/storage.py)
- [harness-sdk/strands-py/src/strands/storage/in_memory_storage.py](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/in_memory_storage.py)
- [harness-sdk/strands-py/src/strands/storage/local_file_storage.py](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/local_file_storage.py)
- [harness-sdk/strands-py/src/strands/storage/s3_storage.py](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/s3_storage.py)

### TypeScript

- [harness-sdk/strands-ts/src/storage/storage.ts](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/storage/storage.ts)
- [harness-sdk/strands-ts/src/storage/in-memory-storage.ts](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/storage/in-memory-storage.ts)
- [harness-sdk/strands-ts/src/storage/local-file-storage.ts](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/storage/local-file-storage.ts)
- [harness-sdk/strands-ts/src/storage/s3-storage.ts](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/storage/s3-storage.ts)
