Session management in Strands Agents provides a robust mechanism for persisting agent state and conversation history across multiple interactions. This enables agents to maintain context and continuity even when the application restarts or when deployed in distributed environments.

## Overview

A session represents all of stateful information that is needed by agents and multi-agent systems to function, including:

**Single Agent Sessions**:

-   Conversation history (messages)
-   Agent state (key-value storage)
-   Other stateful information (like [Conversation Manager](/docs/user-guide/concepts/agents/state/index.md#conversation-manager))

**Multi-Agent Sessions**:

-   Orchestrator state and configuration
-   Individual agent states and result within the orchestrator
-   Cross-agent shared state and context
-   Execution flow and node transition history

Strands provides built-in session persistence capabilities that automatically capture and restore this information, allowing agents to seamlessly continue conversations where they left off.

Beyond the built-in options, [third-party session managers](#third-party-session-managers) provide additional storage and memory capabilities.

## Basic Usage

### Single Agent Sessions

Simply create an agent with a session manager and use it:

(( tab "Python" ))
```python
from strands import Agent
from strands.session.file_session_manager import FileSessionManager

# Create a session manager with a unique session ID
session_manager = FileSessionManager(session_id="test-session")

# Create an agent with the session manager
agent = Agent(session_manager=session_manager)

# Use the agent - all messages and state are automatically persisted
agent("Hello!")  # This conversation is persisted
```
(( /tab "Python" ))

(( tab "TypeScript" ))
`SessionManager` implements both [Plugin](/docs/user-guide/concepts/plugins/index.md) (for agents) and `MultiAgentPlugin` (for orchestrators). The `sessionManager` constructor field is a convenience shorthand — you can also pass it directly in the `plugins` array:

```typescript
const session = new SessionManager({
  sessionId: 'test-session',
  storage: new LocalFileStorage('./sessions/'),
})

const agent = new Agent({ sessionManager: session })

// Use the agent - all messages and state are automatically persisted
await agent.invoke('Hello!') // This conversation is persisted
```

```typescript
const session = new SessionManager({
  sessionId: 'test-session',
  storage: new LocalFileStorage('./sessions/'),
})

// Equivalent to passing via sessionManager field
const agent = new Agent({ plugins: [session] })
await agent.invoke('Hello!')
```
(( /tab "TypeScript" ))

The conversation, and associated state, is persisted to the underlying storage backend.

### Multi-Agent Sessions

Multi-agent systems (Graph/Swarm) can also use session management to persist their state.

(( tab "Python" ))
Caution

Agents inside a multi-agent system must not have their own session manager — only the orchestrator should have one. Python will raise a `ValueError` if an agent with a session manager is added to a Graph or Swarm.

```python
from strands.multiagent import GraphBuilder
from strands.session.file_session_manager import FileSessionManager

# Create agents
agent1 = Agent(name="researcher")
agent2 = Agent(name="writer")

# Create a session manager for the graph
session_manager = FileSessionManager(session_id="multi-agent-session")

# Create graph with session management
graph = Graph(
    agents={"researcher": agent1, "writer": agent2},
    session_manager=session_manager
)

# Use the graph - all orchestrator state is persisted
result = graph("Research and write about AI")
```
(( /tab "Python" ))

(( tab "TypeScript" ))
Caution

Agents inside a multi-agent system must not have their own session manager — only the orchestrator should have one. The orchestrator snapshots and restores each agent node’s state on every execution, so an agent-level session manager would conflict with the orchestrator’s persistence.

```typescript
const session = new SessionManager({
  sessionId: 'graph-session',
  storage: new LocalFileStorage('./sessions/'),
})

const researcher = new Agent({
  id: 'researcher',
  systemPrompt: 'You are a research specialist.',
})
const writer = new Agent({
  id: 'writer',
  systemPrompt: 'You are a writing specialist.',
})

const graph = new Graph({
  nodes: [researcher, writer],
  edges: [['researcher', 'writer']],
  sessionManager: session,
})

// Orchestrator state is automatically persisted after each node completes
const result = await graph.invoke('Research and write about AI')
```

Swarm works the same way:

```typescript
const session = new SessionManager({
  sessionId: 'swarm-session',
  storage: new LocalFileStorage('./sessions/'),
})

const researcher = new Agent({
  id: 'researcher',
  description: 'Researches a topic and gathers key facts.',
  systemPrompt: 'Research the answer, then hand off to the writer.',
})

const writer = new Agent({
  id: 'writer',
  description: 'Writes a polished final answer.',
  systemPrompt: 'Write the final answer. Do not hand off.',
})

const swarm = new Swarm({
  nodes: [researcher, writer],
  start: 'researcher',
  sessionManager: session,
})

const result = await swarm.invoke('Explain quantum computing')
```
(( /tab "TypeScript" ))

Multi-agent session managers only track the current state of the Graph/Swarm execution and do not persist individual agent conversation histories.

## Storage Backends

(( tab "Python" ))
Strands offers two built-in session managers:

| Session Manager | Persistence | Best for |
| --- | --- | --- |
| [`FileSessionManager`](/docs/api/python/strands.session.file_session_manager#FileSessionManager) | Local disk | Development, single-machine |
| [`S3SessionManager`](/docs/api/python/strands.session.s3_session_manager#S3SessionManager) | Amazon S3 | Production, distributed |

```python
from strands import Agent
from strands.session.file_session_manager import FileSessionManager
from strands.session.s3_session_manager import S3SessionManager

# File-based persistence
session_manager = FileSessionManager(
    session_id="user-123",
    storage_dir="/path/to/sessions"
)

# S3-based persistence
session_manager = S3SessionManager(
    session_id="user-123",
    bucket="my-agent-sessions",
    prefix="production/",
)

agent = Agent(session_manager=session_manager)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
Session management accepts any [Storage](/docs/user-guide/concepts/storage/index.md) backend. Choose one based on your durability needs:

| Backend | Persistence | Best for |
| --- | --- | --- |
| `LocalFileStorage` | Local disk | Development, single-machine |
| `S3Storage` | Amazon S3 | Production, distributed |

See [Storage](/docs/user-guide/concepts/storage/index.md) for full details on each backend.

```typescript
const session = new SessionManager({
  sessionId: 'user-123',
  storage: new LocalFileStorage('./sessions/'),
})

const agent = new Agent({ sessionManager: session })
await agent.invoke("Hello, I'm a new user!")
```

```typescript
const session = new SessionManager({
  sessionId: 'user-456',
  storage: new S3Storage('my-agent-sessions', {
    prefix: 'production/',
    s3Client: new S3Client({ region: 'us-west-2' }),
  }),
})

const agent = new Agent({ sessionManager: session })
await agent.invoke('Tell me about AWS S3')
```
(( /tab "TypeScript" ))

### Required S3 Permissions

To use S3-backed session storage, your AWS credentials must have the following permissions:

-   `s3:PutObject` - To create and update session data
-   `s3:GetObject` - To retrieve session data
-   `s3:DeleteObject` - To delete session data
-   `s3:ListBucket` - To list objects in the bucket

Here’s a sample IAM policy that grants these permissions for a specific bucket:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:DeleteObject"
            ],
            "Resource": "arn:aws:s3:::my-agent-sessions/*"
        },
        {
            "Effect": "Allow",
            "Action": "s3:ListBucket",
            "Resource": "arn:aws:s3:::my-agent-sessions"
        }
    ]
}
```

## How Session Management Works

### Session Persistence Triggers

Session persistence is automatically triggered by lifecycle events in the agent:

**Single Agent Events**

(( tab "Python" ))
-   **Agent Initialization**: When an agent is created with a session manager, it automatically restores any existing state and messages from the session.
-   **Message Addition**: When a new message is added to the conversation, it’s automatically persisted to the session.
-   **Agent Invocation**: After each agent invocation, the agent state is synchronized with the session to capture any updates.
-   **Message Redaction**: When sensitive information needs to be redacted, the session manager can replace the original message with a redacted version while maintaining conversation flow.
(( /tab "Python" ))

(( tab "TypeScript" ))
-   **Agent Initialization**: Restores state from `snapshot_latest` if it exists.
-   **Message Addition** (`saveLatestOn: 'message'`): Saves after every message and after model calls with guardrail redactions.
-   **Agent Invocation** (`saveLatestOn: 'invocation'`, default): Saves after each invocation completes.
-   **Snapshot Trigger**: Creates an immutable checkpoint when the `snapshotTrigger` callback returns `true`.

See [Basic Usage](#basic-usage) for configuration examples.
(( /tab "TypeScript" ))

**Multi-Agent Events**:

(( tab "Python" ))
-   **Multi-Agent Initialization**: Restores orchestrator state from the session.
-   **Node Execution**: Synchronizes orchestrator state after node transitions.
-   **Multi-Agent Invocation**: Captures final orchestrator state after execution.
(( /tab "Python" ))

(( tab "TypeScript" ))
-   **Before Multi-Agent Invocation**: Restores orchestrator state from `snapshot_latest` on first invocation.
-   **After Node Call** (`multiAgentSaveLatestOn: 'node'`, default): Saves after each node completes, enabling resume from the last completed node after a crash.
-   **After Multi-Agent Invocation** (`multiAgentSaveLatestOn: 'invocation'`): Saves after the full orchestrator invocation completes (lower I/O, but only captures state at invocation boundaries).

```typescript
const session = new SessionManager({
  sessionId: 'my-session',
  storage: new LocalFileStorage('./sessions/'),
  // Save orchestrator state after each node completes (default)
  multiAgentSaveLatestOn: 'node',
  // Or save only after the full orchestrator invocation completes:
  // multiAgentSaveLatestOn: 'invocation',
})
```
(( /tab "TypeScript" ))

Direct Message Modifications Not Persisted

After initializing the agent, direct modifications to `agent.messages` will not be persisted. Utilize the [Conversation Manager](/docs/user-guide/concepts/agents/conversation-management/index.md) to help manage context of the agent in a way that can be persisted.

## Immutable Snapshots *(TypeScript only)*

In addition to `snapshot_latest`, the TypeScript SDK supports **immutable snapshots** — append-only checkpoints identified by UUID v7. These enable time-travel restore: you can restore the agent to any prior checkpoint, not just the latest state.

### Creating Immutable Snapshots

Use the `snapshotTrigger` callback to control when an immutable snapshot is created. The callback receives the current agent data and returns `true` to trigger a snapshot:

```typescript
const session = new SessionManager({
  sessionId: 'my-session',
  storage: new LocalFileStorage('./sessions/'),
  // Create an immutable snapshot after every 4 messages
  snapshotTrigger: ({ agentData }) => agentData.messages.length % 4 === 0,
})

const agent = new Agent({ sessionManager: session })
await agent.invoke('First message') // 2 messages — no snapshot
await agent.invoke('Second message') // 4 messages — immutable snapshot created
```

### Listing and Restoring Snapshots

Snapshot IDs are UUID v7, so they sort lexicographically in chronological order. Use `listSnapshotIds` on the `SessionManager` to retrieve them, then pass a `snapshotId` to `restoreSnapshot`:

```typescript
const storage = new LocalFileStorage('./sessions/')

const session = new SessionManager({
  sessionId: 'my-session',
  storage,
})
const agent = new Agent({ sessionManager: session })
await agent.initialize()

// List all immutable snapshot IDs (chronological order)
const snapshotIds = await session.listSnapshotIds({
  target: agent,
})

// Restore agent to a specific checkpoint
await session.restoreSnapshot({
  target: agent,
  snapshotId: snapshotIds[0]!,
})
```

## Deleting Sessions *(TypeScript only)*

To remove all snapshots and manifests for a session, call `deleteSession()` on the `SessionManager`. This removes the entire session root directory (filesystem) or all objects under the session prefix (S3):

```typescript
const session = new SessionManager({
  sessionId: 'my-session',
  storage: new LocalFileStorage('./sessions/'),
})

// Remove all snapshots and manifests for this session
await session.deleteSession()
```

## Data Models

(( tab "Python" ))
Session data is stored using these key data models:

**Session**

The [`Session`](/docs/api/python/strands.types.session#Session) model is the top-level container for session data:

-   **Purpose**: Provides a namespace for organizing multiple agents and their interactions
-   **Key Fields**:
    -   `session_id`: Unique identifier for the session
    -   `session_type`: Type of session (currently `"AGENT"` for both agent & multiagent in order to keep backward compatibility)
    -   `created_at`: ISO format timestamp of when the session was created
    -   `updated_at`: ISO format timestamp of when the session was last updated

**SessionAgent**

The [`SessionAgent`](/docs/api/python/strands.types.session#SessionAgent) model stores agent-specific data:

-   **Purpose**: Maintains the state and configuration of a specific agent within a session
-   **Key Fields**:
    -   `agent_id`: Unique identifier for the agent within the session
    -   `state`: Dictionary containing the agent’s state data (key-value pairs)
    -   `conversation_manager_state`: Dictionary containing the state of the conversation manager
    -   `created_at`: ISO format timestamp of when the agent was created
    -   `updated_at`: ISO format timestamp of when the agent was last updated

**SessionMessage**

The [`SessionMessage`](/docs/api/python/strands.types.session#SessionMessage) model stores individual messages in the conversation:

-   **Purpose**: Preserves the conversation history with support for message redaction
-   **Key Fields**:
    -   `message`: The original message content (role, content blocks)
    -   `redact_message`: Optional redacted version of the message (used when sensitive information is detected)
    -   `message_id`: Index of the message in the agent’s messages array
    -   `created_at`: ISO format timestamp of when the message was created
    -   `updated_at`: ISO format timestamp of when the message was last updated

These data models work together to provide a complete representation of an agent’s state and conversation history. The session management system handles serialization and deserialization of these models, including special handling for binary data using base64 encoding.

**Multi-Agent State**

Multi-agent systems serialize their state as JSON objects containing:

-   **Orchestrator Configuration**: Settings, parameters, and execution preferences
-   **Node State**: Current execution state and node transition history
-   **Shared Context**: Cross-agent shared state and variables
(( /tab "Python" ))

(( tab "TypeScript" ))
The TypeScript SDK stores session state as a `Snapshot` object written to JSON. Each snapshot contains:

-   `data.messages`: The full conversation history
-   `data.state`: Agent key-value state
-   `data.systemPrompt`: The agent’s system prompt
-   `schemaVersion`: Schema version for forward compatibility
-   `createdAt`: ISO 8601 timestamp

There are two kinds of snapshots:

-   **`snapshot_latest.json`**: A single mutable file overwritten on each save. Used to resume the most recent state after a restart.
-   **Immutable snapshots** (`immutable_history/snapshot_<uuid7>.json`): Append-only checkpoints created when `snapshotTrigger` fires. Used for time-travel restore.
(( /tab "TypeScript" ))

## Third-Party Session Managers

The following third-party session managers extend Strands with additional storage and memory capabilities:

| Session Manager | Provider | Description | Documentation |
| --- | --- | --- | --- |
| AgentCoreMemorySessionManager | Amazon | Advanced memory with intelligent retrieval using Amazon Bedrock AgentCore Memory. Supports both short-term memory (STM) and long-term memory (LTM) with strategies for user preferences, facts, and session summaries. | [View Documentation](/docs/integrations/session-managers/agentcore-memory/index.md) |
| **Contribute Your Own** | Community | Have you built a session manager? Share it with the community! | [Learn How](/integrations/index.md) |

## Custom Session Repositories

For advanced use cases, you can implement your own session storage backend.

(( tab "Python" ))
Create a custom session repository by implementing the `SessionRepository` interface:

```python
from typing import Optional
from strands import Agent
from strands.session.repository_session_manager import RepositorySessionManager
from strands.session.session_repository import SessionRepository
from strands.types.session import Session, SessionAgent, SessionMessage

class CustomSessionRepository(SessionRepository):
    """Custom session repository implementation."""

    def __init__(self):
        """Initialize with your custom storage backend."""
        # Initialize your storage backend (e.g., database connection)
        self.db = YourDatabaseClient()

    def create_session(self, session: Session) -> Session:
        """Create a new session."""
        self.db.sessions.insert(asdict(session))
        return session

    def read_session(self, session_id: str) -> Optional[Session]:
        """Read a session by ID."""
        data = self.db.sessions.find_one({"session_id": session_id})
        if data:
            return Session.from_dict(data)
        return None

    # Implement other required methods...
    # create_agent, read_agent, update_agent
    # create_message, read_message, update_message, list_messages

# Use your custom repository with RepositorySessionManager
custom_repo = CustomSessionRepository()
session_manager = RepositorySessionManager(
    session_id="user-789",
    session_repository=custom_repo
)

agent = Agent(session_manager=session_manager)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
The simplest approach is to pass any [Storage](/docs/user-guide/concepts/storage/index.md) backend directly — the `SessionManager` wraps it automatically. For full control, you can implement the `SnapshotStorage` interface:

```typescript
// Implement SnapshotStorage to plug in any backend
class MyStorage implements SnapshotStorage {
  async saveSnapshot({
    location,
    snapshotId,
    snapshot,
  }: {
    location: SnapshotLocation
    snapshotId: string
    isLatest: boolean
    snapshot: Snapshot
  }) {
    // Store the snapshot JSON keyed by location + snapshotId
  }

  async loadSnapshot({
    location,
    snapshotId,
  }: {
    location: SnapshotLocation
    snapshotId?: string
  }) {
    // Return the snapshot, or null if not found
    return null
  }

  async listSnapshotIds({
    location,
  }: {
    location: SnapshotLocation
    limit?: number
    startAfter?: string
  }) {
    // Return immutable snapshot IDs sorted chronologically
    return []
  }

  async deleteSession({ sessionId }: { sessionId: string }) {
    // Remove all stored data for this session
  }

  async loadManifest({
    location,
  }: {
    location: SnapshotLocation
  }): Promise<SnapshotManifest> {
    return {
      schemaVersion: '1',
      updatedAt: new Date().toISOString(),
    }
  }

  async saveManifest({
    location,
    manifest,
  }: {
    location: SnapshotLocation
    manifest: SnapshotManifest
  }) {
    // Persist the manifest
  }
}

const agent = new Agent({
  sessionManager: new SessionManager({
    sessionId: 'user-789',
    storage: { snapshot: new MyStorage() },
  }),
})
```
(( /tab "TypeScript" ))

This approach allows you to store session data in any backend system while leveraging the built-in session management logic.

## Data Layout

Both file and S3 backends use the same key structure:

(( tab "Python" ))
```plaintext
<root>/
└── session_<session_id>/
    ├── session.json
    ├── agents/
    │   └── agent_<agent_id>/
    │       ├── agent.json
    │       └── messages/
    │           ├── message_0.json
    │           └── message_1.json
    └── multi_agents/
        └── multi_agent_<orchestrator_id>/
            └── multi_agent.json
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```plaintext
<root>/
└── <sessionId>/
    └── scopes/
        ├── agent/
        │   └── <agentId>/
        │       └── snapshots/
        │           ├── snapshot_latest.json
        │           └── immutable_history/
        │               └── snapshot_<uuid7>.json
        └── multiAgent/
            └── <orchestratorId>/
                └── snapshots/
                    └── snapshot_latest.json
```
(( /tab "TypeScript" ))

## Session Persistence Best Practices

When implementing session persistence in your applications, consider these best practices:

-   **Use Unique Session IDs**: Generate unique session IDs for each user or conversation context to prevent data overlap.
-   **Session Cleanup**: Implement a strategy for cleaning up old or inactive sessions. Consider adding TTL (Time To Live) for sessions in production environments.
-   **Understand Persistence Triggers**: Remember that changes to agent state or messages are only persisted during specific lifecycle events.
-   **Concurrent Access**: Session managers are not thread-safe; use appropriate locking for concurrent access.
-   **Secure Storage Directories**: The session storage directory is a trusted data store. Restrict filesystem permissions so that only the agent process can read and write to it. In shared or multi-tenant environments (shared volumes, containers), be aware that the SDK does not block symlinks in the session storage directory. If an attacker with write access to the storage directory creates a symlink (e.g., `message_0.json` pointing to an arbitrary file), the SDK will follow it, which could cause sensitive file contents to be loaded into the agent’s conversation history.

## Related pages

- [State Management](/docs/user-guide/concepts/agents/state/index.md) (3 shared tags)
- [Bidirectional Streaming Session Management](/docs/user-guide/concepts/bidirectional-streaming/session-management/index.md) (2 shared tags)
- [Serialization](/docs/user-guide/evals-sdk/how-to/serialization/index.md) (1 shared tag)
- [Storage](/docs/user-guide/concepts/storage/index.md) (1 shared tag)
- [OpenAI Responses API](/docs/user-guide/concepts/model-providers/openai-responses/index.md) (1 shared tag)
- [Conversation Management](/docs/user-guide/concepts/agents/conversation-management/index.md) (1 shared tag)


## Implementation

### Python

- [harness-sdk/strands-py/src/strands/session/session_manager.py](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/session_manager.py)
- [harness-sdk/strands-py/src/strands/session/file_session_manager.py](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/file_session_manager.py)
- [harness-sdk/strands-py/src/strands/session/s3_session_manager.py](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/s3_session_manager.py)

### TypeScript

- [harness-sdk/strands-ts/src/session/session-manager.ts](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/session/session-manager.ts)
