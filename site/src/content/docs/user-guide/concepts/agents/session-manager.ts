import { Agent, SessionManager, FileStorage, Graph, Swarm } from '@strands-agents/sdk'
import { S3Storage } from '@strands-agents/sdk/session/s3-storage'
import type {
  SnapshotStorage,
  SnapshotLocation,
  Snapshot,
  SnapshotManifest,
} from '@strands-agents/sdk'
import { S3Client } from '@aws-sdk/client-s3'

// =====================
// Basic Usage
// =====================

async function basicFileStorageExample() {
  // --8<-- [start:basic_file_storage]
  const session = new SessionManager({
    sessionId: 'test-session',
    storage: { snapshot: new FileStorage('./sessions') },
  })

  const agent = new Agent({ sessionManager: session })

  // Use the agent - all messages and state are automatically persisted
  await agent.invoke('Hello!') // This conversation is persisted
  // --8<-- [end:basic_file_storage]
}

// =====================
// FileStorage
// =====================

async function sessionAsPluginExample() {
  // --8<-- [start:session_as_plugin]
  const session = new SessionManager({
    sessionId: 'test-session',
    storage: { snapshot: new FileStorage('./sessions') },
  })

  // Equivalent to passing via sessionManager field
  const agent = new Agent({ plugins: [session] })
  await agent.invoke('Hello!')
  // --8<-- [end:session_as_plugin]
}

async function fileStorageExample() {
  // --8<-- [start:file_storage]
  const session = new SessionManager({
    sessionId: 'user-123',
    storage: { snapshot: new FileStorage('./sessions') },
  })

  const agent = new Agent({ sessionManager: session })
  await agent.invoke("Hello, I'm a new user!")
  // --8<-- [end:file_storage]
}

// =====================
// S3Storage
// =====================

async function s3StorageExample() {
  // --8<-- [start:s3_storage]
  const session = new SessionManager({
    sessionId: 'user-456',
    storage: {
      snapshot: new S3Storage({
        bucket: 'my-agent-sessions',
        prefix: 'production', // Optional key prefix
        s3Client: new S3Client({
          // Optional pre-configured client
          region: 'us-west-2',
        }),
        // Alternatively, use region directly (cannot be combined with s3Client):
        // region: 'us-west-2',
      }),
    },
  })

  const agent = new Agent({ sessionManager: session })
  await agent.invoke('Tell me about AWS S3')
  // --8<-- [end:s3_storage]
}

// =====================
// Multi-Agent Sessions
// =====================

async function multiAgentGraphSessionExample() {
  // --8<-- [start:multi_agent_graph_session]
  const session = new SessionManager({
    sessionId: 'graph-session',
    storage: { snapshot: new FileStorage('./sessions') },
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
  // --8<-- [end:multi_agent_graph_session]
}

async function multiAgentSwarmSessionExample() {
  // --8<-- [start:multi_agent_swarm_session]
  const session = new SessionManager({
    sessionId: 'swarm-session',
    storage: { snapshot: new FileStorage('./sessions') },
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
  // --8<-- [end:multi_agent_swarm_session]
}

// =====================
// SaveLatestStrategy
// =====================

async function saveLatestStrategyExample() {
  // --8<-- [start:save_latest_strategy]
  const session = new SessionManager({
    sessionId: 'my-session',
    storage: { snapshot: new FileStorage('./sessions') },
    saveLatestOn: 'invocation', // default — also: 'message' | 'trigger'
  })
  // --8<-- [end:save_latest_strategy]
}

async function multiAgentSaveLatestStrategyExample() {
  // --8<-- [start:multi_agent_save_latest_strategy]
  const session = new SessionManager({
    sessionId: 'my-session',
    storage: { snapshot: new FileStorage('./sessions') },
    // Save orchestrator state after each node completes (default)
    multiAgentSaveLatestOn: 'node',
    // Or save only after the full orchestrator invocation completes:
    // multiAgentSaveLatestOn: 'invocation',
  })
  // --8<-- [end:multi_agent_save_latest_strategy]
}

// =====================
// Immutable Snapshots
// =====================

async function snapshotTriggerExample() {
  // --8<-- [start:snapshot_trigger]
  const session = new SessionManager({
    sessionId: 'my-session',
    storage: { snapshot: new FileStorage('./sessions') },
    // Create an immutable snapshot after every 4 messages
    snapshotTrigger: ({ agentData }) => agentData.messages.length % 4 === 0,
  })

  const agent = new Agent({ sessionManager: session })
  await agent.invoke('First message') // 2 messages — no snapshot
  await agent.invoke('Second message') // 4 messages — immutable snapshot created
  // --8<-- [end:snapshot_trigger]
}

// =====================
// List and Restore Snapshots
// =====================

async function listAndRestoreExample() {
  // --8<-- [start:list_and_restore]
  const storage = new FileStorage('./sessions')
  const location = {
    sessionId: 'my-session',
    scope: 'agent' as const,
    scopeId: 'default',
  }

  // List all immutable snapshot IDs (chronological order)
  const snapshotIds = await storage.listSnapshotIds({ location })

  // Paginate: get the next 10 snapshots after a cursor
  const page2 = await storage.listSnapshotIds({
    location,
    limit: 10,
    startAfter: snapshotIds.at(-1),
  })

  // Restore agent to a specific checkpoint
  const session = new SessionManager({
    sessionId: 'my-session',
    storage: { snapshot: storage },
  })
  const agent = new Agent({ sessionManager: session })
  await agent.initialize()
  await session.restoreSnapshot({ target: agent, snapshotId: snapshotIds[0]! })
  // --8<-- [end:list_and_restore]
}

// =====================
// Custom Storage
// =====================

async function customStorageExample() {
  // --8<-- [start:custom_storage]
  // Implement SnapshotStorage to plug in any backend (database, Redis, etc.)
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
      // Return the snapshot for the given location, or null if not found
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
      // Return the manifest for the given location
      return { schemaVersion: '1', updatedAt: new Date().toISOString() }
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
  // --8<-- [end:custom_storage]
}

// =====================
// Delete Session
// =====================

async function deleteSessionExample() {
  // --8<-- [start:delete_session]
  const session = new SessionManager({
    sessionId: 'my-session',
    storage: { snapshot: new FileStorage('./sessions') },
  })

  // Remove all snapshots and manifests for this session
  await session.deleteSession()
  // --8<-- [end:delete_session]
}
