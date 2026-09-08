import { Agent, SessionManager } from '@strands-agents/sdk'
import { ContextOffloader } from '@strands-agents/sdk/vended-plugins/context-offloader'
import { InMemoryStorage, LocalFileStorage, S3Storage } from '@strands-agents/sdk/storage'
import { KeywordSearchStrategy } from '@strands-agents/sdk/storage/search'
import { QmdSearchStrategy } from '@strands-agents/sdk/storage/search/qmd'
import { FileMemoryStore } from '@strands-agents/sdk/vended-memory-stores/file-memory-store'

async function agentLevel() {
  // --8<-- [start:agent_level]
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
  // --8<-- [end:agent_level]
}

async function perPlugin() {
  // --8<-- [start:per_plugin]
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
  // --8<-- [end:per_plugin]
}

async function inMemory() {
  // --8<-- [start:in_memory]
  const storage = new InMemoryStorage()
  // --8<-- [end:in_memory]
}

async function localFile() {
  // --8<-- [start:local_file]
  const storage = new LocalFileStorage('./my-data/')
  // --8<-- [end:local_file]
}

async function s3() {
  // --8<-- [start:s3]
  const storage = new S3Storage('my-bucket', {
    prefix: 'agents/prod/',
  })
  // --8<-- [end:s3]
}

async function keywordSearch() {
  // --8<-- [start:keyword_search]
  const storage = new LocalFileStorage('./my-data/')
  const results = await KeywordSearchStrategy.search(
    storage,
    'dark mode toggle',
  )
  // --8<-- [end:keyword_search]
}

async function qmdSearch() {
  // --8<-- [start:qmd_search]
  const storage = new LocalFileStorage('./memory/')
  const search = new QmdSearchStrategy()

  const results = await search.search(
    storage,
    'authentication flow',
  )

  await search.close()
  // --8<-- [end:qmd_search]
}

async function fileMemoryQmd() {
  // --8<-- [start:file_memory_qmd]
  const store = new FileMemoryStore({
    name: 'agent-memory',
    storage: new LocalFileStorage('./memory/'),
    search: new QmdSearchStrategy(),
  })
  // --8<-- [end:file_memory_qmd]
}
