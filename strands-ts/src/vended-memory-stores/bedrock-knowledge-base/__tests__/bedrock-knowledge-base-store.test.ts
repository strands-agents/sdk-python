import { describe, it, expect, beforeEach, vi, type MockedFunction } from 'vitest'
import { BedrockAgentRuntimeClient } from '@aws-sdk/client-bedrock-agent-runtime'
import { BedrockAgentClient } from '@aws-sdk/client-bedrock-agent'
import { BedrockKnowledgeBaseStore } from '../store.js'
import { MemoryManager } from '../../../memory/index.js'
import { logger } from '../../../logging/logger.js'

// Mock the AWS SDK clients. Command classes are stubbed to echo their input as `{ input }`, so a
// test can assert on `send`'s argument — mirroring src/session/__tests__/s3-storage.test.ts. Client
// constructors return a stub with a `send` spy; tests that exercise behavior inject their own client
// instead (the store accepts `config.runtimeClient` / `config.agentClient` / `config.s3.client`).
vi.mock('@aws-sdk/client-bedrock-agent-runtime', () => ({
  BedrockAgentRuntimeClient: vi.fn().mockImplementation(function () {
    return { send: vi.fn() }
  }),
  RetrieveCommand: vi.fn().mockImplementation(function (input) {
    return { input }
  }),
}))

vi.mock('@aws-sdk/client-bedrock-agent', () => ({
  BedrockAgentClient: vi.fn().mockImplementation(function () {
    return { send: vi.fn() }
  }),
  IngestKnowledgeBaseDocumentsCommand: vi.fn().mockImplementation(function (input) {
    return { input }
  }),
}))

vi.mock('@aws-sdk/client-s3', () => ({
  S3Client: vi.fn().mockImplementation(function () {
    return { send: vi.fn() }
  }),
  PutObjectCommand: vi.fn().mockImplementation(function (input) {
    return { input }
  }),
}))

// uuid v7 backs both the S3 object key and the CUSTOM document id; pin it for deterministic assertions.
vi.mock('uuid', () => ({
  v7: vi.fn().mockReturnValue('test-uuid-v7'),
}))

/** A stub AWS client whose `send` is a spy the test can program and inspect. */
function mockClient(): { send: MockedFunction<any> } {
  return { send: vi.fn() }
}

/**
 * Builds a store from a base connection config plus per-store fields. `config` overrides merge onto
 * the shared connection (`knowledgeBaseId: 'kb-1'` + injected runtime/agent clients); `overrides`
 * carry the per-store fields that sit outside `config` (`name`, `scope`, `writable`, ...). The
 * returned `runtime` / `agent` spies are the injected clients, ready to program and inspect.
 */
function makeStore(
  overrides: Record<string, unknown> = {},
  configOverrides: Record<string, unknown> = {}
): {
  store: BedrockKnowledgeBaseStore
  runtime: { send: MockedFunction<any> }
  agent: { send: MockedFunction<any> }
} {
  const runtime = mockClient()
  runtime.send.mockResolvedValue({ retrievalResults: [] })
  const agent = mockClient()
  agent.send.mockResolvedValue({})
  const store = new BedrockKnowledgeBaseStore({
    config: { knowledgeBaseId: 'kb-1', runtimeClient: runtime as any, agentClient: agent as any, ...configOverrides },
    name: 'kb',
    ...overrides,
  })
  return { store, runtime, agent }
}

/** A writable CUSTOM store built via {@link makeStore}, with the data source wired into `config`. */
function makeCustomStore(
  overrides: Record<string, unknown> = {},
  configOverrides: Record<string, unknown> = {}
): { store: BedrockKnowledgeBaseStore; agent: { send: MockedFunction<any> } } {
  const { store, agent } = makeStore(
    { writable: true, ...overrides },
    { dataSourceType: 'CUSTOM', dataSourceId: 'ds-1', ...configOverrides }
  )
  return { store, agent }
}

/** A writable S3 store built via {@link makeStore}, with the data source + bucket wired into `config`. */
function makeS3Store(
  overrides: Record<string, unknown> = {},
  configOverrides: Record<string, unknown> = {}
): { store: BedrockKnowledgeBaseStore; agent: { send: MockedFunction<any> }; s3: { send: MockedFunction<any> } } {
  const s3 = mockClient()
  s3.send.mockResolvedValue({})
  const { store, agent } = makeStore(
    { writable: true, ...overrides },
    {
      dataSourceType: 'S3',
      dataSourceId: 'ds-1',
      s3: { bucket: 'my-bucket', client: s3 as any, prefix: 'memories' },
      ...configOverrides,
    }
  )
  return { store, agent, s3 }
}

/** Reads the filter the most recent RetrieveCommand was sent with. */
function lastSearchFilter(runtime: { send: MockedFunction<any> }): unknown {
  const calls = runtime.send.mock.calls
  return calls[calls.length - 1]?.[0].input.retrievalConfiguration.vectorSearchConfiguration.filter
}

/** Reads the inline attributes the most recent CUSTOM ingestion document carried. */
function lastInlineAttributes(agent: { send: MockedFunction<any> }): any[] {
  const calls = agent.send.mock.calls
  return calls[calls.length - 1]?.[0].input.documents[0].metadata?.inlineAttributes ?? []
}

describe('BedrockKnowledgeBaseStore', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('constructor', () => {
    it('exposes name and defaults writable to false', () => {
      const { store } = makeStore()
      expect(store.name).toBe('kb')
      expect(store.writable).toBe(false)
      expect(store.description).toBeUndefined()
      expect(store.maxSearchResults).toBeUndefined()
    })

    it('keeps name and scope as independent fields', () => {
      const { store } = makeStore({ name: 'explicit', scope: 'user-abc' })
      expect(store.name).toBe('explicit')
      expect(store.scope).toBe('user-abc')
    })

    it('requires name at the type level', () => {
      // @ts-expect-error name is required
      new BedrockKnowledgeBaseStore({ config: { knowledgeBaseId: 'kb-1' }, scope: 'user-abc' })
    })

    it('carries through description and maxSearchResults', () => {
      const { store } = makeStore({ description: 'product docs', maxSearchResults: 7 })
      expect(store.description).toBe('product docs')
      expect(store.maxSearchResults).toBe(7)
    })

    it('throws when writable is true but dataSourceType is omitted', () => {
      expect(() => makeStore({ writable: true })).toThrow("add requires dataSourceType 'CUSTOM' or 'S3'")
    })

    it("throws when writable is true but dataSourceType is 'OTHER'", () => {
      expect(() => makeStore({ writable: true }, { dataSourceType: 'OTHER' })).toThrow(
        "add requires dataSourceType 'CUSTOM' or 'S3'"
      )
    })

    it('throws when maxSearchResults is less than 1', () => {
      expect(() => makeStore({ maxSearchResults: 0 })).toThrow('maxSearchResults must be at least 1')
      expect(() => makeStore({ maxSearchResults: -5 })).toThrow('maxSearchResults must be at least 1')
    })

    it("allows writable with a 'CUSTOM' data source", () => {
      const { store } = makeCustomStore()
      expect(store.writable).toBe(true)
    })

    it("allows writable with an 'S3' data source", () => {
      const { store } = makeS3Store()
      expect(store.writable).toBe(true)
    })

    it('constructs a default runtime client when none is injected', () => {
      new BedrockKnowledgeBaseStore({ config: { knowledgeBaseId: 'kb-1' }, name: 'kb' })
      expect(vi.mocked(BedrockAgentRuntimeClient)).toHaveBeenCalledWith({})
    })

    it('uses the injected runtime client without constructing one', () => {
      makeStore()
      expect(vi.mocked(BedrockAgentRuntimeClient)).not.toHaveBeenCalled()
    })
  })

  describe('search', () => {
    it('issues a RetrieveCommand with the query and a default result limit of 10', async () => {
      const { store, runtime } = makeStore()
      await store.search('how do refunds work')
      expect(runtime.send).toHaveBeenCalledWith(
        expect.objectContaining({
          input: {
            knowledgeBaseId: 'kb-1',
            retrievalQuery: { text: 'how do refunds work' },
            retrievalConfiguration: { vectorSearchConfiguration: { numberOfResults: 10 } },
          },
        })
      )
    })

    it("uses the store's maxSearchResults when the caller omits one", async () => {
      const { store, runtime } = makeStore({ maxSearchResults: 5 })
      await store.search('q')
      expect(runtime.send).toHaveBeenCalledWith(
        expect.objectContaining({
          input: expect.objectContaining({
            retrievalConfiguration: { vectorSearchConfiguration: { numberOfResults: 5 } },
          }),
        })
      )
    })

    it('lets a per-call maxSearchResults override the store default', async () => {
      const { store, runtime } = makeStore({ maxSearchResults: 5 })
      await store.search('q', { maxSearchResults: 2 })
      expect(runtime.send).toHaveBeenCalledWith(
        expect.objectContaining({
          input: expect.objectContaining({
            retrievalConfiguration: { vectorSearchConfiguration: { numberOfResults: 2 } },
          }),
        })
      )
    })

    it('derives a scope filter (default key "namespace") and applies it to retrieval', async () => {
      const { store, runtime } = makeStore({ scope: 'user-123' })
      await store.search('q')
      expect(runtime.send).toHaveBeenCalledWith(
        expect.objectContaining({
          input: expect.objectContaining({
            retrievalConfiguration: {
              vectorSearchConfiguration: {
                numberOfResults: 10,
                filter: { equals: { key: 'namespace', value: 'user-123' } },
              },
            },
          }),
        })
      )
    })

    it('honors a custom scopeMetadataKey when building the scope filter', async () => {
      const { store, runtime } = makeStore({ scope: 'acme' }, { scopeMetadataKey: 'tenant' })
      await store.search('q')
      expect(runtime.send).toHaveBeenCalledWith(
        expect.objectContaining({
          input: expect.objectContaining({
            retrievalConfiguration: expect.objectContaining({
              vectorSearchConfiguration: expect.objectContaining({
                filter: { equals: { key: 'tenant', value: 'acme' } },
              }),
            }),
          }),
        })
      )
    })

    it('prefers an explicit filter over a scope-derived one', async () => {
      const filter = { equals: { key: 'custom', value: 'v' } }
      const { store, runtime } = makeStore({ scope: 'ignored', filter })
      await store.search('q')
      expect(runtime.send).toHaveBeenCalledWith(
        expect.objectContaining({
          input: expect.objectContaining({
            retrievalConfiguration: expect.objectContaining({
              vectorSearchConfiguration: expect.objectContaining({ filter }),
            }),
          }),
        })
      )
    })

    it('maps content, metadata, location, and score onto each entry', async () => {
      const { store, runtime } = makeStore()
      runtime.send.mockResolvedValue({
        retrievalResults: [
          {
            content: { text: 'refunds take 5 days' },
            metadata: { source: 'faq' },
            location: { type: 'S3', s3Location: { uri: 's3://b/k' } },
            score: 0.92,
          },
        ],
      })

      const results = await store.search('q')
      expect(results).toStrictEqual([
        {
          content: 'refunds take 5 days',
          metadata: {
            source: 'faq',
            _sourceLocation: { type: 'S3', s3Location: { uri: 's3://b/k' } },
            _relevanceScore: 0.92,
          },
        },
      ])
    })

    it('defaults missing content to an empty string and omits absent metadata', async () => {
      const { store, runtime } = makeStore()
      runtime.send.mockResolvedValue({ retrievalResults: [{}] })

      const results = await store.search('q')
      expect(results).toStrictEqual([{ content: '', metadata: {} }])
    })

    it('returns an empty array when the knowledge base yields no results', async () => {
      const { store } = makeStore()
      await expect(store.search('q')).resolves.toStrictEqual([])
    })

    it('returns an empty array when the response omits retrievalResults entirely', async () => {
      const { store, runtime } = makeStore()
      runtime.send.mockResolvedValue({})
      await expect(store.search('q')).resolves.toStrictEqual([])
    })

    it('logs and rethrows when the retrieve call fails', async () => {
      const errorSpy = vi.spyOn(logger, 'error').mockImplementation(() => {})
      const { store, runtime } = makeStore()
      runtime.send.mockRejectedValue(new Error('retrieve boom'))

      await expect(store.search('q')).rejects.toThrow('retrieve boom')
      expect(errorSpy).toHaveBeenCalled()
      errorSpy.mockRestore()
    })
  })

  describe('add — CUSTOM data source', () => {
    it('throws when dataSourceId is missing', () => {
      expect(() => makeStore({ writable: true }, { dataSourceType: 'CUSTOM' })).toThrow('dataSourceId is required')
    })

    it('throws when content is empty or whitespace-only', async () => {
      const { store } = makeCustomStore()
      await expect(store.add('')).rejects.toThrow('content must not be empty')
      await expect(store.add('   ')).rejects.toThrow('content must not be empty')
    })

    it('returns the generated custom document id', async () => {
      const { store } = makeCustomStore()
      await expect(store.add('fact')).resolves.toStrictEqual({ documentId: 'test-uuid-v7' })
    })

    it('uses the same id for the document identifier and the returned documentId', async () => {
      const { store, agent } = makeCustomStore()
      const result = await store.add('fact')
      const document = agent.send.mock.calls[0]?.[0].input.documents[0]
      expect(document.content.custom.customDocumentIdentifier.id).toBe(result.documentId)
    })

    it('ingests an inline CUSTOM document with no metadata field when no scope or metadata', async () => {
      const { store, agent } = makeCustomStore()
      await store.add('remember this')
      expect(agent.send).toHaveBeenCalledWith(
        expect.objectContaining({
          input: {
            knowledgeBaseId: 'kb-1',
            dataSourceId: 'ds-1',
            documents: [
              {
                content: {
                  dataSourceType: 'CUSTOM',
                  custom: {
                    customDocumentIdentifier: { id: 'test-uuid-v7' },
                    sourceType: 'IN_LINE',
                    inlineContent: { type: 'TEXT', textContent: { data: 'remember this' } },
                  },
                },
              },
            ],
          },
        })
      )
    })

    it('attaches the scope as a leading inline attribute', async () => {
      const { store, agent } = makeCustomStore({ scope: 'user-123' })
      await store.add('fact')
      const document = agent.send.mock.calls[0]?.[0].input.documents[0]
      expect(document.metadata.inlineAttributes).toStrictEqual([
        { key: 'namespace', value: { type: 'STRING', stringValue: 'user-123' } },
      ])
    })

    it('drops metadata keys that collide with scopeMetadataKey and preserves scope', async () => {
      const { store, agent } = makeCustomStore({ scope: 'tenant-A' })
      const warnSpy = vi.spyOn(logger, 'warn')
      await store.add('fact', { namespace: 'tenant-EVIL', other: 'ok' })
      const document = agent.send.mock.calls[0]?.[0].input.documents[0]
      const attrs = document.metadata.inlineAttributes
      expect(attrs.filter((a: any) => a.key === 'namespace')).toHaveLength(1)
      expect(attrs.find((a: any) => a.key === 'namespace').value.stringValue).toBe('tenant-A')
      expect(attrs.find((a: any) => a.key === 'other')).toBeDefined()
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('collides with scopeMetadataKey'))
      warnSpy.mockRestore()
    })

    it('maps supported metadata value types and skips unsupported ones', async () => {
      const { store, agent } = makeCustomStore()
      await store.add('fact', {
        str: 'a',
        num: 1,
        bool: false,
        arr: ['x', 'y'],
        obj: { nested: true },
        nul: null,
        mixedArr: [1, 'a'],
      })
      const document = agent.send.mock.calls[0]?.[0].input.documents[0]
      expect(document.metadata.inlineAttributes).toStrictEqual([
        { key: 'str', value: { type: 'STRING', stringValue: 'a' } },
        { key: 'num', value: { type: 'NUMBER', numberValue: 1 } },
        { key: 'bool', value: { type: 'BOOLEAN', booleanValue: false } },
        { key: 'arr', value: { type: 'STRING_LIST', stringListValue: ['x', 'y'] } },
      ])
    })

    it('logs and rethrows when ingestion fails', async () => {
      const errorSpy = vi.spyOn(logger, 'error').mockImplementation(() => {})
      const { store, agent } = makeCustomStore()
      agent.send.mockRejectedValue(new Error('ingest boom'))

      await expect(store.add('fact')).rejects.toThrow('ingest boom')
      expect(errorSpy).toHaveBeenCalled()
      errorSpy.mockRestore()
    })

    it('lazily constructs a default agent client when none is injected', async () => {
      // No injected agentClient, so construct directly rather than via the helper.
      const store = new BedrockKnowledgeBaseStore({
        config: { knowledgeBaseId: 'kb-1', dataSourceType: 'CUSTOM', dataSourceId: 'ds-1' },
        name: 'kb',
        writable: true,
      })
      await store.add('fact')
      expect(vi.mocked(BedrockAgentClient)).toHaveBeenCalledWith({})
    })
  })

  describe('add — S3 data source', () => {
    it('throws when the s3 config is missing', () => {
      expect(() => makeStore({ writable: true }, { dataSourceType: 'S3', dataSourceId: 'ds-1' })).toThrow(
        's3 config is required'
      )
    })

    it("returns the uploaded content object's s3:// URI as the document id", async () => {
      const { store } = makeS3Store()
      await expect(store.add('content')).resolves.toStrictEqual({
        documentId: 's3://my-bucket/memories/test-uuid-v7.txt',
      })
    })

    it('uploads the content object and ingests an S3 document referencing it (no sidecar)', async () => {
      const { store, agent, s3 } = makeS3Store()
      await store.add('s3 content')

      expect(s3.send).toHaveBeenCalledTimes(1)
      expect(s3.send).toHaveBeenCalledWith(
        expect.objectContaining({
          input: {
            Bucket: 'my-bucket',
            Key: 'memories/test-uuid-v7.txt',
            Body: 's3 content',
            ContentType: 'text/plain; charset=utf-8',
          },
        })
      )
      expect(agent.send).toHaveBeenCalledWith(
        expect.objectContaining({
          input: {
            knowledgeBaseId: 'kb-1',
            dataSourceId: 'ds-1',
            documents: [
              {
                content: {
                  dataSourceType: 'S3',
                  s3: { s3Location: { uri: 's3://my-bucket/memories/test-uuid-v7.txt' } },
                },
              },
            ],
          },
        })
      )
    })

    it('does not double up the slash when the prefix already ends in one', async () => {
      const client = mockClient()
      client.send.mockResolvedValue({})
      const { store } = makeS3Store({}, { s3: { bucket: 'my-bucket', client: client as any, prefix: 'memories/' } })
      await store.add('content')
      expect(client.send).toHaveBeenCalledWith(
        expect.objectContaining({ input: expect.objectContaining({ Key: 'memories/test-uuid-v7.txt' }) })
      )
    })

    it('writes a sidecar carrying the scope and points the document metadata at it', async () => {
      const { store, agent, s3 } = makeS3Store({ scope: 'team-a' })
      await store.add('content')

      expect(s3.send).toHaveBeenCalledTimes(2)
      expect(s3.send).toHaveBeenNthCalledWith(
        2,
        expect.objectContaining({
          input: {
            Bucket: 'my-bucket',
            Key: 'memories/test-uuid-v7.txt.metadata.json',
            Body: JSON.stringify({
              metadataAttributes: {
                namespace: { value: { type: 'STRING', stringValue: 'team-a' }, includeForEmbedding: false },
              },
            }),
            ContentType: 'application/json',
          },
        })
      )
      const document = agent.send.mock.calls[0]?.[0].input.documents[0]
      expect(document.metadata).toStrictEqual({
        type: 'S3_LOCATION',
        s3Location: { uri: 's3://my-bucket/memories/test-uuid-v7.txt.metadata.json' },
      })
    })

    it('writes a sidecar built from caller metadata', async () => {
      const { store, s3 } = makeS3Store()
      await store.add('content', { priority: 'high' })

      expect(s3.send).toHaveBeenCalledTimes(2)
      expect(s3.send).toHaveBeenNthCalledWith(
        2,
        expect.objectContaining({
          input: expect.objectContaining({
            Body: JSON.stringify({
              metadataAttributes: {
                priority: { value: { type: 'STRING', stringValue: 'high' }, includeForEmbedding: false },
              },
            }),
          }),
        })
      )
    })

    it('omits unsupported metadata values from the sidecar', async () => {
      const { store, s3 } = makeS3Store()
      await store.add('content', { keep: 'yes', drop: { nested: true } })

      expect(s3.send).toHaveBeenNthCalledWith(
        2,
        expect.objectContaining({
          input: expect.objectContaining({
            Body: JSON.stringify({
              metadataAttributes: {
                keep: { value: { type: 'STRING', stringValue: 'yes' }, includeForEmbedding: false },
              },
            }),
          }),
        })
      )
    })

    it('lazily constructs an agent client with no config when neither client nor config is given', async () => {
      const s3 = mockClient()
      s3.send.mockResolvedValue({})
      const store = new BedrockKnowledgeBaseStore({
        config: {
          knowledgeBaseId: 'kb-1',
          dataSourceType: 'S3',
          dataSourceId: 'ds-1',
          s3: { bucket: 'my-bucket', client: s3 as any, prefix: 'memories' },
        },
        name: 'kb',
        writable: true,
      })
      await store.add('content')
      expect(vi.mocked(BedrockAgentClient)).toHaveBeenCalledWith({})
    })

    it('logs and rethrows when the S3 upload fails, before any ingestion', async () => {
      const errorSpy = vi.spyOn(logger, 'error').mockImplementation(() => {})
      const { store, agent, s3 } = makeS3Store()
      s3.send.mockRejectedValue(new Error('upload boom'))

      await expect(store.add('content')).rejects.toThrow('upload boom')
      expect(errorSpy).toHaveBeenCalled()
      expect(agent.send).not.toHaveBeenCalled()
      errorSpy.mockRestore()
    })
  })

  describe('add — non-writable store', () => {
    it('throws when add is called on a non-writable store', async () => {
      const agent = mockClient()
      const store = new BedrockKnowledgeBaseStore({
        config: { knowledgeBaseId: 'kb-1', dataSourceType: 'OTHER', dataSourceId: 'ds-1', agentClient: agent as any },
        name: 'kb',
      })
      await expect(store.add('fact')).rejects.toThrow('store is not writable')
      expect(agent.send).not.toHaveBeenCalled()
    })
  })

  describe('config reuse across namespaces', () => {
    it('reuses an injected runtime client across stores built from the same config', async () => {
      const runtime = mockClient()
      runtime.send.mockResolvedValue({ retrievalResults: [] })
      const config = { knowledgeBaseId: 'kb-1', runtimeClient: runtime as any }

      const personal = new BedrockKnowledgeBaseStore({ config, name: 'personal', scope: 'user-abc' })
      const team = new BedrockKnowledgeBaseStore({ config, name: 'team', scope: 'other' })

      expect(vi.mocked(BedrockAgentRuntimeClient)).not.toHaveBeenCalled()
      await personal.search('q')
      expect(lastSearchFilter(runtime)).toStrictEqual({ equals: { key: 'namespace', value: 'user-abc' } })
      await team.search('q')
      expect(lastSearchFilter(runtime)).toStrictEqual({ equals: { key: 'namespace', value: 'other' } })
    })

    it('constructs a separate default runtime client per store when the config injects none', () => {
      const config = { knowledgeBaseId: 'kb-1' }
      new BedrockKnowledgeBaseStore({ config, name: 'a' })
      new BedrockKnowledgeBaseStore({ config, name: 'b' })
      expect(vi.mocked(BedrockAgentRuntimeClient)).toHaveBeenCalledTimes(2)
    })

    it('inherits the data source from the shared config when writing', async () => {
      const agent = mockClient()
      agent.send.mockResolvedValue({})
      const config = {
        knowledgeBaseId: 'kb-shared',
        dataSourceType: 'CUSTOM' as const,
        dataSourceId: 'ds-shared',
        agentClient: agent as any,
      }
      const store = new BedrockKnowledgeBaseStore({ config, name: 'personal', scope: 'user-abc', writable: true })
      await store.add('fact')
      expect(agent.send).toHaveBeenCalledWith(
        expect.objectContaining({
          input: expect.objectContaining({ knowledgeBaseId: 'kb-shared', dataSourceId: 'ds-shared' }),
        })
      )
    })

    it('registers distinct-name stores together in a MemoryManager', () => {
      const config = { knowledgeBaseId: 'kb-1', runtimeClient: mockClient() as any }
      const personal = new BedrockKnowledgeBaseStore({ config, name: 'personal', scope: 'user-abc' })
      const team = new BedrockKnowledgeBaseStore({ config, name: 'team', scope: 'other' })
      expect(() => new MemoryManager({ stores: [personal, team] })).not.toThrow()
    })

    it('rejects two stores with the same name in a MemoryManager', () => {
      const config = { knowledgeBaseId: 'kb-1', runtimeClient: mockClient() as any }
      const a = new BedrockKnowledgeBaseStore({ config, name: 'dupe', scope: 'user-abc' })
      const b = new BedrockKnowledgeBaseStore({ config, name: 'dupe', scope: 'other' })
      expect(() => new MemoryManager({ stores: [a, b] })).toThrow("duplicate store name 'dupe'")
    })
  })

  describe('scope and filter resolution', () => {
    it('applies no filter when the store has no scope', async () => {
      const { store, runtime } = makeStore()
      await store.search('q')
      expect(lastSearchFilter(runtime)).toBeUndefined()
    })

    it('scopes writes by scope even when an explicit search filter is set (search/write asymmetry)', async () => {
      const { store, agent } = makeCustomStore({ scope: 'tenant-a', filter: { equals: { key: 'custom', value: 'v' } } })
      await store.add('fact')
      expect(lastInlineAttributes(agent)).toStrictEqual([
        { key: 'namespace', value: { type: 'STRING', stringValue: 'tenant-a' } },
      ])
    })
  })

  describe('metadata logging', () => {
    it('logs a debug line when a CUSTOM document drops an unsupported metadata value', async () => {
      const debugSpy = vi.spyOn(logger, 'debug').mockImplementation(() => {})
      const { store } = makeCustomStore()

      await store.add('fact', { good: 'v', bad: { nested: true } })
      expect(debugSpy).toHaveBeenCalledWith(expect.stringContaining('key=<bad>'))
      expect(debugSpy).not.toHaveBeenCalledWith(expect.stringContaining('key=<good>'))
      debugSpy.mockRestore()
    })
  })
})
