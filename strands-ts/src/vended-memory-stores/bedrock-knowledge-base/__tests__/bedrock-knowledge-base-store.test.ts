import { describe, it, expect, beforeEach, vi, type MockedFunction } from 'vitest'
import { BedrockAgentRuntimeClient } from '@aws-sdk/client-bedrock-agent-runtime'
import { BedrockAgentClient } from '@aws-sdk/client-bedrock-agent'
import { BedrockKnowledgeBaseStore } from '../store.js'
import { logger } from '../../../logging/logger.js'

// Mock the AWS SDK clients. Command classes are stubbed to echo their input as `{ input }`, so a
// test can assert on `send`'s argument — mirroring src/session/__tests__/s3-storage.test.ts. Client
// constructors return a stub with a `send` spy; tests that exercise behavior inject their own client
// instead (the store accepts `runtimeClient` / `agentClient` / `s3.client`).
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

describe('BedrockKnowledgeBaseStore', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('constructor', () => {
    it('exposes name and defaults writable to false', () => {
      const store = new BedrockKnowledgeBaseStore({ name: 'kb', knowledgeBaseId: 'kb-1' })
      expect(store.name).toBe('kb')
      expect(store.writable).toBe(false)
      expect(store.description).toBeUndefined()
      expect(store.maxSearchResults).toBeUndefined()
    })

    it('carries through description and maxSearchResults', () => {
      const store = new BedrockKnowledgeBaseStore({
        name: 'kb',
        knowledgeBaseId: 'kb-1',
        description: 'product docs',
        maxSearchResults: 7,
      })
      expect(store.description).toBe('product docs')
      expect(store.maxSearchResults).toBe(7)
    })

    it('throws when writable is true but dataSourceType is omitted', () => {
      expect(() => new BedrockKnowledgeBaseStore({ name: 'kb', knowledgeBaseId: 'kb-1', writable: true })).toThrow(
        "add requires dataSourceType 'CUSTOM' or 'S3'"
      )
    })

    it("throws when writable is true but dataSourceType is 'OTHER'", () => {
      expect(
        () =>
          new BedrockKnowledgeBaseStore({
            name: 'kb',
            knowledgeBaseId: 'kb-1',
            writable: true,
            dataSourceType: 'OTHER',
          })
      ).toThrow("add requires dataSourceType 'CUSTOM' or 'S3'")
    })

    it('throws when maxSearchResults is less than 1', () => {
      expect(() => new BedrockKnowledgeBaseStore({ name: 'kb', knowledgeBaseId: 'kb-1', maxSearchResults: 0 })).toThrow(
        'maxSearchResults must be at least 1'
      )
      expect(
        () => new BedrockKnowledgeBaseStore({ name: 'kb', knowledgeBaseId: 'kb-1', maxSearchResults: -5 })
      ).toThrow('maxSearchResults must be at least 1')
    })

    it("allows writable with a 'CUSTOM' data source", () => {
      const store = new BedrockKnowledgeBaseStore({
        name: 'kb',
        knowledgeBaseId: 'kb-1',
        writable: true,
        dataSourceType: 'CUSTOM',
        dataSourceId: 'ds-1',
      })
      expect(store.writable).toBe(true)
    })

    it("allows writable with an 'S3' data source", () => {
      const store = new BedrockKnowledgeBaseStore({
        name: 'kb',
        knowledgeBaseId: 'kb-1',
        writable: true,
        dataSourceType: 'S3',
        dataSourceId: 'ds-1',
        s3: { bucket: 'b', client: {} as any, prefix: 'p' },
      })
      expect(store.writable).toBe(true)
    })

    it('constructs a default runtime client when none is injected', () => {
      new BedrockKnowledgeBaseStore({
        name: 'kb',
        knowledgeBaseId: 'kb-1',
      })
      expect(vi.mocked(BedrockAgentRuntimeClient)).toHaveBeenCalledWith({})
    })

    it('uses the injected runtime client without constructing one', () => {
      const runtime = mockClient()
      new BedrockKnowledgeBaseStore({ name: 'kb', knowledgeBaseId: 'kb-1', runtimeClient: runtime as any })
      expect(vi.mocked(BedrockAgentRuntimeClient)).not.toHaveBeenCalled()
    })
  })

  describe('search', () => {
    function searchStore(overrides: Record<string, unknown> = {}): {
      store: BedrockKnowledgeBaseStore
      runtime: { send: MockedFunction<any> }
    } {
      const runtime = mockClient()
      runtime.send.mockResolvedValue({ retrievalResults: [] })
      const store = new BedrockKnowledgeBaseStore({
        name: 'kb',
        knowledgeBaseId: 'kb-1',
        runtimeClient: runtime as any,
        ...overrides,
      })
      return { store, runtime }
    }

    it('issues a RetrieveCommand with the query and a default result limit of 10', async () => {
      const { store, runtime } = searchStore()
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
      const { store, runtime } = searchStore({ maxSearchResults: 5 })
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
      const { store, runtime } = searchStore({ maxSearchResults: 5 })
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
      const { store, runtime } = searchStore({ scope: 'user-123' })
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
      const { store, runtime } = searchStore({ scope: 'acme', scopeMetadataKey: 'tenant' })
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
      const { store, runtime } = searchStore({ scope: 'ignored', filter })
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
      const runtime = mockClient()
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
      const store = new BedrockKnowledgeBaseStore({
        name: 'kb',
        knowledgeBaseId: 'kb-1',
        runtimeClient: runtime as any,
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
      const runtime = mockClient()
      runtime.send.mockResolvedValue({ retrievalResults: [{}] })
      const store = new BedrockKnowledgeBaseStore({
        name: 'kb',
        knowledgeBaseId: 'kb-1',
        runtimeClient: runtime as any,
      })

      const results = await store.search('q')
      expect(results).toStrictEqual([{ content: '', metadata: {} }])
    })

    it('returns an empty array when the knowledge base yields no results', async () => {
      const { store } = searchStore()
      await expect(store.search('q')).resolves.toStrictEqual([])
    })

    it('returns an empty array when the response omits retrievalResults entirely', async () => {
      const runtime = mockClient()
      runtime.send.mockResolvedValue({})
      const store = new BedrockKnowledgeBaseStore({
        name: 'kb',
        knowledgeBaseId: 'kb-1',
        runtimeClient: runtime as any,
      })
      await expect(store.search('q')).resolves.toStrictEqual([])
    })

    it('logs and rethrows when the retrieve call fails', async () => {
      const errorSpy = vi.spyOn(logger, 'error').mockImplementation(() => {})
      const runtime = mockClient()
      runtime.send.mockRejectedValue(new Error('retrieve boom'))
      const store = new BedrockKnowledgeBaseStore({
        name: 'kb',
        knowledgeBaseId: 'kb-1',
        runtimeClient: runtime as any,
      })

      await expect(store.search('q')).rejects.toThrow('retrieve boom')
      expect(errorSpy).toHaveBeenCalled()
      errorSpy.mockRestore()
    })
  })

  describe('add — CUSTOM data source', () => {
    function customStore(overrides: Record<string, unknown> = {}): {
      store: BedrockKnowledgeBaseStore
      agent: { send: MockedFunction<any> }
    } {
      const agent = mockClient()
      agent.send.mockResolvedValue({})
      const store = new BedrockKnowledgeBaseStore({
        name: 'kb',
        knowledgeBaseId: 'kb-1',
        writable: true,
        dataSourceType: 'CUSTOM',
        dataSourceId: 'ds-1',
        agentClient: agent as any,
        ...overrides,
      })
      return { store, agent }
    }

    it('throws when dataSourceId is missing', () => {
      expect(
        () =>
          new BedrockKnowledgeBaseStore({
            name: 'kb',
            knowledgeBaseId: 'kb-1',
            writable: true,
            dataSourceType: 'CUSTOM',
          })
      ).toThrow('dataSourceId is required')
    })

    it('throws when content is empty or whitespace-only', async () => {
      const { store } = customStore()
      await expect(store.add('')).rejects.toThrow('content must not be empty')
      await expect(store.add('   ')).rejects.toThrow('content must not be empty')
    })

    it('returns the generated custom document id', async () => {
      const { store } = customStore()
      await expect(store.add('fact')).resolves.toStrictEqual({ documentId: 'test-uuid-v7' })
    })

    it('uses the same id for the document identifier and the returned documentId', async () => {
      const { store, agent } = customStore()
      const result = await store.add('fact')
      const document = agent.send.mock.calls[0]?.[0].input.documents[0]
      expect(document.content.custom.customDocumentIdentifier.id).toBe(result.documentId)
    })

    it('ingests an inline CUSTOM document with no metadata field when no scope or metadata', async () => {
      const { store, agent } = customStore()
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
      const { store, agent } = customStore({ scope: 'user-123' })
      await store.add('fact')
      const document = agent.send.mock.calls[0]?.[0].input.documents[0]
      expect(document.metadata.inlineAttributes).toStrictEqual([
        { key: 'namespace', value: { type: 'STRING', stringValue: 'user-123' } },
      ])
    })

    it('drops metadata keys that collide with scopeMetadataKey and preserves scope', async () => {
      const { store, agent } = customStore({ scope: 'tenant-A' })
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
      const { store, agent } = customStore()
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
      const agent = mockClient()
      agent.send.mockRejectedValue(new Error('ingest boom'))
      const store = new BedrockKnowledgeBaseStore({
        name: 'kb',
        knowledgeBaseId: 'kb-1',
        writable: true,
        dataSourceType: 'CUSTOM',
        dataSourceId: 'ds-1',
        agentClient: agent as any,
      })

      await expect(store.add('fact')).rejects.toThrow('ingest boom')
      expect(errorSpy).toHaveBeenCalled()
      errorSpy.mockRestore()
    })

    it('lazily constructs a default agent client when none is injected', async () => {
      const store = new BedrockKnowledgeBaseStore({
        name: 'kb',
        knowledgeBaseId: 'kb-1',
        writable: true,
        dataSourceType: 'CUSTOM',
        dataSourceId: 'ds-1',
      })
      await store.add('fact')
      expect(vi.mocked(BedrockAgentClient)).toHaveBeenCalledWith({})
    })
  })

  describe('add — S3 data source', () => {
    function s3Store(overrides: Record<string, unknown> = {}): {
      store: BedrockKnowledgeBaseStore
      agent: { send: MockedFunction<any> }
      s3: { send: MockedFunction<any> }
    } {
      const agent = mockClient()
      agent.send.mockResolvedValue({})
      const s3 = mockClient()
      s3.send.mockResolvedValue({})
      const store = new BedrockKnowledgeBaseStore({
        name: 'kb',
        knowledgeBaseId: 'kb-1',
        writable: true,
        dataSourceType: 'S3',
        dataSourceId: 'ds-1',
        agentClient: agent as any,
        s3: { bucket: 'my-bucket', client: s3 as any, prefix: 'memories' },
        ...overrides,
      })
      return { store, agent, s3 }
    }

    it('throws when the s3 config is missing', () => {
      expect(
        () =>
          new BedrockKnowledgeBaseStore({
            name: 'kb',
            knowledgeBaseId: 'kb-1',
            writable: true,
            dataSourceType: 'S3',
            dataSourceId: 'ds-1',
          })
      ).toThrow('s3 config is required')
    })

    it("returns the uploaded content object's s3:// URI as the document id", async () => {
      const { store } = s3Store()
      await expect(store.add('content')).resolves.toStrictEqual({
        documentId: 's3://my-bucket/memories/test-uuid-v7.txt',
      })
    })

    it('uploads the content object and ingests an S3 document referencing it (no sidecar)', async () => {
      const { store, agent, s3 } = s3Store()
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
      const store = new BedrockKnowledgeBaseStore({
        name: 'kb',
        knowledgeBaseId: 'kb-1',
        writable: true,
        dataSourceType: 'S3',
        dataSourceId: 'ds-1',
        agentClient: mockClient() as any,
        s3: { bucket: 'my-bucket', client: client as any, prefix: 'memories/' },
      })
      await store.add('content')
      expect(client.send).toHaveBeenCalledWith(
        expect.objectContaining({ input: expect.objectContaining({ Key: 'memories/test-uuid-v7.txt' }) })
      )
    })

    it('writes a sidecar carrying the scope and points the document metadata at it', async () => {
      const { store, agent, s3 } = s3Store({ scope: 'team-a' })
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
      const { store, s3 } = s3Store()
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
      const { store, s3 } = s3Store()
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
        name: 'kb',
        knowledgeBaseId: 'kb-1',
        writable: true,
        dataSourceType: 'S3',
        dataSourceId: 'ds-1',
        s3: { bucket: 'my-bucket', client: s3 as any, prefix: 'memories' },
      })
      await store.add('content')
      expect(vi.mocked(BedrockAgentClient)).toHaveBeenCalledWith({})
    })

    it('logs and rethrows when the S3 upload fails, before any ingestion', async () => {
      const errorSpy = vi.spyOn(logger, 'error').mockImplementation(() => {})
      const { store, agent, s3 } = s3Store()
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
        name: 'kb',
        knowledgeBaseId: 'kb-1',
        dataSourceType: 'OTHER',
        dataSourceId: 'ds-1',
        agentClient: agent as any,
      })
      await expect(store.add('fact')).rejects.toThrow('store is not writable')
      expect(agent.send).not.toHaveBeenCalled()
    })
  })

  describe('metadata logging', () => {
    it('logs a debug line when a CUSTOM document drops an unsupported metadata value', async () => {
      const debugSpy = vi.spyOn(logger, 'debug').mockImplementation(() => {})
      const agent = mockClient()
      agent.send.mockResolvedValue({})
      const store = new BedrockKnowledgeBaseStore({
        name: 'kb',
        knowledgeBaseId: 'kb-1',
        writable: true,
        dataSourceType: 'CUSTOM',
        dataSourceId: 'ds-1',
        agentClient: agent as any,
      })

      await store.add('fact', { good: 'v', bad: { nested: true } })
      expect(debugSpy).toHaveBeenCalledWith(expect.stringContaining('key=<bad>'))
      expect(debugSpy).not.toHaveBeenCalledWith(expect.stringContaining('key=<good>'))
      debugSpy.mockRestore()
    })
  })
})
