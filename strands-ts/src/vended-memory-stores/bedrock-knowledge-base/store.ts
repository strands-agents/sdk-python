import { BedrockAgentRuntimeClient, RetrieveCommand, type RetrievalFilter } from '@aws-sdk/client-bedrock-agent-runtime'
import {
  BedrockAgentClient,
  type KnowledgeBaseDocument,
  type MetadataAttributeValue,
  IngestKnowledgeBaseDocumentsCommand,
} from '@aws-sdk/client-bedrock-agent'
import type { S3Client } from '@aws-sdk/client-s3'
import { v7 as uuidv7 } from 'uuid'

import type { MemoryEntry, MemoryStore, MemoryStoreConfig, SearchOptions } from '../../memory/types.js'
import type { JSONValue } from '../../types/json.js'
import { logger } from '../../logging/logger.js'

const DEFAULT_MAX_SEARCH_RESULTS = 10

/**
 * An attribute entry in an S3 `.metadata.json` sidecar. `includeForEmbedding` is `false` so the
 * attribute is stored for filtering only and does not influence the embedding (matching how inline
 * attributes behave for `CUSTOM` documents). The sidecar is a plain JSON file with no SDK type, so
 * this is declared here; its `value` reuses Bedrock's `MetadataAttributeValue`.
 */
type S3SidecarAttribute = { value: MetadataAttributeValue; includeForEmbedding: false }

/** Converts a caller metadata value into a Bedrock attribute value, or `undefined` if unsupported. */
function toAttributeValue(value: JSONValue): MetadataAttributeValue | undefined {
  if (typeof value === 'string') return { type: 'STRING', stringValue: value }
  if (typeof value === 'number') return { type: 'NUMBER', numberValue: value }
  if (typeof value === 'boolean') return { type: 'BOOLEAN', booleanValue: value }
  if (Array.isArray(value) && value.length > 0 && value.every((item) => typeof item === 'string')) {
    return { type: 'STRING_LIST', stringListValue: value }
  }
  return undefined
}

/**
 * S3 ingestion settings for {@link BedrockKnowledgeBaseStore}, required when `dataSourceType` is `'S3'`.
 *
 * An S3 data source indexes objects from a bucket — there is no inline-text path — so `add` uploads
 * to this bucket and then ingests directly (`IngestKnowledgeBaseDocuments`). Unlike a `CUSTOM`
 * document, an S3 document can't carry metadata inline, so a single `add` may write **two** objects:
 * - the content, as a `.txt` object; and
 * - when `scope`/`metadata` are present, a `<object-key>.metadata.json` *sidecar* beside it. This
 *   is Bedrock's out-of-band convention for attaching attributes to an S3 object — it's paired to
 *   the content by name and used for retrieval filtering. With no scope/metadata, no sidecar is
 *   written.
 *
 * Direct ingestion indexes whatever object you point it at, so `add` works with any `bucket`/`prefix`
 * the credentials can write to, regardless of where the data source is configured to read.
 *
 * The `bucket`/`prefix` choice only governs durability across future data-source *syncs*
 * (`StartIngestionJob`): a sync reconciles the index to match the data source's scanned location, so
 * an object outside that location is treated as deleted and removed from the index. If syncs will
 * run against this data source, upload to the bucket it reads from and a `prefix` within its
 * inclusion prefixes so directly-ingested memories survive them; otherwise the location is free.
 */
export interface BedrockKnowledgeBaseS3Config {
  /** Bucket the content object and its optional `.metadata.json` sidecar are uploaded to before ingestion. */
  bucket: string
  /** Client used to upload objects. When omitted, a default `S3Client` is constructed using the default credential chain. */
  client?: S3Client
  /**
   * Key prefix for uploaded objects (e.g. `'memories/'`). A trailing slash is added when missing.
   * Both the content object and its sidecar (when written) land under this prefix.
   */
  prefix: string
}

/** Configuration for {@link BedrockKnowledgeBaseStore}. */
export interface BedrockKnowledgeBaseStoreConfig extends MemoryStoreConfig {
  /** The Bedrock Knowledge Base identifier to query and ingest into. */
  knowledgeBaseId: string
  /**
   * The type of data source backing this knowledge base, matching Bedrock's `dataSourceType`. Only
   * `'CUSTOM'` and `'S3'` data sources accept direct document ingestion
   * (`IngestKnowledgeBaseDocuments`), so only those can be written to:
   * - `'CUSTOM'`: `add` ingests its `content` argument as inline text, with scope/metadata attached
   *   as inline attributes.
   * - `'S3'`: `add` uploads its `content` to the configured `s3` bucket and ingests that object, so
   *   the write is self-contained (no separate upload or sync needed). When scope/metadata are
   *   present they're written as a *second* object — a `.metadata.json` sidecar beside the content —
   *   since an S3 document can't carry attributes inline; with none, only the content object is
   *   written. Requires `s3`; see {@link BedrockKnowledgeBaseS3Config}.
   * - `'OTHER'`: any other backend (Confluence, SharePoint, Salesforce, Web, SQL/Redshift, …),
   *   which sync from an external store or are query-only and so are read-only.
   *
   * Effective writability is `writable && (dataSourceType === 'CUSTOM' || dataSourceType === 'S3')`:
   * a store is only writable when the caller opts in *and* the backend supports direct ingestion.
   * When omitted, the store is read-only.
   */
  dataSourceType?: 'CUSTOM' | 'S3' | 'OTHER'
  /**
   * Data source to ingest into when writing. Required for `add` to succeed — without it, write
   * calls throw, since the knowledge base has no destination to ingest into.
   */
  dataSourceId?: string
  /** S3 ingestion settings. Required when `dataSourceType` is `'S3'`; ignored otherwise. */
  s3?: BedrockKnowledgeBaseS3Config
  /** Logical namespace used to isolate documents; applied as a metadata filter on search. */
  scope?: string
  /** Metadata attribute key used for scope-based filtering. Defaults to `'namespace'`. */
  scopeMetadataKey?: string
  /** Explicit retrieval filter; overrides the auto-generated scope filter when provided. */
  filter?: RetrievalFilter
  /** Pre-constructed runtime client for Retrieve calls. When omitted, a default client is constructed. */
  runtimeClient?: BedrockAgentRuntimeClient
  /** Pre-constructed agent client for IngestKnowledgeBaseDocuments calls. When omitted, a default client is constructed lazily on first write. */
  agentClient?: BedrockAgentClient
}

/** Result returned by {@link BedrockKnowledgeBaseStore.add}. */
export interface BedrockKnowledgeBaseAddResult {
  /** `CUSTOM`: the generated document id (UUID). `S3`: the `s3://` URI of the uploaded content object. */
  documentId: string
}

/**
 * A {@link MemoryStore} backed by Amazon Bedrock Knowledge Bases. Supports semantic search via
 * Retrieve and document ingestion via IngestKnowledgeBaseDocuments for CUSTOM and S3 data sources.
 *
 * @example
 * ```typescript
 * import { BedrockKnowledgeBaseStore } from '@strands-agents/sdk/vended-memory-stores/bedrock-knowledge-base'
 *
 * const store = new BedrockKnowledgeBaseStore({
 *   name: 'personal',
 *   knowledgeBaseId: 'KB123',
 *   writable: true,
 *   dataSourceType: 'CUSTOM',
 *   dataSourceId: 'DS456',
 *   scope: 'user-abc',
 * })
 *
 * const results = await store.search('what are my preferences?')
 * const { documentId } = await store.add('User prefers dark mode')
 * ```
 */
export class BedrockKnowledgeBaseStore implements MemoryStore {
  readonly name: string
  readonly description?: string
  readonly maxSearchResults?: number
  readonly writable: boolean

  private readonly _runtimeClient: BedrockAgentRuntimeClient
  private _agentClient: BedrockAgentClient | undefined
  private _s3Client: S3Client | undefined
  private readonly _s3Config: BedrockKnowledgeBaseS3Config | undefined
  private readonly _knowledgeBaseId: string
  private readonly _dataSourceType: 'CUSTOM' | 'S3' | 'OTHER' | undefined
  private readonly _dataSourceId: string | undefined
  private readonly _scope: string | undefined
  private readonly _scopeMetadataKey: string
  private readonly _filter: RetrievalFilter | undefined

  constructor(config: BedrockKnowledgeBaseStoreConfig) {
    this.name = config.name
    if (config.description !== undefined) this.description = config.description
    if (config.maxSearchResults !== undefined) {
      if (config.maxSearchResults < 1) {
        throw new Error('BedrockKnowledgeBaseStore: maxSearchResults must be at least 1.')
      }
      this.maxSearchResults = config.maxSearchResults
    }
    this.writable = config.writable ?? false

    this._runtimeClient = config.runtimeClient ?? new BedrockAgentRuntimeClient({})
    this._agentClient = config.agentClient
    this._s3Client = config.s3?.client
    this._s3Config = config.s3
    this._knowledgeBaseId = config.knowledgeBaseId
    this._dataSourceType = config.dataSourceType
    this._dataSourceId = config.dataSourceId

    if (this.writable) this._validateWriteConfig()

    this._scope = config.scope
    this._scopeMetadataKey = config.scopeMetadataKey ?? 'namespace'

    if (config.filter) {
      this._filter = config.filter
    } else if (config.scope) {
      this._filter = {
        equals: {
          key: this._scopeMetadataKey,
          value: config.scope,
        },
      }
    }
  }

  /**
   * Searches the knowledge base for entries matching the query.
   *
   * @param query - The search query text
   * @param options - Optional search configuration
   * @returns Matching memory entries ordered by relevance. Each entry's `metadata` includes
   *   user-provided attributes plus two reserved synthetic keys: `_relevanceScore` (number) and
   *   `_sourceLocation` (Bedrock retrieval location object).
   */
  async search(query: string, options?: SearchOptions): Promise<MemoryEntry[]> {
    if (options?.maxSearchResults !== undefined && options.maxSearchResults < 1) {
      throw new Error('BedrockKnowledgeBaseStore: maxSearchResults must be at least 1.')
    }
    const limit = options?.maxSearchResults || this.maxSearchResults || DEFAULT_MAX_SEARCH_RESULTS

    let response
    try {
      response = await this._runtimeClient.send(
        new RetrieveCommand({
          knowledgeBaseId: this._knowledgeBaseId,
          retrievalQuery: { text: query },
          retrievalConfiguration: {
            vectorSearchConfiguration: {
              numberOfResults: limit,
              ...(this._filter && { filter: this._filter }),
            },
          },
        })
      )
    } catch (error) {
      logger.error(
        `store=<${this.name}>, knowledgeBaseId=<${this._knowledgeBaseId}>, error=<${error}> | knowledge base retrieve failed`,
        error
      )
      throw error
    }

    return (response.retrievalResults ?? []).map((result) => {
      const metadata: Record<string, JSONValue> = {}
      if (result.metadata) {
        for (const [key, value] of Object.entries(result.metadata)) {
          metadata[key] = value as JSONValue
        }
      }
      if (result.location) {
        metadata._sourceLocation = result.location as unknown as JSONValue
      }
      if (result.score != null) {
        metadata._relevanceScore = result.score
      }

      return {
        content: result.content?.text ?? '',
        metadata,
      }
    })
  }

  /**
   * Ingests `content` (with optional `metadata`) into the knowledge base.
   *
   * Only `CUSTOM` and `S3` data sources support this — they are the sole `dataSourceType`s that
   * accept direct ingestion (`IngestKnowledgeBaseDocuments`). `OTHER` backends sync from an external
   * store or are query-only, so the store is read-only and `add` is unavailable. Requires
   * `dataSourceId` (and, for `S3`, an `s3` config); see {@link BedrockKnowledgeBaseStoreConfig}.
   *
   * @param content - The text content to ingest
   * @param metadata - Optional metadata attributes to attach to the document
   * @returns The document identifier (UUID for CUSTOM, s3:// URI for S3)
   */
  async add(content: string, metadata?: Record<string, JSONValue>): Promise<BedrockKnowledgeBaseAddResult> {
    if (!this.writable) {
      throw new Error('BedrockKnowledgeBaseStore: store is not writable. Set writable: true in config to enable add().')
    }
    if (!content.trim()) {
      throw new Error('BedrockKnowledgeBaseStore: content must not be empty.')
    }
    const { dataSourceId, dataSourceType } = this._validateWriteConfig()

    // S3 and CUSTOM data sources accept fundamentally different documents. S3 ingests objects, so
    // its document references objects uploaded to S3 first; CUSTOM ingests the text inline. Either
    // way the store mints the document's identifier (CUSTOM: a UUID; S3: the content object's URI)
    // and returns it, so the caller has a stable handle to the document Bedrock now tracks.
    let document: KnowledgeBaseDocument
    let documentId: string
    if (dataSourceType === 'S3') {
      const objectUris = await this._uploadS3Objects(content, metadata)
      document = this._buildS3Document(objectUris)
      documentId = objectUris.contentUri
    } else {
      documentId = uuidv7()
      document = this._buildCustomDocument(documentId, content, metadata)
    }

    try {
      await this._getAgentClient().send(
        new IngestKnowledgeBaseDocumentsCommand({
          knowledgeBaseId: this._knowledgeBaseId,
          dataSourceId,
          documents: [document],
        })
      )
    } catch (error) {
      logger.error(
        `store=<${this.name}>, knowledgeBaseId=<${this._knowledgeBaseId}>, dataSourceId=<${dataSourceId}>, dataSourceType=<${dataSourceType}>, error=<${error}> | knowledge base document ingestion failed`,
        error
      )
      throw error
    }

    return { documentId }
  }

  /**
   * Uploads the objects that back one S3 ingestion and returns their `s3://` URIs.
   *
   * A single `add` produces up to *two* objects — hence the plural name and the multi-URI return:
   * - `contentUri`: the content itself, uploaded as a `.txt` object. Always written.
   * - `sidecarUri`: a `<object-key>.metadata.json` sidecar carrying scope/metadata, written *only*
   *   when there is any to attach (so it's optional in the return). Unlike a `CUSTOM` document, an
   *   S3 document can't carry attributes inline, so the sidecar is Bedrock's out-of-band convention
   *   for attaching them: it sits beside the content object and Bedrock pairs the two by name.
   *
   * Bedrock reads and indexes these objects on ingestion.
   *
   * Note: uploads are not transactional. If the sidecar upload (or the subsequent ingestion) fails
   * after the content object lands, the uploaded object(s) remain in the bucket un-ingested. They are
   * inert — a later data-source sync may pick them up, or they can be cleaned up out of band.
   */
  private async _uploadS3Objects(
    content: string,
    metadata?: Record<string, JSONValue>
  ): Promise<{ contentUri: string; sidecarUri?: string }> {
    const s3 = this._s3Config!
    const prefix = s3.prefix.endsWith('/') ? s3.prefix : `${s3.prefix}/`
    const key = `${prefix}${uuidv7()}.txt`

    const contentUri = await this._putObject(s3, key, content, 'text/plain; charset=utf-8')

    const attributes = this._buildS3SidecarAttributes(metadata)
    if (Object.keys(attributes).length === 0) {
      return { contentUri }
    }

    // The sidecar must sit beside the source object and be named `<object-key>.metadata.json`.
    const sidecar = JSON.stringify({ metadataAttributes: attributes })
    const sidecarUri = await this._putObject(s3, `${key}.metadata.json`, sidecar, 'application/json')
    return { contentUri, sidecarUri }
  }

  /** Uploads a single object to the configured bucket and returns its `s3://` URI. */
  private async _putObject(
    s3: BedrockKnowledgeBaseS3Config,
    key: string,
    body: string,
    contentType: string
  ): Promise<string> {
    try {
      const { PutObjectCommand } = await import('@aws-sdk/client-s3')
      const client = await this._getS3Client()
      await client.send(new PutObjectCommand({ Bucket: s3.bucket, Key: key, Body: body, ContentType: contentType }))
    } catch (error) {
      logger.error(
        `store=<${this.name}>, uri=<s3://${s3.bucket}/${key}>, error=<${error}> | S3 upload failed before ingestion`,
        error
      )
      throw error
    }
    return `s3://${s3.bucket}/${key}`
  }

  /**
   * Builds a document for an `S3` data source from the URIs produced by {@link _uploadS3Objects}.
   *
   * Takes both URIs because an S3 document references objects by location rather than carrying data
   * inline: `contentUri` becomes the document's content (the object to index), and `sidecarUri`,
   * when present, becomes its metadata (an `S3_LOCATION` pointing at the sidecar). With no
   * scope/metadata there's no sidecar, so `sidecarUri` is omitted and the document carries no
   * metadata.
   */
  private _buildS3Document({
    contentUri,
    sidecarUri,
  }: {
    contentUri: string
    sidecarUri?: string
  }): KnowledgeBaseDocument {
    const document: KnowledgeBaseDocument = {
      content: {
        dataSourceType: 'S3',
        s3: { s3Location: { uri: contentUri } },
      },
    }

    if (sidecarUri) {
      document.metadata = {
        type: 'S3_LOCATION',
        s3Location: { uri: sidecarUri },
      }
    }

    return document
  }

  /**
   * Resolves scope and caller metadata into a flat list of key-value pairs, handling collision
   * detection and unsupported-type filtering. Shared by both CUSTOM (inline attributes) and S3
   * (sidecar) document builders.
   */
  private _resolveAttributes(
    metadata?: Record<string, JSONValue>
  ): Array<{ key: string; value: MetadataAttributeValue }> {
    const attrs: Array<{ key: string; value: MetadataAttributeValue }> = []

    if (this._scope) {
      attrs.push({ key: this._scopeMetadataKey, value: { type: 'STRING', stringValue: this._scope } })
    }

    if (metadata) {
      for (const [key, value] of Object.entries(metadata)) {
        if (this._scope && key === this._scopeMetadataKey) {
          logger.warn(`store=<${this.name}>, key=<${key}> | dropping metadata key that collides with scopeMetadataKey`)
          continue
        }
        const attributeValue = toAttributeValue(value)
        if (attributeValue) {
          attrs.push({ key, value: attributeValue })
        } else {
          logger.debug(`store=<${this.name}>, key=<${key}> | dropping metadata value of unsupported type`)
        }
      }
    }

    return attrs
  }

  /**
   * Builds a document for a `CUSTOM` data source: the text ingested inline, with the scope and any
   * caller metadata attached as inline attributes for retrieval filtering. The caller supplies
   * `documentId` (so it can return that same id to its own caller) and it becomes the document's
   * `customDocumentIdentifier`.
   */
  private _buildCustomDocument(
    documentId: string,
    content: string,
    metadata?: Record<string, JSONValue>
  ): KnowledgeBaseDocument {
    const attrs = this._resolveAttributes(metadata)

    const document: KnowledgeBaseDocument = {
      content: {
        dataSourceType: 'CUSTOM',
        custom: {
          customDocumentIdentifier: { id: documentId },
          sourceType: 'IN_LINE',
          inlineContent: {
            type: 'TEXT',
            textContent: { data: content },
          },
        },
      },
    }

    if (attrs.length > 0) {
      document.metadata = {
        type: 'IN_LINE_ATTRIBUTE',
        inlineAttributes: attrs.map(({ key, value }) => ({ key, value })),
      }
    }

    return document
  }

  /**
   * Builds the `metadataAttributes` map for an S3 `.metadata.json` sidecar from the scope and caller
   * metadata. Returns an empty map when there's nothing to attach — {@link _uploadS3Objects} treats
   * that as "no sidecar" and skips writing the second object.
   */
  private _buildS3SidecarAttributes(metadata?: Record<string, JSONValue>): Record<string, S3SidecarAttribute> {
    const attrs = this._resolveAttributes(metadata)
    const attributes: Record<string, S3SidecarAttribute> = {}
    for (const { key, value } of attrs) {
      attributes[key] = { value, includeForEmbedding: false }
    }
    return attributes
  }

  private _validateWriteConfig(): { dataSourceId: string; dataSourceType: 'CUSTOM' | 'S3' } {
    if (this._dataSourceType !== 'CUSTOM' && this._dataSourceType !== 'S3') {
      throw new Error(
        `BedrockKnowledgeBaseStore: add requires dataSourceType 'CUSTOM' or 'S3', but it is '${this._dataSourceType ?? 'undefined'}'. ` +
          "'OTHER' backends are read-only."
      )
    }
    if (!this._dataSourceId) {
      throw new Error(
        'BedrockKnowledgeBaseStore: dataSourceId is required for write operations. ' +
          'Provide it in the config to enable add().'
      )
    }
    if (this._dataSourceType === 'S3' && !this._s3Config) {
      throw new Error(
        "BedrockKnowledgeBaseStore: s3 config is required when dataSourceType is 'S3'. " +
          'Provide bucket and prefix to enable add().'
      )
    }
    return { dataSourceId: this._dataSourceId, dataSourceType: this._dataSourceType }
  }

  private async _getS3Client(): Promise<S3Client> {
    if (!this._s3Client) {
      const { S3Client: S3ClientImpl } = await import('@aws-sdk/client-s3')
      this._s3Client = new S3ClientImpl({})
    }
    return this._s3Client
  }

  private _getAgentClient(): BedrockAgentClient {
    if (!this._agentClient) {
      this._agentClient = new BedrockAgentClient({})
    }
    return this._agentClient
  }
}
