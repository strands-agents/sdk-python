/**
 * Storage backends for offloaded tool result content.
 *
 * This module defines the {@link Storage} interface and provides three built-in
 * implementations: {@link InMemoryStorage}, {@link FileStorage}, and {@link S3Storage}.
 * Each content block from a tool result is stored individually with its content type preserved.
 */

/**
 * Backend for storing and retrieving offloaded content blocks.
 *
 * Implement this interface to create custom storage backends (e.g., Redis, DynamoDB).
 * The SDK ships three built-in implementations: {@link InMemoryStorage},
 * {@link FileStorage}, and {@link S3Storage}.
 */
export interface Storage {
  /**
   * Store content and return a reference identifier.
   *
   * @param key - Unique key for this content block
   * @param content - Raw content bytes to store
   * @param contentType - MIME type of the content (e.g., "text/plain", "image/png")
   * @returns Reference string for later retrieval
   */
  store(key: string, content: Uint8Array, contentType?: string): Promise<string>

  /**
   * Retrieve previously stored content by reference.
   *
   * @param reference - Reference returned by a previous {@link store} call
   * @returns Content bytes and content type
   * @throws Error if the reference is not found
   */
  retrieve(reference: string): Promise<{ content: Uint8Array; contentType: string }>
}

function sanitizeId(rawId: string): string {
  return rawId
    .replace(/\.\./g, '_')
    .replace(/[/\\]/g, '_')
    .replace(/[^\w\-.]/g, '_')
}

/**
 * In-memory storage backend.
 *
 * Useful for testing and serverless environments where disk access is not available.
 * Content accumulates for the lifetime of this instance; call {@link clear} to free memory.
 */
export class InMemoryStorage implements Storage {
  private _store = new Map<string, { content: Uint8Array; contentType: string }>()
  private _counter = 0

  /** {@inheritdoc} */
  async store(key: string, content: Uint8Array, contentType: string = 'text/plain'): Promise<string> {
    this._counter++
    const reference = `mem_${this._counter}_${key}`
    this._store.set(reference, { content, contentType })
    return reference
  }

  /** {@inheritdoc} */
  async retrieve(reference: string): Promise<{ content: Uint8Array; contentType: string }> {
    const entry = this._store.get(reference)
    if (!entry) {
      throw new Error(`Reference not found: ${reference}`)
    }
    return entry
  }

  /** Remove all stored content. */
  clear(): void {
    this._store.clear()
  }
}

/**
 * File-based storage backend.
 *
 * Stores offloaded content as files on disk. File extensions are derived from the
 * content type. A `.metadata.json` sidecar file tracks content types across restarts.
 * References are file paths preserving the configured artifact directory form.
 *
 * @param artifactDir - Directory path where artifact files will be stored
 */
export class FileStorage implements Storage {
  private static readonly METADATA_FILE = '.metadata.json'
  private readonly _artifactDir: string
  private _counter = 0
  private _contentTypes: Record<string, string> = {}
  private _metadataLoaded = false
  private _metadataWriteChain: Promise<void> = Promise.resolve()

  constructor(artifactDir: string = './artifacts') {
    this._artifactDir = artifactDir
  }

  private static _extensionFor(contentType: string): string {
    if (contentType === 'text/plain') return '.txt'
    return `.${contentType.split('/').pop()}`
  }

  private async _ensureDir(): Promise<typeof import('node:fs/promises')> {
    const fs = await import('node:fs/promises')
    await fs.mkdir(this._artifactDir, { recursive: true })
    if (!this._metadataLoaded) {
      this._contentTypes = await this._loadMetadata(fs)
      this._metadataLoaded = true
    }
    return fs
  }

  private async _loadMetadata(fs: typeof import('node:fs/promises')): Promise<Record<string, string>> {
    const path = await import('node:path')
    const metadataPath = path.join(this._artifactDir, FileStorage.METADATA_FILE)
    try {
      const raw = await fs.readFile(metadataPath, 'utf-8')
      return JSON.parse(raw) as Record<string, string>
    } catch {
      return {}
    }
  }

  private async _saveMetadata(fs: typeof import('node:fs/promises')): Promise<void> {
    const path = await import('node:path')
    const metadataPath = path.join(this._artifactDir, FileStorage.METADATA_FILE)
    await fs.writeFile(metadataPath, JSON.stringify(this._contentTypes), 'utf-8')
  }

  /** {@inheritdoc} */
  async store(key: string, content: Uint8Array, contentType: string = 'text/plain'): Promise<string> {
    const fs = await this._ensureDir()
    const path = await import('node:path')

    const sanitizedKey = sanitizeId(key)
    const timestampMs = Date.now()
    this._counter++
    const ext = FileStorage._extensionFor(contentType)
    const filename = `${timestampMs}_${this._counter}_${sanitizedKey}${ext}`

    this._contentTypes[filename] = contentType
    this._metadataWriteChain = this._metadataWriteChain.then(() => this._saveMetadata(fs))
    await this._metadataWriteChain

    const filePath = path.join(this._artifactDir, filename)
    await fs.writeFile(filePath, content)

    return filePath
  }

  /** {@inheritdoc} */
  async retrieve(reference: string): Promise<{ content: Uint8Array; contentType: string }> {
    const fs = await this._ensureDir()
    const path = await import('node:path')

    const filePath = path.resolve(this._artifactDir, reference)
    const resolvedDir = path.resolve(this._artifactDir)
    if (!filePath.startsWith(resolvedDir)) {
      throw new Error(`Reference not found: ${reference}`)
    }

    const filename = path.basename(filePath)

    try {
      const content = await fs.readFile(filePath)
      const contentType = this._contentTypes[filename] ?? 'application/octet-stream'
      return { content: new Uint8Array(content), contentType }
    } catch {
      throw new Error(`Reference not found: ${reference}`)
    }
  }
}

/**
 * S3-based storage backend.
 *
 * Stores offloaded content as S3 objects. Content type is preserved as S3 object metadata.
 * References are `s3://` URIs for direct access via AWS CLI or SDK.
 *
 * @param bucket - S3 bucket name
 * @param options - Optional configuration (prefix, region, pre-configured S3Client)
 */
export class S3Storage implements Storage {
  private readonly _bucket: string
  private readonly _prefix: string
  private _client: import('@aws-sdk/client-s3').S3Client | undefined
  private readonly _region: string
  private _counter = 0

  constructor(
    bucket: string,
    options?: { prefix?: string; region?: string; s3Client?: import('@aws-sdk/client-s3').S3Client }
  ) {
    this._bucket = bucket
    this._prefix = options?.prefix ? options.prefix.replace(/\/+$/, '') + '/' : ''
    this._client = options?.s3Client
    this._region = options?.region ?? 'us-east-1'
  }

  private async _getClient(): Promise<import('@aws-sdk/client-s3').S3Client> {
    if (this._client) return this._client
    const { S3Client } = await import('@aws-sdk/client-s3')
    this._client = new S3Client({ region: this._region })
    return this._client
  }

  /** {@inheritdoc} */
  async store(key: string, content: Uint8Array, contentType: string = 'text/plain'): Promise<string> {
    const client = await this._getClient()
    const { PutObjectCommand } = await import('@aws-sdk/client-s3')

    const sanitizedKey = sanitizeId(key)
    const timestampMs = Date.now()
    this._counter++
    const s3Key = `${this._prefix}${timestampMs}_${this._counter}_${sanitizedKey}`

    await client.send(
      new PutObjectCommand({
        Bucket: this._bucket,
        Key: s3Key,
        Body: content,
        ContentType: contentType,
      })
    )

    return `s3://${this._bucket}/${s3Key}`
  }

  /** {@inheritdoc} */
  async retrieve(reference: string): Promise<{ content: Uint8Array; contentType: string }> {
    const client = await this._getClient()
    const { GetObjectCommand } = await import('@aws-sdk/client-s3')

    // Accept both s3:// URIs and raw keys
    let s3Key = reference
    const uriMatch = reference.match(/^s3:\/\/([^/]+)\/(.+)$/)
    if (uriMatch?.[1] && uriMatch[2]) {
      if (uriMatch[1] !== this._bucket) {
        throw new Error(`Reference not found: ${reference} (bucket mismatch)`)
      }
      s3Key = uriMatch[2]
    }

    try {
      const response = await client.send(
        new GetObjectCommand({
          Bucket: this._bucket,
          Key: s3Key,
        })
      )
      const body = await response.Body?.transformToByteArray()
      if (!body) throw new Error(`Reference not found: ${reference}`)
      const contentType = response.ContentType ?? 'application/octet-stream'
      return { content: new Uint8Array(body), contentType }
    } catch (error: unknown) {
      if (error instanceof Error && error.name === 'NoSuchKey') {
        throw new Error(`Reference not found: ${reference}`)
      }
      throw error
    }
  }
}
