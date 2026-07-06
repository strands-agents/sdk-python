/**
 * S3-based storage backend for offloaded tool result content.
 *
 * Split out from `./storage.js` so that importing the core `Agent` (which only
 * needs {@link InMemoryStorage}) does not pull the optional `@aws-sdk/client-s3`
 * peer dependency into a bundler's static import graph. The client is loaded
 * lazily via dynamic `import()`, so it is only required when {@link S3Storage}
 * is actually used.
 */

import type { Storage } from './storage.js'
import { sanitizeId } from './storage.js'

/**
 * Remove any trailing `/` characters from a path-like string.
 *
 * Implemented as a linear scan rather than a `/\/+$/` replacement so that
 * inputs with long runs of slashes are handled in linear time.
 */
function stripTrailingSlashes(value: string): string {
  let end = value.length
  while (end > 0 && value[end - 1] === '/') {
    end--
  }
  return value.slice(0, end)
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
    this._prefix = options?.prefix ? stripTrailingSlashes(options.prefix) + '/' : ''
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
        throw new Error(`Reference not found: ${reference}`, { cause: error })
      }
      throw error
    }
  }
}
