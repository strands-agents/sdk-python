/**
 * L1 stash: durable storage for offloaded context content.
 *
 * When the ContextManager offloads content from L0 (the context window), the
 * original is persisted here so the agent can retrieve it on demand via the
 * retrieval tool.
 *
 * @internal
 */

import { NAMESPACED, namespace, type Storage } from '../storage/storage.js'
import { logger } from '../logging/logger.js'

const STASH_PREFIX = 'context-stash'

/**
 * Frames content bytes with a content-type header for round-trip storage.
 * Format: [2-byte BE content-type length][content-type UTF-8][content bytes]
 */
function frameContent(content: Uint8Array, contentType: string): Uint8Array {
  const ctBytes = new TextEncoder().encode(contentType)
  const frame = new Uint8Array(2 + ctBytes.length + content.length)
  frame[0] = (ctBytes.length >> 8) & 0xff
  frame[1] = ctBytes.length & 0xff
  frame.set(ctBytes, 2)
  frame.set(content, 2 + ctBytes.length)
  return frame
}

/**
 * Unframes stored bytes into content + content-type.
 */
function unframeContent(frame: Uint8Array): { content: Uint8Array; contentType: string } {
  const ctLen = (frame[0]! << 8) | frame[1]!
  const contentType = new TextDecoder().decode(frame.subarray(2, 2 + ctLen))
  const content = frame.subarray(2 + ctLen)
  return { content, contentType }
}

/**
 * Wraps a Storage backend with key management and content framing for the
 * ContextManager's L1 stash.
 *
 * @internal
 */
export class Stash {
  private readonly _storage: Storage
  private _counter = 0

  constructor(storage: Storage) {
    if (NAMESPACED in storage) {
      this._storage = storage
    } else if (storage.namespace) {
      this._storage = storage.namespace(STASH_PREFIX)
    } else {
      this._storage = namespace(storage, STASH_PREFIX)
    }
  }

  /**
   * Store content and return a reference key.
   *
   * @param toolUseId - The toolUseId this content belongs to
   * @param blockIndex - Index of the block within the tool result
   * @param content - Raw content bytes
   * @param contentType - MIME type of the content
   * @returns Reference key for retrieval
   */
  async store(toolUseId: string, blockIndex: number, content: Uint8Array, contentType: string): Promise<string> {
    this._counter++
    const key = `${this._counter}_${toolUseId}_${blockIndex}`
    const framed = frameContent(content, contentType)
    await this._storage.write(key, framed)
    return key
  }

  /**
   * Retrieve previously stashed content.
   *
   * @param reference - Key returned by a previous store call
   * @returns Content bytes and content type, or null if not found
   */
  async retrieve(reference: string): Promise<{ content: Uint8Array; contentType: string } | null> {
    const data = await this._storage.read(reference)
    if (data === null) return null
    return unframeContent(data)
  }

  /**
   * List all stashed references.
   */
  async list(): Promise<string[]> {
    return this._storage.list('')
  }

  /**
   * Delete a stashed entry.
   */
  async delete(reference: string): Promise<void> {
    await this._storage.delete(reference)
    logger.debug(`reference=<${reference}> | stash entry deleted`)
  }
}
