/**
 * L1 stash: durable storage for offloaded context content.
 *
 * When the ContextManager offloads content from the context window, the
 * original is persisted here so the agent can retrieve it on demand via the
 * retrieval tool.
 *
 * @internal
 */

import { resolveNamespace, type Storage } from '../storage/storage.js'
import { Message, TextBlock, JsonBlock, ToolResultBlock } from '../types/messages.js'
import type { ContentBlock, ToolResultContent } from '../types/messages.js'
import { ImageBlock, VideoBlock, DocumentBlock } from '../types/media.js'
import { logger } from '../logging/logger.js'

const STASH_PREFIX = 'context-stash'

/** A reference to stashed content. */
export interface StashRef {
  /** Storage key for retrieval. */
  ref: string
  /** MIME content type of the stored content. */
  contentType: string
}

function frameContent(content: Uint8Array, contentType: string): Uint8Array {
  const ctBytes = new TextEncoder().encode(contentType)
  const frame = new Uint8Array(2 + ctBytes.length + content.length)
  frame[0] = (ctBytes.length >> 8) & 0xff
  frame[1] = ctBytes.length & 0xff
  frame.set(ctBytes, 2)
  frame.set(content, 2 + ctBytes.length)
  return frame
}

function unframeContent(frame: Uint8Array): { content: Uint8Array; contentType: string } {
  const ctLen = (frame[0]! << 8) | frame[1]!
  const contentType = new TextDecoder().decode(frame.subarray(2, 2 + ctLen))
  const content = frame.subarray(2 + ctLen)
  return { content, contentType }
}

function serializeContentBlock(block: ToolResultContent): { bytes: Uint8Array | undefined; contentType: string } {
  if (block instanceof TextBlock) {
    return { bytes: new TextEncoder().encode(block.text), contentType: 'text/plain' }
  }
  if (block instanceof JsonBlock) {
    return { bytes: new TextEncoder().encode(JSON.stringify(block.json, null, 2)), contentType: 'application/json' }
  }
  if (block instanceof ImageBlock) {
    if (block.source.type === 'imageSourceBytes') {
      return { bytes: block.source.bytes, contentType: `image/${block.format}` }
    }
    return { bytes: undefined, contentType: `image/${block.format}` }
  }
  if (block instanceof VideoBlock) {
    if (block.source.type === 'videoSourceBytes') {
      return { bytes: block.source.bytes, contentType: `video/${block.format}` }
    }
    return { bytes: undefined, contentType: `video/${block.format}` }
  }
  if (block instanceof DocumentBlock) {
    if (block.source.type === 'documentSourceBytes') {
      return { bytes: block.source.bytes, contentType: `application/${block.format}` }
    }
    if (block.source.type === 'documentSourceText') {
      return { bytes: new TextEncoder().encode(block.source.text), contentType: 'text/plain' }
    }
    return { bytes: undefined, contentType: `application/${block.format}` }
  }
  return { bytes: undefined, contentType: 'application/octet-stream' }
}

/**
 * Wraps a Storage backend with key management and content framing for the
 * ContextManager's L1 stash.
 *
 * @internal
 */
export class Stash {
  private readonly _storage: Storage
  private readonly _sessionId: string
  private readonly _refsByBlock = new WeakMap<ContentBlock | ToolResultContent, StashRef[]>()
  private _counter = 0

  constructor(storage: Storage) {
    this._sessionId = Math.random().toString(36).slice(2, 8)
    this._storage = resolveNamespace(storage, STASH_PREFIX)
  }

  /**
   * Store raw content and return a reference key.
   *
   * @param toolUseId - The toolUseId this content belongs to
   * @param blockIndex - Index of the block within the tool result
   * @param content - Raw content bytes
   * @param contentType - MIME type of the content
   * @returns Reference key for retrieval
   */
  async store(toolUseId: string, blockIndex: number, content: Uint8Array, contentType: string): Promise<string> {
    this._counter++
    const key = `${this._sessionId}_${this._counter}_${toolUseId}_${blockIndex}`
    const framed = frameContent(content, contentType)
    await this._storage.write(key, framed)
    return key
  }

  /**
   * Look up pre-computed refs for a content block.
   * Returns refs populated by a prior {@link storeMessage} call, or empty if the block
   * was never eagerly stashed (e.g. retrieval results that were excluded).
   *
   * @param block - The content block to look up
   * @returns Array of stash references
   */
  getRefs(block: ContentBlock): StashRef[] {
    return this._refsByBlock.get(block) ?? []
  }

  /**
   * Eagerly stash all content from a message on arrival.
   * Called via MessageAddedEvent so content is persisted before any strategy runs.
   * Refs are keyed by the block object itself (WeakMap), so they auto-GC when
   * the block is replaced during offloading.
   *
   * @param message - The message whose content should be persisted
   */
  async storeMessage(message: Message): Promise<void> {
    for (const block of message.content) {
      if (block instanceof ToolResultBlock) {
        const refs = await this._storeToolResult(block).catch((error) => {
          logger.debug(`toolUseId=<${block.toolUseId}>, error=<${error}> | failed to stash tool result`)
          return [] as StashRef[]
        })
        if (refs.length > 0) this._refsByBlock.set(block, refs)
      } else if (block instanceof TextBlock && block.text.length > 0) {
        try {
          const key = `text_${message.trackingId ?? 'msg'}`
          const ref = await this.store(key, 0, new TextEncoder().encode(block.text), 'text/plain')
          this._refsByBlock.set(block, [{ ref, contentType: 'text/plain' }])
        } catch (error) {
          logger.debug(`trackingId=<${message.trackingId}>, error=<${error}> | failed to stash text block`)
        }
      }
    }
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

  private async _storeToolResult(block: ToolResultBlock): Promise<StashRef[]> {
    if (block.content.length === 0) return []

    const refs: StashRef[] = []
    for (let blockIndex = 0; blockIndex < block.content.length; blockIndex++) {
      const item = block.content[blockIndex]!
      const serialized = serializeContentBlock(item)
      if (!serialized.bytes) {
        logger.debug(
          `toolUseId=<${block.toolUseId}>, blockIndex=<${blockIndex}> | skipped non-byte content (${serialized.contentType})`
        )
        continue
      }
      const ref = await this.store(block.toolUseId, blockIndex, serialized.bytes, serialized.contentType)
      refs.push({ ref, contentType: serialized.contentType })
    }
    return refs
  }
}
