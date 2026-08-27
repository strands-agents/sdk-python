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
import { Message, TextBlock, ToolResultBlock, ToolUseBlock, CachePointBlock, ReasoningBlock } from '../types/messages.js'
import type { ContentBlock, ToolResultContent } from '../types/messages.js'
import { ImageBlock, VideoBlock, DocumentBlock, AudioBlock } from '../types/media.js'
import { logger } from '../logging/logger.js'

const STASH_PREFIX = 'context-stash'

/** A reference to stashed content. */
export interface StashRef {
  /** Storage key for retrieval. */
  ref: string
  /** MIME content type of the stored content. */
  contentType: string
}

function encode(value: unknown): Uint8Array {
  return new TextEncoder().encode(JSON.stringify(value))
}

function decode(bytes: Uint8Array): unknown {
  return JSON.parse(new TextDecoder().decode(bytes))
}

function contentTypeOf(block: ContentBlock | ToolResultContent): string {
  if (block instanceof TextBlock) return 'text/plain'
  if (block instanceof ImageBlock) return `image/${block.format}`
  if (block instanceof VideoBlock) return `video/${block.format}`
  if (block instanceof DocumentBlock) return `application/${block.format}`
  if (block instanceof AudioBlock) return `audio/${block.format}`
  return 'application/json'
}

/** Format stash refs for display in placeholders. Returns '' when refs is empty. */
export function formatStashRefs(refs: StashRef[]): string {
  if (refs.length === 0) return ''
  if (refs.length === 1) return ` ref: ${refs[0]!.ref} (${refs[0]!.contentType})`
  return ` refs: ${refs.map((r) => `${r.ref} (${r.contentType})`).join(', ')}`
}

/**
 * Wraps a Storage backend with key management and content framing for the
 * ContextManager's L1 stash.
 *
 * @internal
 */
export class Stash {
  private readonly _storage: Storage
  private readonly _refsByBlock = new WeakMap<ContentBlock | ToolResultContent, StashRef[]>()

  constructor(storage: Storage, sessionId: string) {
    this._storage = resolveNamespace(storage, `${STASH_PREFIX}/${sessionId}`)
  }

  /**
   * Store a serialized block and return a deterministic reference key.
   *
   * Keys are deterministic (`<id>_<blockIndex>`) so they can be recomputed
   * after a process restart or snapshot restore without the WeakMap cache.
   *
   * @param id - Identifier component for the key (e.g. toolUseId)
   * @param blockIndex - Index of the block within the tool result
   * @param data - Serialized bytes to persist
   * @returns Reference key for retrieval
   */
  async store(id: string, blockIndex: number, data: Uint8Array): Promise<string> {
    const key = `${id}_${blockIndex}`
    await this._storage.write(key, data)
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
   * @param skipToolUseIds - ToolUseIds to skip (e.g. retrieval tool results)
   */
  async storeMessage(message: Message, skipToolUseIds?: ReadonlySet<string>): Promise<void> {
    for (let blockIndex = 0; blockIndex < message.content.length; blockIndex++) {
      const block = message.content[blockIndex]!
      if (block instanceof ToolResultBlock) {
        if (skipToolUseIds?.has(block.toolUseId)) continue
        const refs = await this._storeToolResult(block).catch((error) => {
          logger.debug(`toolUseId=<${block.toolUseId}>, error=<${error}> | failed to stash tool result`)
          return [] as StashRef[]
        })
        if (refs.length > 0) this._refsByBlock.set(block, refs)
      } else if (block instanceof ToolUseBlock || block instanceof CachePointBlock || block instanceof ReasoningBlock) {
        continue
      } else {
        try {
          const ref = await this.store(message.trackingId, blockIndex, encode(block.toJSON()))
          this._refsByBlock.set(block, [{ ref, contentType: contentTypeOf(block) }])
        } catch (error) {
          logger.debug(`trackingId=<${message.trackingId}>, error=<${error}> | failed to stash block`)
        }
      }
    }
  }

  /**
   * Retrieve previously stashed content as its serialized JSON form.
   *
   * @param reference - Key returned by a previous store call
   * @returns The deserialized block data (from `toJSON()`), or null if not found
   */
  async retrieve(reference: string): Promise<{ data: unknown; contentType: string } | null> {
    const bytes = await this._storage.read(reference)
    if (bytes === null) return null
    return { data: decode(bytes), contentType: 'application/json' }
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
      const ref = await this.store(block.toolUseId, blockIndex, encode(item.toJSON()))
      refs.push({ ref, contentType: contentTypeOf(item) })
    }
    return refs
  }
}
