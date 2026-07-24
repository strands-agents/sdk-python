/**
 * L1 transcript writer: writes messages to storage on arrival.
 *
 * @internal
 */

import type { Storage } from '../storage/storage.js'
import type { Message } from '../types/messages.js'
import { logger } from '../logging/logger.js'
import { Manifest, type ManifestEntry } from './manifest.js'

const MANIFEST_KEY = '_manifest'

/**
 * Manages the L1 durable message store for a single scope.
 *
 * Each message is written as a JSON blob keyed by its trackingId.
 * A manifest file tracks the order and metadata of stored messages.
 *
 * The transcript is scoped to a storage namespace — callers are responsible
 * for providing a pre-scoped storage view.
 *
 * @internal
 */
export class Transcript {
  private readonly _storage: Storage
  private _manifest: Manifest | undefined

  constructor(storage: Storage) {
    this._storage = storage
  }

  private async _ensureManifest(): Promise<Manifest> {
    if (this._manifest !== undefined) return this._manifest

    const data = await this._storage.read(MANIFEST_KEY)
    if (data !== null) {
      try {
        this._manifest = Manifest.deserialize(data)
      } catch {
        logger.warn('action=<loadManifest> | corrupt manifest, starting fresh')
        this._manifest = new Manifest()
      }
    } else {
      this._manifest = new Manifest()
    }

    return this._manifest
  }

  /**
   * Write a message to L1 storage and update the manifest.
   * If the message has already been written (by trackingId), this is a no-op.
   */
  async writeMessage(message: Message): Promise<void> {
    const trackingId = message.trackingId
    if (!trackingId) {
      logger.debug('action=<writeMessage> | skipping message without trackingId')
      return
    }

    const manifest = await this._ensureManifest()

    if (manifest.has(trackingId)) return

    const storageKey = trackingId
    const data = new TextEncoder().encode(JSON.stringify(message))

    await this._storage.write(storageKey, data)

    const entry: ManifestEntry = {
      trackingId,
      role: message.role,
      storageKey,
      contentBlocks: message.content.length,
    }
    manifest.add(entry)
    await this._storage.write(MANIFEST_KEY, manifest.serialize())

    logger.debug(`trackingId=<${trackingId}>, role=<${message.role}> | wrote message to L1`)
  }

  /**
   * Read a message from L1 by trackingId.
   *
   * @returns The deserialized message data, or null if not found.
   */
  async readMessage(trackingId: string): Promise<Record<string, unknown> | null> {
    const data = await this._storage.read(trackingId)
    if (data === null) return null
    return JSON.parse(new TextDecoder().decode(data)) as Record<string, unknown>
  }

  /** Return the current manifest (loading from storage if needed). */
  async getManifest(): Promise<Manifest> {
    return this._ensureManifest()
  }
}
