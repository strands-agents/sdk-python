import type { Storage } from './storage.js'

import { normalizeKey, normalizePrefix } from './normalize.js'

/** Configuration for {@link InMemoryStorage}. */
export interface InMemoryStorageConfig {
  /**
   * Maximum number of entries before LRU eviction kicks in.
   * When a `put` would exceed this limit, the least-recently-accessed entry is evicted.
   * Set to `null` for unbounded growth.
   */
  maxEntries: number | null
}

/**
 * In-memory {@link Storage} backend backed by a `Map`.
 *
 * Useful for testing and for serverless environments where disk access is unavailable.
 * Content does not survive process restarts — for persistence use {@link LocalFileStorage}
 * or {@link S3Storage}.
 *
 * Eviction is LRU-based: when a `put` would exceed `maxEntries`, the least-recently-accessed
 * entry is evicted. This is the only eviction mechanism — there is no turn-based or
 * time-based expiration. When used with the `ContextOffloader` plugin, set `maxEntries`
 * to control how many offloaded entries are retained (each offloaded content block uses
 * exactly one key).
 *
 * Keys are normalized identically to {@link LocalFileStorage}: slash runs are collapsed,
 * leading/trailing slashes are stripped, and `..` segments are rejected.
 *
 * @example
 * ```typescript
 * const storage = new InMemoryStorage({ maxEntries: 100 })
 * await storage.put('memory/notes.json', new TextEncoder().encode('[]'))
 * const bytes = await storage.get('memory/notes.json')
 * ```
 */
export class InMemoryStorage implements Storage {
  private readonly _store = new Map<string, Uint8Array>()
  private readonly _maxEntries: number | null

  constructor(config: InMemoryStorageConfig) {
    const maxEntries = config.maxEntries
    if (maxEntries !== null && (!Number.isInteger(maxEntries) || maxEntries < 1)) {
      throw new Error('maxEntries must be a positive integer')
    }
    this._maxEntries = maxEntries
  }

  /**
   * Stores `data` under `key`, overwriting any existing value.
   * Bytes are copied on write to prevent aliasing with the caller's buffer.
   * If `maxEntries` is set and the store is full, the least-recently-accessed entry is evicted.
   *
   * @param key - Opaque, `/`-separated key identifying the value
   * @param data - Raw bytes to persist
   * @throws {@link StorageError} if the key is empty or contains `..` segments
   */
  async put(key: string, data: Uint8Array): Promise<void> {
    const normalized = normalizeKey(key)
    this._store.delete(normalized)
    if (this._maxEntries !== null && this._store.size >= this._maxEntries) {
      const oldest = this._store.keys().next().value!
      this._store.delete(oldest)
    }
    this._store.set(normalized, data.slice())
  }

  /**
   * Retrieves the bytes previously stored under `key`.
   * Returns a copy to prevent aliasing with the internal buffer.
   * Accessing a key moves it to the most-recently-used position.
   *
   * @param key - The key to read
   * @returns The stored bytes, or `null` if no value exists for `key`
   * @throws {@link StorageError} if the key is empty or contains `..` segments
   */
  async get(key: string): Promise<Uint8Array | null> {
    const normalized = normalizeKey(key)
    const value = this._store.get(normalized)
    if (value === undefined) return null
    this._store.delete(normalized)
    this._store.set(normalized, value)
    return value.slice()
  }

  /**
   * Deletes the value stored under `key`. A no-op if the key does not exist.
   *
   * @param key - The key to delete
   * @throws {@link StorageError} if the key is empty or contains `..` segments
   */
  async delete(key: string): Promise<void> {
    this._store.delete(normalizeKey(key))
  }

  /**
   * Lists the keys whose names begin with `prefix`, sorted lexicographically.
   *
   * @param prefix - Key prefix to match. An empty string matches all keys.
   * @returns The matching keys, sorted ascending
   * @throws {@link StorageError} if the prefix contains `..` segments
   */
  async list(prefix: string): Promise<string[]> {
    const normalized = normalizePrefix(prefix)
    const keys: string[] = []
    for (const key of this._store.keys()) {
      if (key.startsWith(normalized)) keys.push(key)
    }
    return keys.sort()
  }

  /**
   * Removes all stored entries. Useful for resetting state between tests.
   */
  clear(): void {
    this._store.clear()
  }
}
