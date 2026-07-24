import { StorageError } from '../errors.js'

/**
 * Symbol present on namespaced storage views. Constructs use this to detect
 * whether the caller already scoped the storage, skipping the default prefix.
 *
 * @internal
 */
export const NAMESPACED: unique symbol = Symbol.for('strands.storage.namespaced')

/**
 * Validates and normalizes a storage key: collapses runs of `/`, strips leading
 * and trailing `/`, and rejects empty keys and any `..` segment.
 *
 * @param key - The raw key to normalize
 * @returns The normalized key
 * @throws {@link StorageError} if the key is empty or contains a `..` segment
 */
export function normalizeKey(key: string): string {
  const segments = key.split('/').filter(Boolean)
  if (segments.length === 0) {
    throw new StorageError('Storage key must not be empty')
  }
  if (segments.includes('..')) {
    throw new StorageError(`Invalid storage key '${key}': '..' path segments are not allowed`)
  }
  return segments.join('/')
}

/**
 * Normalizes a list prefix: collapses slash runs, strips leading slashes.
 * Unlike a key, an empty prefix is valid and matches everything.
 *
 * @param prefix - The raw prefix to normalize
 * @returns The normalized prefix
 * @throws {@link StorageError} if the prefix contains a `..` segment
 */
export function normalizePrefix(prefix: string): string {
  const parts = prefix.split('/')
  const segments = parts.filter(Boolean)
  if (segments.includes('..')) {
    throw new StorageError(`Invalid storage prefix '${prefix}': '..' path segments are not allowed`)
  }
  const joined = segments.join('/')
  const normalized = parts[parts.length - 1] === '' && joined ? joined + '/' : joined
  return normalized
}

/**
 * A backend for storing and retrieving raw bytes under string keys.
 *
 * The interface is deliberately minimal — four operations over opaque `Uint8Array`
 * values. Implementations must treat keys as opaque path-like strings (segments
 * separated by `/`) and must round-trip the bytes they are given unchanged.
 *
 * The `ListQuery` type parameter controls what `list` accepts. It defaults to
 * `string` (a key prefix), which every backend supports. Implementations may
 * widen it to accept a richer query object (e.g. a DynamoDB partition/sort-key
 * filter) while still accepting a plain string for SDK-internal callers.
 *
 * Implement this to add a custom backend; the SDK ships {@link InMemoryStorage},
 * {@link LocalFileStorage}, and {@link S3Storage}.
 */
export interface Storage<ListQuery = string> {
  /**
   * Stores `data` under `key`, overwriting any existing value.
   *
   * @param key - Opaque, `/`-separated key identifying the value
   * @param data - Raw bytes to persist
   * @throws {@link StorageError} if the write fails
   */
  write(key: string, data: Uint8Array): Promise<void>

  /**
   * Retrieves the bytes previously stored under `key`.
   *
   * @param key - The key to read
   * @returns The stored bytes, or `null` if no value exists for `key`
   * @throws {@link StorageError} if the read fails for a reason other than a missing key
   */
  read(key: string): Promise<Uint8Array | null>

  /**
   * Deletes the value stored under `key`. A no-op if the key does not exist.
   *
   * @param key - The key to delete
   * @throws {@link StorageError} if the delete fails
   */
  delete(key: string): Promise<void>

  /**
   * Lists keys matching the given query.
   *
   * When `ListQuery` is `string` (the default), this is a prefix match — returns
   * full keys (not the suffix after the prefix), sorted lexicographically. An empty
   * string lists every key.
   *
   * Implementations may accept richer query objects (e.g. partition + sort-key filters)
   * while still supporting a plain string prefix for SDK-internal callers.
   *
   * @param query - A string prefix or backend-specific query object
   * @returns The matching keys, sorted ascending
   * @throws {@link StorageError} if the listing fails
   */
  list(query: ListQuery): Promise<string[]>

  /**
   * Returns a view of this storage with all keys prefixed by `prefix`.
   * The original storage is not mutated.
   *
   * Optional — shipped backends implement this, custom backends may omit it.
   *
   * @param prefix - Prefix to prepend to all keys
   * @returns A Storage view scoped to the given prefix
   */
  namespace?(prefix: string): Storage
}

/**
 * Returns a {@link Storage} view with all keys prefixed by `prefix`.
 *
 * Composable — calling `namespace()` on the result nests prefixes.
 *
 * @internal
 * @param storage - The underlying storage to delegate to
 * @param prefix - Prefix to prepend to all keys
 * @returns A namespaced Storage view
 */
export function namespace(storage: Storage, prefix: string): Storage {
  const normalized = normalizePrefix(prefix)
  const p = normalized ? `${normalized}/` : ''
  const view: Storage & { [NAMESPACED]: true } = {
    write: (key, data) => storage.write(`${p}${key}`, data),
    read: (key) => storage.read(`${p}${key}`),
    delete: (key) => storage.delete(`${p}${key}`),
    list: (query) => storage.list(`${p}${query}`).then((keys) => keys.map((key) => key.slice(p.length))),
    namespace: (sub) => namespace(storage, `${p}${sub}`),
    [NAMESPACED]: true,
  }
  return view
}
