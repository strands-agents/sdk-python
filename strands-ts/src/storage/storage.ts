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
  put(key: string, data: Uint8Array): Promise<void>

  /**
   * Retrieves the bytes previously stored under `key`.
   *
   * @param key - The key to read
   * @returns The stored bytes, or `null` if no value exists for `key`
   * @throws {@link StorageError} if the read fails for a reason other than a missing key
   */
  get(key: string): Promise<Uint8Array | null>

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
}
