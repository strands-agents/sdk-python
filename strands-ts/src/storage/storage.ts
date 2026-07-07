/**
 * A backend for storing and retrieving raw bytes under string keys.
 *
 * The interface is deliberately minimal — four operations over opaque `Uint8Array`
 * values. Implementations must treat keys as opaque path-like strings (segments
 * separated by `/`) and must round-trip the bytes they are given unchanged.
 *
 * Implement this to add a custom backend; the SDK ships {@link InMemoryStorage},
 * {@link LocalFileStorage}, and {@link S3Storage}.
 */
export interface Storage {
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
   * Lists the keys whose names begin with `prefix`.
   *
   * Returns full keys (not the suffix after the prefix), sorted lexicographically.
   * An empty `prefix` lists every key.
   *
   * @param prefix - Key prefix to match
   * @returns The matching keys, sorted ascending
   * @throws {@link StorageError} if the listing fails
   */
  list(prefix: string): Promise<string[]>
}
