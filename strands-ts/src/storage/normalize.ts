import { StorageError } from '../errors.js'

/**
 * Validates and normalizes a storage key: collapses runs of `/`, strips leading
 * and trailing `/`, and rejects empty keys and any `..` segment.
 *
 * @param key - The raw key to normalize
 * @returns The normalized key
 * @throws {@link StorageError} if the key is empty or contains a `..` segment
 */
export function normalizeKey(key: string): string {
  const normalized = key.replace(/\/+/g, '/').replace(/^\/+|\/+$/g, '')
  if (normalized.length === 0) {
    throw new StorageError('Storage key must not be empty')
  }
  if (normalized.split('/').includes('..')) {
    throw new StorageError(`Invalid storage key '${key}': '..' path segments are not allowed`)
  }
  return normalized
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
  const normalized = prefix.replace(/\/+/g, '/').replace(/^\/+/, '')
  if (normalized.split('/').includes('..')) {
    throw new StorageError(`Invalid storage prefix '${prefix}': '..' path segments are not allowed`)
  }
  return normalized
}
