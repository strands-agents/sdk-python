import type { Storage } from './storage.js'
import { normalizePrefix } from './normalize.js'

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
export function namespace(storage: Storage, prefix: string): Storage & { namespace(prefix: string): Storage } {
  const normalized = normalizePrefix(prefix)
  const p = normalized ? `${normalized}/` : ''
  return {
    write: (key, data) => storage.write(`${p}${key}`, data),
    read: (key) => storage.read(`${p}${key}`),
    delete: (key) => storage.delete(`${p}${key}`),
    list: (query) => storage.list(`${p}${query}`).then((keys) => keys.map((key) => key.slice(p.length))),
    namespace: (sub) => namespace(storage, `${p}${sub}`),
  }
}
