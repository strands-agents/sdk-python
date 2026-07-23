/**
 * Types for the file-based memory store.
 */

import type { MemoryStoreConfig } from '../../memory/types.js'
import type { Storage } from '../../storage/storage.js'

/**
 * Configuration for {@link FileMemoryStore}.
 */
export interface FileMemoryStoreConfig extends MemoryStoreConfig {
  /**
   * The unified Storage backend for file operations. Defaults to LocalFileStorage at `./.strands/`.
   * Keys are auto-scoped under `memory/<name>/` unless the provided storage is already namespaced, so
   * stores with distinct names safely share one backend. Two stores with the same name on the same
   * backend share storage — give them different names (or separate storage) to isolate them.
   */
  storage?: Storage
}
