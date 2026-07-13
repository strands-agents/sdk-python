/**
 * Types for the file-based memory store.
 */

import type { MemoryStoreConfig } from '../../memory/types.js'
import type { Storage } from '../../storage/storage.js'

/**
 * Configuration for {@link FileMemoryStore}.
 */
export interface FileMemoryStoreConfig extends MemoryStoreConfig {
  /** The unified Storage backend for file operations. Defaults to LocalFileStorage at `~/.strands/`. */
  storage?: Storage
}
