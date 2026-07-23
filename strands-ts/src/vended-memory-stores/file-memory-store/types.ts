/**
 * Types for the file-based memory store.
 */

import type { MemoryStoreConfig } from '../../memory/types.js'
import type { Model } from '../../models/model.js'
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

/**
 * A maintenance operation that the consolidation agent can perform.
 *
 * - `deduplicate` — merge files with overlapping content
 * - `resolveContradictions` — fix conflicting facts across files
 * - `deriveInsights` — synthesize new knowledge from patterns across files
 * - `prune` — remove stale or irrelevant entries
 * - `reorganize` — move files to better subdirectories
 */
export type ConsolidateOperation = 'deduplicate' | 'resolveContradictions' | 'deriveInsights' | 'prune' | 'reorganize'

/**
 * Configuration for {@link FileMemoryStore.consolidate}.
 */
export interface ConsolidateConfig {
  /** The model to use for consolidation reasoning. */
  model: Model

  /** Which maintenance operations to run. Defaults to all operations. */
  operations?: ConsolidateOperation[]

  /** Maximum subdirectories allowed under `knowledge/`. Defaults to 8. */
  maxDirectories?: number

  /**
   * Maximum number of knowledge files allowed as planner input. Defaults to 100.
   *
   * Bounds the single-call planner input; plan output scales with touched files.
   */
  maxFiles?: number

  /**
   * Maximum total UTF-8 byte size of all knowledge file contents allowed as planner input.
   * Defaults to 131072 (128 KiB).
   *
   * Bounds the single-call planner input; plan output scales with touched files.
   */
  maxInputBytes?: number
}
