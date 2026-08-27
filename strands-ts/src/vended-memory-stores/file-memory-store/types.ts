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
  /**
   * Whether to serve retrieval by progressive disclosure: inject the file listing each turn and
   * register the store's read file tool. `false` withholds both, leaving retrieval to the
   * {@link MemoryManager}'s search-based paths — its context injection and the `search_memory` tool.
   *
   * @defaultValue true
   */
  progressiveDisclosure?: boolean
  /**
   * Maximum files progressive disclosure reads and injects per turn — each is one storage read and one
   * line of injected context, so this bounds the recurring per-turn cost (notably storage GETs on a
   * remote backend). When more files exist, the first `maxListedFiles` by sorted path are shown and the
   * listing reports how many it omitted. Set to `Infinity` to disable. Does not cap
   * {@link FileMemoryStore.listFiles}, which always returns every file.
   *
   * @defaultValue 100
   */
  maxListedFiles?: number
}

/**
 * Every maintenance operation the consolidation agent can perform, in application order. The default
 * set {@link FileMemoryStore.consolidate} runs when `operations` is omitted, and the source of truth
 * {@link ConsolidateOperation} is derived from, so the two can never drift.
 *
 * - `deduplicate` — merge files with overlapping content
 * - `resolveContradictions` — fix conflicting facts across files
 * - `deriveInsights` — synthesize new knowledge from patterns across files
 * - `prune` — remove stale or irrelevant entries
 * - `reorganize` — move files to better subdirectories
 */
export const CONSOLIDATE_OPERATIONS = [
  'deduplicate',
  'resolveContradictions',
  'deriveInsights',
  'prune',
  'reorganize',
] as const

/**
 * A maintenance operation that the consolidation agent can perform. See {@link CONSOLIDATE_OPERATIONS}
 * for the full set and per-operation descriptions.
 */
export type ConsolidateOperation = (typeof CONSOLIDATE_OPERATIONS)[number]

/**
 * Configuration for {@link FileMemoryStore.consolidate}.
 */
export interface ConsolidateConfig {
  /** The model to use for consolidation reasoning. Defaults to the {@link Agent} default model. */
  model?: Model

  /** Which maintenance operations to run. Defaults to all operations. */
  operations?: ConsolidateOperation[]

  /**
   * Maximum subdirectories allowed under the store's namespace. Defaults to 8 — enough to group
   * knowledge by topic while keeping the tree shallow, so it never fragments into sparse directories.
   */
  maxDirectories?: number

  /**
   * Maximum number of knowledge files allowed as planner input. Defaults to 100. Bounds the file
   * *count*, not payload size — a few very large files can still overflow the model context and
   * surface as a provider error rather than a {@link ConsolidationError}.
   */
  maxFiles?: number

  /**
   * Maximum number of actions a single consolidation plan may contain. Defaults to 1000. Bounds the
   * planner *output* (`maxFiles` caps the input); an exceeding plan is rejected before any mutation.
   */
  maxActionsPerPlan?: number
}
