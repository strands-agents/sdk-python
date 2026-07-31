/**
 * File-based memory store implementing the {@link MemoryStore} interface.
 *
 * Organizes knowledge as a structured file hierarchy under a `memory/` storage namespace. Provides
 * keyword-based search via `search_memory` (registered by {@link MemoryManager}).
 *
 * Consolidation is a separate concern and lives under `consolidation/`: this file holds the public
 * {@link FileMemoryStore.consolidate} entry point and the run's orchestration, delegating planning,
 * validation, and execution to those modules.
 */

import type { JSONValue } from '../../types/json.js'
import type { MemoryEntry, MemoryStore, SearchOptions } from '../../memory/types.js'
import type { ExtractionConfig } from '../../memory/extraction/types.js'
import type { Storage } from '../../storage/storage.js'
import type { ConsolidateConfig, FileMemoryStoreConfig } from './types.js'
import { CONSOLIDATE_OPERATIONS } from './types.js'
import { LocalFileStorage } from '../../storage/local-file-storage.js'
import { NAMESPACED, namespace, normalizeKey } from '../../storage/storage.js'
import { DEFAULT_MAX_SEARCH_RESULTS, tokenize, tokenOverlapScore } from '../../memory/search/keyword.js'
import {
  CONSOLIDATION_CHANGELOG,
  containsDotSegments,
  decoder,
  encoder,
  isConsolidationChangelog,
  mapWithConcurrency,
  parseFrontmatter,
  STORAGE_READ_CONCURRENCY,
} from './internal.js'
import { generatePlan } from './consolidation/planner.js'
import { executePlan, readAllFiles, recordChangelog } from './consolidation/execute.js'

/**
 * Top-level storage namespace shared by every file memory store, isolating them as a group from
 * other subsystems (sessions, context offloading) that may share the same backend. See the storage
 * design doc's key-prefix convention (`team/designs/0014-storage.md`). Each store further scopes
 * under its own `name` within this namespace — see {@link FileMemoryStore._resolveStorage}.
 */
const STORAGE_NAMESPACE = 'memory'

/** Default subdirectory (within the store's namespace) for entries added without an explicit path. */
const FACTS_PREFIX = 'facts/'

/** Extract the filename stem (without `.md` extension) from a storage key. */
function basename(key: string): string {
  const filename = key.split('/').pop() ?? key
  return filename.replace(/\.md$/, '')
}

/** Convert text to a URL-safe kebab-case slug, truncated to 50 characters. */
function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, '')
    .trim()
    .replace(/\s+/g, '-')
    .slice(0, 50)
}

/**
 * A file-based memory store backed by the unified {@link Storage} interface.
 *
 * Implements {@link MemoryStore} for use with {@link MemoryManager}. Knowledge is stored as
 * markdown files with YAML frontmatter under a `memory/` storage namespace. Retrieval is via the
 * `search_memory` tool registered by {@link MemoryManager}, which calls {@link search} (keyword-based).
 *
 * The storage backend defaults to {@link LocalFileStorage} when no custom {@link Storage}
 * implementation is provided. Keys are auto-scoped under `memory/<name>/` (so a store named
 * `agent-memory` with the default backend lands knowledge under `./.strands/memory/agent-memory/`),
 * isolating it from other subsystems — and from differently-named file memory stores — that share
 * the same backend. Two stores with the same name on one backend share storage. Pass a storage view
 * that is already namespaced to override this.
 *
 * @example
 * ```typescript
 * import { Agent, MemoryManager } from '@strands-agents/sdk'
 * import { FileMemoryStore } from '@strands-agents/sdk/vended-memory-stores/file-memory-store'
 *
 * const memoryStore = new FileMemoryStore({ name: 'agent-memory' })
 *
 * const agent = new Agent({
 *   model,
 *   memoryManager: new MemoryManager({ stores: [memoryStore], injection: false }),
 * })
 * ```
 */
export class FileMemoryStore implements MemoryStore {
  readonly name: string
  readonly writable: boolean
  readonly description?: string
  readonly maxSearchResults?: number
  readonly extraction?: boolean | ExtractionConfig

  private readonly _storage: Storage

  /**
   * Guards against overlapping {@link consolidate} runs on this instance. Set synchronously before
   * the first `await` and cleared in a `finally`, so a second concurrent call observes it and throws
   * rather than planning against a stale snapshot and racing on the same keys.
   *
   * Instance-scoped only, and not a lock — it is unaware of {@link add}, of other store instances
   * over the same storage, and of other processes. See {@link consolidate}'s remarks for what that
   * leaves exposed.
   */
  private _consolidating = false

  constructor(config: FileMemoryStoreConfig) {
    this.name = config.name
    this.writable = config.writable ?? true
    if (config.description !== undefined) this.description = config.description
    if (config.maxSearchResults !== undefined) this.maxSearchResults = config.maxSearchResults
    if (config.extraction !== undefined) this.extraction = config.extraction
    this._storage = this._resolveStorage(config.storage ?? new LocalFileStorage())
  }

  /**
   * Auto-scopes keys under `memory/<name>/` so this store never collides with other subsystems
   * (sessions, context offloading) sharing the same backend, nor with a differently-named file
   * memory store on it — distinct {@link name}s yield non-overlapping scopes. Two stores sharing
   * both a name and a backend still share storage. Storage that is already namespaced — e.g. handed
   * down pre-scoped by a future router — is used as-is, so scoping never stacks twice.
   */
  private _resolveStorage(storage: Storage): Storage {
    if (NAMESPACED in storage) return storage
    const prefix = `${STORAGE_NAMESPACE}/${this.name}`
    if (storage.namespace) return storage.namespace(prefix)
    return namespace(storage, prefix)
  }

  /**
   * Search knowledge files by keyword matching against filenames, descriptions, and content.
   *
   * Returns the top matches ranked by distinct token overlap. Each result's `metadata.path`
   * reflects the entry's current storage location and may change after consolidation.
   */
  async search(query: string, options?: SearchOptions): Promise<MemoryEntry[]> {
    const maxResults = options?.maxSearchResults ?? this.maxSearchResults ?? DEFAULT_MAX_SEARCH_RESULTS
    const queryTokens = tokenize(query)
    if (queryTokens.size === 0) return []

    const allKeys = await this._storage.list('')

    const scored = (
      await mapWithConcurrency(allKeys, STORAGE_READ_CONCURRENCY, async (key) => {
        // The changelog is an audit artifact, not knowledge — never return it as a memory
        if (isConsolidationChangelog(key)) return null
        try {
          const bytes = await this._storage.read(key)
          if (!bytes) return null

          const content = decoder.decode(bytes)
          const { description, body } = parseFrontmatter(content)
          const searchable = `${basename(key)} ${description} ${body}`

          const relevanceScore = tokenOverlapScore(queryTokens, searchable)
          if (relevanceScore === 0) return null
          return {
            entry: {
              content: body.trim(),
              metadata: { path: key, description, _relevanceScore: relevanceScore },
            } as MemoryEntry,
            relevanceScore,
          }
        } catch {
          return null
        }
      })
    ).filter((s): s is { entry: MemoryEntry; relevanceScore: number } => s !== null)

    scored.sort((a, b) => b.relevanceScore - a.relevanceScore)
    return scored.slice(0, maxResults).map((s) => s.entry)
  }

  /**
   * Add a knowledge entry to the store.
   *
   * Writes a markdown file with YAML frontmatter. By default writes to `facts/` within the store's
   * namespace. Pass `metadata.path` to write to a custom location within the namespace.
   *
   * @param content - The knowledge content to store
   * @param metadata - Optional metadata: `title`, `description`, and `path` (custom target path)
   * @returns The canonical storage-relative key the entry was written under, normalized to
   *   match what {@link search} and the backend's `list` report (slash runs collapsed, leading
   *   and trailing slashes stripped)
   */
  async add(content: string, metadata?: Record<string, JSONValue>): Promise<string> {
    const customPath = metadata?.['path'] as string | undefined
    const firstSentence = content.split(/[.\n]/)[0]!
    const title = (metadata?.['title'] as string | undefined) ?? firstSentence.slice(0, 60)
    const description = (metadata?.['description'] as string | undefined) ?? firstSentence.slice(0, 120)

    let key: string
    if (customPath) {
      key = customPath
      if (!key.endsWith('.md')) key += '.md'
    } else {
      const slug = slugify(title) || `entry-${Date.now()}`
      key = `${FACTS_PREFIX}${slug}.md`

      // Probe with read() so the backend resolves key identity — a case-insensitive filesystem
      // treats Topic.md and topic.md as one file, which comparing list()'s spellings would miss.
      // Best-effort: two concurrent adds can settle on the same key, as Storage has no create-if-absent.
      let suffix = 1
      while (await this._storage.read(key)) {
        key = `${FACTS_PREFIX}${slug}-${suffix}.md`
        suffix++
      }
    }

    // Canonicalize with the same helper the shipped backends apply internally, so the
    // returned receipt matches the key search() and the backend's list() report.
    const canonicalKey = normalizeKey(key)

    // Reject single-dot path segments that normalizeKey does not strip. The OS collapses './' so
    // a key like './consolidation-changelog.md' would alias the reserved changelog on disk despite
    // failing the string-equality guard below. Consolidation's validatePath already rejects dots;
    // this closes the same gap in the public add() path.
    if (containsDotSegments(canonicalKey)) {
      throw new Error("Path must not contain '.' segments: use a direct path without dot-directory references")
    }

    if (isConsolidationChangelog(canonicalKey)) {
      throw new Error(`Path must not be the reserved '${CONSOLIDATION_CHANGELOG}' file`)
    }

    const fileContent = `---\ndescription: ${JSON.stringify(description)}\n---\n\n${content}\n`
    await this._storage.write(canonicalKey, encoder.encode(fileContent))
    return canonicalKey
  }

  /**
   * Run consolidation to maintain knowledge quality.
   *
   * Plan-then-execute: one structured-output call produces an action plan over all files,
   * guardrails validate the whole plan before anything is mutated, then deterministic code
   * executes it (writes before deletes). A plan that fails validation throws without mutating.
   * The planning agent is bounded by a turn limit (default 3) to prevent runaway loops.
   *
   * @remarks
   * Not concurrency-safe. Each run snapshots the store up front and mutates it later, so a second
   * call on this instance throws rather than planning against a stale snapshot — but that guard is
   * instance-scoped, not a lock. {@link Storage} offers only unconditional read/write/delete/list,
   * so nothing here can observe a writer in another process, on another instance, or calling
   * {@link add} on this one. A concurrent {@link add} minting a fresh key is safe (the snapshot
   * never saw it, so no action can name it); an `add` with an explicit `metadata.path` naming a
   * snapshotted file is not, and can be silently discarded by a merge or delete. Do not write to
   * the store while consolidation is in flight.
   *
   * @param config - Model and operation configuration
   * @throws Error when a consolidation is already running on this store instance
   * @throws TypeError when maxFiles, maxActionsPerPlan, or maxDirectories is not a positive finite number
   * @throws Error when the knowledge store exceeds the file count limit (maxFiles)
   * @throws Error when structured output is undefined (model did not return a plan)
   * @throws Error when the consolidation plan exceeds the action limit (maxActionsPerPlan)
   * @throws Error when the consolidation plan fails validation
   * @throws Error when the consolidation agent exceeds its turn limit without producing a plan
   * @throws Error when a path the plan would create was claimed by a writer outside this run
   *   (the store is left unchanged — no write or delete runs)
   * @throws Error when a path the plan writes matches several stored keys differing only by case
   *   (the store is left unchanged — no write or delete runs)
   *
   * @example
   * ```typescript
   * // Run periodically (e.g. from a scheduled job), not on the agent's hot path
   * await memoryStore.consolidate({
   *   model,
   *   operations: ['deduplicate', 'prune'], // omit to run all operations
   * })
   * ```
   */
  async consolidate(config: ConsolidateConfig): Promise<void> {
    if (!this.writable) {
      throw new Error(
        'FileMemoryStore: consolidate requires a writable store (writable: false is searchable only, never written to)'
      )
    }
    // Set synchronously before the first await so a concurrent call cannot slip past the check
    if (this._consolidating) {
      throw new Error('A consolidation is already running on this store instance; run consolidation one at a time')
    }
    this._consolidating = true
    try {
      await this._consolidate(config)
    } finally {
      this._consolidating = false
    }
  }

  /**
   * Execute a single consolidation run. The guard in {@link consolidate} keeps this from overlapping
   * another run on the same instance; it says nothing about runs on other instances or processes.
   *
   * @param config - Model and operation configuration
   */
  private async _consolidate(config: ConsolidateConfig): Promise<void> {
    const operations = config.operations ?? [...CONSOLIDATE_OPERATIONS]
    const maxDirectories = config.maxDirectories ?? 8

    const maxFiles = config.maxFiles ?? 100
    const maxActionsPerPlan = config.maxActionsPerPlan ?? 1000

    // A NaN or non-positive cap would silently disable the guardrail it configures — NaN fails every
    // comparison, so `files.size > NaN` is false and the gate never fires
    const assertPositiveFinite = (name: string, value: number): void => {
      if (!Number.isFinite(value) || value <= 0) {
        throw new TypeError(`${name} must be a positive finite number, got ${value}`)
      }
    }
    assertPositiveFinite('maxFiles', maxFiles)
    assertPositiveFinite('maxActionsPerPlan', maxActionsPerPlan)
    assertPositiveFinite('maxDirectories', maxDirectories)

    const files = await readAllFiles(this._storage)
    if (files.size === 0) return

    // Bounds the planner's input so a large corpus cannot blow past the model's context window
    if (files.size > maxFiles) {
      throw new Error(`Knowledge store exceeds consolidation file limit: ${files.size} files (maxFiles: ${maxFiles})`)
    }

    const plan = await generatePlan(config, operations, files, maxDirectories, maxActionsPerPlan)

    const deleteErrors = await executePlan(this._storage, plan, files)
    // Record the changelog even on partial failure — writes and some deletes already hit disk,
    // so an accurate audit trail must capture the run before surfacing the error
    await recordChangelog(this._storage, operations, plan, deleteErrors)

    if (deleteErrors.length > 0) {
      const paths = deleteErrors.map((deleteError) => deleteError.path).join(', ')
      throw new Error(
        `Plan executed but ${deleteErrors.length} delete(s) failed: ${paths}. ` +
          `Writes succeeded — duplicate content may remain until next consolidation.`
      )
    }
  }
}
