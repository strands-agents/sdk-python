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
  decoder,
  encoder,
  isConsolidationChangelog,
  mapWithConcurrency,
  parseFrontmatter,
  STORAGE_READ_CONCURRENCY,
} from './internal.js'
import { generatedByteSize } from './consolidation/plan.js'
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

/** Default cap on total UTF-8 bytes of knowledge files accepted as planner input. */
const DEFAULT_MAX_INPUT_BYTES = 128 * 1024

/**
 * Default cap for total generated content bytes across all write actions in a plan. Consolidation
 * reorganizes the corpus it was given, so 2x the input cap leaves headroom for content split across
 * merge targets while still catching a planner that generates instead of reorganizing.
 */
const DEFAULT_MAX_GENERATED_BYTES = 2 * DEFAULT_MAX_INPUT_BYTES

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

      // Probe with read() so the backend resolves key identity: a case-insensitive
      // filesystem treats Topic.md and topic.md as the same file, which comparing
      // against list()'s exact key spellings in memory would miss. A miss returns null
      // without transferring a body, so the only full reads are on genuine collisions.
      // Best-effort: two concurrent adds can settle on the same free key and the later
      // write wins, since Storage offers no create-if-absent to make the claim atomic.
      let suffix = 1
      while (await this._storage.read(key)) {
        key = `${FACTS_PREFIX}${slug}-${suffix}.md`
        suffix++
      }
    }

    // Canonicalize with the same helper the shipped backends apply internally, so the
    // returned receipt matches the key search() and the backend's list() report.
    const canonicalKey = normalizeKey(key)

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
   * Uses a plan-then-execute strategy: one structured-output call produces an action plan
   * over all files, programmatic guardrails validate the whole plan before anything is mutated,
   * then deterministic code executes it (writes before deletes). On validation failure, one
   * revise-retry is attempted; if the revised plan also fails validation, the method throws.
   *
   * Only one consolidation may run at a time per store instance. Each run snapshots the store at
   * the start (`readAllFiles`) and later mutates it, so overlapping runs would plan against stale
   * snapshots and race on the same keys. A call that starts while another is still in flight throws
   * immediately rather than executing against a snapshot the other run has invalidated.
   *
   * @remarks
   * Consolidation is not concurrency-safe, and the instance guard above is the only serialization it
   * has. There is no lock: the {@link Storage} contract exposes only unconditional read/write/delete/
   * list, so nothing here can hold off or even observe a writer on another store instance, in another
   * process, or calling {@link add} on this one. Scheduling a run as a background job does not change
   * this — it is a placement recommendation, not a quiescence guarantee.
   *
   * A concurrent {@link add} nevertheless survives, because of where it writes rather than any
   * coordination: it mints a fresh key, which the snapshot never captured, so no plan action can name
   * it. Two gaps follow from that, neither closed by a lock:
   *
   * - A plan can also create new files (a merge or move target absent from the snapshot), so its
   *   chosen name can collide with a concurrent `add`'s fresh key. `executePlan` re-reads those
   *   targets immediately before writing and aborts the run untouched if one is already claimed.
   *   This is best-effort detection, not mutual exclusion: it shrinks the exposure from the whole
   *   run (which spans a model call) to the gap between that check and the write, and a write
   *   landing inside that gap is still overwritten.
   * - An `add` carrying an explicit `metadata.path` that names a file already in the snapshot is not
   *   covered at all. The plan was built from that file's pre-`add` content and may merge or delete
   *   it, discarding what the `add` wrote. Detecting this needs a conditional write or version check
   *   (ETag, generation number) that {@link Storage} cannot currently express.
   *
   * To avoid both, do not write to the store while a consolidation is in flight — either quiesce
   * writers for the duration, or issue consolidation and explicit-path writes from a single caller
   * that serializes them.
   *
   * The internal planning agent is bounded by a turn limit (default 3 turns) to prevent runaway
   * loops. The planning agent has no tools registered and is expected to complete in a single turn;
   * if it does not produce a valid structured plan within the limit, consolidation throws.
   *
   * @param config - Model and operation configuration
   * @throws Error when a consolidation is already running on this store instance
   * @throws Error when the knowledge store exceeds the file count limit (maxFiles)
   * @throws Error when the knowledge store exceeds the input byte size limit (maxInputBytes)
   * @throws Error when structured output is undefined (model did not return a plan)
   * @throws Error when the consolidation plan exceeds the action limit (maxActionsPerPlan)
   * @throws Error when the plan's generated content exceeds the byte limit (maxGeneratedBytes)
   * @throws Error when the consolidation plan fails validation after retry
   * @throws Error when the consolidation agent exceeds its turn limit without producing a plan
   * @throws Error when a path the plan would create was claimed by a writer outside this run
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
    const maxInputBytes = config.maxInputBytes ?? DEFAULT_MAX_INPUT_BYTES
    const maxActionsPerPlan = config.maxActionsPerPlan ?? 1000
    const maxGeneratedBytes = config.maxGeneratedBytes ?? DEFAULT_MAX_GENERATED_BYTES

    const assertPositiveFinite = (name: string, value: number): void => {
      if (!Number.isFinite(value) || value <= 0) {
        throw new TypeError(`${name} must be a positive finite number, got ${value}`)
      }
    }
    assertPositiveFinite('maxFiles', maxFiles)
    assertPositiveFinite('maxInputBytes', maxInputBytes)
    assertPositiveFinite('maxActionsPerPlan', maxActionsPerPlan)
    assertPositiveFinite('maxGeneratedBytes', maxGeneratedBytes)
    assertPositiveFinite('maxDirectories', maxDirectories)

    const files = await readAllFiles(this._storage)
    if (files.size === 0) return

    if (files.size > maxFiles) {
      throw new Error(`Knowledge store exceeds consolidation file limit: ${files.size} files (maxFiles: ${maxFiles})`)
    }

    let totalBytes = 0
    for (const content of files.values()) {
      totalBytes += encoder.encode(content).byteLength
    }
    if (totalBytes > maxInputBytes) {
      throw new Error(
        `Knowledge store exceeds consolidation input size limit: ${totalBytes} bytes (maxInputBytes: ${maxInputBytes})`
      )
    }

    const plan = await generatePlan(config, operations, files, maxDirectories, maxActionsPerPlan)

    // Like the action-count guard, an oversized plan is a runaway signal rather than a fixable
    // mistake, so this throws instead of routing into the revise-retry
    const generatedBytes = generatedByteSize(plan)
    if (generatedBytes > maxGeneratedBytes) {
      throw new Error(
        `Consolidation plan exceeds generated content limit: ${generatedBytes} bytes (maxGeneratedBytes: ${maxGeneratedBytes})`
      )
    }

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
