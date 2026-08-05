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
import { ConsolidationError } from '../../errors.js'
import { LocalFileStorage } from '../../storage/local-file-storage.js'
import { NAMESPACED, namespace, normalizeKey } from '../../storage/storage.js'
import { DEFAULT_MAX_SEARCH_RESULTS, tokenize, tokenOverlapScore } from '../../memory/search/keyword.js'
import { generatePlan } from './consolidation/planner.js'
import {
  CONSOLIDATION_CHANGELOG,
  executePlan,
  mapWithConcurrency,
  readAllFiles,
  recordChangelog,
  STORAGE_READ_CONCURRENCY,
} from './consolidation/execute.js'
import { FRONTMATTER_DESCRIPTION_PATTERN } from './consolidation/validate.js'

const encoder = new TextEncoder()
const decoder = new TextDecoder()

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

/** Extract description from YAML frontmatter and return the remaining body. */
function parseFrontmatter(content: string): { description: string; body: string } {
  const match = content.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/)
  if (!match) return { description: '', body: content }

  const frontmatter = match[1] ?? ''
  const body = match[2] ?? ''

  const descMatch = frontmatter.match(FRONTMATTER_DESCRIPTION_PATTERN)
  if (!descMatch) return { description: '', body }

  const rawDesc = descMatch[1] ?? ''
  let description: string
  try {
    description = JSON.parse(rawDesc) as string
  } catch {
    description = rawDesc.slice(1, -1)
  }
  return { description, body }
}

/**
 * Whether a path contains single-dot segments that the OS would collapse — e.g. `./foo.md` resolves
 * to `foo.md`. {@link add} uses it to prevent changelog aliasing that {@link normalizeKey} does not
 * strip. Does NOT check `..` — {@link normalizeKey} already handles that.
 */
function containsDotSegments(key: string): boolean {
  return key.split('/').some((seg) => seg === '.')
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
   * the first `await` and cleared in a `finally`, so a second concurrent call throws rather than
   * racing on a stale snapshot. Instance-scoped, not a lock — see {@link consolidate}'s remarks.
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
   * Auto-scopes keys under `memory/<name>/` so this store never collides with other subsystems or
   * differently-named stores on the same backend. Two stores sharing both a name and a backend still
   * share storage. Already-namespaced storage is used as-is, so scoping never stacks twice.
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
        if (key === CONSOLIDATION_CHANGELOG) return null
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
   * namespace. Pass `metadata.path` to write to a custom location within the namespace; the key is
   * lowercased, so `Projects/Roadmap.md` and `projects/roadmap.md` address the same file.
   *
   * @param content - The knowledge content to store
   * @param metadata - Optional metadata: `title`, `description`, and `path` (custom target path)
   * @returns The canonical storage-relative key the entry was written under, normalized to match
   *   what {@link search} and the backend's `list` report (slash runs collapsed, leading and
   *   trailing slashes stripped, lowercased)
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

      // Probe with read() so the backend resolves key identity rather than comparing list()'s
      // spellings, then suffix on a hit so a new slug never overwrites a stored entry.
      // Best-effort: two concurrent adds can settle on the same key, as Storage has no create-if-absent.
      let suffix = 1
      while (await this._storage.read(key)) {
        key = `${FACTS_PREFIX}${slug}-${suffix}.md`
        suffix++
      }
    }

    // Canonicalize, then lowercase so the store holds at most one spelling per case-fold.
    const canonicalKey = normalizeKey(key).toLowerCase()

    // Reject single-dot segments normalizeKey does not strip: the OS collapses './', so
    // './consolidation-changelog.md' would alias the reserved changelog past the guard below.
    if (containsDotSegments(canonicalKey)) {
      throw new Error("Path must not contain '.' segments: use a direct path without dot-directory references")
    }

    if (canonicalKey === CONSOLIDATION_CHANGELOG) {
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
   * Not safe for concurrent use. Each run snapshots the store up front and mutates it later, and
   * {@link Storage} has no conditional write, so a writer in another process or instance — or an
   * {@link add} on this one — can be silently overwritten. A second `consolidate` on this instance
   * throws, but that guard is instance-scoped, not a lock. Do not write while consolidation runs.
   *
   * @param config - Model and operation configuration
   * @throws TypeError when maxFiles, maxActionsPerPlan, or maxDirectories is not a positive finite number
   * @throws StructuredOutputError when the model returns no plan
   * @throws ConsolidationError when the store is not writable, a run is already in flight, the store
   *   exceeds maxFiles, or the plan is unusable (over the action limit, fails validation, or the turn
   *   limit is hit without a plan)
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
    const maxFiles = config.maxFiles ?? 100
    const maxActionsPerPlan = config.maxActionsPerPlan ?? 1000
    const maxDirectories = config.maxDirectories ?? 8

    // Validate before any I/O so a malformed config fails at the call site. A NaN cap would silently
    // disable its guardrail — `files.size > NaN` is always false, so the gate never fires.
    const assertPositiveFinite = (name: string, value: number): void => {
      if (!Number.isFinite(value) || value <= 0) {
        throw new TypeError(`${name} must be a positive finite number, got ${value}`)
      }
    }
    assertPositiveFinite('maxFiles', maxFiles)
    assertPositiveFinite('maxActionsPerPlan', maxActionsPerPlan)
    assertPositiveFinite('maxDirectories', maxDirectories)

    if (!this.writable) {
      throw new ConsolidationError(
        'FileMemoryStore: consolidate requires a writable store (writable: false is searchable only, never written to)'
      )
    }
    // Set synchronously before the first await so a concurrent call cannot slip past the check
    if (this._consolidating) {
      throw new ConsolidationError(
        'A consolidation is already running on this store instance; run consolidation one at a time'
      )
    }
    this._consolidating = true
    try {
      await this._consolidate(config, maxFiles, maxActionsPerPlan, maxDirectories)
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
  private async _consolidate(
    config: ConsolidateConfig,
    maxFiles: number,
    maxActionsPerPlan: number,
    maxDirectories: number
  ): Promise<void> {
    const operations = config.operations ?? [...CONSOLIDATE_OPERATIONS]

    const files = await readAllFiles(this._storage)
    if (files.size === 0) return

    // Bounds the planner's input so a large corpus cannot blow past the model's context window
    if (files.size > maxFiles) {
      throw new ConsolidationError(
        `Knowledge store exceeds consolidation file limit: ${files.size} files (maxFiles: ${maxFiles})`
      )
    }

    const plan = await generatePlan(config, operations, files, maxDirectories, maxActionsPerPlan)

    const deleteErrors = await executePlan(this._storage, plan, files)
    // Record the changelog even on partial failure — writes and some deletes already hit disk,
    // so an accurate audit trail must capture the run before surfacing the error
    await recordChangelog(this._storage, operations, plan, deleteErrors)

    if (deleteErrors.length > 0) {
      const paths = deleteErrors.map((deleteError) => deleteError.path).join(', ')
      throw new ConsolidationError(
        `Plan executed but ${deleteErrors.length} delete(s) failed: ${paths}. ` +
          `Writes succeeded — duplicate content may remain until next consolidation.`
      )
    }
  }
}
