/**
 * File-based memory store implementing the {@link MemoryStore} interface.
 *
 * Organizes knowledge as a structured file hierarchy under a `memory/` storage namespace. Provides
 * progressive disclosure — the file listing injected each turn, read on demand with the store's own
 * tool — plus keyword-based search via `search_memory` (registered by {@link MemoryManager}).
 *
 * Consolidation lives under `consolidation/`; progressive disclosure in `progressive-disclosure.ts`.
 * This file holds the store itself plus the public {@link FileMemoryStore.consolidate} entry point.
 */

import type { JSONValue } from '../../types/json.js'
import type { MemoryEntry, MemoryStore, SearchOptions } from '../../memory/types.js'
import type { ExtractionConfig } from '../../memory/extraction/types.js'
import type { Plugin } from '../../plugins/plugin.js'
import type { Storage } from '../../storage/storage.js'
import type { Tool } from '../../tools/tool.js'
import type { ConsolidateConfig, FileMemoryStoreConfig } from './types.js'
import { CONSOLIDATE_OPERATIONS } from './types.js'
import { ConsolidationError } from '../../errors.js'
import { logger } from '../../logging/logger.js'
import { LocalFileStorage } from '../../storage/local-file-storage.js'
import { NAMESPACED, namespace, normalizeKey } from '../../storage/storage.js'
import { DEFAULT_MAX_SEARCH_RESULTS, tokenize, tokenOverlapScore } from '../../memory/search/keyword.js'
import { generatePlan } from './consolidation/planner.js'
import {
  CONSOLIDATION_CHANGELOG,
  executePlan,
  isKnowledgeKey,
  readAllFiles,
  recordChangelog,
} from './consolidation/execute.js'
import { FRONTMATTER_DESCRIPTION_PATTERN } from './consolidation/validate.js'
import { mapWithConcurrency, STORAGE_READ_CONCURRENCY } from './concurrency.js'
import { createProgressiveDisclosureInjector, createReadTool } from './progressive-disclosure.js'

const encoder = new TextEncoder()
const decoder = new TextDecoder()

/**
 * Top-level storage namespace shared by every file memory store, isolating them as a group from
 * other subsystems (sessions, context offloading) that may share the same backend. Each store
 * further scopes under its own `name` within this namespace.
 */
const STORAGE_NAMESPACE = 'memory'

/** Default subdirectory (within the store's namespace) for entries added without an explicit path. */
const FACTS_PREFIX = 'facts/'

/**
 * Default cap on files the per-turn progressive-disclosure injection reads and shows — each is one
 * storage read and one line of context, so this bounds the recurring cost on a large store. Override
 * per store with {@link FileMemoryStoreConfig.maxListedFiles}. {@link ConsolidateConfig.maxFiles} sets
 * the same scale; {@link FileMemoryStore.listFiles} is not capped.
 */
const DEFAULT_MAX_LISTED_FILES = 100

/** Extract the filename stem (without `.md` extension) from a storage key. */
function basename(key: string): string {
  const filename = key.split('/').pop() ?? key
  return filename.replace(/\.md$/, '')
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
 * Canonicalize a caller- or model-supplied path to the key the store reads and writes under.
 * `normalizeKey` collapses slash runs and rejects `..`; this lowercases (so the store holds one
 * spelling per case-fold), rejects `.` segments (which the OS collapses, aliasing another key), and
 * rejects the reserved changelog. {@link FileMemoryStore.add} and {@link FileMemoryStore.readFile}
 * both go through it, so a file is readable under the same spelling it was written with.
 *
 * @throws {@link StorageError} when the path is empty or contains a `..` segment
 * @throws Error when the path contains a `.` segment or addresses the reserved changelog
 */
function canonicalizeKnowledgePath(path: string): string {
  const key = normalizeKey(path).toLowerCase()
  if (key.split('/').some((segment) => segment === '.')) {
    throw new Error(`Invalid memory path '${path}': must not contain '.' segments`)
  }
  if (key === CONSOLIDATION_CHANGELOG) {
    throw new Error(`Invalid memory path '${path}': must not be the reserved '${CONSOLIDATION_CHANGELOG}' file`)
  }
  return key
}

/**
 * A file-based memory store backed by the unified {@link Storage} interface.
 *
 * Implements {@link MemoryStore} for use with {@link MemoryManager}. Knowledge is stored as
 * markdown files with YAML frontmatter under a `memory/` storage namespace.
 *
 * Retrieval is by progressive disclosure: the store injects the file listing each turn and registers
 * a read tool named after it, so the model judges what is relevant and pulls only that. It composes with
 * the manager's `injection`: keep it on (the default) for higher quality, or set `injection: false` for
 * lower cost. `search_memory` searches inside bodies.
 *
 * The storage backend defaults to {@link LocalFileStorage} when no custom {@link Storage}
 * implementation is provided. Keys are auto-scoped under `memory/<name>/`, isolating the store from
 * other subsystems and from differently-named file memory stores on the same backend.
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
  private readonly _progressiveDisclosure: boolean
  private readonly _maxListedFiles: number

  /**
   * Guards against overlapping {@link consolidate} runs on this instance. Set synchronously before
   * the first `await` and cleared in a `finally`, so a second concurrent call throws rather than
   * racing on a stale snapshot. Instance-scoped, not a lock — see {@link consolidate}'s remarks.
   */
  private _consolidating = false

  /** Set once the injected listing has been truncated, so the size warning logs at most once. */
  private _listingTruncationWarned = false

  constructor(config: FileMemoryStoreConfig) {
    this.name = config.name
    this.writable = config.writable ?? true
    if (config.description !== undefined) this.description = config.description
    if (config.maxSearchResults !== undefined) this.maxSearchResults = config.maxSearchResults
    if (config.extraction !== undefined) this.extraction = config.extraction
    this._progressiveDisclosure = config.progressiveDisclosure ?? true
    // Fail at construction, not agent-wide at first invoke(): name derives the read tool name, capped
    // to fit the registry's 64-char limit.
    if (this._progressiveDisclosure && (!/[a-zA-Z0-9]/.test(this.name) || this.name.length > 54)) {
      throw new RangeError(
        `FileMemoryStore: name must contain a letter or digit and be at most 54 characters (got ${JSON.stringify(config.name)}); it derives the read tool name`
      )
    }
    this._maxListedFiles = config.maxListedFiles ?? DEFAULT_MAX_LISTED_FILES
    this._storage = this._resolveStorage(config.storage ?? new LocalFileStorage())
  }

  /**
   * Auto-scopes keys under `memory/<name>/` so this store never collides with other subsystems or
   * differently-named stores on the same backend. Already-namespaced storage is used as-is.
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
        if (!isKnowledgeKey(key)) return null
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
   * List every knowledge file's path and description, without content — enough to judge relevance
   * cheaply. Sorted by path, changelog excluded. Never capped, unlike the per-turn injection (see
   * {@link FileMemoryStoreConfig.maxListedFiles}); both are built by {@link _collectListing}.
   *
   * @returns Every knowledge file's path and description, sorted by path
   */
  async listFiles(): Promise<{ path: string; description: string }[]> {
    return (await this._collectListing()).files
  }

  /**
   * Shared builder for {@link listFiles} and the injector: list keys, sort, read each description.
   * `limit` caps how many are read (the first by sorted key), bounding per-turn cost; `total` is the
   * full count so the caller can report what it omitted.
   *
   * @param limit - Maximum files to read and return; omit for all
   * @returns The (possibly capped) files sorted by path, and the total eligible file count
   */
  private async _collectListing(
    limit?: number
  ): Promise<{ files: { path: string; description: string }[]; total: number }> {
    const keys = (await this._storage.list('')).filter(isKnowledgeKey).sort((a, b) => a.localeCompare(b))
    const truncated = limit !== undefined && keys.length > limit
    const selected = truncated ? keys.slice(0, limit) : keys

    if (truncated && !this._listingTruncationWarned) {
      this._listingTruncationWarned = true
      logger.warn(
        `store=<${this.name}>, total=<${keys.length}>, shown=<${limit}> | memory store exceeds the injected-listing cap, injecting a truncated listing`
      )
    }

    const infos = await mapWithConcurrency(selected, STORAGE_READ_CONCURRENCY, async (key) => {
      try {
        const bytes = await this._storage.read(key)
        if (!bytes) return null
        const { description } = parseFrontmatter(decoder.decode(bytes))
        return { path: key, description }
      } catch {
        return null
      }
    })

    return { files: infos.filter((info): info is NonNullable<typeof info> => info !== null), total: keys.length }
  }

  /**
   * Read one knowledge file's body by path — the on-demand half of progressive disclosure. Tries the
   * canonical (lowercased) key {@link add} writes under, then on a miss the path as {@link listFiles}
   * advertised it, so a file seeded outside the store's API stays readable. Strips the frontmatter.
   *
   * @param path - Path of the file to read, as rendered in the injected listing
   * @returns The file's body, without frontmatter
   * @throws Error when the path is invalid or no file exists at it
   */
  async readFile(path: string): Promise<string> {
    const key = canonicalizeKnowledgePath(path)
    const rawKey = normalizeKey(path)

    // Canonical key first; on a miss, retry the raw listed key so a file seeded outside the store's API
    // (never lowercased) stays readable. The second read only fires when the two differ.
    const bytes = (await this._storage.read(key)) ?? (rawKey === key ? null : await this._storage.read(rawKey))
    if (!bytes) {
      throw new Error(`No memory file at '${key}'. Paths must match the memory file listing exactly.`)
    }

    return parseFrontmatter(decoder.decode(bytes)).body.trim()
  }

  /**
   * Returns the read tool — the on-demand half of progressive disclosure. Read-only, so it is
   * available on a `writable: false` store.
   *
   * @returns The store's read tool, or nothing when `progressiveDisclosure` is off
   */
  getTools(): Tool[] {
    return this._progressiveDisclosure ? [createReadTool(this.name, (path) => this.readFile(path))] : []
  }

  /**
   * Returns the file-listing injector for {@link MemoryManager} to register. Independent of the
   * manager's own `injection` setting.
   *
   * @returns The listing injector, or nothing when `progressiveDisclosure` is off
   */
  getPlugins(): Plugin[] {
    return this._progressiveDisclosure
      ? [createProgressiveDisclosureInjector(this.name, () => this._collectListing(this._maxListedFiles))]
      : []
  }

  /**
   * Add a knowledge entry to the store.
   *
   * Writes a markdown file with YAML frontmatter. By default writes to `facts/` within the store's
   * namespace. Pass `metadata.path` to write to a custom location.
   *
   * @param content - The knowledge content to store
   * @param metadata - Optional metadata: `title`, `description`, and `path` (custom target path)
   * @returns The canonical storage-relative key the entry was written under
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
      let suffix = 1
      while (await this._storage.read(key)) {
        key = `${FACTS_PREFIX}${slug}-${suffix}.md`
        suffix++
      }
    }

    const canonicalKey = canonicalizeKnowledgePath(key)

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

    const files = await readAllFiles(this._storage, maxFiles)
    if (files.size === 0) return

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
