/**
 * File-based memory store implementing the {@link MemoryStore} interface.
 *
 * Organizes knowledge as a structured file hierarchy under a `memory/` storage namespace. Provides
 * keyword-based search via `search_memory` (registered by {@link MemoryManager}).
 */

import type { JSONValue } from '../../types/json.js'
import type { MemoryEntry, MemoryStore, SearchOptions } from '../../memory/types.js'
import type { ExtractionConfig } from '../../memory/extraction/types.js'
import type { Storage } from '../../storage/storage.js'
import type { ConsolidateConfig, ConsolidateOperation, FileMemoryStoreConfig } from './types.js'
import { CONSOLIDATE_OPERATIONS } from './types.js'
import { z } from 'zod'
import { Agent } from '../../agent/agent.js'
import { logger } from '../../logging/logger.js'
import { LocalFileStorage } from '../../storage/local-file-storage.js'
import { NAMESPACED, namespace, normalizeKey } from '../../storage/storage.js'
import { DEFAULT_MAX_SEARCH_RESULTS, tokenize, tokenOverlapScore } from '../../memory/search/keyword.js'

/**
 * Top-level storage namespace shared by every file memory store, isolating them as a group from
 * other subsystems (sessions, context offloading) that may share the same backend. See the storage
 * design doc's key-prefix convention (`team/designs/0014-storage.md`). Each store further scopes
 * under its own `name` within this namespace — see {@link FileMemoryStore._resolveStorage}.
 */
const STORAGE_NAMESPACE = 'memory'

/** Default subdirectory (within the store's namespace) for entries added without an explicit path. */
const FACTS_PREFIX = 'facts/'

/**
 * Subdirectory (within the store's namespace) reserved for the consolidation audit log.
 * {@link FileMemoryStore.consolidate} excludes this prefix from its working set so it never ingests
 * and rewrites its own changelog as though it were a knowledge file.
 */
const CONSOLIDATION_PREFIX = 'consolidation/'

/** Path of the consolidation changelog, within {@link CONSOLIDATION_PREFIX}. */
const CONSOLIDATION_CHANGELOG = `${CONSOLIDATION_PREFIX}changelog.md`

/**
 * Cap on concurrent storage reads during search. The Storage contract makes no guarantee
 * about concurrent-read capacity, so an unbounded fan-out (one read per key) can exhaust a
 * backend's connection pool or trip throttling on a large corpus. Reads still run in parallel,
 * just no more than this many at once.
 */
const SEARCH_READ_CONCURRENCY = 8

/** Default cap for total generated content bytes across all write actions in a plan. */
const DEFAULT_MAX_GENERATED_BYTES = 262_144

/**
 * Frontmatter opening delimiter. Matches the convention used by {@link FileMemoryStore.add}:
 * files start with `---\n`, followed by YAML fields, then a closing `---\n`.
 */
const FRONTMATTER_OPEN = '---\n'

/** Frontmatter closing delimiter, including the newline that must precede it. */
const FRONTMATTER_CLOSE = '\n---\n'

const encoder = new TextEncoder()
const decoder = new TextDecoder()

const ConsolidationPlanSchema = z.object({
  actions: z.array(
    z.discriminatedUnion('action', [
      z.object({
        action: z.literal('merge'),
        sources: z.array(z.string()),
        target: z.string(),
        content: z.string(),
        reason: z.string(),
      }),
      z.object({
        action: z.literal('update'),
        path: z.string(),
        content: z.string(),
        reason: z.string(),
      }),
      z.object({
        action: z.literal('delete'),
        path: z.string(),
        reason: z.string(),
      }),
      z.object({
        action: z.literal('move'),
        from: z.string(),
        to: z.string(),
        reason: z.string(),
      }),
    ])
  ),
  summary: z.string(),
})

type ConsolidationPlan = z.infer<typeof ConsolidationPlanSchema>
type ConsolidationAction = ConsolidationPlan['actions'][number]

/** Extract description from YAML frontmatter and return the remaining body. */
function parseFrontmatter(content: string): { description: string; body: string } {
  const match = content.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/)
  if (!match) return { description: '', body: content }

  const frontmatter = match[1] ?? ''
  const body = match[2] ?? ''

  const descMatch = frontmatter.match(/^description:\s*(".*")\s*$/m)
  if (!descMatch) return { description: '', body }

  let description: string
  try {
    description = JSON.parse(descMatch[1]!) as string
  } catch {
    description = descMatch[1]!.slice(1, -1)
  }
  return { description, body }
}

/** Extract the filename stem (without `.md` extension) from a storage key. */
function basename(key: string): string {
  const filename = key.split('/').pop() ?? key
  return filename.replace(/\.md$/, '')
}

/**
 * Map `items` through `fn` running at most `limit` calls concurrently, preserving input order.
 * A worker pool pulls from a shared cursor so a slow item never blocks others in its batch.
 */
async function mapWithConcurrency<T, R>(items: T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const results = new Array<R>(items.length)
  let cursor = 0
  const worker = async (): Promise<void> => {
    while (cursor < items.length) {
      const index = cursor++
      results[index] = await fn(items[index]!)
    }
  }
  const workers = Array.from({ length: Math.min(limit, items.length) }, () => worker())
  await Promise.all(workers)
  return results
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
 * Case-normalized path identity comparison. Returns true when two paths would resolve to the same
 * file on a case-insensitive filesystem. This is a conservative approximation — backend-resolved
 * identity (probing the storage layer for true equivalence) is future work.
 */
function pathsResolveSame(a: string, b: string): boolean {
  return a.toLowerCase() === b.toLowerCase()
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
      await mapWithConcurrency(allKeys, SEARCH_READ_CONCURRENCY, async (key) => {
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
      // Best-effort for a single-writer local store (TOCTOU is acceptable).
      let suffix = 1
      while (await this._storage.read(key)) {
        key = `${FACTS_PREFIX}${slug}-${suffix}.md`
        suffix++
      }
    }

    // Canonicalize with the same helper the shipped backends apply internally, so the
    // returned receipt matches the key search() and the backend's list() report.
    const canonicalKey = normalizeKey(key)
    const fileContent = `---\ndescription: ${JSON.stringify(description)}\n---\n\n${content}\n`
    await this._storage.write(canonicalKey, encoder.encode(fileContent))
    return canonicalKey
  }

  /**
   * Run offline consolidation to maintain knowledge quality.
   *
   * Uses a plan-then-execute strategy: one structured-output call produces an action plan
   * over all files, programmatic guardrails validate the plan atomically, then deterministic
   * code executes it (writes before deletes). On validation failure, one revise-retry is
   * attempted; if the revised plan also fails validation, the method throws.
   *
   * Only one consolidation may run at a time per store instance. Each run snapshots the store at
   * the start ({@link _readAllFiles}) and later mutates it, so overlapping runs would plan against
   * stale snapshots and race on the same keys. A call that starts while another is still in flight
   * throws immediately rather than corrupting the store.
   *
   * @param config - Model and operation configuration
   * @throws Error when a consolidation is already running on this store instance
   * @throws Error when the knowledge store exceeds the file count limit (maxFiles)
   * @throws Error when the knowledge store exceeds the input byte size limit (maxInputBytes)
   * @throws Error when structured output is undefined (model did not return a plan)
   * @throws Error when the consolidation plan exceeds the action limit (maxActionsPerPlan)
   * @throws Error when the plan's generated content exceeds the byte limit (maxGeneratedBytes)
   * @throws Error when the consolidation plan fails validation after retry
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
   * Execute a single consolidation run. Assumes the concurrency guard in {@link consolidate} holds,
   * so it never overlaps another run on the same instance.
   *
   * @param config - Model and operation configuration
   */
  private async _consolidate(config: ConsolidateConfig): Promise<void> {
    const operations = config.operations ?? [...CONSOLIDATE_OPERATIONS]
    const maxDirectories = config.maxDirectories ?? 8

    const maxFiles = config.maxFiles ?? 100
    const maxInputBytes = config.maxInputBytes ?? 128 * 1024
    const maxActionsPerPlan = config.maxActionsPerPlan ?? 1000
    const maxGeneratedBytes = config.maxGeneratedBytes ?? DEFAULT_MAX_GENERATED_BYTES

    const files = await this._readAllFiles()
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

    const plan = await this._generatePlan(config, operations, files, maxDirectories, maxActionsPerPlan)

    // Like the action-count guard, an oversized plan is a runaway signal rather than a fixable
    // mistake, so this throws instead of routing into the revise-retry
    const generatedBytes = generatedByteSize(plan)
    if (generatedBytes > maxGeneratedBytes) {
      throw new Error(
        `Consolidation plan exceeds generated content limit: ${generatedBytes} bytes (maxGeneratedBytes: ${maxGeneratedBytes})`
      )
    }

    const deleteErrors = await this._executePlan(plan, files)
    // Record the changelog even on partial failure — writes and some deletes already hit disk,
    // so an accurate audit trail must capture the run before surfacing the error
    await this._recordChangelog(operations, plan, deleteErrors)

    if (deleteErrors.length > 0) {
      const paths = deleteErrors.map((deleteError) => deleteError.path).join(', ')
      throw new Error(
        `Plan executed but ${deleteErrors.length} delete(s) failed: ${paths}. ` +
          `Writes succeeded — duplicate content may remain until next consolidation.`
      )
    }
  }

  /**
   * Read every knowledge file into memory as a `path → content` map.
   *
   * This snapshot is the working set for one consolidation run: it is handed to the planner,
   * the validator, and the executor so they all reason over the same view of the store.
   */
  private async _readAllFiles(): Promise<Map<string, string>> {
    const files = new Map<string, string>()
    const allKeys = await this._storage.list('')
    for (const key of allKeys) {
      // Exclude the changelog dir so consolidation never ingests and rewrites its own audit log
      if (key.startsWith(CONSOLIDATION_PREFIX)) continue
      const bytes = await this._storage.read(key)
      if (bytes) files.set(key, decoder.decode(bytes))
    }
    return files
  }

  /**
   * Produce a validated action plan from the model via a single structured-output call.
   *
   * The plan is validated against the guardrails before being returned; if validation fails,
   * one revise-retry is attempted. A returned plan is always guaranteed to have passed validation.
   *
   * @throws Error when the model returns no structured output, the plan exceeds the action limit,
   *   or the plan fails validation after retry
   */
  private async _generatePlan(
    config: ConsolidateConfig,
    operations: ConsolidateOperation[],
    files: Map<string, string>,
    maxDirectories: number,
    maxActionsPerPlan: number
  ): Promise<ConsolidationPlan> {
    const systemPrompt = buildPlannerSystemPrompt(operations)
    const userMessage = buildPlannerUserMessage(files)

    const agent = new Agent({
      model: config.model,
      systemPrompt,
      printer: false,
      structuredOutputSchema: ConsolidationPlanSchema,
    })

    const result = await agent.invoke(userMessage)
    let plan = extractPlan(result, maxActionsPerPlan)

    const validationError = validatePlan(plan, files, operations, maxDirectories)
    if (validationError) {
      logger.warn(
        `validation_errors=<${validationError}>, plan=<${JSON.stringify(plan)}> | consolidation plan rejected on initial attempt`
      )
      plan = await this._revisePlan(agent, plan, validationError, files, operations, maxDirectories, maxActionsPerPlan)
    }

    return plan
  }

  /**
   * Ask the model to fix a rejected plan, feeding back the validation error and the prior plan.
   *
   * Only one retry is attempted: if the revised plan also fails validation, this throws rather
   * than looping, so consolidation never runs an unvalidated plan.
   *
   * @throws Error when the revised plan exceeds the action limit, or also fails validation
   */
  private async _revisePlan(
    agent: Agent,
    originalPlan: ConsolidationPlan,
    validationError: string,
    files: Map<string, string>,
    operations: ConsolidateOperation[],
    maxDirectories: number,
    maxActionsPerPlan: number
  ): Promise<ConsolidationPlan> {
    const reviseResult = await agent.invoke(
      `Your plan was rejected: ${validationError}. Here is the plan you produced:\n\n${JSON.stringify(originalPlan)}\n\nModify ONLY the offending actions to fix the violations above. Keep all other actions unchanged.\n\nRevise your plan to fix this issue.`
    )
    const revisedPlan = extractPlan(reviseResult, maxActionsPerPlan)

    const revisedValidationError = validatePlan(revisedPlan, files, operations, maxDirectories)
    if (revisedValidationError) {
      logger.warn(
        `validation_errors=<${revisedValidationError}>, plan=<${JSON.stringify(revisedPlan)}> | consolidation plan rejected after retry`
      )
      throw new Error(`Consolidation plan validation failed after retry: ${revisedValidationError}`)
    }

    return revisedPlan
  }

  /**
   * Apply a validated plan to storage deterministically.
   *
   * All writes run before any deletes so merged/moved content lands before its sources are
   * removed — a crash between the two passes leaves duplicated content, never lost content.
   *
   * A failed write throws immediately, before any delete runs and before the changelog is recorded.
   * That is intentional: writes-before-deletes means an aborted write pass has removed nothing, so
   * the store is unchanged and the run can simply be retried — there is no partial state to audit.
   *
   * Deletes use best-effort semantics: every delete is attempted even if earlier ones fail.
   * A missing key is a no-op (per the {@link Storage} contract), so a delete only fails on a
   * genuine backend error — permissions, a read-only or broken disk, or a remote backend
   * (S3, DynamoDB) throttling or refusing the call. The failures are returned rather than thrown
   * so the caller can still record the changelog for the partial run before surfacing the error.
   *
   * @returns The paths whose deletes failed, each with the underlying error (empty when all succeed)
   */
  private async _executePlan(
    plan: ConsolidationPlan,
    files: Map<string, string>
  ): Promise<Array<{ path: string; error: unknown }>> {
    // Writes before deletes — merged content lands before sources are removed
    for (const action of plan.actions) {
      if (action.action === 'merge') {
        await this._storage.write(action.target, encoder.encode(action.content))
      } else if (action.action === 'update') {
        await this._storage.write(action.path, encoder.encode(action.content))
      } else if (action.action === 'move') {
        // validatePlan guarantees every move source exists in `files`; a miss here means
        // validation and execution have diverged, so fail loud rather than write empty content
        const content = files.get(action.from)
        if (content === undefined) {
          throw new Error(
            `Invariant violated: move source '${action.from}' missing from working set — plan not validated`
          )
        }
        await this._storage.write(action.to, encoder.encode(content))
      }
    }

    // Best-effort deletes — attempt all, then report failures
    const deleteErrors: Array<{ path: string; error: unknown }> = []
    for (const action of plan.actions) {
      if (action.action === 'delete') {
        try {
          await this._storage.delete(action.path)
        } catch (error) {
          deleteErrors.push({ path: action.path, error })
        }
      } else if (action.action === 'merge') {
        for (const source of action.sources) {
          // Skip the target when it is one of its own sources — the merge folded into an existing
          // file, so deleting it here would remove the content just written
          if (!pathsResolveSame(source, action.target)) {
            try {
              await this._storage.delete(source)
            } catch (error) {
              deleteErrors.push({ path: source, error })
            }
          }
        }
      } else if (action.action === 'move') {
        // Skip delete when source and target resolve to the same identity (case-only rename) —
        // deleting would remove the content the write pass just produced
        if (!pathsResolveSame(action.from, action.to)) {
          try {
            await this._storage.delete(action.from)
          } catch (error) {
            deleteErrors.push({ path: action.from, error })
          }
        }
      }
    }

    return deleteErrors
  }

  /**
   * Append a human-readable summary of an applied plan to the consolidation changelog.
   *
   * Provides an audit trail of what each run changed and why, one dated entry per consolidation.
   * When deletes failed, they are recorded too so the log reflects the partial run rather than
   * implying every action succeeded.
   */
  private async _recordChangelog(
    operations: ConsolidateOperation[],
    plan: ConsolidationPlan,
    deleteErrors: Array<{ path: string; error: unknown }>
  ): Promise<void> {
    const timestamp = new Date().toISOString().slice(0, 16).replace('T', ' ')
    const actionSummaries = plan.actions.map((action) => {
      switch (action.action) {
        case 'merge':
          return `  - merge: ${action.sources.join(' + ')} → ${action.target} (${action.reason})`
        case 'update':
          return `  - update: ${action.path} (${action.reason})`
        case 'delete':
          return `  - delete: ${action.path} (${action.reason})`
        case 'move':
          return `  - move: ${action.from} → ${action.to} (${action.reason})`
      }
    })

    const parts = [
      `\n## ${timestamp}`,
      ``,
      `Operations: ${operations.join(', ')}`,
      `Actions (${plan.actions.length}):`,
      ...actionSummaries,
    ]

    // summary is a required string; the guard only suppresses an empty-string value
    if (plan.summary) {
      parts.push(``, `Summary: ${plan.summary}`)
    }

    if (deleteErrors.length > 0) {
      parts.push(``, `Failed deletes (${deleteErrors.length}) — sources may remain until next consolidation:`)
      for (const deleteError of deleteErrors) {
        parts.push(`  - ${deleteError.path}: ${String(deleteError.error)}`)
      }
    }
    parts.push('')

    const entry = parts.join('\n')
    // The changelog is an audit artifact written after the plan's mutations already landed. A
    // failure to record it must not throw: doing so would mask the real run outcome (a partial
    // delete failure the caller needs to see, or a fully successful run reported as failed).
    try {
      const existing = await this._storage.read(CONSOLIDATION_CHANGELOG)
      const content = existing ? decoder.decode(existing) + entry : `# Consolidation Changelog\n${entry}`
      await this._storage.write(CONSOLIDATION_CHANGELOG, encoder.encode(content))
    } catch (error) {
      logger.warn(`error=<${error}> | failed to record consolidation changelog, audit log not updated`)
    }
  }
}

/**
 * Extract and validate the plan from a raw agent result.
 *
 * Runs the untrusted model output through the schema so everything downstream can rely on the
 * plan's shape being correct, then bounds the action count. The count guard throws rather than
 * routing into the revise-retry: an oversized plan is an abuse/runaway signal, not a fixable
 * mistake, and feeding it back to the model would re-incur the same unbounded cost.
 *
 * @throws Error when the result carries no structured output
 * @throws ZodError when the structured output does not match {@link ConsolidationPlanSchema}
 * @throws Error when the plan's action count exceeds `maxActionsPerPlan`
 */
function extractPlan(result: { structuredOutput?: unknown }, maxActionsPerPlan: number): ConsolidationPlan {
  if (!result.structuredOutput) {
    throw new Error('Model did not return structured output — cannot produce a consolidation plan')
  }
  const plan = ConsolidationPlanSchema.parse(result.structuredOutput)
  if (plan.actions.length > maxActionsPerPlan) {
    throw new Error(
      `Consolidation plan exceeds action limit: ${plan.actions.length} actions (maxActionsPerPlan: ${maxActionsPerPlan})`
    )
  }
  return plan
}

/**
 * Total UTF-8 bytes of model-generated content across a plan's write actions.
 *
 * Bounds planner output volume independently of the action count: a plan within the action limit
 * can still carry a few very large writes.
 */
function generatedByteSize(plan: ConsolidationPlan): number {
  let bytes = 0
  for (const action of plan.actions) {
    if (action.action === 'merge' || action.action === 'update') {
      bytes += encoder.encode(action.content).byteLength
    }
  }
  return bytes
}

/**
 * Validate a plan against the guardrails before any storage mutation.
 *
 * Guards four properties: every action is permitted by the requested operations, every path is
 * well-formed and references files that exist, every write carries well-formed content, and no two
 * actions collide on a write target. The model is untrusted, so this is the gate that keeps a bad
 * plan from corrupting the store.
 *
 * All violations are accumulated so the revision prompt can present a complete repair spec in one
 * shot rather than requiring iterative single-error fixes.
 *
 * @returns A newline-joined string of all violations when the plan is invalid, or `undefined` when it passes
 */
function validatePlan(
  plan: ConsolidationPlan,
  files: Map<string, string>,
  operations: ConsolidateOperation[],
  maxDirectories: number
): string | undefined {
  const violations: string[] = []

  const allowedActions = new Set<string>()
  if (operations.includes('deduplicate') || operations.includes('deriveInsights')) allowedActions.add('merge')
  if (operations.includes('resolveContradictions')) {
    allowedActions.add('update')
    allowedActions.add('delete')
  }
  if (operations.includes('prune')) allowedActions.add('delete')
  if (operations.includes('reorganize')) allowedActions.add('move')
  // 'update' is only allowed when resolveContradictions or deriveInsights is active —
  // deriveInsights may rewrite merged content as an update to an existing file
  if (operations.includes('deriveInsights')) allowedActions.add('update')

  // Seed with the directories already on disk, then let validateActionPaths add any new directory a
  // write introduces — so the maxDirectories budget is enforced against the plan's cumulative effect,
  // not each action in isolation.
  const plannedDirs = new Set<string>()
  for (const key of files.keys()) {
    const idx = key.indexOf('/')
    if (idx !== -1) plannedDirs.add(key.slice(0, idx))
  }

  for (const action of plan.actions) {
    if (!allowedActions.has(action.action)) {
      violations.push(`Action '${action.action}' is not allowed for operations: ${operations.join(', ')}`)
    }

    // Kept here rather than as a schema `.min(2)` so a too-short merge flows through the same
    // accumulate-and-revise path as every other guardrail, instead of failing as a raw ZodError
    // at parse time before the model gets a chance to fix it.
    if (action.action === 'merge' && action.sources.length < 2) {
      violations.push('Merge action requires at least 2 sources')
    }

    const pathErrors = validateActionPaths(action, files, plannedDirs, maxDirectories)
    violations.push(...pathErrors)

    const contentError = validateActionContent(action)
    if (contentError) violations.push(contentError)
  }

  // Reject plans where multiple actions write to the same target path
  const collisionErrors = validateNoTargetCollisions(plan, files)
  violations.push(...collisionErrors)

  return violations.length > 0 ? violations.join('\n') : undefined
}

/**
 * Validate the paths of a single action: sources/targets must exist where read, and new write
 * targets must be well-formed paths within the directory budget.
 *
 * @returns An array of human-readable error strings for every invalid path (empty when all pass)
 */
function validateActionPaths(
  action: ConsolidationAction,
  files: Map<string, string>,
  existingDirs: Set<string>,
  maxDirectories: number
): string[] {
  const errors: string[] = []

  switch (action.action) {
    case 'merge': {
      for (const source of action.sources) {
        if (!files.has(source)) errors.push(`Merge source '${source}' does not exist`)
      }
      const pathError = validatePath(action.target, existingDirs, maxDirectories)
      if (pathError) errors.push(pathError)
      return errors
    }
    case 'update': {
      if (!files.has(action.path)) errors.push(`Update target '${action.path}' does not exist`)
      return errors
    }
    case 'delete': {
      if (!files.has(action.path)) errors.push(`Delete target '${action.path}' does not exist`)
      return errors
    }
    case 'move': {
      if (!files.has(action.from)) errors.push(`Move source '${action.from}' does not exist`)
      const pathError = validatePath(action.to, existingDirs, maxDirectories)
      if (pathError) errors.push(pathError)
      return errors
    }
  }
}

/**
 * Validate the content a write action (merge or update) would put on disk.
 *
 * A schema-valid plan can still carry content that erases knowledge: the schema accepts any string,
 * so an empty or structurally broken value would be written verbatim — and for a merge, its sources
 * deleted afterwards. This requires the frontmatter shape {@link FileMemoryStore.add} produces
 * (`---\n`, YAML, `\n---\n`) plus a non-empty body, so a written file stays parseable by
 * {@link parseFrontmatter} and searchable.
 *
 * Non-write actions carry no content and are skipped, so delete, move, and prune stay unaffected.
 *
 * @returns A human-readable error string when the content is invalid, or `undefined` when it passes
 */
function validateActionContent(action: ConsolidationAction): string | undefined {
  if (action.action !== 'merge' && action.action !== 'update') return undefined

  const label = action.action === 'merge' ? `Merge target '${action.target}'` : `Update target '${action.path}'`

  if (action.content.trim().length === 0) {
    return `${label} has empty content — a write must not blank out a file`
  }
  if (!action.content.startsWith(FRONTMATTER_OPEN)) {
    return `${label} must start with YAML frontmatter ('---' on the first line)`
  }
  const closingIndex = action.content.indexOf(FRONTMATTER_CLOSE, FRONTMATTER_OPEN.length - 1)
  if (closingIndex === -1) {
    return `${label} is missing the closing frontmatter delimiter ('---' on its own line)`
  }
  if (action.content.slice(closingIndex + FRONTMATTER_CLOSE.length).trim().length === 0) {
    return `${label} has no body after its frontmatter`
  }

  return undefined
}

/**
 * Reject plans that would clobber data: two actions writing the same path, or a write landing on
 * an existing file that no other action vacates (a self-overwrite like an update is allowed).
 *
 * @returns An array of human-readable error strings for every collision (empty when the plan is safe)
 */
function validateNoTargetCollisions(plan: ConsolidationPlan, files: Map<string, string>): string[] {
  const errors: string[] = []

  // Collect all paths that are written to by the plan
  const writeTargets: string[] = []
  // Collect all paths vacated (deleted/moved away) by the plan
  const vacatedPaths = new Set<string>()

  for (const action of plan.actions) {
    if (action.action === 'merge') {
      writeTargets.push(action.target)
      // Non-target merge sources are deleted during execution (case-normalized —
      // a source that resolves to the same identity as the target is not vacated)
      for (const source of action.sources) {
        if (!pathsResolveSame(source, action.target)) {
          vacatedPaths.add(source)
        }
      }
    } else if (action.action === 'update') {
      writeTargets.push(action.path)
    } else if (action.action === 'move') {
      writeTargets.push(action.to)
      // A case-only rename (different string but same normalized identity) does not delete the
      // source in execution — skip vacating to keep validation consistent with the executor.
      // An exact-string identity move (from === to) still vacates, triggering the write-vs-vacate
      // guard that rejects it as a no-op that would destroy content.
      if (action.from === action.to || !pathsResolveSame(action.from, action.to)) {
        vacatedPaths.add(action.from)
      }
    } else if (action.action === 'delete') {
      vacatedPaths.add(action.path)
    }
  }

  // Check for two actions writing the same path (case-insensitive — divergent casing resolves
  // to the same file on case-insensitive backends)
  const seen = new Set<string>()
  for (const target of writeTargets) {
    const normalized = target.toLowerCase()
    if (seen.has(normalized)) {
      errors.push(`Multiple actions write to the same path '${target}'`)
    }
    seen.add(normalized)
  }

  // A path both written and vacated in one plan is destroyed: writes run before deletes, so the
  // delete pass removes the content the write pass just produced. This single rule covers write +
  // delete on one path, an identity move (from === to), chained moves (one move's target is another
  // move's source), an update whose target is later moved away, and a move onto a since-deleted file.
  // Case-normalized so case-only aliases are caught on case-insensitive backends.
  const writeSetNormalized = new Set(writeTargets.map((t) => t.toLowerCase()))
  for (const path of vacatedPaths) {
    if (writeSetNormalized.has(path.toLowerCase())) {
      errors.push(
        `Path '${path}' is both written to and removed by the same plan (one action writes it, another deletes or moves it away), which would destroy its content`
      )
    }
  }

  // Check for a write landing on a pre-existing file the plan does not vacate (vacated-and-written
  // targets are already rejected above). A self-overwrite — an update, or a merge into one of its
  // own sources — is the one legitimate case. Case-normalized to catch aliases on case-insensitive
  // backends.
  const vacatedNormalized = new Set([...vacatedPaths].map((p) => p.toLowerCase()))
  for (const target of writeTargets) {
    const existsInFiles = [...files.keys()].some((key) => pathsResolveSame(key, target))
    const isVacated = vacatedNormalized.has(target.toLowerCase())
    if (existsInFiles && !isVacated && !planOverwritesSelf(plan, target)) {
      errors.push(`Target path '${target}' already exists and is not vacated by another action in the plan`)
    }
  }

  return errors
}

/**
 * Returns true when the action that writes to `target` also reads from it, making the overwrite
 * intentional: an update to an existing file, or a merge whose target is one of its own sources.
 *
 * A merge onto an existing file that is NOT one of its sources is deliberately excluded — the model
 * was never shown that file's content, so its `content` cannot have folded it in, and overwriting it
 * would silently destroy data. Merging into a brand-new path is unaffected (the caller's existence
 * check short-circuits before reaching here).
 */
function planOverwritesSelf(plan: ConsolidationPlan, target: string): boolean {
  for (const action of plan.actions) {
    if (
      action.action === 'merge' &&
      pathsResolveSame(action.target, target) &&
      action.sources.some((s) => pathsResolveSame(s, target))
    ) {
      return true
    }
    if (action.action === 'update' && pathsResolveSame(action.path, target)) {
      return true
    }
    // A case-only move (from and to resolve to same identity) is an in-place rename — writing
    // the target is overwriting the source's own file, which is intentional
    if (action.action === 'move' && pathsResolveSame(action.to, target) && pathsResolveSame(action.from, target)) {
      return true
    }
  }
  return false
}

/**
 * Validate a single write-target path against the store's layout rules: a namespace-relative `.md`
 * file, at most one level of nesting, outside the reserved `consolidation/` dir, with a well-formed
 * directory name that stays within the budget.
 *
 * @returns A human-readable error string when the path is invalid, or `undefined` when it passes
 */
function validatePath(path: string, existingDirs: Set<string>, maxDirectories: number): string | undefined {
  if (path.includes('\\')) {
    return `Path must not contain backslashes: ${path}`
  }

  const rawSegments = path.split('/')
  if (rawSegments.some((seg) => seg === '.' || seg === '..')) {
    return `Path must not contain dot segments ('.' or '..'): ${path}`
  }

  if (path.startsWith(CONSOLIDATION_PREFIX)) {
    return `Path must not be under the reserved '${CONSOLIDATION_PREFIX}' directory: ${path}`
  }
  if (!path.endsWith('.md')) {
    return `Path must end with .md: ${path}`
  }

  const segments = path.split('/')

  if (segments.length > 2) {
    return `Only one level of nesting allowed: ${path}`
  }

  if (segments.length === 2) {
    const dirName = segments[0]!
    if (!/^[a-z0-9-]{1,30}$/.test(dirName)) {
      return `Directory name must be lowercase alphanumeric + hyphens, ≤30 chars: '${dirName}'`
    }
    if (!existingDirs.has(dirName) && existingDirs.size >= maxDirectories) {
      return `Cannot create directory '${dirName}': maximum of ${maxDirectories} directories reached`
    }
    // Track the new directory so subsequent actions in the same plan see it
    existingDirs.add(dirName)
  }

  return undefined
}

/**
 * Build the planner's system prompt, including only the directives for the requested operations.
 *
 * Scoping the prompt to the active operations keeps the model from proposing actions the validator
 * would reject, and pairs with the allowed-action check in {@link validatePlan}.
 */
function buildPlannerSystemPrompt(operations: ConsolidateOperation[]): string {
  const directives: string[] = [
    'You are a knowledge maintenance agent. Your job is to improve the quality of stored knowledge files.',
    'Each file is markdown with YAML frontmatter containing a `description` field.',
    '',
    'Apply the following operations to the knowledge files below:',
  ]

  for (const op of operations) {
    switch (op) {
      case 'deduplicate':
        directives.push(
          '- DEDUPLICATE: Merge files that express the same fact. Keep the most complete version and delete the redundant one(s). Use the `merge` action with all source paths and the merged content.'
        )
        break
      case 'resolveContradictions':
        directives.push(
          '- RESOLVE CONTRADICTIONS: When files contain conflicting information, keep the more recent or more specific fact and delete the outdated one. Use `update` to rewrite the kept file or `delete` to remove the outdated one.'
        )
        break
      case 'deriveInsights':
        directives.push(
          '- DERIVE INSIGHTS: When multiple files together reveal a higher-level pattern, synthesize them into a new file that captures the insight. Keep or remove originals as appropriate. Use the `merge` action. Example: files noting "prefers dark theme", "uses a high-contrast editor", and "increased default font size" together support a new file "prefers high-visibility UI settings".'
        )
        break
      case 'prune':
        directives.push(
          '- PRUNE: Delete files whose content is fully covered by another file or that are no longer relevant. Use the `delete` action. Example: a note "investigating flaky test X" is stale once another file records "flaky test X fixed"; a one-off "temporarily using staging endpoint" is no longer relevant.'
        )
        break
      case 'reorganize':
        directives.push('- REORGANIZE: Move files that belong in a different subdirectory. Use the `move` action.')
        break
    }
  }

  directives.push(
    '',
    'Instructions:',
    '1. Read each knowledge file.',
    '2. Reason about which operations apply.',
    '3. Produce a plan with the appropriate actions.',
    '4. Every `content` you write must be a complete markdown file: a `---` line, YAML fields including a `description`, a closing `---` line, then a non-empty body. Never emit empty or frontmatter-only content — it would erase the file.',
    '5. All paths must end with `.md` and must not be under the reserved `consolidation/` directory.',
    '6. Only one level of subdirectory nesting is allowed.',
    '7. Each action fully transforms one path. Never write to and delete the same path in one plan, and never move a file onto its own path. To rewrite a file in place use `update`; to relocate it use `move` to a different path.',
    '8. Only make changes that clearly improve quality. When in doubt, leave files as-is.',
    '9. For each action, provide a concise reason explaining WHY.'
  )

  return directives.join('\n')
}

/**
 * Render the full working set into the planner's user message.
 *
 * Each file becomes a labeled, fenced block so the model sees its path alongside verbatim content
 * and can reference exact paths in the actions it produces.
 */
function buildPlannerUserMessage(files: Map<string, string>): string {
  const fileEntries: string[] = []
  let totalBytes = 0
  for (const [path, content] of files) {
    totalBytes += encoder.encode(content).byteLength
    fileEntries.push(`### ${path}\n\`\`\`\n${content}\`\`\``)
  }
  const totalKiB = (totalBytes / 1024).toFixed(1)
  return `Review the following ${files.size} knowledge files (${totalKiB} KiB total) and produce a maintenance plan:\n\n${fileEntries.join('\n\n')}`
}
