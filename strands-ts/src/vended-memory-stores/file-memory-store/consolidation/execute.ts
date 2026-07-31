/**
 * The execution half of consolidation: apply a validated plan to storage, and record what it did.
 *
 * Everything here is deterministic — no model is consulted. The plan arrives already validated, so
 * these functions are concerned only with the order operations hit disk and what survives a partial
 * failure. Storage is passed in rather than reached for, so the store's namespacing applies without
 * this module knowing about it.
 *
 * @internal
 */

import type { Storage } from '../../../storage/storage.js'
import type { ConsolidateOperation } from '../types.js'
import type { ConsolidationPlan } from './plan.js'
import { clipWithCount } from './plan.js'
import { logger } from '../../../logging/logger.js'
import {
  CONSOLIDATION_CHANGELOG,
  decoder,
  encoder,
  isConsolidationChangelog,
  mapWithConcurrency,
  pathsResolveSame,
  resolveCanonicalKey,
  resolveWriteTarget,
  STORAGE_READ_CONCURRENCY,
} from '../internal.js'

/**
 * Cap on a single interpolated changelog field — a path, reason, summary, or backend error message.
 * Generous for a real one-line audit entry, tight enough that no field can dominate the log.
 */
const MAX_CHANGELOG_FIELD_LENGTH = 500

/**
 * A delete that failed during execution, paired with the error the backend raised.
 *
 * @internal
 */
export interface DeleteFailure {
  /** The path whose delete failed. */
  path: string
  /** The error the storage backend raised. */
  error: unknown
}

/**
 * Read every knowledge file into memory as a `path → content` map.
 *
 * This snapshot is the working set for one consolidation run: it is handed to the planner,
 * the validator, and the executor so they all reason over the same view of the store.
 *
 * @internal
 */
export async function readAllFiles(storage: Storage): Promise<Map<string, string>> {
  const files = new Map<string, string>()
  const allKeys = await storage.list('')
  const keysToRead = allKeys.filter((key) => !isConsolidationChangelog(key))

  const entries = await mapWithConcurrency(keysToRead, STORAGE_READ_CONCURRENCY, async (key) => {
    const bytes = await storage.read(key)
    return bytes ? ([key, decoder.decode(bytes)] as const) : null
  })

  for (const entry of entries) {
    if (entry) files.set(entry[0], entry[1])
  }
  return files
}

/**
 * Apply a validated plan to storage deterministically.
 *
 * Two pre-flight passes run first, so an abort leaves the store untouched: every write target must
 * resolve unambiguously, and every path the plan expected to create must still be unclaimed.
 *
 * Writes then run before any deletes, so merged content lands before its sources are removed — a
 * crash between the passes leaves duplicates rather than dropping content that had nowhere else to
 * live. A failed write throws before any delete runs. This protects sources, not overwrite targets:
 * an `update` rewrites in place by design, so an interrupted run can leave that file changed while
 * the rest of the plan never ran.
 *
 * Deletes are best-effort — all are attempted and failures returned, not thrown, so the caller can
 * still record the changelog for a partial run. A missing key is a no-op per the {@link Storage}
 * contract, so a delete only fails on a genuine backend error.
 *
 * @returns The paths whose deletes failed, each with the underlying error (empty when all succeed)
 * @throws Error when a path the plan expected to create was claimed by a writer outside this run
 *
 * @internal
 */
export async function executePlan(
  storage: Storage,
  plan: ConsolidationPlan,
  files: Map<string, string>
): Promise<DeleteFailure[]> {
  assertPathsUnambiguous(plan, files)
  await assertNewTargetsUnclaimed(storage, plan, files)

  // Writes before deletes — merged content lands before sources are removed
  for (const action of plan.actions) {
    if (action.action === 'merge') {
      // A new path resolves to itself; a differently-cased one folds into the stored file rather
      // than duplicating it on case-sensitive backends
      const canonicalTarget = resolveWriteTarget(files, action.target)
      await storage.write(canonicalTarget, encoder.encode(action.content))
    } else if (action.action === 'update') {
      const canonicalPath = resolveWriteTarget(files, action.path)
      await storage.write(canonicalPath, encoder.encode(action.content))
    } else if (action.action === 'move') {
      // validatePlan guarantees every move source exists in `files`
      const canonicalFrom = resolveCanonicalKey(files, action.from)
      const content = canonicalFrom !== undefined ? files.get(canonicalFrom) : undefined
      if (content === undefined) {
        throw new Error(
          `Invariant violated: move source '${action.from}' missing from working set — plan not validated`
        )
      }
      // A case-only rename rewrites in place rather than minting a key the delete pass skips
      const canonicalTo = resolveWriteTarget(files, action.to)
      await storage.write(canonicalTo, encoder.encode(content))
    }
  }

  // Collected before deleting so the failure-recording try/catch is written once rather than per
  // action shape. Appended one at a time, never spread — a plan with a huge sources array would
  // blow the argument limit.
  const pathsToDelete: string[] = []
  for (const action of plan.actions) {
    if (action.action === 'delete') {
      pathsToDelete.push(action.path)
    } else if (action.action === 'merge') {
      for (const source of action.sources) {
        // Skip a source that is also the target — deleting it would remove the content just written
        if (!pathsResolveSame(source, action.target)) pathsToDelete.push(source)
      }
    } else if (action.action === 'move' && !pathsResolveSame(action.from, action.to)) {
      // A case-only rename already rewrote the file in place — deleting would undo the write
      pathsToDelete.push(action.from)
    }
  }

  // Best-effort deletes — attempt all, then report failures
  const deleteErrors: DeleteFailure[] = []
  for (const path of pathsToDelete) {
    const canonicalPath = resolveCanonicalKey(files, path) ?? path
    try {
      await storage.delete(canonicalPath)
    } catch (error) {
      deleteErrors.push({ path: canonicalPath, error })
    }
  }

  return deleteErrors
}

/**
 * Verify that every path the plan writes resolves to exactly one key, before anything is written.
 *
 * {@link resolveWriteTarget} throws on an ambiguous path; doing that from inside the write loop would
 * abort a run that had already written earlier actions. Resolving every target up front aborts with
 * the store untouched, and names every ambiguous path at once rather than only the first.
 *
 * @throws Error when a write target matches two or more stored keys differing only by case
 */
function assertPathsUnambiguous(plan: ConsolidationPlan, files: Map<string, string>): void {
  const ambiguityErrors: string[] = []
  for (const action of plan.actions) {
    // Each variant names its write target differently; 'delete' has none. Declaring `target` without
    // an initializer makes a variant added without a case here fail to compile.
    let target: string
    switch (action.action) {
      case 'merge':
        target = action.target
        break
      case 'update':
        target = action.path
        break
      case 'move':
        target = action.to
        break
      case 'delete':
        continue
    }
    try {
      resolveWriteTarget(files, target)
    } catch (error) {
      ambiguityErrors.push(String(error instanceof Error ? error.message : error))
    }
  }
  if (ambiguityErrors.length > 0) {
    throw new Error(ambiguityErrors.join('\n'))
  }
}

/**
 * Verify that every path the plan expected to create is still unclaimed in storage.
 *
 * A path absent from the run's snapshot but present now was written by something outside this run,
 * and the planner never saw its content — writing over it would silently destroy knowledge.
 *
 * Detection, not mutual exclusion: the losing write has already happened by the time it is noticed,
 * and a write landing between this read and the write pass is still overwritten. Checking here
 * rather than at snapshot time shrinks that window to the gap between the two. Closing it entirely
 * needs a conditional write that {@link Storage} does not offer.
 *
 * Paths already in the snapshot are excluded — validation proved the plan vacates or rewrites them.
 *
 * @throws Error naming the claimed paths, before any write or delete has run
 */
async function assertNewTargetsUnclaimed(
  storage: Storage,
  plan: ConsolidationPlan,
  files: Map<string, string>
): Promise<void> {
  const snapshotKeys = [...files.keys()]
  const inSnapshot = (target: string): boolean => snapshotKeys.some((key) => pathsResolveSame(key, target))

  const newTargets = new Set<string>()
  for (const action of plan.actions) {
    const target = action.action === 'merge' ? action.target : action.action === 'move' ? action.to : undefined
    // 'update' and 'delete' only ever address snapshot files, so they can never create a path
    if (target !== undefined && !inSnapshot(target)) newTargets.add(target)
  }
  if (newTargets.size === 0) return

  const claimed = (
    await mapWithConcurrency([...newTargets], STORAGE_READ_CONCURRENCY, async (target) => {
      // A read error is not evidence the key is claimed — let the write pass surface it
      try {
        return (await storage.read(target)) ? target : null
      } catch {
        return null
      }
    })
  ).filter((target): target is string => target !== null)

  if (claimed.length > 0) {
    throw new Error(
      `Consolidation aborted before writing: ${claimed.length} target path(s) were created by another writer ` +
        `since this run began: ${claimed.join(', ')}. The store is unchanged — re-run consolidation to plan ` +
        `against the current contents.`
    )
  }
}

/**
 * Append a human-readable summary of an applied plan to the consolidation changelog.
 *
 * Provides an audit trail of what each run changed and why, one dated entry per consolidation.
 * When deletes failed, they are recorded too so the log reflects the partial run rather than
 * implying every action succeeded.
 *
 * @internal
 */
export async function recordChangelog(
  storage: Storage,
  operations: ConsolidateOperation[],
  plan: ConsolidationPlan,
  deleteErrors: DeleteFailure[]
): Promise<void> {
  const timestamp = new Date().toISOString().slice(0, 16).replace('T', ' ')

  // Strip newlines and leading '#' so injected content cannot forge a '## ' run header or other
  // markdown structure. Paths need this as much as reason/summary — validatePath constrains the
  // directory segment but not the filename charset. Clipping keeps one field from dominating an
  // entry, which matters because every run rewrites the whole log.
  const sanitizeChangelogField = (value: string): string =>
    clipWithCount(value.replace(/[\r\n]+/g, ' ').replace(/^#+\s*/g, ''), MAX_CHANGELOG_FIELD_LENGTH)

  const actionSummaries = plan.actions.map((action) => {
    const reason = sanitizeChangelogField(action.reason)
    switch (action.action) {
      case 'merge': {
        const sources = action.sources.map(sanitizeChangelogField).join(' + ')
        return `  - merge: ${sources} → ${sanitizeChangelogField(action.target)} (${reason})`
      }
      case 'update':
        return `  - update: ${sanitizeChangelogField(action.path)} (${reason})`
      case 'delete':
        return `  - delete: ${sanitizeChangelogField(action.path)} (${reason})`
      case 'move':
        return `  - move: ${sanitizeChangelogField(action.from)} → ${sanitizeChangelogField(action.to)} (${reason})`
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
    parts.push(``, `Summary: ${sanitizeChangelogField(plan.summary)}`)
  }

  if (deleteErrors.length > 0) {
    parts.push(``, `Failed deletes (${deleteErrors.length}) — sources may remain until next consolidation:`)
    for (const deleteError of deleteErrors) {
      // The path can carry a newline (normalizeKey does not strip them), and the backend's error
      // text echoes the key it was given
      parts.push(
        `  - ${sanitizeChangelogField(deleteError.path)}: ${sanitizeChangelogField(String(deleteError.error))}`
      )
    }
  }
  parts.push('')

  const entry = parts.join('\n')
  // Written after the plan's mutations already landed, so a failure here must not throw — it would
  // mask the real run outcome
  try {
    const existing = await storage.read(CONSOLIDATION_CHANGELOG)
    const content = existing ? decoder.decode(existing) + entry : `# Consolidation Changelog\n${entry}`
    await storage.write(CONSOLIDATION_CHANGELOG, encoder.encode(content))
  } catch (error) {
    logger.warn(`error=<${error}> | failed to record consolidation changelog, audit log not updated`)
  }
}
