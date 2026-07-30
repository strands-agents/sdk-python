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
 * Opens with a pre-flight pass that re-reads every target the plan expected to create — a path
 * absent from the snapshot. Validation already proved those paths were free when the plan was
 * built, so a hit means a writer outside this run (an {@link FileMemoryStore.add} on another
 * instance, a second process) claimed the key since. Overwriting it would destroy content the
 * planner never saw, so the run aborts here, before any write or delete, leaving the store
 * untouched. Targets that already existed in the snapshot are skipped: validation proved the plan
 * either vacates them or overwrites them from their own content.
 *
 * That pass detects a collision; it does not prevent one. A write landing between the check and
 * the write below is still overwritten — see {@link assertNewTargetsUnclaimed}.
 *
 * All writes run before any deletes so merged/moved content lands before its sources are
 * removed — a crash between the two passes leaves duplicated content rather than dropping a
 * source whose content had nowhere else to live.
 *
 * A failed write throws immediately, before any delete runs and before the changelog is recorded.
 * That is intentional: writes-before-deletes means an aborted write pass has removed nothing, so
 * the worst case is leftover duplicates from partial writes that already landed. A later run can
 * fold those away, though only if its planner chooses to — nothing schedules or forces the
 * cleanup. Note this ordering protects sources, not overwrite targets: an `update` (and a merge
 * into one of its own sources) replaces a file's content in place by design, so an interrupted
 * run can leave that file rewritten while the rest of the plan never ran.
 *
 * Deletes use best-effort semantics: every delete is attempted even if earlier ones fail.
 * A missing key is a no-op (per the {@link Storage} contract), so a delete only fails on a
 * genuine backend error — permissions, a read-only or broken disk, or a remote backend
 * (S3, DynamoDB) throttling or refusing the call. The failures are returned rather than thrown
 * so the caller can still record the changelog for the partial run before surfacing the error.
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
      // Resolve to the stored canonical key so a differently-cased target folds into the existing
      // file instead of creating a duplicate on case-sensitive backends. A target absent from the
      // snapshot resolves to itself, so brand-new paths are written verbatim; an ambiguous target
      // (several stored keys differing only by case) throws rather than minting a third spelling
      const canonicalTarget = resolveWriteTarget(files, action.target)
      await storage.write(canonicalTarget, encoder.encode(action.content))
    } else if (action.action === 'update') {
      // Resolve to the stored canonical key so a differently-cased echo from the model overwrites
      // the existing file instead of creating a duplicate on case-sensitive backends
      const canonicalPath = resolveWriteTarget(files, action.path)
      await storage.write(canonicalPath, encoder.encode(action.content))
    } else if (action.action === 'move') {
      // validatePlan guarantees every move source exists in `files`; resolve via canonical key so
      // a differently-cased path the model echoed still finds the stored content
      const canonicalFrom = resolveCanonicalKey(files, action.from)
      const content = canonicalFrom !== undefined ? files.get(canonicalFrom) : undefined
      if (content === undefined) {
        throw new Error(
          `Invariant violated: move source '${action.from}' missing from working set — plan not validated`
        )
      }
      // Resolve the destination the same way, so a case-only rename rewrites the file in place
      // rather than minting a second key the delete pass then skips
      const canonicalTo = resolveWriteTarget(files, action.to)
      await storage.write(canonicalTo, encoder.encode(content))
    }
  }

  // Best-effort deletes — attempt all, then report failures
  const deleteErrors: DeleteFailure[] = []
  for (const action of plan.actions) {
    if (action.action === 'delete') {
      // Resolve via canonical key so a case-variant delete path still removes the stored file
      const canonicalPath = resolveCanonicalKey(files, action.path) ?? action.path
      try {
        await storage.delete(canonicalPath)
      } catch (error) {
        deleteErrors.push({ path: canonicalPath, error })
      }
    } else if (action.action === 'merge') {
      for (const source of action.sources) {
        // Skip the target when it is one of its own sources — the merge folded into an existing
        // file, so deleting it here would remove the content just written
        if (!pathsResolveSame(source, action.target)) {
          // Resolve via canonical key so a case-variant source path still deletes the stored file
          const canonicalSource = resolveCanonicalKey(files, source) ?? source
          try {
            await storage.delete(canonicalSource)
          } catch (error) {
            deleteErrors.push({ path: canonicalSource, error })
          }
        }
      }
    } else if (action.action === 'move') {
      // Skip delete when source and target resolve to the same identity (case-only rename) —
      // deleting would remove the content the write pass just produced
      if (!pathsResolveSame(action.from, action.to)) {
        // Resolve via canonical key so a case-variant source path still deletes the stored file
        const canonicalFrom = resolveCanonicalKey(files, action.from) ?? action.from
        try {
          await storage.delete(canonicalFrom)
        } catch (error) {
          deleteErrors.push({ path: canonicalFrom, error })
        }
      }
    }
  }

  return deleteErrors
}

/**
 * Verify that every path the plan writes resolves to exactly one key, before anything is written.
 *
 * {@link resolveWriteTarget} throws on an ambiguous path, and doing that from inside the write loop
 * would abort a run that had already written earlier actions. Running the same resolution over every
 * write target up front means the abort happens with the store untouched, the way the pre-flight
 * claim check does — and the error names every ambiguous path at once rather than only the first.
 *
 * @throws Error when a write target matches two or more stored keys differing only by case
 */
function assertPathsUnambiguous(plan: ConsolidationPlan, files: Map<string, string>): void {
  const ambiguityErrors: string[] = []
  for (const action of plan.actions) {
    const target =
      action.action === 'merge'
        ? action.target
        : action.action === 'update'
          ? action.path
          : action.action === 'move'
            ? action.to
            : undefined
    if (target === undefined) continue
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
 * The plan was validated against a snapshot taken at the start of the run. A path absent from
 * that snapshot but present now was written by something outside this run, and its content was
 * never shown to the planner — so writing over it would silently destroy knowledge.
 *
 * This is best-effort detection rather than mutual exclusion, and it is not a lock: the losing
 * write has already happened by the time it is noticed, and the only remedy is to abort before
 * adding a second loss. Checking here rather than at snapshot time shrinks the exposure from the
 * whole run (which spans a model call) to the gap between this read and the write that follows;
 * a write landing inside that gap is still overwritten. Closing it entirely needs a conditional
 * write or version check that {@link Storage} does not offer.
 *
 * Paths the snapshot already held are excluded: validation proved the plan either vacates them
 * or rewrites them from their own content, so their presence here is expected. A plan that
 * creates no new paths reads nothing.
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
      // A read error is not evidence the key is claimed — let the write pass surface it instead
      // of aborting a valid plan on a transient backend failure
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

  // Sanitize every interpolated string: strip newlines and leading '#' so injected content cannot
  // forge a '## ' run header or inject arbitrary markdown structure into the changelog. Paths need
  // this as much as reason/summary do — validatePath constrains the directory segment but not the
  // filename's charset, so a target could otherwise carry a newline into the log. Backend error
  // text gets it too: the message typically echoes the key, carrying that key's newlines with it.
  const sanitizeChangelogField = (value: string): string => value.replace(/[\r\n]+/g, ' ').replace(/^#+\s*/g, '')

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
      // Both fields need the same treatment as the action summaries above: the path can carry a
      // newline (normalizeKey does not strip them, so an add() with an explicit metadata.path puts
      // one in the store), and the backend's error text is influenced by the key it was given
      parts.push(
        `  - ${sanitizeChangelogField(deleteError.path)}: ${sanitizeChangelogField(String(deleteError.error))}`
      )
    }
  }
  parts.push('')

  const entry = parts.join('\n')
  // The changelog is an audit artifact written after the plan's mutations already landed. A
  // failure to record it must not throw: doing so would mask the real run outcome (a partial
  // delete failure the caller needs to see, or a fully successful run reported as failed).
  try {
    const existing = await storage.read(CONSOLIDATION_CHANGELOG)
    const content = existing ? decoder.decode(existing) + entry : `# Consolidation Changelog\n${entry}`
    await storage.write(CONSOLIDATION_CHANGELOG, encoder.encode(content))
  } catch (error) {
    logger.warn(`error=<${error}> | failed to record consolidation changelog, audit log not updated`)
  }
}
