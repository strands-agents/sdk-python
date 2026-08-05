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
import type { ConsolidationAction, ConsolidationPlan } from './plan.js'
import { ConsolidationError } from '../../../errors.js'
import { logger } from '../../../logging/logger.js'
import {
  CONSOLIDATION_CHANGELOG,
  decoder,
  encoder,
  isConsolidationChangelog,
  mapWithConcurrency,
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
 * Apply a validated plan to storage deterministically, in two passes:
 *
 * 1. **Writes** — merged content lands before its sources are removed, so an interrupted run leaves
 *    duplicates rather than dropping content that had nowhere else to live. This protects sources,
 *    not overwrite targets: an `update` rewrites in place by design.
 * 2. **Deletes** — best-effort. All are attempted and failures returned, not thrown, so the caller
 *    can still record the changelog for a partial run. A missing key is a no-op per the
 *    {@link Storage} contract, so a delete only fails on a genuine backend error.
 *
 * Not safe for concurrent use — writes land unconditionally, so a file created after this run's
 * snapshot can be overwritten by a plan that never saw it. {@link Storage} offers no conditional
 * write, so nothing here can detect it.
 *
 * @returns The paths whose deletes failed, each with the underlying error (empty when all succeed)
 *
 * @internal
 */
export async function executePlan(
  storage: Storage,
  plan: ConsolidationPlan,
  files: Map<string, string>
): Promise<DeleteFailure[]> {
  // Writes before deletes — merged content lands before sources are removed. Every path is already
  // lowercased (stored keys by add(), plan paths by extractPlan), so a path is its own storage key.
  for (const action of plan.actions) {
    if (action.action === 'merge') {
      await storage.write(action.target, encoder.encode(action.content))
    } else if (action.action === 'update') {
      await storage.write(action.path, encoder.encode(action.content))
    } else if (action.action === 'move') {
      // validatePlan guarantees every move source exists in `files`
      const content = files.get(action.from)
      if (content === undefined) {
        throw new ConsolidationError(
          `Invariant violated: move source '${action.from}' missing from working set — plan not validated`
        )
      }
      await storage.write(action.to, encoder.encode(content))
    }
  }

  // Best-effort deletes — attempt all, then report failures
  const deleteErrors: DeleteFailure[] = []
  for (const path of collectDeletePaths(plan)) {
    try {
      await storage.delete(path)
    } catch (error) {
      deleteErrors.push({ path, error })
    }
  }

  return deleteErrors
}

/**
 * Collect every path the plan removes, in action order.
 *
 * Gathering them up front keeps the failure-recording try/catch in one place rather than repeating it
 * per action shape.
 *
 * @returns The paths to delete, which may repeat when two actions name the same source
 */
function collectDeletePaths(plan: ConsolidationPlan): string[] {
  // Appended one at a time, never spread — a plan with a huge sources array would blow the argument
  // limit
  const pathsToDelete: string[] = []
  for (const action of plan.actions) {
    switch (action.action) {
      case 'delete':
        pathsToDelete.push(action.path)
        break
      case 'merge':
        for (const source of action.sources) {
          // Skip a source that is also the target — deleting it would remove the content just written
          if (source !== action.target) pathsToDelete.push(source)
        }
        break
      case 'move':
        // A no-op rename (from === to) already rewrote the file in place — deleting would undo the write
        if (action.from !== action.to) pathsToDelete.push(action.from)
        break
    }
  }
  return pathsToDelete
}

/**
 * Append a human-readable summary of an applied plan to the consolidation changelog.
 *
 * Provides an audit trail of what each run changed and why, one UTC-timestamped entry per
 * consolidation. When deletes failed, they are recorded too so the log reflects the partial run
 * rather than implying every action succeeded.
 *
 * @internal
 */
export async function recordChangelog(
  storage: Storage,
  operations: ConsolidateOperation[],
  plan: ConsolidationPlan,
  deleteErrors: DeleteFailure[]
): Promise<void> {
  // Full ISO 8601 rather than a prettier form — the trailing 'Z' states the timezone is UTC, which an
  // audit entry needs more than it needs to read nicely
  const timestamp = new Date().toISOString()

  // Each variant names its paths with different fields, so only that part varies — the action name and
  // reason are shaped identically for all four
  const describePaths = (action: ConsolidationAction): string => {
    switch (action.action) {
      case 'merge':
        return `${action.sources.join(' + ')} → ${action.target}`
      case 'update':
      case 'delete':
        return action.path
      case 'move':
        return `${action.from} → ${action.to}`
    }
  }

  const actionSummaries = plan.actions.map(
    (action) => `  - ${action.action}: ${describePaths(action)} (${action.reason})`
  )

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
