/**
 * Guardrails a consolidation plan must clear before anything is written.
 *
 * The model is untrusted, so this is the gate that keeps a bad plan from corrupting the store.
 * Everything here is a pure function over the plan and the run's file snapshot — no storage or
 * model access — so a plan's validity is fully determined by those two inputs.
 *
 * @internal
 */

import type { ConsolidateOperation } from '../types.js'
import type { ConsolidationAction, ConsolidationPlan } from './plan.js'
import {
  CONSOLIDATION_CHANGELOG,
  FRONTMATTER_CLOSE,
  FRONTMATTER_DESCRIPTION_PATTERN,
  FRONTMATTER_OPEN,
  isConsolidationChangelog,
} from '../internal.js'

/**
 * Actions each operation is permitted to emit. `deriveInsights` includes `update` because it may
 * rewrite merged content as an update to an existing file.
 */
const OPERATION_ACTIONS: Record<ConsolidateOperation, string[]> = {
  deduplicate: ['merge'],
  deriveInsights: ['merge', 'update'],
  resolveContradictions: ['update', 'delete'],
  prune: ['delete'],
  reorganize: ['move'],
}

/**
 * Validate a plan against the guardrails before any storage mutation.
 *
 * Guards four properties: every action is permitted by the requested operations, every path is
 * well-formed and references files that exist, every write carries well-formed content, and no two
 * actions collide on a write target.
 *
 * All violations are accumulated so the rejection names every offending action at once rather than
 * surfacing them one at a time.
 *
 * @returns An array of every violation found; empty when the plan passes
 *
 * @internal
 */
export function validatePlan(
  plan: ConsolidationPlan,
  files: Map<string, string>,
  operations: ConsolidateOperation[],
  maxDirectories: number
): string[] {
  const violations: string[] = []

  const allowedActions = new Set(operations.flatMap((operation) => OPERATION_ACTIONS[operation]))

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

    // Count distinct sources: duplicate sources would otherwise launder an in-place overwrite past
    // the operations allow-list.
    if (action.action === 'merge') {
      const distinctSources = new Set(action.sources)
      if (distinctSources.size < 2) {
        violations.push('Merge action requires at least 2 distinct source paths')
      }
    }

    // Append without spreading: a huge sources array would blow the argument limit and crash
    // with RangeError instead of being rejected.
    const pathErrors = validateActionPaths(action, files, plannedDirs, maxDirectories)
    for (const pathError of pathErrors) violations.push(pathError)

    const contentError = validateActionContent(action)
    if (contentError) violations.push(contentError)
  }

  // Reject plans where multiple actions write to the same target path
  const collisionErrors = validateNoTargetCollisions(plan, files)
  for (const collisionError of collisionErrors) violations.push(collisionError)

  return violations
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
 * The schema accepts any string, so an empty or structurally broken value would be written verbatim
 * (and for a merge, its sources deleted afterwards). This requires the frontmatter shape
 * {@link FileMemoryStore.add} produces plus a non-empty body, so the file stays parseable and
 * searchable. Non-write actions carry no content and are skipped.
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
  // Offset past the open so the close cannot overlap it ('---\n---\n' reusing the opening newline)
  const closingIndex = action.content.indexOf(FRONTMATTER_CLOSE, FRONTMATTER_OPEN.length)
  if (closingIndex === -1) {
    return `${label} is missing the closing frontmatter delimiter ('---' on its own line)`
  }
  const frontmatterRegion = action.content.slice(FRONTMATTER_OPEN.length, closingIndex + 1)
  if (!FRONTMATTER_DESCRIPTION_PATTERN.test(frontmatterRegion)) {
    return `${label} frontmatter needs a quoted description field (description: "a short summary")`
  }
  if (action.content.slice(closingIndex + FRONTMATTER_CLOSE.length).trim().length === 0) {
    return `${label} has no body after its frontmatter`
  }

  return undefined
}

/**
 * The path an action writes to, or `undefined` for a `delete`, which writes nothing.
 *
 * Each variant names its target with a different field, so every pass that reasons about write targets
 * needs this mapping. Keeping it in one place is what stops those passes from disagreeing about what an
 * action writes — a disagreement between validation and execution is how a plan slips past a guardrail.
 *
 * A variant added to {@link ConsolidationAction} without a case here fails to compile: `noImplicitReturns`
 * makes the missing branch a `TS7030` error rather than a silent `undefined`.
 *
 * @internal
 */
export function writeTargetOf(action: ConsolidationAction): string | undefined {
  switch (action.action) {
    case 'merge':
      return action.target
    case 'update':
      return action.path
    case 'move':
      return action.to
    case 'delete':
      return undefined
  }
}

/**
 * Reject plans that would clobber data: two actions writing the same path, a path both written and
 * vacated in one plan, or a write landing on an existing file that no other action vacates (a
 * self-overwrite like an update is allowed).
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
    const writeTarget = writeTargetOf(action)
    if (writeTarget !== undefined) writeTargets.push(writeTarget)

    if (action.action === 'merge') {
      // Non-target merge sources are deleted during execution — a source equal to the target is not vacated
      for (const source of action.sources) {
        if (source !== action.target) {
          vacatedPaths.add(source)
        }
      }
    } else if (action.action === 'move') {
      // A no-op rename (from === to) still vacates, triggering the write-vs-vacate guard that rejects
      // it as a move that would destroy content
      vacatedPaths.add(action.from)
    } else if (action.action === 'delete') {
      vacatedPaths.add(action.path)
    }
  }

  // Check for two actions writing the same path
  const seen = new Set<string>()
  for (const target of writeTargets) {
    if (seen.has(target)) {
      errors.push(`Multiple actions write to the same path '${target}'`)
    }
    seen.add(target)
  }

  // A path both written and vacated in one plan is destroyed: writes run before deletes, so the
  // delete pass removes the content the write pass just produced. This single rule covers write +
  // delete on one path, an identity move (from === to), chained moves (one move's target is another
  // move's source), an update whose target is later moved away, and a move onto a since-deleted file.
  const writeSet = new Set(writeTargets)
  for (const path of vacatedPaths) {
    if (writeSet.has(path)) {
      errors.push(
        `Path '${path}' is both written to and removed by the same plan (one action writes it, another deletes or moves it away), which would destroy its content`
      )
    }
  }

  // Check for a write landing on a pre-existing file the plan does not vacate (vacated-and-written
  // targets are already rejected above). A self-overwrite — an update, or a merge into one of its
  // own sources — is the one legitimate case.
  for (const target of writeTargets) {
    if (files.has(target) && !vacatedPaths.has(target) && !planOverwritesSelf(plan, target)) {
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
    if (action.action === 'merge' && action.target === target && action.sources.includes(target)) {
      return true
    }
    if (action.action === 'update' && action.path === target) {
      return true
    }
    // A no-op move (from === to === target) writes the target onto the source's own file, which is intentional
    if (action.action === 'move' && action.to === target && action.from === target) {
      return true
    }
  }
  return false
}

/**
 * Validate a single write-target path against the store's layout rules: a namespace-relative `.md`
 * file, at most one level of nesting, not the reserved changelog path, with a well-formed
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

  if (isConsolidationChangelog(path)) {
    return `Path must not be the reserved '${CONSOLIDATION_CHANGELOG}' file: ${path}`
  }
  if (!path.endsWith('.md')) {
    return `Path must end with .md: ${path}`
  }

  // Validate the filename stem: reject control characters, leading/trailing whitespace, empty stems,
  // and over-long names. Mirrors the directory segment's charset discipline so hostile filenames
  // cannot crash the storage backend mid-write.
  const filename = rawSegments[rawSegments.length - 1]!
  const stem = filename.slice(0, -3) // strip '.md'
  if (stem.length === 0) {
    return `Filename must have a non-empty stem before .md: ${path}`
  }
  if (stem.length > 80) {
    return `Filename stem exceeds 80 characters: ${path}`
  }
  // A control character in a filename corrupts or crashes the write on most storage backends
  // eslint-disable-next-line no-control-regex -- intentional: reject NUL, BEL, newlines, tabs, etc.
  if (/[\u0000-\u001F\u007F]/.test(stem)) {
    return `Filename stem must not contain control characters: ${path}`
  }
  if (stem !== stem.trim()) {
    return `Filename stem must not have leading or trailing whitespace: ${path}`
  }
  if (/[\\/:*?"<>|]/.test(stem)) {
    return `Filename stem contains path-hostile characters: ${path}`
  }

  if (rawSegments.length > 2) {
    return `Only one level of nesting allowed: ${path}`
  }

  if (rawSegments.length === 2) {
    const dirName = rawSegments[0]!
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
