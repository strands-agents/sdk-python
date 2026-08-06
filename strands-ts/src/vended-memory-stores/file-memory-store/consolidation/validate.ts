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
import { CONSOLIDATION_CHANGELOG } from './execute.js'

/**
 * Frontmatter opening delimiter. Matches the convention used by {@link FileMemoryStore.add}:
 * files start with `---\n`, followed by YAML fields, then a closing `---\n`.
 *
 * @internal
 */
export const FRONTMATTER_OPEN = '---\n'

/**
 * Frontmatter closing delimiter, including the newline that must precede it.
 *
 * @internal
 */
export const FRONTMATTER_CLOSE = '\n---\n'

/**
 * The only `description` form {@link FileMemoryStore}'s frontmatter parser reads. Owned here because
 * this module enforces the frontmatter contract on every write; the store imports it so a plan cannot
 * write a description its parser would read as empty.
 *
 * @internal
 */
export const FRONTMATTER_DESCRIPTION_PATTERN = /^description:\s*(".*")\s*$/m

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

  // Seed with directories already on disk; validateActionPaths adds each new one a write introduces,
  // so the maxDirectories budget covers the plan's cumulative effect, not each action alone.
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

    // Append without spreading: a huge sources array would blow the argument limit with a RangeError
    const pathErrors = validateActionPaths(action, files, plannedDirs, maxDirectories)
    for (const pathError of pathErrors) violations.push(pathError)

    const contentErrors = validateActionContent(action)
    for (const contentError of contentErrors) violations.push(contentError)
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
      errors.push(...validatePath(action.target, existingDirs, maxDirectories))
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
      errors.push(...validatePath(action.to, existingDirs, maxDirectories))
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
 * @returns An array with the single content error, or empty when the content passes (or the action
 *   writes no content)
 */
function validateActionContent(action: ConsolidationAction): string[] {
  if (action.action !== 'merge' && action.action !== 'update') return []

  const label = action.action === 'merge' ? `Merge target '${action.target}'` : `Update target '${action.path}'`

  if (action.content.trim().length === 0) {
    return [`${label} has empty content — a write must not blank out a file`]
  }
  if (!action.content.startsWith(FRONTMATTER_OPEN)) {
    return [`${label} must start with YAML frontmatter ('---' on the first line)`]
  }
  // Offset past the open so the close cannot overlap it ('---\n---\n' reusing the opening newline)
  const closingIndex = action.content.indexOf(FRONTMATTER_CLOSE, FRONTMATTER_OPEN.length)
  if (closingIndex === -1) {
    return [`${label} is missing the closing frontmatter delimiter ('---' on its own line)`]
  }
  const frontmatterRegion = action.content.slice(FRONTMATTER_OPEN.length, closingIndex + 1)
  if (!FRONTMATTER_DESCRIPTION_PATTERN.test(frontmatterRegion)) {
    return [`${label} frontmatter needs a quoted description field (description: "a short summary")`]
  }
  if (action.content.slice(closingIndex + FRONTMATTER_CLOSE.length).trim().length === 0) {
    return [`${label} has no body after its frontmatter`]
  }

  return []
}

/**
 * The path an action writes to, or `undefined` for a `delete`, which writes nothing. One shared
 * mapping so validation and execution never disagree about what an action writes.
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

  const writeTargets: string[] = []
  const vacatedPaths = new Set<string>()
  // Writes that legitimately land on an existing file because the action also reads it: an update,
  // a merge folding in one of its own sources, or a no-op move. A merge onto a non-source file is
  // excluded — the model never saw its content, so overwriting would destroy data.
  const safeOverwrites = new Set<string>()

  for (const action of plan.actions) {
    const writeTarget = writeTargetOf(action)
    if (writeTarget !== undefined) writeTargets.push(writeTarget)

    if (action.action === 'merge') {
      // A source equal to the target is not vacated — it is overwritten in place
      for (const source of action.sources) {
        if (source !== action.target) vacatedPaths.add(source)
      }
      if (action.sources.includes(action.target)) safeOverwrites.add(action.target)
    } else if (action.action === 'move') {
      vacatedPaths.add(action.from)
      if (action.from === action.to) safeOverwrites.add(action.to)
    } else if (action.action === 'delete') {
      vacatedPaths.add(action.path)
    } else {
      // update
      safeOverwrites.add(action.path)
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
  // delete pass removes the content the write pass just produced.
  const writeSet = new Set(writeTargets)
  for (const path of vacatedPaths) {
    if (writeSet.has(path)) {
      errors.push(`Path '${path}' is both written to and removed by the same plan`)
    }
  }

  // Reject a write onto a pre-existing file the plan does not vacate, unless it is a self-overwrite
  // (an update, or a merge into one of its own sources).
  for (const target of writeTargets) {
    if (files.has(target) && !vacatedPaths.has(target) && !safeOverwrites.has(target)) {
      errors.push(`Target path '${target}' already exists and is not vacated by another action in the plan`)
    }
  }

  return errors
}

/**
 * Validate a single write-target path against the store's layout rules: a namespace-relative `.md`
 * file, at most one level of nesting, not the reserved changelog path, with a well-formed
 * directory name that stays within the budget.
 *
 * @returns An array with the single path error, or empty when the path passes
 */
function validatePath(path: string, existingDirs: Set<string>, maxDirectories: number): string[] {
  if (path.includes('\\')) {
    return [`Path must not contain backslashes: ${path}`]
  }

  const rawSegments = path.split('/')
  if (rawSegments.some((seg) => seg === '.' || seg === '..')) {
    return [`Path must not contain dot segments ('.' or '..'): ${path}`]
  }

  if (path === CONSOLIDATION_CHANGELOG) {
    return [`Path must not be the reserved '${CONSOLIDATION_CHANGELOG}' file: ${path}`]
  }
  if (!path.endsWith('.md')) {
    return [`Path must end with .md: ${path}`]
  }

  // Validate the filename stem, mirroring the directory segment's charset discipline.
  const filename = rawSegments[rawSegments.length - 1]!
  const stem = filename.slice(0, -3) // strip '.md'
  if (stem.length === 0) {
    return [`Filename must have a non-empty stem before .md: ${path}`]
  }
  if (stem.length > 80) {
    return [`Filename stem exceeds 80 characters: ${path}`]
  }
  // Reject path-hostile characters that corrupt or crash the write on common storage backends.
  if (/[\\/:*?"<>|]/.test(stem)) {
    return [`Filename stem contains path-hostile characters: ${path}`]
  }
  if (stem !== stem.trim()) {
    return [`Filename stem must not have leading or trailing whitespace: ${path}`]
  }

  if (rawSegments.length > 2) {
    return [`Only one level of nesting allowed: ${path}`]
  }

  if (rawSegments.length === 2) {
    const dirName = rawSegments[0]!
    if (!/^[a-z0-9-]{1,30}$/.test(dirName)) {
      return [`Directory name must be lowercase alphanumeric + hyphens, ≤30 chars: '${dirName}'`]
    }
    if (!existingDirs.has(dirName) && existingDirs.size >= maxDirectories) {
      return [`Cannot create directory '${dirName}': maximum of ${maxDirectories} directories reached`]
    }
    // Track the new directory so subsequent actions in the same plan see it
    existingDirs.add(dirName)
  }

  return []
}
