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
  pathsResolveSame,
  resolveCanonicalKey,
} from '../internal.js'

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
 *
 * @internal
 */
export function validatePlan(
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

    // A merge must draw on two genuinely different files. Counting distinct sources covers both a
    // too-short list and a padded one — duplicate or case-variant sources would otherwise launder an
    // in-place overwrite past the operations allow-list (a self-overwrite under 'deduplicate', where
    // 'update' is not authorized). Kept here rather than as a schema `.min(2)` so it flows through the
    // same accumulate-and-revise path as every other guardrail instead of failing as a raw ZodError.
    if (action.action === 'merge') {
      const distinctSources = new Set(action.sources.map((source) => source.toLowerCase()))
      if (distinctSources.size < 2) {
        violations.push('Merge action requires at least 2 distinct source paths (case-insensitive)')
      }
    }

    // Append without spreading: an action carries one violation per source path, so a plan with a
    // huge sources array would blow the argument limit and crash with RangeError instead of being
    // rejected with the message it was about to produce
    const pathErrors = validateActionPaths(action, files, plannedDirs, maxDirectories)
    for (const pathError of pathErrors) violations.push(pathError)

    const contentError = validateActionContent(action)
    if (contentError) violations.push(contentError)
  }

  // Reject plans where multiple actions write to the same target path
  const collisionErrors = validateNoTargetCollisions(plan, files)
  for (const collisionError of collisionErrors) violations.push(collisionError)

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
        if (!resolveCanonicalKey(files, source)) errors.push(`Merge source '${source}' does not exist`)
      }
      const pathError = validatePath(action.target, existingDirs, maxDirectories)
      if (pathError) errors.push(pathError)
      return errors
    }
    case 'update': {
      if (!resolveCanonicalKey(files, action.path)) errors.push(`Update target '${action.path}' does not exist`)
      return errors
    }
    case 'delete': {
      if (!resolveCanonicalKey(files, action.path)) errors.push(`Delete target '${action.path}' does not exist`)
      return errors
    }
    case 'move': {
      if (!resolveCanonicalKey(files, action.from)) errors.push(`Move source '${action.from}' does not exist`)
      const pathError = validatePath(action.to, existingDirs, maxDirectories)
      if (pathError) errors.push(pathError)
      return errors
    }
  }
}

/**
 * Unicode classes that render as nothing, matched by property rather than enumeration.
 *
 * An explicit codepoint list is the wrong shape for this: the set of invisible codepoints is large
 * and grows with each Unicode revision, so any list is a denylist that a plan can step around by
 * picking a codepoint nobody thought of. Matching the categories closes the class instead of moving
 * its boundary.
 *
 * - `Cc` C0/C1 control characters, including NUL
 * - `Cf` format characters — soft hyphen, zero-width joiners, bidi embeddings and isolates, tag chars
 * - `Cn` unassigned codepoints, which no font can render
 * - `Cs` lone surrogates, which are not renderable text
 * - `Default_Ignorable_Code_Point` the Unicode-designated invisibles: Hangul fillers, variation
 *   selectors, and Mongolian vowel separator — several of which fall outside `Cc`/`Cf`/`Cn`
 *
 * U+2800 BRAILLE PATTERN BLANK is named explicitly: it is category `Lo` and not default-ignorable,
 * so no property covers it, yet it renders as blank space.
 */
const INVISIBLE_CLASSES = String.raw`\p{Cc}\p{Cf}\p{Cn}\p{Cs}\p{Default_Ignorable_Code_Point}\u{2800}`

/**
 * Strip characters that render as nothing, so content carrying no readable text is treated as empty.
 *
 * Covers {@link INVISIBLE_CLASSES} plus non-spacing and enclosing marks (`Mn`/`Me`). Bare combining
 * marks are included because a body of accents with no base characters to attach to displays as
 * nothing; they are stripped only here, never from filenames, where a mark following a base letter
 * is legitimate (`café`).
 *
 * Trims surrounding whitespace so whitespace-and-invisibles-only content is treated as empty.
 */
function stripInvisible(text: string): string {
  return text.replace(new RegExp(`[${INVISIBLE_CLASSES}\\p{Mn}\\p{Me}]`, 'gu'), '').trim()
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

  if (stripInvisible(action.content).length === 0) {
    return `${label} has empty content — a write must not blank out a file`
  }
  if (!action.content.startsWith(FRONTMATTER_OPEN)) {
    return `${label} must start with YAML frontmatter ('---' on the first line)`
  }
  // Search from FRONTMATTER_OPEN.length (not length-1) so the close delimiter cannot overlap the
  // open — '---\n---\n' would otherwise match by reusing the opening newline as the close's prefix
  const closingIndex = action.content.indexOf(FRONTMATTER_CLOSE, FRONTMATTER_OPEN.length)
  if (closingIndex === -1) {
    return `${label} is missing the closing frontmatter delimiter ('---' on its own line)`
  }
  // Require the description in the form parseFrontmatter reads — empty frontmatter, unrelated fields,
  // and an unquoted value all parse to an empty description, dropping a field add() always writes and
  // search() ranks against
  const frontmatterRegion = action.content.slice(FRONTMATTER_OPEN.length, closingIndex + 1)
  if (!FRONTMATTER_DESCRIPTION_PATTERN.test(frontmatterRegion)) {
    return `${label} frontmatter needs a quoted description field (description: "a short summary")`
  }
  if (stripInvisible(action.content.slice(closingIndex + FRONTMATTER_CLOSE.length)).length === 0) {
    return `${label} has no body after its frontmatter`
  }

  return undefined
}

/**
 * Reject plans that would clobber or amplify data: two actions writing the same path, a write landing
 * on an existing file that no other action vacates (a self-overwrite like an update is allowed), or
 * two actions claiming the same source.
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

  // Reject plans where two content-producing actions claim the same source. A merge and a move both
  // consume their sources — they write the source's content (or a synthesis of it) elsewhere and then
  // delete it — so a file can only be claimed once. Without this guard a plan amplifies one source
  // into arbitrarily many copies (N merges over the same pair write N targets and still delete the
  // pair), which wedges the store past maxFiles and leaves it permanently unconsolidatable: every
  // later run aborts on the file-count limit before a plan can fold the copies away.
  //
  // Deliberately strict: two merges drawing on one shared source may be a well-meant plan, but it is
  // indistinguishable from the amplifying one, and the planner prompt directs one action per path.
  const claimedSources = new Set<string>()
  const claimSource = (path: string): void => {
    const normalized = path.toLowerCase()
    if (claimedSources.has(normalized)) {
      errors.push(
        `Multiple actions claim '${path}' as a source — a file can only be merged or moved into one destination`
      )
    }
    claimedSources.add(normalized)
  }
  for (const action of plan.actions) {
    if (action.action === 'move') {
      claimSource(action.from)
    } else if (action.action === 'merge') {
      // A merge folding into one of its own sources rewrites that file in place rather than consuming
      // it, so the target spelling is not a claim — the write-vs-vacate rules above already govern it
      for (const source of action.sources) {
        if (!pathsResolveSame(source, action.target)) claimSource(source)
      }
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

  // Validate the filename stem: reject control characters, zero-width/invisible codepoints,
  // leading/trailing whitespace, empty stems, and over-long names. Mirrors the directory segment's
  // charset discipline so hostile filenames cannot crash the storage backend mid-write.
  const filename = rawSegments[rawSegments.length - 1]!
  const stem = filename.slice(0, -3) // strip '.md'
  if (stem.length === 0) {
    return `Filename must have a non-empty stem before .md: ${path}`
  }
  if (stem.length > 80) {
    return `Filename stem exceeds 80 characters: ${path}`
  }
  // A strict subset of the `Cc` class the next check matches, tested first only to name what is
  // wrong precisely: validation errors are fed back to the planner as a repair spec, and 'control
  // character' is actionable where calling a NUL 'invisible or zero-width' would misdirect.
  // eslint-disable-next-line no-control-regex -- intentional: reject NUL, BEL, newlines, tabs, etc.
  if (/[\u0000-\u001F\u007F]/.test(stem)) {
    return `Filename stem must not contain control characters: ${path}`
  }
  // Property-matched rather than enumerated, for the reason given on INVISIBLE_CLASSES. Combining
  // marks are deliberately not rejected here — unlike a file body, a stem like 'café' is legitimate.
  if (new RegExp(`[${INVISIBLE_CLASSES}]`, 'u').test(stem)) {
    return `Filename stem must not contain invisible or zero-width characters: ${path}`
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
