import { tool } from '../../tools/tool-factory.js'
import { z } from 'zod'
import { Sandbox } from '../../sandbox/base.js'
import { SandboxPathNotFoundError } from '../../sandbox/errors.js'
import * as path from 'path'
import * as fs from 'fs'
import { Buffer } from 'buffer'

const SNIPPET_LINES = 4
const DEFAULT_MAX_FILE_SIZE = 1 * 1024 * 1024 // 1 MB
const MAX_DIRECTORY_DEPTH = 2
const MAX_PATTERN_LENGTH = 1000
const MAX_PATTERN_MATCHES = 10_000
const DEFAULT_MAX_UNDO_ENTRIES = 32
const DEFAULT_MAX_UNDO_BYTES = 32 * 1024 * 1024 // 32 MB

/**
 * Zod schema for file editor input validation.
 */
const fileEditorInputSchema = z.object({
  command: z
    .enum(['view', 'create', 'str_replace', 'insert', 'pattern_replace', 'find_line', 'undo_edit'])
    .describe(
      'The operation to perform: `view`, `create`, `str_replace`, `insert`, `pattern_replace`, `find_line`, or `undo_edit`.'
    ),
  path: z.string().describe('Absolute path to the file or directory.'),
  file_text: z.string().optional().describe('Content for new file (required for create command).'),
  view_range: z
    .tuple([z.number(), z.number()])
    .optional()
    .describe('Line range to view [start, end]. 1-indexed. End can be -1 for end of file.'),
  old_str: z
    .string()
    .optional()
    .describe(
      'Exact string to find and replace (required for str_replace). Must be unique unless replace_all is true.'
    ),
  new_str: z
    .string()
    .optional()
    .describe('Replacement string (for str_replace, pattern_replace, and insert commands).'),
  insert_line: z
    .number()
    .optional()
    .describe('Line number where text should be inserted (0-indexed, required for insert command).'),
  pattern: z.string().optional().describe('Regex pattern to match (required for pattern_replace command).'),
  search_text: z.string().optional().describe('Text to search for (required for find_line command).'),
  fuzzy: z.boolean().optional().describe('Enable whitespace-tolerant, case-insensitive matching for find_line.'),
  replace_all: z
    .boolean()
    .optional()
    .describe(
      'For str_replace and pattern_replace, allow replacing every occurrence. Defaults to false; a match count > 1 is rejected without this flag to prevent silent broad edits.'
    ),
})

/**
 * File editor tool for viewing, creating, and editing files programmatically.
 *
 * Provides commands for viewing files/directories, creating files, string and
 * regex replacement, line insertion, search, and single-step undo. All I/O
 * routes through the agent's configured sandbox.
 *
 * @example
 * ```typescript
 * import { fileEditor } from '@strands-agents/sdk/vended-tools/file-editor'
 * import { Agent } from '@strands-agents/sdk'
 *
 * const agent = new Agent({
 *   model: new BedrockModel({ region: 'us-east-1' }),
 *   tools: [fileEditor],
 * })
 *
 * await agent.invoke('View the file /tmp/test.txt')
 * await agent.invoke('Create a file /tmp/notes.txt with content "Hello World"')
 * await agent.invoke('Replace "Hello" with "Hi" in /tmp/notes.txt')
 * ```
 */
export const DEFAULT_FILE_EDITOR_DESCRIPTION =
  'Filesystem editor for viewing, creating, and editing files. Supports view (with line ranges), create, str_replace (exact match; ambiguous matches must opt in via replace_all), insert, pattern_replace (regex), find_line, and undo_edit. Files must use absolute paths.'

export interface MakeFileEditorOptions {
  name?: string
  description?: string
  /**
   * Optional absolute directory that confines every operation. String-level
   * checks reject non-absolute paths and any `..` traversal on the raw input;
   * when the resolved target exists on the local host, `fs.realpathSync` is
   * also applied and the result must still be inside `root`. Confinement is
   * enforced against the raw path input before I/O; concurrent rename attacks
   * between the confinement check and the read/write are the sandbox's
   * responsibility. When `undefined`, only absolute-path and `..`-traversal
   * checks apply; the underlying sandbox's symlink policy governs escape.
   */
  root?: string
  /**
   * Maximum file size (bytes) accepted by view/edit commands. Defaults to
   * 1 MB. Anything larger is rejected with a clean error rather than being
   * loaded into memory.
   */
  maxFileSize?: number
  /**
   * Maximum number of distinct paths retained in the in-memory undo history.
   * Oldest entry is evicted on overflow. Defaults to 32.
   */
  maxUndoEntries?: number
  /**
   * Approximate cap on total bytes of file content held in the undo history
   * (measured as UTF-8 byte length, matching the Python side). Oldest entries
   * are evicted until the cap is met. Defaults to 32 MB.
   */
  maxUndoBytes?: number
}

/**
 * Create a file editor tool. If a sandbox is passed, it's bound at creation time.
 * Otherwise, the tool reads from `context.agent.sandbox` at call time.
 * Used by sandbox implementations in `getTools()` and by users who want a customized file editor.
 */
export function makeFileEditor(options?: MakeFileEditorOptions): ReturnType<typeof tool>
export function makeFileEditor(sandbox: Sandbox | undefined, options?: MakeFileEditorOptions): ReturnType<typeof tool>
export function makeFileEditor(
  sandboxOrOptions?: Sandbox | MakeFileEditorOptions,
  maybeOptions?: MakeFileEditorOptions
): ReturnType<typeof tool> {
  const boundSandbox = sandboxOrOptions instanceof Sandbox ? sandboxOrOptions : undefined
  const options = sandboxOrOptions instanceof Sandbox || maybeOptions ? (maybeOptions ?? {}) : (sandboxOrOptions ?? {})
  const maxFileSize = options.maxFileSize ?? DEFAULT_MAX_FILE_SIZE
  const maxUndoEntries = options.maxUndoEntries ?? DEFAULT_MAX_UNDO_ENTRIES
  const maxUndoBytes = options.maxUndoBytes ?? DEFAULT_MAX_UNDO_BYTES

  if (options.root !== undefined && !path.isAbsolute(options.root)) {
    throw new Error(`root must be an absolute path, got: ${options.root}`)
  }
  // Use platform-native normalization so a Windows-style `root`
  // (e.g. `C:\Users\...\workspace`) survives round-tripping. Trailing separators
  // are stripped so subsequent `startsWith(root + sep)` checks work.
  const normalizedRoot: string | undefined =
    options.root === undefined ? undefined : stripTrailingSep(path.normalize(options.root))

  // Bounded LRU: previous file content keyed by path. `Map` preserves insertion
  // order, so re-inserting a key after `delete` gives LRU behavior; eviction
  // removes the oldest key when either cap is exceeded.
  const undoHistory = new Map<string, string>()

  return tool({
    name: options.name ?? 'fileEditor',
    description: options.description ?? DEFAULT_FILE_EDITOR_DESCRIPTION,
    inputSchema: fileEditorInputSchema,
    callback: async (input, context) => {
      if (!context) throw new Error('Tool context is required for fileEditor operations')
      const sandbox = boundSandbox ?? context.agent.sandbox
      const filePath = resolvePath(input.path, normalizedRoot)

      switch (input.command) {
        case 'view':
          return handleView(sandbox, filePath, input.view_range, maxFileSize)
        case 'create':
          return handleCreate(sandbox, filePath, input.file_text!, undoHistory, maxFileSize)
        case 'str_replace':
          return handleStrReplace(
            sandbox,
            filePath,
            input.old_str!,
            input.new_str,
            input.replace_all === true,
            maxFileSize,
            undoHistory,
            maxUndoEntries,
            maxUndoBytes
          )
        case 'insert':
          return handleInsert(
            sandbox,
            filePath,
            input.insert_line!,
            input.new_str!,
            maxFileSize,
            undoHistory,
            maxUndoEntries,
            maxUndoBytes
          )
        case 'pattern_replace':
          return handlePatternReplace(
            sandbox,
            filePath,
            input.pattern!,
            input.new_str,
            input.replace_all === true,
            maxFileSize,
            undoHistory,
            maxUndoEntries,
            maxUndoBytes
          )
        case 'find_line':
          return handleFindLine(sandbox, filePath, input.search_text!, input.fuzzy === true, maxFileSize)
        case 'undo_edit':
          return handleUndo(sandbox, filePath, undoHistory)
        default:
          throw new Error(`Unknown command: ${(input as { command: string }).command}`)
      }
    },
  })
}

/**
 * Default file editor tool. Reads the sandbox from the agent's context at call time.
 */
export const fileEditor = makeFileEditor()

/**
 * Normalize a path, rejecting traversal and out-of-root inputs.
 *
 * This is the single funnel every command routes through — it exists so path
 * validation cannot be bypassed by a new command forgetting to call it. When
 * `root` is set and the target (or its parent, for a not-yet-existing file)
 * exists locally, symlinks are also resolved via `fs.realpathSync` and the
 * result must still be inside `root` — a symlink inside `root` pointing at
 * `/etc/passwd` is rejected. When `root` does not exist locally (e.g. a Docker
 * sandbox whose paths are container-side), the realpath layer is skipped and
 * the sandbox owns its own symlink policy — otherwise every operation would
 * fail an ENOENT-on-realpath check.
 */
function resolvePath(filePath: string, root: string | undefined): string {
  // Strip trailing slashes on the raw input (compat with pre-existing
  // behavior). Uses `stripTrailingSep` so a Windows drive root like `C:\` is
  // preserved rather than collapsed to `C:` (which is not absolute).
  const stripped = stripTrailingSep(filePath)

  if (!path.isAbsolute(stripped)) {
    const suggestedPath = path.resolve(stripped)
    throw new Error(
      `The path ${filePath} is not an absolute path, it should start with \`/\` (or a drive letter on Windows). Maybe you meant ${suggestedPath}?`
    )
  }

  // Reject `..` segments on the raw input — path.normalize resolves them away
  // and could silently permit escape past the root.
  if (stripped.split(/[/\\]/).includes('..')) {
    throw new Error(`Invalid path: path traversal is not allowed`)
  }

  // Platform-native normalization so Windows drive-letter paths survive.
  const normalized = stripTrailingSep(path.normalize(stripped))

  if (root !== undefined) {
    // String-level confinement: normalized must equal root or start with root
    // + platform separator. Case-insensitive on Windows; case-sensitive on POSIX.
    if (!isInsideRoot(normalized, root)) {
      throw new Error(`Invalid path: ${filePath} is outside the configured root ${root}`)
    }

    // Symlink confinement is a best-effort, local-fs-only layer. If `root`
    // itself does not exist on the local filesystem, the caller is running
    // against a non-local sandbox (Docker, SSH, etc.) whose paths do not map
    // to this host — realpath would ENOENT on every call and reject every
    // operation. In that case the string-level check above is the whole
    // policy; the sandbox owns its own symlink handling.
    if (!fs.existsSync(root)) {
      return normalized
    }

    // Resolve the deepest existing ancestor via realpath and confirm it is
    // still inside root. This is what catches a symlink `inside-root/link ->
    // /etc/passwd` — the string-level check above only sees `inside-root/link`.
    let probe = normalized
    while (probe && !existsSyncLstat(probe)) {
      const parent = path.dirname(probe)
      if (parent === probe) break
      probe = parent
    }
    if (probe && existsSyncLstat(probe)) {
      let real: string
      let rootReal: string
      try {
        real = fs.realpathSync(probe)
        rootReal = fs.realpathSync(root)
      } catch {
        // realpath can fail on a broken symlink; treat that as an escape.
        throw new Error(`Invalid path: ${filePath} could not be resolved for symlink confinement.`)
      }
      if (!isInsideRoot(real, rootReal)) {
        throw new Error(
          `Invalid path: ${filePath} resolves via symlink to ${real}, outside the configured root ${root}`
        )
      }
    }
  }

  return normalized
}

function stripTrailingSep(p: string): string {
  // Preserve drive roots (`C:\`, `/`) but drop a trailing separator otherwise.
  if (p.length <= 1) return p
  if (process.platform === 'win32' && /^[A-Za-z]:[\\/]$/.test(p)) return p
  return p.replace(/[\\/]+$/, '') || p
}

function isInsideRoot(target: string, root: string): boolean {
  const t = process.platform === 'win32' ? target.toLowerCase() : target
  const r = process.platform === 'win32' ? stripTrailingSep(root).toLowerCase() : stripTrailingSep(root)
  if (t === r) return true
  // Root can already end in a separator when it is the filesystem root (`/`)
  // or a Windows drive root (`C:\`) — stripTrailingSep preserves those. In
  // that case, concatenating another separator would produce `//` or `C:\\`
  // and every valid in-root path would fail startsWith.
  const rEndsWithSep = r.endsWith('/') || r.endsWith('\\')
  if (rEndsWithSep) return t.startsWith(r)
  // Otherwise, match either separator so a Windows normalized path
  // (`C:\a\b`) matches a root that could be written with either style.
  return t.startsWith(r + path.sep) || t.startsWith(r + '/')
}

/**
 * Reject a replacement payload whose UTF-8 length would exceed `maxSize`. The
 * read-side cap already protects against pulling a huge file into memory; this
 * mirrors it on the write side so a model can't ship an unbounded `new_str`
 * or `file_text` through the tool.
 */
function rejectOversizeReplacement(text: string | undefined, maxSize: number, label = 'new_str'): void {
  if (text === undefined) return
  const bytes = Buffer.byteLength(text, 'utf-8')
  if (bytes > maxSize) {
    throw new Error(`${label} (${bytes} bytes) exceeds maximum allowed size (${maxSize} bytes)`)
  }
}

/**
 * Reject a post-edit buffer whose UTF-8 length would exceed `maxSize`. The
 * write-side cap catches the case where the individual `new_str` fits but the
 * resulting file (after all substitutions) is larger than the read cap — a
 * `replace_all` that expands every match, for instance.
 */
function rejectOversizeResult(content: string, maxSize: number, filePath: string): void {
  const bytes = Buffer.byteLength(content, 'utf-8')
  if (bytes > maxSize) {
    throw new Error(
      `The edit would produce a ${bytes}-byte file at ${filePath}, exceeding the maximum allowed size of ${maxSize} bytes.`
    )
  }
}

function existsSyncLstat(p: string): boolean {
  try {
    fs.lstatSync(p)
    return true
  } catch {
    return false
  }
}

/**
 * Approximate check that a pattern contains a nested quantifier (`(...+)+`,
 * `(...*)*`, `(...+)*`, `(...*)+`). Catastrophic backtracking in JS regex
 * (`(a+)+b` on all-a input) hangs a single `.exec()` call, and V8 has no
 * regex timeout — the match-count cap only fires between exec calls, so this
 * pre-compile heuristic is the only sync-time defense. False positives on
 * legitimate patterns are acceptable — the caller can loosen the pattern.
 */
function looksLikeCatastrophicPattern(pattern: string): boolean {
  // A quantified group whose interior also contains a quantifier is the
  // classic ReDoS shape. `(a+)+`, `(a*)*`, `(a|a)*`, `(a+)*b`, etc.
  return /\([^()]*[+*][^()]*\)[+*]/.test(pattern)
}

/**
 * Validates a view_range tuple and slices the file content to it. Returns the
 * visible content along with the line number to use as the first line in the
 * formatted output.
 */
function applyViewRange(
  fileContent: string,
  viewRange: [number, number] | undefined
): { content: string; initLine: number } {
  if (!viewRange) {
    return { content: fileContent, initLine: 1 }
  }
  const lines = fileContent.split('\n')
  const nLines = lines.length
  const [start, end] = viewRange

  if (start < 1 || start > nLines) {
    throw new Error(
      `Invalid \`view_range\`: [${start}, ${end}]. Its first element \`${start}\` should be within the range of lines of the file: [1, ${nLines}]`
    )
  }
  if (end !== -1 && end > nLines) {
    throw new Error(
      `Invalid \`view_range\`: [${start}, ${end}]. Its second element \`${end}\` should be smaller than the number of lines in the file: \`${nLines}\``
    )
  }
  if (end !== -1 && end < start) {
    throw new Error(
      `Invalid \`view_range\`: [${start}, ${end}]. Its second element \`${end}\` should be larger or equal than its first \`${start}\``
    )
  }

  const content = end === -1 ? lines.slice(start - 1).join('\n') : lines.slice(start - 1, end).join('\n')
  return { content, initLine: start }
}

/**
 * Performs a str_replace transformation on file content. Requires exactly one
 * match unless `replaceAll` is true. Returns the new content, a context
 * snippet, the snippet's 0-indexed start line, and the number of substitutions
 * performed.
 */
function buildStrReplaceResult(
  originalContent: string,
  oldStr: string,
  newStr: string | undefined,
  filePath: string,
  replaceAll: boolean
): { newContent: string; snippet: string; startLine: number; count: number } {
  const newStrValue = newStr ?? ''

  const occurrences = (originalContent.match(new RegExp(escapeRegExp(oldStr), 'g')) || []).length
  if (occurrences === 0) {
    throw new Error(`No replacement was performed, old_str \`${oldStr}\` did not appear verbatim in ${filePath}.`)
  }
  if (occurrences > 1 && !replaceAll) {
    const lines = originalContent.split('\n')
    const lineNumbers = lines.map((line, index) => (line.includes(oldStr) ? index + 1 : -1)).filter((num) => num !== -1)
    throw new Error(
      `No replacement was performed. Multiple occurrences of old_str \`${oldStr}\` in lines ${JSON.stringify(lineNumbers)}. Pass replace_all=true to replace every occurrence, or make old_str unique.`
    )
  }

  const count = replaceAll ? occurrences : 1
  // Single replacement uses a replacer function so `$&`/`$1`/`$$` in newStr
  // survive verbatim (String.prototype.replace with a string would interpret
  // them). replace_all is a plain split/join for the same reason.
  const newContent = replaceAll
    ? originalContent.split(oldStr).join(newStrValue)
    : originalContent.replace(oldStr, () => newStrValue)
  const replacementLine = originalContent.substring(0, originalContent.indexOf(oldStr)).split('\n').length - 1
  const insertedLines = newStrValue.split('\n').length
  const originalLines = oldStr.split('\n').length
  const lineDifference = insertedLines - originalLines

  const newLines = newContent.split('\n')
  const startLine = Math.max(0, replacementLine - SNIPPET_LINES)
  const endLine = Math.min(newLines.length, replacementLine + SNIPPET_LINES + lineDifference + 1)
  const snippet = newLines.slice(startLine, endLine).join('\n')

  return { newContent, snippet, startLine, count }
}

/**
 * Inserts text at a 0-indexed line in file content. Validates the insertion
 * point. Returns the new content plus a snippet around the insertion site
 * (with 0-indexed `startLine`).
 */
function buildInsertResult(
  originalContent: string,
  insertLine: number,
  newStr: string
): { newContent: string; snippet: string; startLine: number } {
  const fileTextLines = originalContent.split('\n')
  const nLines = fileTextLines.length

  if (insertLine < 0 || insertLine > nLines) {
    throw new Error(
      `Invalid \`insert_line\` parameter: ${insertLine}. It should be within the range of lines of the file: [0, ${nLines}]`
    )
  }

  const newStrLines = newStr.split('\n')
  const newFileTextLines =
    originalContent === ''
      ? newStrLines
      : [...fileTextLines.slice(0, insertLine), ...newStrLines, ...fileTextLines.slice(insertLine)]

  const newContent = newFileTextLines.join('\n')
  const snippetStartLine = Math.max(0, insertLine - SNIPPET_LINES)
  const snippetEndLine = Math.min(newFileTextLines.length, insertLine + newStrLines.length + SNIPPET_LINES)
  const snippet = newFileTextLines.slice(snippetStartLine, snippetEndLine).join('\n')

  return { newContent, snippet, startLine: snippetStartLine }
}

/**
 * Formats file content with line numbers (cat -n style).
 */
function makeOutput(fileContent: string, fileDescriptor: string, initLine: number = 1): string {
  // Expand tabs to spaces in content
  const expandedContent = fileContent.replace(/\t/g, '        ')

  const numberedLines = expandedContent.split('\n').map((line, index) => {
    const lineNum = index + initLine
    // Use two spaces instead of tab to avoid any tabs in output
    return `${lineNum.toString().padStart(6)}  ${line}`
  })

  return `Here's the result of running \`cat -n\` on ${fileDescriptor}:\n${numberedLines.join('\n')}\n`
}

/**
 * Escapes special regex characters in a string.
 */
function escapeRegExp(string: string): string {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/**
 * Returns the 0-indexed line number where `searchText` first appears, else -1.
 *
 * When `fuzzy` is true, whitespace between tokens is collapsed and matching
 * is case-insensitive.
 */
function findLineNumber(content: string, searchText: string, fuzzy: boolean): number {
  const lines = content.split('\n')
  if (fuzzy) {
    const tokens = searchText.trim().split(/\s+/).filter(Boolean)
    if (tokens.length === 0) return -1
    const pattern = new RegExp(tokens.map(escapeRegExp).join('.*'), 'i')
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i]
      if (line !== undefined && pattern.test(line)) return i
    }
  } else {
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i]
      if (line !== undefined && line.includes(searchText)) return i
    }
  }
  return -1
}

// ---- Undo history bookkeeping ----

/**
 * Record a pre-edit snapshot in the LRU undo history, evicting oldest on overflow.
 *
 * `Map` iteration follows insertion order; a re-inserted key after `delete` moves
 * to the end. `keys().next()` therefore yields the oldest entry for eviction.
 */
function storeUndoSnapshot(
  undoHistory: Map<string, string>,
  filePath: string,
  content: string,
  maxEntries: number,
  maxBytes: number
): void {
  if (undoHistory.has(filePath)) undoHistory.delete(filePath)
  undoHistory.set(filePath, content)
  let totalBytes = 0
  for (const v of undoHistory.values()) totalBytes += Buffer.byteLength(v, 'utf-8')
  while (undoHistory.size > 0 && (undoHistory.size > maxEntries || totalBytes > maxBytes)) {
    const oldestKey = undoHistory.keys().next().value
    if (oldestKey === undefined) break
    const evicted = undoHistory.get(oldestKey)!
    undoHistory.delete(oldestKey)
    totalBytes -= Buffer.byteLength(evicted, 'utf-8')
  }
}

// ---- Sandbox-routed I/O helpers ----

/**
 * Probes a path through the sandbox, reporting existence and directory-ness by listing
 * the parent directory. A missing parent or entry resolves to `exists: false`; permission,
 * transport, and other failures propagate so they aren't disguised as non-existence.
 */
async function probeSandboxPath(sandbox: Sandbox, filePath: string): Promise<{ exists: boolean; isDir: boolean }> {
  const normalized = filePath.replaceAll('\\', '/')
  const parent = normalized.split('/').slice(0, -1).join('/') || '/'
  const name = normalized.split('/').pop()!
  try {
    const entry = (await sandbox.listFiles(parent)).find((e) => e.name === name)
    if (!entry) {
      return { exists: false, isDir: false }
    }
    return { exists: true, isDir: entry.isDir ?? false }
  } catch (err) {
    if (err instanceof SandboxPathNotFoundError) {
      return { exists: false, isDir: false }
    }
    throw err
  }
}

/**
 * Read text through the sandbox, rejecting binary files and oversize inputs.
 *
 * Reads raw bytes first so the size cap and encoding detection run before
 * UTF-8 decoding — a corrupted UTF-8 error is worse than a clean rejection.
 * UTF-16 BOMs are detected up front so a valid UTF-16 file is reported as an
 * unsupported encoding rather than misclassified as binary. Otherwise the
 * classic NUL-in-first-8-KB heuristic identifies binary content.
 */
async function readTextOrRejectBinary(sandbox: Sandbox, filePath: string, maxSize: number): Promise<string> {
  const raw = await sandbox.readFile(filePath)
  if (raw.byteLength > maxSize) {
    throw new Error(`File size (${raw.byteLength} bytes) exceeds maximum allowed size (${maxSize} bytes)`)
  }
  if (raw.byteLength >= 2 && ((raw[0] === 0xff && raw[1] === 0xfe) || (raw[0] === 0xfe && raw[1] === 0xff))) {
    throw new Error(`Refusing to read non-UTF-8 file (detected UTF-16 BOM): ${filePath}`)
  }
  const scanLen = Math.min(raw.byteLength, 8192)
  for (let i = 0; i < scanLen; i++) {
    if (raw[i] === 0) {
      throw new Error(`Refusing to read binary file: ${filePath}`)
    }
  }
  const decoder = new TextDecoder('utf-8', { fatal: true })
  try {
    return decoder.decode(raw)
  } catch {
    throw new Error(`Refusing to read non-UTF-8 file: ${filePath}`)
  }
}

/**
 * Lists directory contents up to 2 levels deep, excluding hidden files.
 */
async function listDirectory(sandbox: Sandbox, dirPath: string): Promise<string> {
  const items: string[] = []

  async function walk(currentPath: string, prefix: string, depth: number): Promise<void> {
    let entries
    try {
      entries = await sandbox.listFiles(currentPath)
    } catch {
      // Ignore permission errors and continue
      return
    }

    for (const entry of entries) {
      if (entry.name.startsWith('.')) continue

      const relativePath = prefix ? `${prefix}/${entry.name}` : entry.name
      items.push(relativePath)

      if (entry.isDir && depth < MAX_DIRECTORY_DEPTH) {
        await walk(`${currentPath}/${entry.name}`, relativePath, depth + 1)
      }
    }
  }

  await walk(dirPath, '', 0)

  const result = items.sort().join('\n')
  return `Here's the files and directories up to 2 levels deep in ${dirPath}, excluding hidden items:\n${result}\n`
}

// ---- Sandbox-path handlers ----

async function handleView(
  sandbox: Sandbox,
  filePath: string,
  viewRange: [number, number] | undefined,
  maxSize: number
): Promise<string> {
  const { exists, isDir } = await probeSandboxPath(sandbox, filePath)
  if (!exists) {
    throw new Error(`The path ${filePath} does not exist. Please provide a valid path.`)
  }

  if (isDir) {
    if (viewRange) {
      throw new Error('The `view_range` parameter is not allowed when `path` points to a directory.')
    }
    return listDirectory(sandbox, filePath)
  }

  const fileContent = await readTextOrRejectBinary(sandbox, filePath, maxSize)
  const { content, initLine } = applyViewRange(fileContent, viewRange)
  return makeOutput(content, filePath, initLine)
}

async function handleCreate(
  sandbox: Sandbox,
  filePath: string,
  fileText: string,
  undoHistory: Map<string, string>,
  maxSize: number
): Promise<string> {
  if (fileText === undefined) {
    throw new Error('Parameter `file_text` is required for command: create')
  }
  const writeBytes = Buffer.byteLength(fileText, 'utf-8')
  if (writeBytes > maxSize) {
    throw new Error(`file_text (${writeBytes} bytes) exceeds maximum allowed size (${maxSize} bytes)`)
  }

  const { exists } = await probeSandboxPath(sandbox, filePath)
  if (exists) {
    throw new Error(`File already exists at: ${filePath}. Cannot overwrite files using command \`create\`.`)
  }

  await sandbox.writeText(filePath, fileText)
  // `create` is intentionally not snapshotted for undo: rolling back a create
  // means deleting the file, which is a different operation from "restore
  // prior content" and is easy for the caller to do themselves.
  undoHistory.delete(filePath)
  return `File created successfully at: ${filePath}`
}

async function handleStrReplace(
  sandbox: Sandbox,
  filePath: string,
  oldStr: string,
  newStr: string | undefined,
  replaceAll: boolean,
  maxSize: number,
  undoHistory: Map<string, string>,
  maxUndoEntries: number,
  maxUndoBytes: number
): Promise<string> {
  if (oldStr === undefined) {
    throw new Error('Parameter `old_str` is required for command: str_replace')
  }
  if (oldStr === '') {
    throw new Error('Parameter `old_str` must not be empty for command: str_replace')
  }
  if (newStr !== undefined) rejectOversizeReplacement(newStr, maxSize, 'new_str')

  const { exists, isDir } = await probeSandboxPath(sandbox, filePath)
  if (!exists) {
    throw new Error(`The path ${filePath} does not exist. Please provide a valid path.`)
  }
  if (isDir) {
    throw new Error(`The path ${filePath} is a directory and only the \`view\` command can be used on directories`)
  }

  const fileContent = await readTextOrRejectBinary(sandbox, filePath, maxSize)

  const { newContent, snippet, startLine, count } = buildStrReplaceResult(
    fileContent,
    oldStr,
    newStr,
    filePath,
    replaceAll
  )

  rejectOversizeResult(newContent, maxSize, filePath)
  storeUndoSnapshot(undoHistory, filePath, fileContent, maxUndoEntries, maxUndoBytes)
  await sandbox.writeText(filePath, newContent)

  const suffix = replaceAll && count > 1 ? ` (${count} occurrences replaced)` : ''
  return `The file ${filePath} has been edited.${suffix} ${makeOutput(snippet, `a snippet of ${filePath}`, startLine + 1)}Review the changes and make sure they are as expected. Edit the file again if necessary.`
}

async function handleInsert(
  sandbox: Sandbox,
  filePath: string,
  insertLine: number,
  newStr: string,
  maxSize: number,
  undoHistory: Map<string, string>,
  maxUndoEntries: number,
  maxUndoBytes: number
): Promise<string> {
  if (insertLine === undefined || newStr === undefined) {
    throw new Error('Parameters `insert_line` and `new_str` are required for command: insert')
  }
  rejectOversizeReplacement(newStr, maxSize, 'new_str')

  const { exists, isDir } = await probeSandboxPath(sandbox, filePath)
  if (!exists) {
    throw new Error(`The path ${filePath} does not exist. Please provide a valid path.`)
  }
  if (isDir) {
    throw new Error(`The path ${filePath} is a directory and only the \`view\` command can be used on directories`)
  }

  const fileText = await readTextOrRejectBinary(sandbox, filePath, maxSize)

  const { newContent, snippet, startLine } = buildInsertResult(fileText, insertLine, newStr)

  rejectOversizeResult(newContent, maxSize, filePath)
  storeUndoSnapshot(undoHistory, filePath, fileText, maxUndoEntries, maxUndoBytes)
  await sandbox.writeText(filePath, newContent)

  return `The file ${filePath} has been edited. ${makeOutput(snippet, 'a snippet of the edited file', startLine + 1)}Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary.`
}

async function handlePatternReplace(
  sandbox: Sandbox,
  filePath: string,
  pattern: string,
  newStr: string | undefined,
  replaceAll: boolean,
  maxSize: number,
  undoHistory: Map<string, string>,
  maxUndoEntries: number,
  maxUndoBytes: number
): Promise<string> {
  if (pattern === undefined) {
    throw new Error('Parameter `pattern` is required for command: pattern_replace')
  }
  if (pattern === '') {
    throw new Error('Parameter `pattern` must not be empty for command: pattern_replace')
  }
  if (newStr === undefined) {
    throw new Error('Parameter `new_str` is required for command: pattern_replace')
  }
  rejectOversizeReplacement(newStr, maxSize, 'new_str')
  if (pattern.length > MAX_PATTERN_LENGTH) {
    throw new Error(`Pattern is ${pattern.length} chars, exceeds maximum of ${MAX_PATTERN_LENGTH}.`)
  }
  if (looksLikeCatastrophicPattern(pattern)) {
    // Deliberate cross-SDK divergence: V8 has no in-exec regex timeout and JS
    // has no way to cancel a running .exec() call, so a catastrophic pattern
    // would hang the event loop indefinitely. The Python side runs regexes on
    // a worker thread under an asyncio.wait_for timeout; TypeScript cannot
    // replicate that safely, so it statically rejects the classic ReDoS
    // shapes ((...+)+, (...*)*, ...) before compiling. A pattern accepted by
    // Python with `pattern_replace_timeout` may be refused here — see the
    // vended-tools doc for the reasoning.
    throw new Error(
      `Pattern \`${pattern}\` contains a nested quantifier that risks catastrophic backtracking; refusing to compile. Rewrite without a quantified group whose interior also has a quantifier.`
    )
  }

  // Compile once with the `g` flag; a shared `.exec()` loop drives both the
  // ambiguity check (with a match cap so a runaway regex can't produce an
  // unbounded array) and the eventual substitution. V8 has no timeout, so the
  // pre-compile heuristic above plus the hard match cap below are the defense
  // against catastrophic backtracking.
  let regex: RegExp
  try {
    regex = new RegExp(pattern, 'g')
  } catch (e) {
    throw new Error(`Invalid regex pattern \`${pattern}\`: ${(e as Error).message}`, { cause: e })
  }

  const { exists, isDir } = await probeSandboxPath(sandbox, filePath)
  if (!exists) {
    throw new Error(`The path ${filePath} does not exist. Please provide a valid path.`)
  }
  if (isDir) {
    throw new Error(`The path ${filePath} is a directory and only the \`view\` command can be used on directories`)
  }

  const fileContent = await readTextOrRejectBinary(sandbox, filePath, maxSize)

  const matches: RegExpExecArray[] = []
  regex.lastIndex = 0
  let m: RegExpExecArray | null
  while ((m = regex.exec(fileContent)) !== null) {
    matches.push(m)
    if (matches.length > MAX_PATTERN_MATCHES) {
      throw new Error(`Pattern \`${pattern}\` produced more than ${MAX_PATTERN_MATCHES} matches; refusing to continue.`)
    }
    // Guard against a zero-width match causing an infinite loop.
    if (m.index === regex.lastIndex) regex.lastIndex++
  }

  if (matches.length === 0) {
    throw new Error(`No replacement was performed, pattern \`${pattern}\` did not match in ${filePath}.`)
  }
  if (matches.length > 1 && !replaceAll) {
    const lineNumbers = matches.map((mm) => fileContent.substring(0, mm.index).split('\n').length)
    throw new Error(
      `No replacement was performed. Pattern \`${pattern}\` matched ${matches.length} times (lines ${JSON.stringify(lineNumbers)}). Pass replace_all=true to replace every occurrence, or tighten the pattern.`
    )
  }

  // Build the replacement without recompiling the regex: substitute in reverse
  // order using the captured match indices when we're replacing all; otherwise
  // slice around the first match to avoid re-running the engine.
  const first = matches[0]!
  let newContent: string
  if (replaceAll) {
    newContent = ''
    let cursor = 0
    for (const mm of matches) {
      newContent += fileContent.slice(cursor, mm.index) + expandBackreferences(newStr, mm)
      cursor = mm.index + mm[0].length
    }
    newContent += fileContent.slice(cursor)
  } else {
    newContent =
      fileContent.slice(0, first.index) +
      expandBackreferences(newStr, first) +
      fileContent.slice(first.index + first[0].length)
  }

  const count = replaceAll ? matches.length : 1

  rejectOversizeResult(newContent, maxSize, filePath)
  storeUndoSnapshot(undoHistory, filePath, fileContent, maxUndoEntries, maxUndoBytes)
  await sandbox.writeText(filePath, newContent)

  const replacementLine = fileContent.substring(0, first.index).split('\n').length - 1
  const newLines = newContent.split('\n')
  const startLine = Math.max(0, replacementLine - SNIPPET_LINES)
  const endLine = Math.min(newLines.length, replacementLine + SNIPPET_LINES + 1)
  const snippet = newLines.slice(startLine, endLine).join('\n')

  const suffix = replaceAll && count > 1 ? ` (${count} matches replaced)` : ''
  return `The file ${filePath} has been edited via pattern_replace.${suffix} ${makeOutput(snippet, `a snippet of ${filePath}`, startLine + 1)}Review the changes and make sure they are as expected. Edit the file again if necessary.`
}

/**
 * Expand `$&`, `$$`, and `$1..$9` backreferences against a match, mirroring
 * String.prototype.replace's replacement-string semantics. Used so a single
 * compiled regex + captured matches drives both the ambiguity check and the
 * substitution (no second `.replace()` call, no third regex compile).
 */
function expandBackreferences(replacement: string, match: RegExpExecArray): string {
  return replacement.replace(/\$(\$|&|\d)/g, (_full, token: string) => {
    if (token === '$') return '$'
    if (token === '&') return match[0]
    const idx = parseInt(token, 10)
    return match[idx] ?? ''
  })
}

async function handleFindLine(
  sandbox: Sandbox,
  filePath: string,
  searchText: string,
  fuzzy: boolean,
  maxSize: number
): Promise<string> {
  if (searchText === undefined) {
    throw new Error('Parameter `search_text` is required for command: find_line')
  }

  const { exists, isDir } = await probeSandboxPath(sandbox, filePath)
  if (!exists) {
    throw new Error(`The path ${filePath} does not exist. Please provide a valid path.`)
  }
  if (isDir) {
    throw new Error(`The path ${filePath} is a directory and only the \`view\` command can be used on directories`)
  }

  const fileContent = await readTextOrRejectBinary(sandbox, filePath, maxSize)

  const lineIndex = findLineNumber(fileContent, searchText, fuzzy)
  if (lineIndex === -1) {
    throw new Error(`Could not find \`${searchText}\` in ${filePath}.`)
  }

  const lines = fileContent.split('\n')
  const startLine = Math.max(0, lineIndex - SNIPPET_LINES)
  const endLine = Math.min(lines.length, lineIndex + SNIPPET_LINES + 1)
  const snippet = lines.slice(startLine, endLine).join('\n')
  return `Found \`${searchText}\` at line ${lineIndex + 1} of ${filePath}.\n${makeOutput(snippet, `a snippet of ${filePath}`, startLine + 1)}`
}

// Restores the last in-memory snapshot for `filePath`. The snapshot is
// unconditionally written back through the sandbox: if the file was deleted
// (or moved) outside the tool since the snapshot was captured, `undo_edit`
// will re-create it at that path. Undo tracks content-per-path, not the
// file's presence.
async function handleUndo(sandbox: Sandbox, filePath: string, undoHistory: Map<string, string>): Promise<string> {
  const previous = undoHistory.get(filePath)
  if (previous === undefined) {
    throw new Error(`No undo history available for ${filePath} in this session.`)
  }
  undoHistory.delete(filePath)
  await sandbox.writeText(filePath, previous)
  return `Reverted ${filePath} to its previous in-memory snapshot.`
}
