import { tool } from '../../tools/tool-factory.js'
import { z } from 'zod'
import { Buffer } from 'buffer'
import { Sandbox } from '../../sandbox/base.js'
import { SandboxPathNotFoundError } from '../../sandbox/errors.js'
import * as path from 'path'

const SNIPPET_LINES = 4
const DEFAULT_MAX_FILE_SIZE = 1048576 // 1MB
const MAX_DIRECTORY_DEPTH = 2

/**
 * Zod schema for file editor input validation.
 */
const fileEditorInputSchema = z.object({
  command: z
    .enum(['view', 'create', 'str_replace', 'insert'])
    .describe('The operation to perform: `view`, `create`, `str_replace`, `insert`.'),
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
      'String to find and replace (required for str_replace). Must resolve to exactly one location. Copy it verbatim from a `view` of the file, including indentation. Matching is exact first; if that finds nothing it falls back to a whitespace-tolerant match (line endings, trailing whitespace, tabs) and only proceeds when a single location matches.'
    ),
  new_str: z.string().optional().describe('Replacement string (for str_replace and insert commands).'),
  insert_line: z
    .number()
    .optional()
    .describe('Line number where text should be inserted (0-indexed, required for insert command).'),
})

/**
 * File editor tool for viewing, creating, and editing files programmatically.
 *
 * Provides commands for viewing files/directories, creating files, string replacement,
 * and line insertion. All I/O routes through the agent's configured sandbox.
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
  'Filesystem editor tool for viewing, creating, and editing files. Supports view (with line ranges), create, str_replace, and insert operations. Files must use absolute paths.'

export interface MakeFileEditorOptions {
  name?: string
  description?: string
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

  return tool({
    name: options.name ?? 'fileEditor',
    description: options.description ?? DEFAULT_FILE_EDITOR_DESCRIPTION,
    inputSchema: fileEditorInputSchema,
    callback: async (input, context) => {
      if (!context) throw new Error('Tool context is required for fileEditor operations')
      const sandbox = boundSandbox ?? context.agent.sandbox
      const filePath = input.path.replace(/[/\\]+$/, '')

      switch (input.command) {
        case 'view':
          return handleView(sandbox, filePath, input.view_range)
        case 'create':
          return handleCreate(sandbox, filePath, input.file_text!)
        case 'str_replace':
          return handleStrReplace(sandbox, filePath, input.old_str!, input.new_str)
        case 'insert':
          return handleInsert(sandbox, filePath, input.insert_line!, input.new_str!)
        default:
          throw new Error(`Unknown command: ${input.command}`)
      }
    },
  })
}

/**
 * Default file editor tool. Reads the sandbox from the agent's context at call time.
 */
export const fileEditor = makeFileEditor()

/**
 * Validates that a path is absolute and doesn't contain directory traversal.
 */
function validatePath(filePath: string): void {
  // Check if it's an absolute path
  if (!path.isAbsolute(filePath)) {
    const suggestedPath = path.resolve(filePath)
    throw new Error(
      `The path ${filePath} is not an absolute path, it should start with \`/\`. Maybe you meant ${suggestedPath}?`
    )
  }

  // Check for '..' segments on the raw input — path.normalize resolves them away,
  // so checking after normalize is ineffective.
  if (filePath.split(/[/\\]/).includes('..')) {
    throw new Error(`Invalid path: path traversal is not allowed`)
  }
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

// View-style tab expansion: `view` renders each tab as 8 spaces, so the model
// reconstructs indentation as spaces. The tolerant fallback expands tabs the
// same way when comparing, but replacements are always applied to original bytes.
const TAB_AS_SPACES = '        '

/**
 * Returns the start indices of every non-overlapping exact occurrence of
 * `needle` in `haystack`.
 */
function exactMatchIndices(haystack: string, needle: string): number[] {
  const indices: number[] = []
  if (needle === '') return indices
  let from = 0
  for (let i = haystack.indexOf(needle, from); i !== -1; i = haystack.indexOf(needle, from)) {
    indices.push(i)
    from = i + needle.length
  }
  return indices
}

/** 1-based line number of a character offset within `content`. */
function lineNumberAt(content: string, index: number): number {
  return content.slice(0, index).split('\n').length
}

interface LineRecord {
  /** Line content excluding the trailing `\n`, but including a `\r` for CRLF files. */
  raw: string
  /** Offset in the original content where this line begins. */
  start: number
  /** Offset where the line's text ends (before its `\r`/`\n` terminator). */
  contentEnd: number
}

/** Splits content into line records with original byte offsets, so a tolerant
 * (normalized) match can be mapped back to an exact slice of the original. */
function splitLineRecords(content: string): LineRecord[] {
  const parts = content.split('\n')
  const records: LineRecord[] = []
  let offset = 0
  parts.forEach((raw, i) => {
    const hasCR = raw.endsWith('\r')
    records.push({ raw, start: offset, contentEnd: offset + raw.length - (hasCR ? 1 : 0) })
    offset += raw.length
    if (i < parts.length - 1) offset += 1 // the '\n' delimiter
  })
  return records
}

/** Normalizes a single line for tolerant matching: drops a trailing CR (CRLF),
 * expands tabs the way `view` does, and strips trailing whitespace. */
function normalizeLine(line: string): string {
  return line
    .replace(/\r$/, '')
    .replace(/\t/g, TAB_AS_SPACES)
    .replace(/[ \t]+$/, '')
}

type TolerantResult =
  { kind: 'unique'; start: number; end: number } | { kind: 'ambiguous'; lines: number[] } | { kind: 'none' }

/** Line-oriented whitespace-tolerant search used only when the exact match
 * finds nothing. Tolerates CRLF/LF, trailing whitespace, and tab-vs-8-spaces.
 * Returns original-byte offsets for a UNIQUE match; never guesses when ambiguous. */
function tolerantMatch(content: string, oldStr: string): TolerantResult {
  const records = splitLineRecords(content)
  const normOrig = records.map((r) => normalizeLine(r.raw))
  const normOld = oldStr.replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n').map(normalizeLine)
  const k = normOld.length
  if (k === 0 || k > normOrig.length) return { kind: 'none' }

  const starts: number[] = []
  const lastStart = normOrig.length - k
  for (let i = 0; i <= lastStart; i++) {
    if (normOrig.slice(i, i + k).every((line, j) => line === normOld[j])) starts.push(i)
  }

  if (starts.length === 0) return { kind: 'none' }
  if (starts.length > 1) return { kind: 'ambiguous', lines: starts.map((i) => i + 1) }
  const matchStart = starts[0]
  const first = matchStart === undefined ? undefined : records[matchStart]
  const last = matchStart === undefined ? undefined : records[matchStart + k - 1]
  if (!first || !last) return { kind: 'none' }
  return { kind: 'unique', start: first.start, end: last.contentEnd }
}

/** Builds an actionable hint pointing at lines that match the first line of
 * `oldStr` ignoring leading indentation — the usual culprit behind a near-miss. */
function nearMissHint(content: string, oldStr: string): string {
  const firstOld = oldStr.replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n')[0] ?? ''
  const target = normalizeLine(firstOld).trim()
  if (target === '') return ''
  const lines = splitLineRecords(content)
    .map((r, i) => (normalizeLine(r.raw).trim() === target ? i + 1 : -1))
    .filter((n) => n !== -1)
  if (lines.length === 0) return ''
  const where = lines.length === 1 ? `line ${lines[0]}` : `lines ${lines.slice(0, 5).join(', ')}`
  return ` A similar line was found at ${where}, differing only in leading indentation, trailing whitespace, or line endings — re-copy the exact text (including indentation) using the \`view\` command.`
}

/**
 * Computes a str_replace transformation. Tries an exact byte match first
 * (backward-compatible); only if that finds nothing does it fall back to a
 * conservative whitespace-tolerant search, and only when that resolves to a
 * single location. The replacement is always applied to the ORIGINAL bytes, so
 * tabs and line endings elsewhere in the file are left untouched. Returns the
 * new content plus a snippet around the change site (with 0-indexed `startLine`).
 */
function buildStrReplaceResult(
  originalContent: string,
  oldStr: string,
  newStr: string | undefined,
  filePath: string
): { newContent: string; snippet: string; startLine: number } {
  const replacement = newStr ?? ''

  let start: number
  let end: number

  const exact = exactMatchIndices(originalContent, oldStr)
  if (exact.length > 1) {
    const lineNumbers = exact.map((idx) => lineNumberAt(originalContent, idx))
    const firstLine = (originalContent.split('\n')[(lineNumbers[0] ?? 1) - 1] ?? '').trim()
    throw new Error(
      `No replacement was performed. Multiple occurrences of old_str \`${oldStr}\` — it appears ${exact.length} times, starting at lines ${JSON.stringify(lineNumbers)} (first at: \`${firstLine}\`). Include more surrounding context so exactly one location matches.`
    )
  }

  const [firstExact] = exact
  if (firstExact !== undefined) {
    start = firstExact
    end = firstExact + oldStr.length
  } else {
    const tolerant = tolerantMatch(originalContent, oldStr)
    if (tolerant.kind === 'ambiguous') {
      throw new Error(
        `No replacement was performed. old_str \`${oldStr}\` did not match exactly, and a whitespace-insensitive search found multiple candidates at lines ${JSON.stringify(tolerant.lines)}. Re-copy the exact text with more surrounding context so exactly one location matches.`
      )
    }
    if (tolerant.kind === 'none') {
      throw new Error(
        `No replacement was performed, old_str \`${oldStr}\` did not appear in ${filePath} (exact or whitespace-insensitive).${nearMissHint(originalContent, oldStr)}`
      )
    }
    start = tolerant.start
    end = tolerant.end
  }

  const newContent = originalContent.slice(0, start) + replacement + originalContent.slice(end)

  const replacementLine = originalContent.slice(0, start).split('\n').length - 1
  const insertedLines = replacement.split('\n').length
  const matchedLines = originalContent.slice(start, end).split('\n').length
  const lineDifference = insertedLines - matchedLines

  const newLines = newContent.split('\n')
  const startLine = Math.max(0, replacementLine - SNIPPET_LINES)
  const endLine = Math.min(newLines.length, replacementLine + SNIPPET_LINES + lineDifference + 1)
  const snippet = newLines.slice(startLine, endLine).join('\n')

  return { newContent, snippet, startLine }
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
 * Asserts content size is within the limit, checked after read since `listFiles`
 * does not reliably report size across sandbox backends.
 */
function assertWithinSizeLimit(content: string, maxSize: number = DEFAULT_MAX_FILE_SIZE): void {
  const size = Buffer.byteLength(content, 'utf-8')
  if (size > maxSize) {
    throw new Error(`File size (${size} bytes) exceeds maximum allowed size (${maxSize} bytes)`)
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
  viewRange: [number, number] | undefined
): Promise<string> {
  validatePath(filePath)

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

  const fileContent = await sandbox.readText(filePath)
  assertWithinSizeLimit(fileContent)

  const { content, initLine } = applyViewRange(fileContent, viewRange)
  return makeOutput(content, filePath, initLine)
}

async function handleCreate(sandbox: Sandbox, filePath: string, fileText: string): Promise<string> {
  if (fileText === undefined) {
    throw new Error('Parameter `file_text` is required for command: create')
  }

  validatePath(filePath)

  const { exists } = await probeSandboxPath(sandbox, filePath)
  if (exists) {
    throw new Error(`File already exists at: ${filePath}. Cannot overwrite files using command \`create\`.`)
  }

  await sandbox.writeText(filePath, fileText)
  return `File created successfully at: ${filePath}`
}

async function handleStrReplace(
  sandbox: Sandbox,
  filePath: string,
  oldStr: string,
  newStr: string | undefined
): Promise<string> {
  if (oldStr === undefined) {
    throw new Error('Parameter `old_str` is required for command: str_replace')
  }

  validatePath(filePath)

  const { exists, isDir } = await probeSandboxPath(sandbox, filePath)
  if (!exists) {
    throw new Error(`The path ${filePath} does not exist. Please provide a valid path.`)
  }
  if (isDir) {
    throw new Error(`The path ${filePath} is a directory and only the \`view\` command can be used on directories`)
  }

  const fileContent = await sandbox.readText(filePath)
  assertWithinSizeLimit(fileContent)

  const { newContent, snippet, startLine } = buildStrReplaceResult(fileContent, oldStr, newStr, filePath)

  await sandbox.writeText(filePath, newContent)

  return `The file ${filePath} has been edited. ${makeOutput(snippet, `a snippet of ${filePath}`, startLine + 1)}Review the changes and make sure they are as expected. Edit the file again if necessary.`
}

async function handleInsert(sandbox: Sandbox, filePath: string, insertLine: number, newStr: string): Promise<string> {
  if (insertLine === undefined || newStr === undefined) {
    throw new Error('Parameters `insert_line` and `new_str` are required for command: insert')
  }

  validatePath(filePath)

  const { exists, isDir } = await probeSandboxPath(sandbox, filePath)
  if (!exists) {
    throw new Error(`The path ${filePath} does not exist. Please provide a valid path.`)
  }
  if (isDir) {
    throw new Error(`The path ${filePath} is a directory and only the \`view\` command can be used on directories`)
  }

  const fileText = await sandbox.readText(filePath)
  assertWithinSizeLimit(fileText)

  const { newContent, snippet, startLine } = buildInsertResult(fileText, insertLine, newStr)

  await sandbox.writeText(filePath, newContent)

  return `The file ${filePath} has been edited. ${makeOutput(snippet, 'a snippet of the edited file', startLine + 1)}Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary.`
}
