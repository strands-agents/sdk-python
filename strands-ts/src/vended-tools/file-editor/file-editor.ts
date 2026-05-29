import { tool } from '../../tools/tool-factory.js'
import { z } from 'zod'
import type { IFileReader } from './types.js'
import { promises as fs } from 'fs'
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
  old_str: z.string().optional().describe('Exact string to find and replace (required for str_replace command).'),
  new_str: z.string().optional().describe('Replacement string (for str_replace and insert commands).'),
  insert_line: z
    .number()
    .optional()
    .describe('Line number where text should be inserted (0-indexed, required for insert command).'),
})

/**
 * Text file reader implementation.
 * Reads files as UTF-8 encoded text.
 */
class TextFileReader implements IFileReader {
  async read(filePath: string): Promise<string> {
    return await fs.readFile(filePath, 'utf-8')
  }
}

/**
 * File editor tool for viewing, creating, and editing files programmatically.
 *
 * Provides commands for viewing files/directories, creating files, string replacement,
 * and line insertion.
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
export const fileEditor = tool({
  name: 'fileEditor',
  description:
    'Filesystem editor tool for viewing, creating, and editing files. Supports view (with line ranges), create, str_replace, and insert operations. Files must use absolute paths.',
  inputSchema: fileEditorInputSchema,
  callback: async (input, context) => {
    if (!context) {
      throw new Error('Tool context is required for file editor operations')
    }

    const fileReader = new TextFileReader()

    let result: string

    switch (input.command) {
      case 'view':
        result = await handleView(input.path, input.view_range, fileReader)
        break

      case 'create':
        result = await handleCreate(input.path, input.file_text!)
        break

      case 'str_replace':
        result = await handleStrReplace(input.path, input.old_str!, input.new_str, fileReader)
        break

      case 'insert':
        result = await handleInsert(input.path, input.insert_line!, input.new_str!, fileReader)
        break

      default:
        throw new Error(`Unknown command: ${input.command}`)
    }

    return result
  },
})

/**
 * Validates that a path is absolute and doesn't contain directory traversal.
 */
function validatePath(command: string, filePath: string): void {
  // Check if it's an absolute path
  if (!path.isAbsolute(filePath)) {
    const suggestedPath = path.resolve(filePath)
    throw new Error(
      `The path ${filePath} is not an absolute path, it should start with \`/\`. Maybe you meant ${suggestedPath}?`
    )
  }

  // Check for directory traversal - reject paths containing '..' segments
  const normalized = path.normalize(filePath)
  if (normalized.includes('..')) {
    throw new Error(`Invalid path: path traversal is not allowed`)
  }
}

/**
 * Checks if a file exists.
 */
async function fileExists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath)
    return true
  } catch {
    return false
  }
}

/**
 * Checks if a path is a directory.
 */
async function isDirectory(filePath: string): Promise<boolean> {
  try {
    const stats = await fs.stat(filePath)
    return stats.isDirectory()
  } catch {
    return false
  }
}

/**
 * Checks file size against limit.
 */
async function checkFileSize(filePath: string, maxSize: number = DEFAULT_MAX_FILE_SIZE): Promise<void> {
  const stats = await fs.stat(filePath).catch((err) => {
    throw new Error(`Failed to check file size: ${err}`)
  })

  if (stats.size > maxSize) {
    throw new Error(`File size (${stats.size} bytes) exceeds maximum allowed size (${maxSize} bytes)`)
  }
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
 * Lists directory contents up to 2 levels deep, excluding hidden files.
 */
async function listDirectory(dirPath: string): Promise<string> {
  const items: string[] = []

  async function walk(currentPath: string, depth: number): Promise<void> {
    try {
      const entries = await fs.readdir(currentPath, { withFileTypes: true })

      for (const entry of entries) {
        // Skip hidden files/directories
        if (entry.name.startsWith('.')) continue

        const fullPath = path.join(currentPath, entry.name)
        const relativePath = path.relative(dirPath, fullPath)
        items.push(relativePath || entry.name)

        // Continue walking if we haven't reached max depth yet
        if (entry.isDirectory() && depth < MAX_DIRECTORY_DEPTH) {
          await walk(fullPath, depth + 1)
        }
      }
    } catch {
      // Ignore permission errors and continue
    }
  }

  await walk(dirPath, 0)

  const result = items.sort().join('\n')
  return `Here's the files and directories up to 2 levels deep in ${dirPath}, excluding hidden items:\n${result}\n`
}

/**
 * Handles the view command.
 */
async function handleView(
  filePath: string,
  viewRange: [number, number] | undefined,
  fileReader: IFileReader
): Promise<string> {
  validatePath('view', filePath)

  const exists = await fileExists(filePath)
  if (!exists) {
    throw new Error(`The path ${filePath} does not exist. Please provide a valid path.`)
  }

  const isDir = await isDirectory(filePath)

  if (isDir) {
    if (viewRange) {
      throw new Error('The `view_range` parameter is not allowed when `path` points to a directory.')
    }
    return await listDirectory(filePath)
  }

  // Check file size before reading
  await checkFileSize(filePath)

  // Read file content - only if not a directory
  const fileContent = await fileReader.read(filePath)

  let initLine = 1
  let contentToShow = fileContent

  if (viewRange) {
    const lines = fileContent.split('\n')
    const nLines = lines.length
    let [start, end] = viewRange

    // Validate range
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

    initLine = start
    if (end === -1) {
      contentToShow = lines.slice(start - 1).join('\n')
    } else {
      contentToShow = lines.slice(start - 1, end).join('\n')
    }
  }

  return makeOutput(contentToShow, filePath, initLine)
}

/**
 * Handles the create command.
 */
async function handleCreate(filePath: string, fileText: string): Promise<string> {
  if (fileText === undefined) {
    throw new Error('Parameter `file_text` is required for command: create')
  }

  validatePath('create', filePath)

  const exists = await fileExists(filePath)
  if (exists) {
    throw new Error(`File already exists at: ${filePath}. Cannot overwrite files using command \`create\`.`)
  }

  // Create parent directories if needed
  const dir = path.dirname(filePath)
  await fs.mkdir(dir, { recursive: true })

  // Write file
  await fs.writeFile(filePath, fileText, 'utf-8')

  return `File created successfully at: ${filePath}`
}

/**
 * Handles the str_replace command.
 */
async function handleStrReplace(
  filePath: string,
  oldStr: string,
  newStr: string | undefined,
  fileReader: IFileReader
): Promise<string> {
  if (oldStr === undefined) {
    throw new Error('Parameter `old_str` is required for command: str_replace')
  }

  validatePath('str_replace', filePath)

  const exists = await fileExists(filePath)
  if (!exists) {
    throw new Error(`The path ${filePath} does not exist. Please provide a valid path.`)
  }

  const isDir = await isDirectory(filePath)
  if (isDir) {
    throw new Error(`The path ${filePath} is a directory and only the \`view\` command can be used on directories`)
  }

  await checkFileSize(filePath)

  // Read file content
  let fileContent = await fileReader.read(filePath)

  // Expand tabs in content and search string
  fileContent = fileContent.replace(/\t/g, '        ')
  const expandedOldStr = oldStr.replace(/\t/g, '        ')
  const expandedNewStr = newStr ? newStr.replace(/\t/g, '        ') : ''

  // Check if old_str is unique
  const occurrences = (fileContent.match(new RegExp(escapeRegExp(expandedOldStr), 'g')) || []).length

  if (occurrences === 0) {
    throw new Error(`No replacement was performed, old_str \`${oldStr}\` did not appear verbatim in ${filePath}.`)
  }

  if (occurrences > 1) {
    const lines = fileContent.split('\n')
    const lineNumbers = lines
      .map((line, index) => (line.includes(expandedOldStr) ? index + 1 : -1))
      .filter((num) => num !== -1)
    throw new Error(
      `No replacement was performed. Multiple occurrences of old_str \`${oldStr}\` in lines ${JSON.stringify(lineNumbers)}. Please ensure it is unique`
    )
  }

  // Perform replacement
  const newFileContent = fileContent.replace(expandedOldStr, () => expandedNewStr)

  // Write back to file
  await fs.writeFile(filePath, newFileContent, 'utf-8')

  // Create snippet
  const replacementLine = fileContent.substring(0, fileContent.indexOf(expandedOldStr)).split('\n').length - 1
  const insertedLines = expandedNewStr.split('\n').length
  const originalLines = expandedOldStr.split('\n').length
  const lineDifference = insertedLines - originalLines

  const lines = newFileContent.split('\n')
  const startLine = Math.max(0, replacementLine - SNIPPET_LINES)
  const endLine = Math.min(lines.length, replacementLine + SNIPPET_LINES + lineDifference + 1)
  const snippetLines = lines.slice(startLine, endLine)
  const snippet = snippetLines.join('\n')

  const successMsg = `The file ${filePath} has been edited. ${makeOutput(snippet, `a snippet of ${filePath}`, startLine + 1)}Review the changes and make sure they are as expected. Edit the file again if necessary.`

  return successMsg
}

/**
 * Handles the insert command.
 */
async function handleInsert(
  filePath: string,
  insertLine: number,
  newStr: string,
  fileReader: IFileReader
): Promise<string> {
  if (insertLine === undefined || newStr === undefined) {
    throw new Error('Parameters `insert_line` and `new_str` are required for command: insert')
  }

  validatePath('insert', filePath)

  const exists = await fileExists(filePath)
  if (!exists) {
    throw new Error(`The path ${filePath} does not exist. Please provide a valid path.`)
  }

  const isDir = await isDirectory(filePath)
  if (isDir) {
    throw new Error(`The path ${filePath} is a directory and only the \`view\` command can be used on directories`)
  }

  await checkFileSize(filePath)

  // Read file content
  let fileText = await fileReader.read(filePath)

  // Expand tabs
  fileText = fileText.replace(/\t/g, '        ')
  const expandedNewStr = newStr.replace(/\t/g, '        ')

  const fileTextLines = fileText.split('\n')
  const nLines = fileTextLines.length

  // Validate insert_line
  if (insertLine < 0 || insertLine > nLines) {
    throw new Error(
      `Invalid \`insert_line\` parameter: ${insertLine}. It should be within the range of lines of the file: [0, ${nLines}]`
    )
  }

  // Perform insertion
  const newStrLines = expandedNewStr.split('\n')

  // Handle empty file case
  let newFileTextLines: string[]
  if (fileText === '') {
    newFileTextLines = newStrLines
  } else {
    newFileTextLines = [...fileTextLines.slice(0, insertLine), ...newStrLines, ...fileTextLines.slice(insertLine)]
  }

  const newFileText = newFileTextLines.join('\n')

  // Write back to file
  await fs.writeFile(filePath, newFileText, 'utf-8')

  // Create snippet - show lines around the insertion point
  // Show 4 lines before the insertion line and 4 lines after
  const snippetStartLine = Math.max(0, insertLine - SNIPPET_LINES)
  const snippetEndLine = Math.min(newFileTextLines.length, insertLine + newStrLines.length + SNIPPET_LINES)
  const snippetLines = newFileTextLines.slice(snippetStartLine, snippetEndLine)
  const snippet = snippetLines.join('\n')
  const startLine = snippetStartLine + 1

  const successMsg = `The file ${filePath} has been edited. ${makeOutput(snippet, 'a snippet of the edited file', startLine)}Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary.`

  return successMsg
}

/**
 * Escapes special regex characters in a string.
 */
function escapeRegExp(string: string): string {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}
