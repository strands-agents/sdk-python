/**
 * Retrieval tool for accessing stashed (L1) content.
 *
 * Registered automatically when the ContextManager has storage configured.
 *
 * @internal
 */

import { z } from 'zod'
import { tool } from '../tools/tool-factory.js'
import { isSearchableContent, searchContent } from '../vended-plugins/context-offloader/search.js'
import type { Tool } from '../tools/tool.js'
import type { Stash } from './stash.js'

export const RETRIEVAL_TOOL_NAME = 'retrieve_context'

const CHARS_PER_TOKEN = 4
const DEFAULT_MAX_RESULT_TOKENS = 10_000

const retrievalInputSchema = z.object({
  reference: z.string().describe('The reference key from the offload placeholder.'),
  pattern: z.string().optional().describe('Regex or keyword to grep for. Returns matching lines with context.'),
  line_range: z
    .object({
      start: z.number().int().min(1).describe('First line to return (1-indexed).'),
      end: z.number().int().min(1).describe('Last line to return (1-indexed).'),
    })
    .optional()
    .describe('Return only this span of lines.'),
  context_lines: z.number().int().min(0).optional().describe('Lines of context around each match (default: 5).'),
})

/**
 * Creates the retrieval tool for the given stash.
 *
 * @internal
 */
export function createRetrievalTool(stash: Stash, maxResultTokens?: number): Tool {
  const maxChars = (maxResultTokens ?? DEFAULT_MAX_RESULT_TOKENS) * CHARS_PER_TOKEN

  return tool({
    name: RETRIEVAL_TOOL_NAME,
    description:
      'Retrieve content that was offloaded from context to save space.\n\n' +
      'When a tool result was too large, it was replaced with a preview and a reference key. ' +
      'Use this tool with that reference to access the original content.\n\n' +
      'Options:\n' +
      '  - pattern: regex/keyword to find matching lines with context\n' +
      '  - line_range: { start, end } to read a specific span\n' +
      '  - Without pattern/line_range: returns the full original content\n\n' +
      'Examples:\n' +
      '  { reference: "1_toolu_abc_0", pattern: "error" }\n' +
      '  { reference: "1_toolu_abc_0", line_range: { start: 10, end: 25 } }\n' +
      '  { reference: "1_toolu_abc_0" }',
    inputSchema: retrievalInputSchema,
    callback: async (input) => {
      const result = await stash.retrieve(input.reference)
      if (!result) {
        return `Error: reference not found: ${input.reference}`
      }

      if (!input.pattern && !input.line_range && input.context_lines === undefined) {
        if (result.contentType.startsWith('text/') || result.contentType === 'application/json') {
          return new TextDecoder().decode(result.content)
        }
        return `[Binary content: ${result.contentType}, ${result.content.length.toLocaleString()} bytes]`
      }

      if (!isSearchableContent(result.contentType)) {
        return `Error: cannot search binary content (${result.contentType}). Omit pattern/line_range to retrieve full content.`
      }

      const text = new TextDecoder().decode(result.content)
      const contextLines = input.context_lines ?? 5
      const lineRange = input.line_range ?? (!input.pattern ? { start: 1, end: Math.max(1, contextLines) } : undefined)

      return searchContent(
        text,
        { pattern: input.pattern, line_range: lineRange, context_lines: contextLines },
        maxChars
      )
    },
  })
}
