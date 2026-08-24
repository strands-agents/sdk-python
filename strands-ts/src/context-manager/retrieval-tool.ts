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
import { ImageBlock, VideoBlock, DocumentBlock } from '../types/media.js'
import type { ImageFormat, VideoFormat, DocumentFormat } from '../types/media.js'
import type { JSONValue } from '../types/json.js'
import type { Tool } from '../tools/tool.js'
import type { Stash } from './stash.js'

export const RETRIEVAL_TOOL_NAME = 'retrieve_context'

const DEFAULT_MAX_RESULT_TOKENS = 10_000
const CHARS_PER_TOKEN = 4

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
 * @param stash - The stash instance to retrieve from
 * @param maxResultTokens - Maximum tokens in the retrieval result
 * @returns A Tool that can retrieve stashed content
 *
 * @internal
 */
export function createRetrievalTool(stash: Stash, maxResultTokens?: number): Tool {
  const maxChars = (maxResultTokens ?? DEFAULT_MAX_RESULT_TOKENS) * CHARS_PER_TOKEN

  return tool({
    name: RETRIEVAL_TOOL_NAME,
    description:
      'Retrieve content that was offloaded from context.\n\n' +
      'When content is offloaded (truncated, dropped, or summarized), the original is ' +
      'persisted and a reference key is left in its place. Use this tool with that ' +
      'reference to access the original content.\n\n' +
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

      if (!input.pattern && !input.line_range) {
        return decodeContent(result.content, result.contentType, input.reference)
      }

      if (!isSearchableContent(result.contentType)) {
        return `Error: cannot search binary content (${result.contentType}). Omit pattern/line_range to retrieve full content.`
      }

      const text = new TextDecoder().decode(result.content)
      const contextLines = input.context_lines ?? 5

      return searchContent(
        text,
        { pattern: input.pattern, line_range: input.line_range, context_lines: contextLines },
        maxChars
      )
    },
  })
}

function decodeContent(content: Uint8Array, contentType: string, reference: string): JSONValue {
  if (contentType.startsWith('text/')) {
    return new TextDecoder().decode(content)
  }
  if (contentType === 'application/json') {
    const text = new TextDecoder().decode(content)
    try {
      return JSON.parse(text) as JSONValue
    } catch {
      return text
    }
  }
  if (contentType.startsWith('image/')) {
    const format = contentType.slice(contentType.indexOf('/') + 1)
    return new ImageBlock({ format: format as ImageFormat, source: { bytes: content } }) as unknown as JSONValue
  }
  if (contentType.startsWith('video/')) {
    const format = contentType.slice(contentType.indexOf('/') + 1)
    return new VideoBlock({ format: format as VideoFormat, source: { bytes: content } }) as unknown as JSONValue
  }
  if (contentType.startsWith('application/')) {
    const format = contentType.slice(contentType.indexOf('/') + 1)
    return new DocumentBlock({
      format: format as DocumentFormat,
      name: reference,
      source: { bytes: content },
    }) as unknown as JSONValue
  }
  return new TextDecoder('utf-8', { fatal: false }).decode(content)
}
