import type { Plugin } from '../../plugins/plugin.js'
import type { Tool } from '../../tools/tool.js'
import type { LocalAgent } from '../../types/agent.js'
import { AfterToolCallEvent } from '../../hooks/events.js'
import { TextBlock, JsonBlock, ToolResultBlock, Message } from '../../types/messages.js'
import type { ToolResultContent } from '../../types/messages.js'
import { ImageBlock, VideoBlock, DocumentBlock } from '../../types/media.js'
import type { ImageFormat, VideoFormat, DocumentFormat } from '../../types/media.js'
import { tool } from '../../tools/tool-factory.js'
import { z } from 'zod'
import { logger } from '../../logging/logger.js'
import type { JSONValue } from '../../types/json.js'
import type { Storage } from './storage.js'
import { isSearchableContent, searchContent } from './search.js'

const CHARS_PER_TOKEN = 4
const DEFAULT_MAX_RESULT_TOKENS = 2_500
const DEFAULT_PREVIEW_TOKENS = 1_000
const RETRIEVAL_TOOL_NAME = 'retrieve_offloaded_content'

const retrievalInputSchema = z.object({
  reference: z.string().describe('The reference string from the offload placeholder (e.g. "mem_1_tool-123_0").'),
  pattern: z
    .string()
    .optional()
    .describe('Regex or keyword to grep for. Returns only matching lines with context — not the full content.'),
  line_range: z
    .object({
      start: z.number().int().min(1).describe('First line to return (1-indexed).'),
      end: z.number().int().min(1).describe('Last line to return (1-indexed).'),
    })
    .optional()
    .describe('Return only this span of lines. Combine with pattern to search within the range.'),
  context_lines: z
    .number()
    .int()
    .min(0)
    .optional()
    .describe(
      'Lines before AND after each match (like grep -C). Default: 5. Without pattern/line_range, returns first N lines.'
    ),
})

function slicePreview(text: string, previewTokens: number): string {
  const maxChars = previewTokens * CHARS_PER_TOKEN
  if (text.length <= maxChars) return text
  return text.slice(0, maxChars)
}

function getBytes(block: ToolResultContent): Uint8Array | undefined {
  if (block instanceof ImageBlock && block.source.type === 'imageSourceBytes') {
    return block.source.bytes
  }
  if (block instanceof VideoBlock && block.source.type === 'videoSourceBytes') {
    return block.source.bytes
  }
  if (block instanceof DocumentBlock) {
    if (block.source.type === 'documentSourceBytes') return block.source.bytes
    if (block.source.type === 'documentSourceText') return new TextEncoder().encode(block.source.text)
  }
  return undefined
}

function decodeStoredContent(content: Uint8Array, contentType: string, reference: string): JSONValue {
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
  // Return native content blocks for binary types so the agent sees the actual content.
  // FunctionTool._wrapInToolResult passes ImageBlock/VideoBlock/DocumentBlock through as-is
  // at runtime, even though the callback type signature only accepts JSONValue.
  if (contentType.startsWith('image/')) {
    const format = contentType.split('/').pop()!
    return new ImageBlock({
      format: format as ImageFormat,
      source: { bytes: content },
    }) as unknown as JSONValue
  }
  if (contentType.startsWith('video/')) {
    const format = contentType.split('/').pop()!
    return new VideoBlock({
      format: format as VideoFormat,
      source: { bytes: content },
    }) as unknown as JSONValue
  }
  if (contentType.startsWith('application/')) {
    const format = contentType.split('/').pop()!
    return new DocumentBlock({
      format: format as DocumentFormat,
      name: reference,
      source: { bytes: content },
    }) as unknown as JSONValue
  }
  return new TextDecoder('utf-8', { fatal: false }).decode(content)
}

/** Configuration for the {@link ContextOffloader} plugin. */
export interface ContextOffloaderConfig {
  /** Storage backend for persisting offloaded content. */
  storage: Storage
  /** Token threshold above which tool results are offloaded. Defaults to 2,500. */
  maxResultTokens?: number
  /** Number of tokens to keep as an inline preview. Defaults to 1,000. */
  previewTokens?: number
  /** Whether to register the `retrieve_offloaded_content` tool. Defaults to true. */
  includeRetrievalTool?: boolean
}

/**
 * Plugin that offloads oversized tool results to reduce context consumption.
 *
 * When a tool result exceeds the configured token threshold, this plugin stores
 * each content block to a storage backend and replaces the in-context result with
 * a truncated text preview plus per-block storage references.
 *
 * @example
 * ```typescript
 * import { ContextOffloader, InMemoryStorage } from '@strands-agents/sdk/vended-plugins/context-offloader'
 *
 * const agent = new Agent({
 *   model,
 *   plugins: [new ContextOffloader({ storage: new InMemoryStorage() })],
 * })
 * ```
 */
export class ContextOffloader implements Plugin {
  readonly name = 'strands:context-offloader'

  private readonly _storage: Storage
  private readonly _maxResultTokens: number
  private readonly _previewTokens: number
  private readonly _includeRetrievalTool: boolean
  private _retrievalTool: Tool | undefined

  constructor(config: ContextOffloaderConfig) {
    const maxResultTokens = config.maxResultTokens ?? DEFAULT_MAX_RESULT_TOKENS
    const previewTokens = config.previewTokens ?? DEFAULT_PREVIEW_TOKENS

    if (maxResultTokens <= 0) throw new Error('maxResultTokens must be positive')
    if (previewTokens < 0) throw new Error('previewTokens must be non-negative')
    if (previewTokens >= maxResultTokens) throw new Error('previewTokens must be less than maxResultTokens')

    this._storage = config.storage
    this._maxResultTokens = maxResultTokens
    this._previewTokens = previewTokens
    this._includeRetrievalTool = config.includeRetrievalTool ?? true
  }

  initAgent(agent: LocalAgent): void {
    agent.addHook(AfterToolCallEvent, (event) => this._handleToolResult(event))
  }

  getTools(): Tool[] {
    if (!this._includeRetrievalTool) return []
    if (!this._retrievalTool) this._retrievalTool = this._createRetrievalTool()
    return [this._retrievalTool]
  }

  private _createRetrievalTool(): Tool {
    const storage = this._storage
    const maxChars = this._maxResultTokens * CHARS_PER_TOKEN

    return tool({
      name: RETRIEVAL_TOOL_NAME,
      description:
        'When a tool result was too large to keep in context, it was stored externally and replaced with a preview and a reference. ' +
        'Use this tool with that reference to access the stored content.\n\n' +
        'Returns:\n' +
        '  - With pattern: matching lines with line numbers and surrounding context\n' +
        '  - With line_range: the specified span of lines with line numbers\n' +
        '  - Without pattern/line_range: the full original content (use sparingly — re-injects all tokens)\n\n' +
        'Constraints:\n' +
        '  - pattern/line_range/context_lines only work on text content. For binary content, omit them.\n' +
        '  - Line numbers in results are 1-indexed and can be used in follow-up line_range calls.\n\n' +
        'Examples:\n' +
        '  { reference: "ref_1", pattern: "error" } → lines containing "error" with 5 lines context\n' +
        '  { reference: "ref_1", pattern: "error|warning", context_lines: 3 } → regex, 3 lines context\n' +
        '  { reference: "ref_1", line_range: { start: 10, end: 25 } } → lines 10-25\n' +
        '  { reference: "ref_1", pattern: "TODO", line_range: { start: 1, end: 50 } } → search within range',
      inputSchema: retrievalInputSchema,
      callback: async (input) => {
        try {
          const result = await storage.retrieve(input.reference)

          if (!input.pattern && !input.line_range && input.context_lines === undefined) {
            return decodeStoredContent(result.content, result.contentType, input.reference)
          }

          if (!isSearchableContent(result.contentType)) {
            return `Error: cannot search binary content (${result.contentType}). Omit pattern/line_range/context_lines to retrieve the full content.`
          }

          const text = new TextDecoder().decode(result.content)
          const contextLines = input.context_lines ?? 5
          const lineRange =
            input.line_range ?? (!input.pattern ? { start: 1, end: Math.max(1, contextLines) } : undefined)

          return searchContent(
            text,
            { pattern: input.pattern, line_range: lineRange, context_lines: contextLines },
            maxChars
          )
        } catch {
          return `Error: reference not found: ${input.reference}`
        }
      },
    })
  }

  private async _storeBlock(
    block: ToolResultContent,
    key: string
  ): Promise<{ ref: string; contentType: string; description: string }> {
    if (block instanceof TextBlock && block.text) {
      const ref = await this._storage.store(key, new TextEncoder().encode(block.text), 'text/plain')
      return { ref, contentType: 'text/plain', description: `text, ${block.text.length.toLocaleString()} chars` }
    }
    if (block instanceof JsonBlock) {
      const jsonStr = JSON.stringify(block.json, null, 2)
      const jsonBytes = new TextEncoder().encode(jsonStr)
      const ref = await this._storage.store(key, jsonBytes, 'application/json')
      return { ref, contentType: 'application/json', description: `json, ${jsonBytes.length.toLocaleString()} bytes` }
    }
    if (block instanceof ImageBlock || block instanceof VideoBlock || block instanceof DocumentBlock) {
      const bytes = getBytes(block)
      const contentType =
        block instanceof ImageBlock
          ? `image/${block.format}`
          : block instanceof VideoBlock
            ? `video/${block.format}`
            : `application/${block.format}`
      const label = block instanceof DocumentBlock ? block.name : contentType
      if (bytes) {
        const ref = await this._storage.store(key, bytes, contentType)
        return { ref, contentType, description: `${label}, ${bytes.length.toLocaleString()} bytes` }
      }
      return { ref: '', contentType, description: `${label}, 0 bytes` }
    }
    logger.warn('unsupported content block type encountered during offloading, skipping')
    return { ref: '', contentType: 'unknown', description: 'unknown block type' }
  }

  private _buildPreviewText(
    content: ToolResultContent[],
    references: Array<{ ref: string; description: string }>,
    tokenCount: number,
    fullText: string
  ): string {
    const preview = fullText ? slicePreview(fullText, this._previewTokens) : ''
    const refLines = references
      .filter((r) => r.ref)
      .map((r) => `  ${r.ref} (${r.description})`)
      .join('\n')

    let guidance =
      'Tool result was offloaded to external storage due to size.\n' +
      'Use the preview below if it answers your question.\n'
    if (this._includeRetrievalTool) {
      guidance +=
        'If you need more detail, use retrieve_offloaded_content with a reference and:\n' +
        '  - pattern: regex or keyword to find matching lines with context\n' +
        '  - line_range: { start, end } to read a specific span of lines\n' +
        'Retrieve full content (omit pattern/line_range) as a last resort.'
    } else {
      guidance += 'If you need more detail, use your available tools to access specific data.'
    }

    return (
      `[Offloaded: ${content.length} blocks, ~${tokenCount.toLocaleString()} tokens]\n` +
      `${guidance}\n\n` +
      `${preview}\n\n` +
      `[Stored references:]\n${refLines}`
    )
  }

  private async _handleToolResult(event: AfterToolCallEvent): Promise<void> {
    if (event.result.status === 'error') return

    // Skip results from the retrieval tool to prevent circular offloading
    if (this._includeRetrievalTool && event.toolUse.name === RETRIEVAL_TOOL_NAME) return

    const content = event.result.content
    const toolUseId = event.result.toolUseId

    const tokenCount = await event.agent.model.countTokens([new Message({ role: 'user', content: [event.result] })])

    if (tokenCount <= this._maxResultTokens) return

    // Extract text preview from text/JSON blocks
    const textParts: string[] = []
    for (const block of content) {
      if (block instanceof TextBlock && block.text) textParts.push(block.text)
      else if (block instanceof JsonBlock) textParts.push(JSON.stringify(block.json, null, 2))
    }
    const fullText = textParts.join('\n')

    // Store each content block to the storage backend
    let references: Array<{ ref: string; contentType: string; description: string }>
    try {
      references = await Promise.all(content.map((block, i) => this._storeBlock(block, `${toolUseId}_${i}`)))
    } catch (err) {
      logger.warn(`tool_use_id=<${toolUseId}> | failed to offload tool result, keeping original`, err)
      return
    }

    logger.debug(
      `tool_use_id=<${toolUseId}>, blocks=<${references.length}>, tokens=<${tokenCount}> | tool result offloaded`
    )

    // Build replacement content: preview text + media placeholders
    const newContent: ToolResultContent[] = [
      new TextBlock(this._buildPreviewText(content, references, tokenCount, fullText)),
    ]
    for (let i = 0; i < content.length; i++) {
      const block = content[i]!
      const ref = references[i]?.ref ?? ''
      if (block instanceof TextBlock || block instanceof JsonBlock) continue

      const bytes = getBytes(block)
      const size = bytes ? bytes.length : 0
      let label: string | undefined
      if (block instanceof ImageBlock) label = `image: ${block.format}`
      else if (block instanceof VideoBlock) label = `video: ${block.format}`
      else if (block instanceof DocumentBlock) label = `document: ${block.format}, ${block.name}`
      if (label) {
        newContent.push(new TextBlock(`[${label}, ${size} bytes${ref ? ` | ref: ${ref}` : ''}]`))
      }
    }

    event.result = new ToolResultBlock({
      toolUseId: event.result.toolUseId,
      status: event.result.status,
      content: newContent,
    })
  }
}
