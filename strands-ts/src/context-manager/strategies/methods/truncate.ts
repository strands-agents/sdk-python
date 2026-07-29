/**
 * Truncate reduction method.
 *
 * Replaces content with a preview (head, tail, or head-tail).
 *
 * @internal
 */

import { TextBlock, ToolResultBlock } from '../../../types/messages.js'

export const TRUNCATED_PREFIX = '[Truncated:'
export const DROPPED_MARKER = '[Dropped]'
export const SUMMARIZED_PREFIX = '[Summarized:'

const DEFAULT_PREVIEW_TOKENS = 1000
const CHARS_PER_TOKEN = 4

/**
 * Configuration for the truncate method.
 */
export interface TruncateConfig {
  /** Number of tokens to keep as preview text. Defaults to 1,000. */
  previewTokens?: number

  /** Which portion of the text to keep as preview. Defaults to 'headTail'. */
  preview?: 'head' | 'tail' | 'headTail'
}

/**
 * Estimates token count for a tool result block.
 */
export function estimateBlockTokens(block: ToolResultBlock): number {
  let chars = 0
  for (const content of block.content) {
    if (content instanceof TextBlock) {
      chars += content.text.length
    } else {
      chars += JSON.stringify(content.toJSON()).length
    }
  }
  return Math.ceil(chars / CHARS_PER_TOKEN)
}

/**
 * Estimates token count for a text block.
 */
export function estimateTextBlockTokens(block: TextBlock): number {
  return Math.ceil(block.text.length / CHARS_PER_TOKEN)
}

/**
 * Extracts full text content from a tool result block.
 */
export function extractBlockText(block: ToolResultBlock): string {
  const parts: string[] = []
  for (const content of block.content) {
    if (content instanceof TextBlock) {
      parts.push(content.text)
    } else {
      parts.push(JSON.stringify(content.toJSON(), null, 2))
    }
  }
  return parts.join('\n')
}

/**
 * Checks whether a block has already been processed (truncated, dropped, or summarized).
 */
export function isAlreadyProcessed(block: ToolResultBlock | TextBlock): boolean {
  if (block instanceof TextBlock) {
    return isProcessedText(block.text)
  }
  if (block.content.length === 1 && block.content[0] instanceof TextBlock) {
    return isProcessedText(block.content[0].text)
  }
  return false
}

function isProcessedText(text: string): boolean {
  return text.startsWith(TRUNCATED_PREFIX) || text.startsWith(DROPPED_MARKER) || text.startsWith(SUMMARIZED_PREFIX)
}

/**
 * Builds a head-tail preview of the given text.
 *
 * @returns The preview string with truncation metadata header.
 */
export function buildPreview(fullText: string, blockCount: number, config?: TruncateConfig): string {
  const previewTokens = config?.previewTokens ?? DEFAULT_PREVIEW_TOKENS
  const previewChars = previewTokens * CHARS_PER_TOKEN
  const previewMode = config?.preview ?? 'headTail'
  const totalChars = fullText.length

  if (totalChars <= previewChars) {
    return fullText
  }

  let preview: string
  if (previewMode === 'head') {
    const head = fullText.slice(0, previewChars)
    const elided = totalChars - previewChars
    preview = `${head}\n\n[... ${elided.toLocaleString()} chars elided ...]`
  } else if (previewMode === 'tail') {
    const tail = previewChars > 0 ? fullText.slice(-previewChars) : ''
    const elided = totalChars - previewChars
    preview = `[... ${elided.toLocaleString()} chars elided ...]\n\n${tail}`
  } else {
    const headChars = Math.floor(previewChars * 0.6)
    const tailChars = previewChars - headChars
    const head = fullText.slice(0, headChars)
    const tail = tailChars > 0 ? fullText.slice(-tailChars) : ''
    const elided = totalChars - headChars - tailChars
    preview = `${head}\n\n[... ${elided.toLocaleString()} chars elided ...]\n\n${tail}`
  }

  return (
    `${TRUNCATED_PREFIX} ${blockCount} blocks, ~${Math.ceil(totalChars / CHARS_PER_TOKEN).toLocaleString()} tokens]\n\n` +
    preview
  )
}

/**
 * Creates a replacement ToolResultBlock containing the truncated preview.
 */
export function truncateToolResultBlock(block: ToolResultBlock, config?: TruncateConfig): ToolResultBlock {
  const fullText = extractBlockText(block)
  const preview = buildPreview(fullText, block.content.length, config)
  return new ToolResultBlock({
    toolUseId: block.toolUseId,
    status: block.status,
    content: [new TextBlock(preview)],
  })
}

/**
 * Creates a replacement TextBlock containing the truncated preview.
 */
export function truncateTextBlock(block: TextBlock, config?: TruncateConfig): TextBlock {
  const preview = buildPreview(block.text, 1, config)
  return new TextBlock(preview)
}
