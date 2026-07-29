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
 * Estimates token count for text content in a tool result block.
 */
export function estimateBlockTokens(block: ToolResultBlock): number {
  let chars = 0
  for (const content of block.content) {
    if (content instanceof TextBlock) {
      chars += content.text.length
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
 * Extracts text content from a tool result block (non-text blocks are skipped).
 */
export function extractBlockText(block: ToolResultBlock): string {
  const parts: string[] = []
  for (const content of block.content) {
    if (content instanceof TextBlock) {
      parts.push(content.text)
    }
  }
  return parts.join('\n')
}

/**
 * Builds a preview of the given text (head, tail, or head-tail depending on config).
 *
 * @returns The preview string with truncation metadata header, or the original text if already within budget.
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
 * Creates a replacement ToolResultBlock with text blocks truncated and non-text blocks preserved.
 */
export function truncateToolResultBlock(block: ToolResultBlock, config?: TruncateConfig): ToolResultBlock {
  const textParts: string[] = []
  const nonTextBlocks: Array<{ index: number; block: ToolResultBlock['content'][number] }> = []

  for (let index = 0; index < block.content.length; index++) {
    const content = block.content[index]!
    if (content instanceof TextBlock) {
      textParts.push(content.text)
    } else {
      nonTextBlocks.push({ index, block: content })
    }
  }

  if (textParts.length === 0) {
    return block
  }

  const fullText = textParts.join('\n')
  const preview = buildPreview(fullText, textParts.length, config)
  const newContent: ToolResultBlock['content'] = [new TextBlock(preview), ...nonTextBlocks.map((entry) => entry.block)]

  return new ToolResultBlock({
    toolUseId: block.toolUseId,
    status: block.status,
    content: newContent,
  })
}

/**
 * Creates a replacement TextBlock containing the truncated preview.
 */
export function truncateTextBlock(block: TextBlock, config?: TruncateConfig): TextBlock {
  const preview = buildPreview(block.text, 1, config)
  return new TextBlock(preview)
}
