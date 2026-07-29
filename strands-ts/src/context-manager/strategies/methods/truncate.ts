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
 * Estimates token count for all content in a tool result block.
 * Text blocks use character length; non-text blocks use their JSON serialization size.
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
 * Extracts a text representation from a tool result block.
 * Text blocks contribute their text directly; non-text blocks contribute their JSON serialization.
 */
export function extractBlockText(block: ToolResultBlock): string {
  const parts: string[] = []
  for (const content of block.content) {
    if (content instanceof TextBlock) {
      parts.push(content.text)
    } else {
      parts.push(JSON.stringify(content.toJSON()))
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
  const previewTokens =
    typeof config?.previewTokens === 'number' && Number.isFinite(config.previewTokens)
      ? Math.max(0, config.previewTokens)
      : DEFAULT_PREVIEW_TOKENS
  const previewChars = previewTokens * CHARS_PER_TOKEN
  const previewMode = config?.preview ?? 'headTail'
  const totalChars = fullText.length

  if (totalChars <= previewChars) {
    return fullText
  }

  let preview: string
  if (previewMode === 'head') {
    const headChars = Math.max(0, previewChars)
    const head = headChars > 0 ? fullText.slice(0, headChars) : ''
    const elided = totalChars - headChars
    preview = `${head}\n\n[... ${elided.toLocaleString()} chars elided ...]`
  } else if (previewMode === 'tail') {
    const tailChars = Math.max(0, previewChars)
    const tail = tailChars > 0 ? fullText.slice(-tailChars) : ''
    const elided = totalChars - tailChars
    preview = `[... ${elided.toLocaleString()} chars elided ...]\n\n${tail}`
  } else {
    const headChars = Math.max(0, Math.floor(previewChars * 0.6))
    const tailChars = Math.max(0, previewChars - headChars)
    const head = headChars > 0 ? fullText.slice(0, headChars) : ''
    const tail = tailChars > 0 ? fullText.slice(-tailChars) : ''
    const elided = totalChars - headChars - tailChars
    preview = `${head}\n\n[... ${elided.toLocaleString()} chars elided ...]\n\n${tail}`
  }

  const result =
    `${TRUNCATED_PREFIX} ${blockCount} ${blockCount === 1 ? 'block' : 'blocks'}, ~${Math.ceil(totalChars / CHARS_PER_TOKEN).toLocaleString()} tokens]\n\n` +
    preview

  if (result.length >= totalChars) {
    return fullText
  }

  return result
}

/**
 * Creates a replacement ToolResultBlock with text blocks truncated and non-text blocks preserved in place.
 */
export function truncateToolResultBlock(block: ToolResultBlock, config?: TruncateConfig): ToolResultBlock {
  const textParts: string[] = []
  let hasText = false

  for (const content of block.content) {
    if (content instanceof TextBlock) {
      textParts.push(content.text)
      hasText = true
    }
  }

  if (!hasText) {
    return block
  }

  const fullText = textParts.join('\n')
  const preview = buildPreview(fullText, textParts.length, config)

  if (preview === fullText) {
    return block
  }

  const newContent: ToolResultBlock['content'] = []
  let textReplaced = false

  for (const content of block.content) {
    if (content instanceof TextBlock) {
      if (!textReplaced) {
        newContent.push(new TextBlock(preview))
        textReplaced = true
      }
    } else {
      newContent.push(content)
    }
  }

  return new ToolResultBlock({
    toolUseId: block.toolUseId,
    status: block.status,
    content: newContent,
  })
}

/**
 * Creates a replacement TextBlock containing the truncated preview.
 * Returns the original block unchanged if preview would not reduce size.
 */
export function truncateTextBlock(block: TextBlock, config?: TruncateConfig): TextBlock {
  const preview = buildPreview(block.text, 1, config)
  if (preview === block.text) {
    return block
  }
  return new TextBlock(preview)
}
