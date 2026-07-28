/**
 * Truncate reduction method.
 *
 * Replaces content with a head-tail preview. Used by Offload to shrink
 * oversized tool results in L0.
 *
 * @internal
 */

import { TextBlock, ToolResultBlock } from '../../../types/messages.js'

const DEFAULT_PREVIEW_TOKENS = 1000
const CHARS_PER_TOKEN = 4

/**
 * Configuration for the truncate method.
 */
export interface TruncateConfig {
  /** Number of tokens to keep as preview text. Defaults to 1,000. */
  previewTokens?: number
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
 * Checks whether a block has already been truncated (offloaded).
 */
export function isAlreadyTruncated(block: ToolResultBlock): boolean {
  if (block.content.length === 1 && block.content[0] instanceof TextBlock) {
    return block.content[0].text.startsWith('[Offloaded:')
  }
  return false
}

/**
 * Builds a head-tail preview of the given text.
 *
 * @returns The preview string with offload metadata header.
 */
export function buildPreview(
  fullText: string,
  block: ToolResultBlock,
  storageReference: string,
  config?: TruncateConfig
): string {
  const previewTokens = config?.previewTokens ?? DEFAULT_PREVIEW_TOKENS
  const previewChars = previewTokens * CHARS_PER_TOKEN
  const totalChars = fullText.length

  let preview: string
  if (totalChars <= previewChars) {
    preview = fullText
  } else {
    const headChars = Math.floor(previewChars * 0.6)
    const tailChars = previewChars - headChars
    const head = fullText.slice(0, headChars)
    const tail = fullText.slice(-tailChars)
    const elided = totalChars - headChars - tailChars
    preview = `${head}\n\n[... ${elided.toLocaleString()} chars elided ...]\n\n${tail}`
  }

  return (
    `[Offloaded: ${block.content.length} blocks, ~${Math.ceil(totalChars / CHARS_PER_TOKEN).toLocaleString()} tokens]\n` +
    `Full content available at storage reference "${storageReference}".\n\n` +
    preview
  )
}

/**
 * Creates a replacement ToolResultBlock containing the truncated preview.
 */
export function truncateBlock(
  block: ToolResultBlock,
  storageReference: string,
  config?: TruncateConfig
): ToolResultBlock {
  const fullText = extractBlockText(block)
  const preview = buildPreview(fullText, block, storageReference, config)
  return new ToolResultBlock({
    toolUseId: block.toolUseId,
    status: block.status,
    content: [new TextBlock(preview)],
  })
}
