/**
 * Sliding window conversation history management.
 *
 * This module provides a sliding window strategy for managing conversation history
 * that preserves tool usage pairs and avoids invalid window states.
 */

import { Message, TextBlock, ToolResultBlock, type ToolResultContent } from '../types/messages.js'
import { DocumentBlock, ImageBlock, VideoBlock } from '../types/media.js'
import type { LocalAgent } from '../types/agent.js'
import { AfterInvocationEvent } from '../hooks/events.js'
import {
  ConversationManager,
  type ProactiveCompressionConfig,
  type ConversationManagerReduceOptions,
} from './conversation-manager.js'
import { logger } from '../logging/logger.js'

const PRESERVE_CHARS = 200
// Max plausible marker length, including newlines. Used as the minimum reduction
// a re-truncation would need to produce in order to be worth running.
const MIN_TRUNCATION_GAIN = 50
// Text payloads at or below this length aren't worth truncating: the savings
// would be smaller than the marker itself, and already-truncated output (which
// lands just above `2 * PRESERVE_CHARS`) falls under this threshold so a
// second pass is a natural no-op.
const TRUNCATION_THRESHOLD = 2 * PRESERVE_CHARS + MIN_TRUNCATION_GAIN

/**
 * Build a short textual stand-in for an image block, used when truncating tool
 * results. The placeholder identifies the image format and its source kind
 * (bytes/url/s3) so the model can reason about what was dropped. For inline
 * bytes the size is included; URL and S3 sources only report the kind since
 * their byte count isn't known locally.
 */
function imagePlaceholder(image: ImageBlock): string {
  const source = image.source
  if (source.type === 'imageSourceBytes') {
    return `[image: ${image.format}, source: bytes, ${source.bytes.byteLength} bytes]`
  }
  if (source.type === 'imageSourceUrl') {
    return `[image: ${image.format}, source: url]`
  }
  return `[image: ${image.format}, source: s3]`
}

/**
 * Build a short textual stand-in for a video block. Binary payloads can't be
 * partially inspected, so videos are replaced wholesale. The placeholder
 * reports format and source kind; byte count is included for inline bytes.
 */
function videoPlaceholder(video: VideoBlock): string {
  const source = video.source
  if (source.type === 'videoSourceBytes') {
    return `[video: ${video.format}, source: bytes, ${source.bytes.byteLength} bytes]`
  }
  return `[video: ${video.format}, source: s3]`
}

/**
 * Build a short textual stand-in for a document block with a binary or remote
 * source. Text-based document sources (text / content) are truncated in place
 * instead of replaced, so this is only called for bytes / s3.
 */
function documentPlaceholder(doc: DocumentBlock): string {
  const source = doc.source
  if (source.type === 'documentSourceBytes') {
    return `[document: ${doc.name}, ${doc.format}, source: bytes, ${source.bytes.byteLength} bytes]`
  }
  return `[document: ${doc.name}, ${doc.format}, source: s3]`
}

/**
 * Build a short textual stand-in for a JSON block. The serialized length is
 * reported so the model knows how much was dropped; truncating JSON
 * mid-structure would produce invalid output, so the whole block is replaced.
 */
function jsonPlaceholder(serializedLength: number): string {
  return `[json: ${serializedLength} chars]`
}

/**
 * Configuration for the sliding window conversation manager.
 */
export type SlidingWindowConversationManagerConfig = {
  /**
   * Maximum number of messages to keep in the conversation history.
   * Defaults to 40 messages.
   */
  windowSize?: number

  /**
   * Whether to truncate tool results when a message is too large for the model's context window.
   * Defaults to true.
   */
  shouldTruncateResults?: boolean

  /**
   * Enable proactive context compression before the model call.
   *
   * - `true`: compress when 70% of the context window is used (default threshold).
   * - `{ compressionThreshold: number }`: compress at the specified ratio (0, 1].
   * - `false` or omitted: disabled, only reactive overflow recovery is used.
   */
  proactiveCompression?: boolean | ProactiveCompressionConfig
}

/**
 * Implements a sliding window strategy for managing conversation history.
 *
 * This class handles the logic of maintaining a conversation window that preserves
 * tool usage pairs and avoids invalid window states. When the message count exceeds
 * the window size, it will either truncate large tool results or remove the oldest
 * messages while ensuring tool use/result pairs remain valid.
 *
 * Registers hooks for:
 * - AfterInvocationEvent: Applies sliding window management after each invocation
 * - AfterModelCallEvent: Reduces context on overflow errors and requests retry (via super)
 * - BeforeModelCallEvent: Proactive compression when threshold is exceeded (via super)
 */
export class SlidingWindowConversationManager extends ConversationManager {
  private readonly _windowSize: number
  private readonly _shouldTruncateResults: boolean

  /**
   * Unique identifier for this conversation manager.
   */
  readonly name = 'strands:sliding-window-conversation-manager'

  /**
   * Initialize the sliding window conversation manager.
   *
   * @param config - Configuration options for the sliding window manager.
   */
  constructor(config?: SlidingWindowConversationManagerConfig) {
    super(config)
    this._windowSize = config?.windowSize ?? 40
    this._shouldTruncateResults = config?.shouldTruncateResults ?? true
  }

  /**
   * Initialize the plugin by registering hooks with the agent.
   *
   * Registers:
   * - AfterInvocationEvent callback to apply sliding window management
   * - AfterModelCallEvent callback to handle context overflow and request retry (via super)
   * - BeforeModelCallEvent callback for proactive compression (via super)
   *
   * @param agent - The agent to register hooks with
   */
  public override initAgent(agent: LocalAgent): void {
    super.initAgent(agent)

    agent.addHook(AfterInvocationEvent, (event) => {
      this._applyManagement(event.agent.messages)
    })
  }

  /**
   * Reduce the conversation history.
   *
   * When `error` is set (reactive overflow recovery), attempts to truncate large tool results
   * first before falling back to message trimming.
   *
   * When `error` is undefined (proactive compression), only trims messages without attempting
   * tool result truncation.
   *
   * @param options - The reduction options
   * @returns `true` if the history was reduced, `false` otherwise
   */
  reduce({ agent, error }: ConversationManagerReduceOptions): boolean {
    return this._reduceContext(agent.messages, error)
  }

  /**
   * Apply the sliding window to the messages array to maintain a manageable history size.
   *
   * Called after every agent invocation. No-op if within the window size.
   *
   * @param messages - The message array to manage. Modified in-place.
   */
  private _applyManagement(messages: Message[]): void {
    if (messages.length <= this._windowSize) {
      return
    }

    this._reduceContext(messages, undefined)
  }

  /**
   * Trim the oldest messages to reduce the conversation context size.
   *
   * The method handles special cases where trimming the messages leads to:
   * - toolResult with no corresponding toolUse
   * - toolUse with no corresponding toolResult
   *
   * The strategy is:
   * 1. First, attempt to truncate large tool results if shouldTruncateResults is true
   * 2. If truncation is not possible or doesn't help, trim oldest messages
   * 3. When trimming, skip invalid trim points (toolResult at start, or toolUse without following toolResult)
   *
   * @param messages - The message array to reduce. Modified in-place.
   * @param _error - The error that triggered the context reduction, if any.
   * @returns `true` if any reduction occurred, `false` otherwise.
   */
  private _reduceContext(messages: Message[], _error?: Error): boolean {
    // Only truncate tool results when handling a context overflow error, not for window size enforcement
    const oldestMessageIdxWithToolResults = this._findOldestMessageWithToolResults(messages)
    if (_error && oldestMessageIdxWithToolResults !== undefined && this._shouldTruncateResults) {
      const resultsTruncated = this._truncateToolResults(messages, oldestMessageIdxWithToolResults)
      if (resultsTruncated) {
        return true
      }
    }

    // Try to trim messages when tool result cannot be truncated anymore
    // If the number of messages is less than the window_size, then we default to 2, otherwise, trim to window size
    let trimIndex = messages.length <= this._windowSize ? 2 : messages.length - this._windowSize

    // Find the next valid trim point that:
    // 1. Starts with a user message (required by some models)
    // 2. Does not start with an orphaned toolResult
    // 3. Does not start with a toolUse unless its toolResult immediately follows
    while (trimIndex < messages.length) {
      const oldestMessage = messages[trimIndex]
      if (!oldestMessage) {
        break
      }

      // Must start with a user message
      if (oldestMessage.role !== 'user') {
        trimIndex++
        continue
      }

      // Cannot start with an orphaned toolResult
      const hasToolResult = oldestMessage.content.some((block) => block.type === 'toolResultBlock')
      if (hasToolResult) {
        trimIndex++
        continue
      }

      // toolUse is only valid if the next message is its toolResult
      const hasToolUse = oldestMessage.content.some((block) => block.type === 'toolUseBlock')
      if (hasToolUse) {
        const nextMessage = messages[trimIndex + 1]
        const nextHasToolResult = nextMessage && nextMessage.content.some((block) => block.type === 'toolResultBlock')
        if (!nextHasToolResult) {
          trimIndex++
          continue
        }
      }

      // Valid trim point found
      break
    }

    // If no valid trim point was found, return false and let the caller handle it.
    // When windowSize is 0, trimIndex === messages.length is expected (remove all), so allow it through.
    if (trimIndex > messages.length || (trimIndex === messages.length && this._windowSize > 0)) {
      logger.warn(
        `window_size=<${this._windowSize}>, messages=<${messages.length}> | unable to trim conversation context, no valid trim point found`
      )
      return false
    }

    // trimIndex is guaranteed to be < messages.length here, so splice always removes at least one message
    messages.splice(0, trimIndex)
    return true
  }

  /**
   * Apply head/tail truncation to a string if it exceeds the size threshold.
   *
   * Returns the truncated form (first {@link PRESERVE_CHARS} + marker + last
   * {@link PRESERVE_CHARS}) when the input exceeds {@link TRUNCATION_THRESHOLD},
   * otherwise `undefined`.
   */
  private _truncateLongText(text: string): string | undefined {
    if (text.length <= TRUNCATION_THRESHOLD) {
      return undefined
    }
    const prefix = text.slice(0, PRESERVE_CHARS)
    const suffix = text.slice(-PRESERVE_CHARS)
    const removed = text.length - 2 * PRESERVE_CHARS
    return `${prefix}\n<truncated chars="${removed}"/>\n${suffix}`
  }

  /**
   * Truncate tool result content in a message to reduce context size.
   *
   * Rule: preserve head/tail when the payload is plain-text-shaped; replace
   * wholesale when it's binary or remote. Specifically:
   * - Text blocks: partial head/tail truncation if over threshold.
   * - Image, Video blocks: wholesale replacement with a textual placeholder.
   * - Document blocks with bytes/s3 source: wholesale replacement.
   * - Document blocks with text source: partial truncation of the inner text.
   * - Document blocks with content source (TextBlock[]): partial truncation of
   *   each nested block.
   * - JSON blocks: wholesale replacement if serialized length is over threshold;
   *   mid-structure truncation would produce invalid JSON.
   *
   * The tool result `status` and `error` fields are preserved.
   *
   * @param messages - The conversation message history.
   * @param msgIdx - Index of the message containing tool results to truncate.
   * @returns True if any changes were made to the message, false otherwise.
   */
  private _truncateToolResults(messages: Message[], msgIdx: number): boolean {
    if (msgIdx >= messages.length || msgIdx < 0) {
      return false
    }

    const message = messages[msgIdx]
    if (!message) {
      return false
    }

    let changesMade = false
    const newContent = message.content.map((block) => {
      if (block.type !== 'toolResultBlock') {
        return block
      }

      const toolResultBlock = block as ToolResultBlock
      const newItems: ToolResultContent[] = []
      let itemChanged = false

      for (const item of toolResultBlock.content) {
        if (item.type === 'imageBlock') {
          newItems.push(new TextBlock(imagePlaceholder(item)))
          itemChanged = true
          continue
        }

        if (item.type === 'videoBlock') {
          newItems.push(new TextBlock(videoPlaceholder(item)))
          itemChanged = true
          continue
        }

        if (item.type === 'documentBlock') {
          const source = item.source
          if (source.type === 'documentSourceBytes' || source.type === 'documentSourceS3Location') {
            newItems.push(new TextBlock(documentPlaceholder(item)))
            itemChanged = true
            continue
          }
          if (source.type === 'documentSourceText') {
            const truncated = this._truncateLongText(source.text)
            if (truncated !== undefined) {
              newItems.push(
                new DocumentBlock({
                  name: item.name,
                  format: item.format,
                  source: { text: truncated },
                  ...(item.citations !== undefined ? { citations: item.citations } : {}),
                  ...(item.context !== undefined ? { context: item.context } : {}),
                })
              )
              itemChanged = true
              continue
            }
          }
          if (source.type === 'documentSourceContentBlock') {
            let nestedChanged = false
            const newContentBlocks = source.content.map((nested) => {
              const truncated = this._truncateLongText(nested.text)
              if (truncated !== undefined) {
                nestedChanged = true
                return new TextBlock(truncated)
              }
              return nested
            })
            if (nestedChanged) {
              newItems.push(
                new DocumentBlock({
                  name: item.name,
                  format: item.format,
                  source: { content: newContentBlocks },
                  ...(item.citations !== undefined ? { citations: item.citations } : {}),
                  ...(item.context !== undefined ? { context: item.context } : {}),
                })
              )
              itemChanged = true
              continue
            }
          }
          newItems.push(item)
          continue
        }

        if (item.type === 'jsonBlock') {
          const serializedLength = JSON.stringify(item.json).length
          if (serializedLength > TRUNCATION_THRESHOLD) {
            newItems.push(new TextBlock(jsonPlaceholder(serializedLength)))
            itemChanged = true
            continue
          }
          newItems.push(item)
          continue
        }

        if (item.type === 'textBlock') {
          const truncated = this._truncateLongText(item.text)
          if (truncated !== undefined) {
            newItems.push(new TextBlock(truncated))
            itemChanged = true
            continue
          }
        }

        newItems.push(item)
      }

      if (!itemChanged) {
        return block
      }

      changesMade = true
      return new ToolResultBlock({
        toolUseId: toolResultBlock.toolUseId,
        status: toolResultBlock.status,
        content: newItems,
        ...(toolResultBlock.error !== undefined ? { error: toolResultBlock.error } : {}),
      })
    })

    if (!changesMade) {
      return false
    }

    messages[msgIdx] = new Message({
      role: message.role,
      content: newContent,
    })

    return true
  }

  /**
   * Find the index of the oldest message containing tool results.
   *
   * Truncation targets the least-recent tool result first so the most relevant
   * recent context is preserved as long as possible.
   *
   * @param messages - The conversation message history.
   * @returns Index of the oldest message with tool results, or undefined if no such message exists.
   */
  private _findOldestMessageWithToolResults(messages: Message[]): number | undefined {
    for (let idx = 0; idx < messages.length; idx++) {
      const currentMessage = messages[idx]!

      const hasToolResult = currentMessage.content.some((block) => block.type === 'toolResultBlock')

      if (hasToolResult) {
        return idx
      }
    }

    return undefined
  }
}
