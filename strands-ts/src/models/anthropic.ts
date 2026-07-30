import Anthropic, { type ClientOptions } from '@anthropic-ai/sdk'
import {
  Model,
  type BaseModelConfig,
  type CountTokensOptions,
  type StreamOptions,
  resolveConfigMetadata,
  type CacheConfig,
  type CacheToolsConfig,
} from '../models/model.js'
import type { Message, ContentBlock } from '../types/messages.js'
import type { ToolSpec } from '../tools/types.js'
import type { ModelStreamEvent } from '../models/streaming.js'
import { createEmptyUsage } from '../models/streaming.js'
import { ContextWindowOverflowError, ModelThrottledError, normalizeError } from '../errors.js'
import type { ImageBlock, DocumentBlock } from '../types/media.js'
import { encodeBase64 } from '../types/media.js'
import { logger } from '../logging/logger.js'
import { warnOnce } from '../logging/warn-once.js'
import { MODEL_DEFAULTS, defaultMaxTokensWarningMessage, defaultModelWarningMessage } from './defaults.js'

// Union of overflow phrases observed across Anthropic responses, matched
// case-insensitively. Kept in lowercase so the comparison is a single
// ``toLowerCase`` on the error message.
const CONTEXT_WINDOW_OVERFLOW_ERRORS = [
  'prompt is too long',
  'max_tokens exceeded',
  'input too long',
  'input is too long',
  'input length exceeds context window',
  'input and output tokens exceed your context limit',
]
/**
 * Block discriminators Anthropic accepts `cache_control` on. A cache point placed after any other
 * block (for example a reasoning block) is rejected by the API.
 *
 * @see https://docs.claude.com/en/docs/build-with-claude/prompt-caching
 */
const CACHEABLE_BLOCK_TYPES = ['textBlock', 'imageBlock', 'documentBlock', 'toolUseBlock', 'toolResultBlock']

/**
 * `ephemeral` is the only cache type the Anthropic API supports. Bedrock's cache point type
 * (e.g. `'default'`) has no Anthropic equivalent and is normalized to it.
 */
const ANTHROPIC_CACHE_TYPE = 'ephemeral' as const

const TEXT_FILE_FORMATS = ['txt', 'md', 'markdown', 'csv', 'json', 'xml', 'html', 'yml', 'yaml', 'js', 'ts', 'py']

export interface AnthropicModelConfig extends BaseModelConfig {
  /**
   * Maximum number of tokens the model can generate in a response.
   *
   * @defaultValue 64000 — subject to change between versions.
   * Set this explicitly to avoid unexpected changes.
   */
  maxTokens?: number
  stopSequences?: string[]
  params?: Record<string, unknown>

  /**
   * Beta features to enable via the `anthropic-beta` header.
   *
   * No header is sent by default. Provide a list of beta identifiers to opt into
   * features such as `interleaved-thinking-2025-05-14` or `mcp-client-2025-11-20`.
   *
   * @see https://docs.anthropic.com/en/api/beta-headers
   */
  betas?: string[]

  /**
   * Whether to use the native Anthropic countTokens API.
   *
   * When `true`, `countTokens()` calls the Anthropic token counting API for
   * accurate counts. When `false` or not set (default), skips the API call and uses
   * the character-based heuristic estimator.
   *
   * @defaultValue false
   */
  useNativeTokenCount?: boolean

  /**
   * Prompt caching configuration. When set, a cache breakpoint is added to the last user message so
   * the conversation prefix is read from cache instead of reprocessed. Caching is off when unset.
   *
   * Both strategies behave the same on this provider. `'auto'` carries a model-support check on
   * Bedrock, but the Anthropic API caches on every active Claude model, so there is nothing for that
   * check to decide here; `'anthropic'` is accepted so a config can move between providers unchanged.
   *
   * @example
   * ```typescript
   * new AnthropicModel({ cacheConfig: { strategy: 'auto', ttl: '1h' } })
   * ```
   */
  cacheConfig?: CacheConfig

  /**
   * Cache point applied to tool definitions. Pass a string (e.g. `'default'`) for the API default
   * TTL, or an object to set one. Independent of {@link AnthropicModelConfig.cacheConfig}.
   *
   * @example
   * ```typescript
   * new AnthropicModel({ cacheTools: { ttl: '1h' } })
   * ```
   */
  cacheTools?: string | CacheToolsConfig
}

export interface AnthropicModelOptions extends AnthropicModelConfig {
  apiKey?: string
  client?: Anthropic
  clientConfig?: ClientOptions
}

export class AnthropicModel extends Model<AnthropicModelConfig> {
  private _config: AnthropicModelConfig
  private _client: Anthropic

  constructor(options?: AnthropicModelOptions) {
    super()
    const { apiKey, client, clientConfig, ...modelConfig } = options || {}

    this._config = {
      modelId: MODEL_DEFAULTS.anthropic.modelId,
      maxTokens: MODEL_DEFAULTS.anthropic.maxTokens,
      ...modelConfig,
    }

    if (modelConfig.modelId === undefined) {
      warnOnce(logger, defaultModelWarningMessage(MODEL_DEFAULTS.anthropic.modelId))
    }

    if (modelConfig.maxTokens === undefined) {
      warnOnce(logger, defaultMaxTokensWarningMessage(MODEL_DEFAULTS.anthropic.maxTokens))
    }

    if (client) {
      this._client = client
    } else {
      const hasEnvKey =
        typeof process !== 'undefined' && typeof process.env !== 'undefined' && process.env.ANTHROPIC_API_KEY

      if (!apiKey && !hasEnvKey) {
        throw new Error(
          "Anthropic API key is required. Provide it via the 'apiKey' option or set the ANTHROPIC_API_KEY environment variable."
        )
      }

      this._client = new Anthropic({
        ...(apiKey ? { apiKey } : {}),
        ...clientConfig,
      })
    }
  }

  updateConfig(modelConfig: AnthropicModelConfig): void {
    this._config = { ...this._config, ...modelConfig }
  }

  getConfig(): AnthropicModelConfig {
    return resolveConfigMetadata(this._config, this._config.modelId ?? MODEL_DEFAULTS.anthropic.modelId)
  }

  /**
   * Count tokens using Anthropic's native countTokens API.
   *
   * Uses the same message format as the Messages API to get accurate token counts
   * directly from the Anthropic service. Falls back to the base class heuristic on failure.
   *
   * @param messages - Array of conversation messages to count tokens for
   * @param options - Optional options containing system prompt and tool specs
   * @returns Total input token count
   */
  override async countTokens(messages: Message[], options?: CountTokensOptions): Promise<number> {
    if (this._config.useNativeTokenCount !== true) return super.countTokens(messages, options)

    try {
      const request = this._formatRequest(messages, options)
      const params: Anthropic.MessageCountTokensParams = {
        model: request.model,
        messages: request.messages,
        ...(request.system && { system: request.system }),
        ...(request.tools && { tools: request.tools }),
        ...(request.tool_choice && { tool_choice: request.tool_choice }),
      }

      const requestOptions = this._buildRequestOptions()
      const response = requestOptions
        ? await this._client.messages.countTokens(params, requestOptions)
        : await this._client.messages.countTokens(params)

      logger.debug(`total_tokens=<${response.input_tokens}> | native token count`)
      return response.input_tokens
    } catch (error) {
      logger.debug(`error=<${error}> | native token counting failed, falling back to estimation`)
      return super.countTokens(messages, options)
    }
  }

  async *stream(messages: Message[], options?: StreamOptions): AsyncIterable<ModelStreamEvent> {
    try {
      const request = this._formatRequest(messages, options)
      const requestOptions = this._buildRequestOptions()
      const stream = requestOptions
        ? this._client.messages.stream(request, requestOptions)
        : this._client.messages.stream(request)

      const usage = createEmptyUsage()

      let stopReason = 'endTurn'

      for await (const event of stream) {
        switch (event.type) {
          case 'message_start': {
            usage.inputTokens = event.message.usage.input_tokens

            const rawUsage = event.message.usage as unknown as Record<string, number | undefined>
            if (rawUsage.cache_creation_input_tokens !== undefined) {
              usage.cacheWriteInputTokens = rawUsage.cache_creation_input_tokens
            }
            if (rawUsage.cache_read_input_tokens !== undefined) {
              usage.cacheReadInputTokens = rawUsage.cache_read_input_tokens
            }

            yield {
              type: 'modelMessageStartEvent',
              role: event.message.role,
            }
            break
          }

          case 'content_block_start':
            if (event.content_block.type === 'tool_use') {
              yield {
                type: 'modelContentBlockStartEvent',
                start: {
                  type: 'toolUseStart',
                  name: event.content_block.name,
                  toolUseId: event.content_block.id,
                },
              }
            } else if (event.content_block.type === 'thinking') {
              yield { type: 'modelContentBlockStartEvent' }
              if (event.content_block.thinking) {
                yield {
                  type: 'modelContentBlockDeltaEvent',
                  delta: {
                    type: 'reasoningContentDelta',
                    text: event.content_block.thinking,
                    signature: event.content_block.signature,
                  },
                }
              }
            } else if (event.content_block.type === 'redacted_thinking') {
              yield { type: 'modelContentBlockStartEvent' }
              yield {
                type: 'modelContentBlockDeltaEvent',
                delta: {
                  type: 'reasoningContentDelta',
                  redactedContent: event.content_block.data as unknown as Uint8Array,
                },
              }
            } else {
              yield { type: 'modelContentBlockStartEvent' }
              if (event.content_block.type === 'text' && event.content_block.text) {
                yield {
                  type: 'modelContentBlockDeltaEvent',
                  delta: { type: 'textDelta', text: event.content_block.text },
                }
              }
            }
            break

          case 'content_block_delta':
            if (event.delta.type === 'text_delta') {
              yield {
                type: 'modelContentBlockDeltaEvent',
                delta: { type: 'textDelta', text: event.delta.text },
              }
            } else if (event.delta.type === 'input_json_delta') {
              yield {
                type: 'modelContentBlockDeltaEvent',
                delta: { type: 'toolUseInputDelta', input: event.delta.partial_json },
              }
            } else if (event.delta.type === 'thinking_delta') {
              yield {
                type: 'modelContentBlockDeltaEvent',
                delta: { type: 'reasoningContentDelta', text: event.delta.thinking },
              }
            } else if (event.delta.type === 'signature_delta') {
              yield {
                type: 'modelContentBlockDeltaEvent',
                delta: { type: 'reasoningContentDelta', signature: event.delta.signature },
              }
            }
            break

          case 'content_block_stop':
            yield { type: 'modelContentBlockStopEvent' }
            break

          case 'message_delta':
            if (event.usage) {
              usage.outputTokens = event.usage.output_tokens
            }
            if (event.delta.stop_reason) {
              stopReason = this._mapStopReason(event.delta.stop_reason)
            }
            break

          case 'message_stop':
            usage.totalTokens = usage.inputTokens + usage.outputTokens
            yield {
              type: 'modelMetadataEvent',
              usage,
            }
            yield {
              type: 'modelMessageStopEvent',
              stopReason,
            }
            break
        }
      }
    } catch (unknownError) {
      const error = normalizeError(unknownError)

      const lowered = error.message.toLowerCase()
      if (CONTEXT_WINDOW_OVERFLOW_ERRORS.some((msg) => lowered.includes(msg))) {
        throw new ContextWindowOverflowError(error.message)
      }

      const err = unknownError as Error & { status?: number }
      if (err.status === 429) {
        const message = error.message ?? 'Request was throttled by the model provider'
        logger.debug(`throttled | error_message=<${message}>`)
        throw new ModelThrottledError(message, { cause: err })
      }

      throw error
    }
  }

  private _buildRequestOptions(): Anthropic.RequestOptions | undefined {
    const betas = this._config.betas
    if (!betas || betas.length === 0) return undefined
    return { headers: { 'anthropic-beta': betas.join(',') } }
  }

  /**
   * Whether `cacheConfig` asks for a cache breakpoint on the last user message.
   *
   * @returns True if a cache point should be applied to the messages.
   */
  private _cachingEnabled(): boolean {
    const strategy = this._config.cacheConfig?.strategy
    if (strategy === undefined) {
      return false
    }

    if (strategy !== 'auto' && strategy !== 'anthropic') {
      logger.warn(`strategy=<${strategy}> | unknown cache strategy, prompt caching disabled`)
      return false
    }

    return true
  }

  /**
   * Builds an Anthropic `cache_control` value.
   *
   * @param ttl - Optional TTL duration (e.g. `'5m'`, `'1h'`). Omitted leaves the API default.
   * @returns An Anthropic cache_control value.
   */
  private _formatCacheControl(ttl?: string): Anthropic.CacheControlEphemeral {
    const cacheControl: Anthropic.CacheControlEphemeral = { type: ANTHROPIC_CACHE_TYPE }
    if (ttl !== undefined) {
      // The API validates TTL values server-side, so accept any string here rather than requiring an
      // SDK bump whenever a new duration ships.
      cacheControl.ttl = ttl as NonNullable<Anthropic.CacheControlEphemeral['ttl']>
    }
    return cacheControl
  }

  /**
   * Locates the block that should carry the auto-injected cache breakpoint.
   *
   * Picks the last block Anthropic accepts `cache_control` on, in the last user message. Existing
   * cache point blocks are ignored, so the breakpoint count stays constant as a conversation grows
   * rather than accumulating one per turn.
   *
   * @param messages - Conversation messages to search.
   * @returns The target message and block index, or undefined when there is nothing cacheable.
   */
  private _findCacheTarget(messages: Message[]): { messageIndex: number; blockIndex: number } | undefined {
    for (let messageIndex = messages.length - 1; messageIndex >= 0; messageIndex--) {
      const message = messages[messageIndex]
      if (!message) continue

      // A tool result arrives with role 'tool' but is sent to Anthropic as a user turn.
      const role = (message.role as string) === 'tool' ? 'user' : message.role
      if (role !== 'user') continue

      for (let blockIndex = message.content.length - 1; blockIndex >= 0; blockIndex--) {
        const block = message.content[blockIndex]
        if (block && CACHEABLE_BLOCK_TYPES.includes(block.type)) {
          return { messageIndex, blockIndex }
        }
      }
    }

    logger.debug('no cacheable content block in a user message | skipped cache point')
    return undefined
  }

  /**
   * Formats tool definitions, caching them when `cacheTools` is configured.
   *
   * A `cache_control` on the final tool caches the whole tool block, so one breakpoint is enough.
   *
   * @param toolSpecs - Tool specifications to make available to the model.
   * @returns An Anthropic tools array.
   */
  private _formatTools(toolSpecs: ToolSpec[]): Anthropic.ToolUnion[] {
    const tools = toolSpecs.map((tool) => ({
      name: tool.name,
      description: tool.description,
      input_schema: tool.inputSchema as Anthropic.Tool.InputSchema,
    })) as Anthropic.Tool[]

    const cacheTools = this._config.cacheTools
    const lastTool = tools[tools.length - 1]
    if (cacheTools !== undefined && lastTool) {
      const ttl = typeof cacheTools === 'string' ? undefined : cacheTools.ttl
      lastTool.cache_control = this._formatCacheControl(ttl)
    }

    return tools
  }

  private _formatRequest(messages: Message[], options?: StreamOptions): Anthropic.MessageStreamParams {
    if (!this._config.modelId) throw new Error('Model ID is required')

    const request: Anthropic.MessageStreamParams = {
      model: this._config.modelId,
      max_tokens: this._config.maxTokens ?? MODEL_DEFAULTS.anthropic.maxTokens,
      messages: this._formatMessages(messages, this._cachingEnabled() ? this._findCacheTarget(messages) : undefined),
      stream: true,
    }

    if (options?.systemPrompt) {
      if (typeof options.systemPrompt === 'string') {
        request.system = options.systemPrompt
      } else if (Array.isArray(options.systemPrompt)) {
        const systemBlocks: Anthropic.TextBlockParam[] = []
        for (let i = 0; i < options.systemPrompt.length; i++) {
          const block = options.systemPrompt[i]
          if (!block) continue

          if (block.type === 'textBlock') {
            const nextBlock = options.systemPrompt[i + 1]
            const cacheControl =
              nextBlock?.type === 'cachePointBlock' ? this._formatCacheControl(nextBlock.ttl) : undefined

            systemBlocks.push({
              type: 'text',
              text: block.text,
              ...(cacheControl && { cache_control: cacheControl }),
            })

            if (cacheControl) i++
          } else if (block.type === 'guardContentBlock') {
            logger.warn(
              'block_type=<guardContentBlock> | guard content not supported in anthropic system prompt | skipping'
            )
          }
        }
        if (systemBlocks.length > 0) request.system = systemBlocks
      }
    }

    if (options?.toolSpecs?.length) {
      request.tools = this._formatTools(options.toolSpecs)

      if (options.toolChoice) {
        if ('auto' in options.toolChoice) {
          request.tool_choice = { type: 'auto' }
        } else if ('any' in options.toolChoice) {
          request.tool_choice = { type: 'any' }
        } else if ('tool' in options.toolChoice) {
          request.tool_choice = { type: 'tool', name: options.toolChoice.tool.name }
        }
      }
    }

    if (this._config.temperature !== undefined) request.temperature = this._config.temperature
    if (this._config.topP !== undefined) request.top_p = this._config.topP
    if (this._config.stopSequences !== undefined) request.stop_sequences = this._config.stopSequences
    if (this._config.params) Object.assign(request, this._config.params)

    return request
  }

  private _formatMessages(
    messages: Message[],
    cacheTarget?: { messageIndex: number; blockIndex: number }
  ): Anthropic.MessageParam[] {
    return messages.map((msg, messageIndex) => {
      const role = (msg.role as string) === 'tool' ? 'user' : msg.role

      const content: Anthropic.ContentBlockParam[] = []

      for (let i = 0; i < msg.content.length; i++) {
        const block = msg.content[i]
        if (!block) continue

        // While cacheConfig manages placement it owns every message breakpoint, so hand-placed cache
        // points are dropped instead of adding to the count.
        if (cacheTarget && block.type === 'cachePointBlock') continue

        const nextBlock = msg.content[i + 1]
        const cachePointTTL = nextBlock?.type === 'cachePointBlock' ? nextBlock.ttl : undefined
        const hasCachePoint = !cacheTarget && nextBlock?.type === 'cachePointBlock'
        const isCacheTarget = cacheTarget?.messageIndex === messageIndex && cacheTarget.blockIndex === i

        const formattedBlock = this._formatContentBlock(block)

        if (formattedBlock) {
          if (hasCachePoint && this._isCacheableBlock(formattedBlock)) {
            formattedBlock.cache_control = this._formatCacheControl(cachePointTTL)
            i++
          } else if (isCacheTarget && this._isCacheableBlock(formattedBlock)) {
            formattedBlock.cache_control = this._formatCacheControl(this._config.cacheConfig?.ttl)
          }
          content.push(formattedBlock)
        }
      }

      return {
        role: role as 'user' | 'assistant',
        content,
      }
    })
  }

  private _isCacheableBlock(
    block: Anthropic.ContentBlockParam | Anthropic.ToolResultBlockParam
  ): block is (
    | Anthropic.TextBlockParam
    | Anthropic.ImageBlockParam
    | Anthropic.ToolUseBlockParam
    | Anthropic.ToolResultBlockParam
    | Anthropic.DocumentBlockParam
  ) & { cache_control?: { type: 'ephemeral' } } {
    return ['text', 'image', 'tool_use', 'tool_result', 'document'].includes(block.type)
  }

  private _formatContentBlock(
    block: ContentBlock
  ): Anthropic.ContentBlockParam | Anthropic.ToolResultBlockParam | undefined {
    switch (block.type) {
      case 'textBlock':
        return { type: 'text', text: block.text }

      case 'imageBlock': {
        const imgBlock = block as ImageBlock
        let mediaType: 'image/jpeg' | 'image/png' | 'image/gif' | 'image/webp'

        switch (imgBlock.format) {
          case 'jpeg':
          case 'jpg':
            mediaType = 'image/jpeg'
            break
          case 'png':
            mediaType = 'image/png'
            break
          case 'gif':
            mediaType = 'image/gif'
            break
          case 'webp':
            mediaType = 'image/webp'
            break
          default:
            throw new Error(`Unsupported image format for Anthropic: ${imgBlock.format}`)
        }

        if (imgBlock.source.type === 'imageSourceBytes') {
          return {
            type: 'image',
            source: {
              type: 'base64',
              media_type: mediaType,
              data: encodeBase64(imgBlock.source.bytes),
            },
          }
        }
        logger.warn('source_type=<imageSourceUrl> | anthropic requires image bytes | url sources not fully supported')
        return undefined
      }

      case 'documentBlock': {
        const docBlock = block as DocumentBlock

        if (docBlock.format === 'pdf' && docBlock.source.type === 'documentSourceBytes') {
          return {
            type: 'document',
            source: {
              type: 'base64',
              media_type: 'application/pdf',
              data: encodeBase64(docBlock.source.bytes),
            },
            ...(docBlock.name && { title: docBlock.name }),
          } as unknown as Anthropic.ContentBlockParam
        }

        if (TEXT_FILE_FORMATS.includes(docBlock.format)) {
          let textContent: string | undefined

          if (docBlock.source.type === 'documentSourceText') {
            textContent = docBlock.source.text
          } else if (docBlock.source.type === 'documentSourceBytes') {
            if (typeof TextDecoder !== 'undefined') {
              textContent = new TextDecoder().decode(docBlock.source.bytes)
            } else {
              logger.warn(`format=<${docBlock.format}> | cannot decode document bytes | TextDecoder not available`)
            }
          }

          if (textContent) {
            return {
              type: 'text',
              text: textContent,
            }
          }
        }

        logger.warn(`format=<${docBlock.format}> | unsupported document format or source for anthropic`)
        return undefined
      }

      case 'toolUseBlock':
        return {
          type: 'tool_use',
          id: block.toolUseId,
          name: block.name,
          input: block.input as Record<string, unknown>,
        }

      case 'videoBlock':
        logger.warn('block_type=<videoBlock> | video blocks not supported by anthropic, skipping')
        return undefined

      case 'toolResultBlock': {
        const innerContent = block.content
          .map((c) => {
            if (c.type === 'textBlock') return { type: 'text' as const, text: c.text }
            if (c.type === 'jsonBlock') return { type: 'text' as const, text: JSON.stringify(c.json) }

            // Recursively format any other content block (image, document, video, etc.)
            const formatted = this._formatContentBlock(c as unknown as ContentBlock)
            return formatted
          })
          .filter((c): c is NonNullable<typeof c> => !!c)

        let contentVal: string | Anthropic.ContentBlockParam[]

        const firstItem = innerContent[0]
        if (innerContent.length === 1 && firstItem && firstItem.type === 'text') {
          contentVal = firstItem.text
        } else {
          contentVal = innerContent
        }

        return {
          type: 'tool_result',
          tool_use_id: block.toolUseId,
          content: contentVal,
          is_error: block.status === 'error',
        } as Anthropic.ToolResultBlockParam
      }

      case 'reasoningBlock':
        if (block.signature) {
          return {
            type: 'thinking',
            thinking: block.text ?? '',
            signature: block.signature,
          } as unknown as Anthropic.ContentBlockParam
        } else if (block.redactedContent) {
          return {
            type: 'redacted_thinking',
            data: block.redactedContent,
          } as unknown as Anthropic.ContentBlockParam
        }
        return undefined

      case 'cachePointBlock':
        return undefined

      default:
        return undefined
    }
  }

  private _mapStopReason(anthropicReason: string): string {
    switch (anthropicReason) {
      case 'end_turn':
        return 'endTurn'
      case 'max_tokens':
        return 'maxTokens'
      case 'stop_sequence':
        return 'stopSequence'
      case 'tool_use':
        return 'toolUse'
      case 'pause_turn':
        return 'pauseTurn'
      case 'refusal':
        return 'refusal'
      default:
        logger.warn(`stop_reason=<${anthropicReason}> | unknown anthropic stop reason`)
        return anthropicReason
    }
  }
}
