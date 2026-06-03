import Anthropic, { type ClientOptions } from '@anthropic-ai/sdk'
import {
  Model,
  type BaseModelConfig,
  type CountTokensOptions,
  type StreamOptions,
  resolveConfigMetadata,
} from '../models/model.js'
import type { Message, ContentBlock } from '../types/messages.js'
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

  private _formatRequest(messages: Message[], options?: StreamOptions): Anthropic.MessageStreamParams {
    if (!this._config.modelId) throw new Error('Model ID is required')

    const request: Anthropic.MessageStreamParams = {
      model: this._config.modelId,
      max_tokens: this._config.maxTokens ?? MODEL_DEFAULTS.anthropic.maxTokens,
      messages: this._formatMessages(messages),
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
            const cacheControl = nextBlock?.type === 'cachePointBlock' ? { type: 'ephemeral' as const } : undefined

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
      request.tools = options.toolSpecs.map((tool) => ({
        name: tool.name,
        description: tool.description,
        input_schema: tool.inputSchema as Anthropic.Tool.InputSchema,
      }))

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

  private _formatMessages(messages: Message[]): Anthropic.MessageParam[] {
    return messages.map((msg) => {
      const role = (msg.role as string) === 'tool' ? 'user' : msg.role

      const content: Anthropic.ContentBlockParam[] = []

      for (let i = 0; i < msg.content.length; i++) {
        const block = msg.content[i]
        if (!block) continue

        const nextBlock = msg.content[i + 1]
        const hasCachePoint = nextBlock?.type === 'cachePointBlock'

        const formattedBlock = this._formatContentBlock(block)

        if (formattedBlock) {
          if (hasCachePoint && this._isCacheableBlock(formattedBlock)) {
            formattedBlock.cache_control = { type: 'ephemeral' }
            i++
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
        if (block.text && block.signature) {
          return {
            type: 'thinking',
            thinking: block.text,
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
