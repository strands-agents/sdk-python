/** `InvokableAgent` adapter for the AgentCore `InvokeHarness` API. */

import {
  BedrockAgentCoreClient,
  InvokeHarnessCommand,
  type BedrockAgentCoreClientConfig,
  type InvokeHarnessStreamOutput,
} from '@aws-sdk/client-bedrock-agentcore'
import type { InvokableAgent, InvokeArgs, InvokeOptions } from '../types/agent.js'
import { AgentResult } from '../types/agent.js'
import { Message, contentBlockFromData } from '../types/messages.js'
import type { ContentBlock, ContentBlockData, MessageData, TextBlock } from '../types/messages.js'
import { isInterruptResponseContent } from '../types/interrupt.js'
import {
  ConcurrentInvocationError,
  ContextWindowOverflowError,
  MaxTokensError,
  ModelError,
  ModelThrottledError,
  normalizeError,
} from '../errors.js'
import {
  AgentCoreHarnessResultEvent,
  AgentCoreHarnessStreamUpdateEvent,
  type AgentCoreHarnessEventData,
  type AgentCoreHarnessStreamEvent,
} from './events.js'
import { HarnessStreamDecoder, type DecodedHarnessInvocation } from './stream-decoder.js'

const THROTTLING_ERROR_NAMES = new Set(['ThrottlingException', 'ThrottledException'])
const CONTEXT_WINDOW_OVERFLOW_ERRORS = [
  'input is too long for requested model',
  'input length and `max_tokens` exceed context limit',
  'too many total text bytes',
  'prompt is too long',
]

/**
 * Configuration for {@link AgentCoreHarnessAgent}.
 */
export interface AgentCoreHarnessAgentConfig {
  /** ARN of the deployed AgentCore Harness. */
  harnessArn: string
  /** Conversation session ID. Must be 33-100 Harness-compatible characters. */
  runtimeSessionId: string
  /** Optional unique identifier. Defaults to the Harness ARN and runtime session ID. */
  id?: string
  /** AgentCore client to use. Takes precedence over `clientConfig`. */
  client?: BedrockAgentCoreClient
  /** AgentCore client configuration. Uses standard AWS region resolution; `maxAttempts` defaults to 1. */
  clientConfig?: BedrockAgentCoreClientConfig
}

/**
 * Adapts one deployed AgentCore Harness to the Strands {@link InvokableAgent} interface.
 *
 * Each `invoke()` or `stream()` call sends one non-empty text message through `InvokeHarness`.
 * Text blocks are joined with newlines. Message-history input uses the latest user message.
 * Non-text content, interrupt responses, and checkpoint resumes are rejected before the request.
 * The deployed Harness owns its model, prompt, tools, skills, limits, and agent loop.
 * Tool execution and continuation must happen in the deployed Harness; this adapter does not
 * execute or continue inline tool calls locally.
 *
 * @example
 * ```typescript
 * import { randomUUID } from 'node:crypto'
 * import { AgentCoreHarnessAgent } from '@strands-agents/sdk/agentcore-harness'
 *
 * const agent = new AgentCoreHarnessAgent({
 *   harnessArn: process.env.AGENTCORE_HARNESS_ARN!,
 *   runtimeSessionId: randomUUID(),
 * })
 *
 * const result = await agent.invoke('Summarize the worktree.')
 * ```
 */
export class AgentCoreHarnessAgent implements InvokableAgent {
  private readonly _client: BedrockAgentCoreClient
  private _isInvoking = false

  /** ARN of the deployed Harness. */
  readonly harnessArn: string

  /** Session ID used by the Harness-owned conversation. */
  readonly runtimeSessionId: string

  /** Identifier unique to this Harness session. */
  readonly id: string

  /**
   * Creates a Harness adapter.
   *
   * @param config - Harness identity, session, and optional AgentCore client configuration
   */
  constructor(config: AgentCoreHarnessAgentConfig) {
    this.harnessArn = config.harnessArn
    this.runtimeSessionId = config.runtimeSessionId
    this.id = config.id ?? `${config.harnessArn}:${config.runtimeSessionId}`
    this._client =
      config.client ??
      new BedrockAgentCoreClient({
        ...config.clientConfig,
        maxAttempts: config.clientConfig?.maxAttempts ?? 1,
      })
  }

  /**
   * Invokes the deployed Harness and returns its final result.
   *
   * @param args - Non-empty text as a string, text blocks, or message history
   * @param options - Cancellation and invocation state; other options are unsupported
   * @returns Final Harness result
   * @throws {@link ConcurrentInvocationError} If this instance is already processing an invocation
   * @throws TypeError If input or invocation options are unsupported
   * @throws {@link ModelThrottledError} If AgentCore reports throttling
   * @throws {@link ModelError} If AgentCore or the Harness stream fails
   */
  async invoke(args: InvokeArgs, options?: InvokeOptions): Promise<AgentResult> {
    const generator = this.stream(args, options)
    let next = await generator.next()
    while (!next.done) {
      next = await generator.next()
    }
    return next.value
  }

  /**
   * Streams one `InvokeHarness` request.
   *
   * @param args - Non-empty text as a string, text blocks, or message history
   * @param options - Cancellation and invocation state; other options are unsupported
   * @returns Raw Harness events followed by the final result event and generator return value
   * @throws {@link ConcurrentInvocationError} If this instance is already processing an invocation
   * @throws TypeError If input or invocation options are unsupported
   * @throws {@link ModelThrottledError} If AgentCore reports throttling
   * @throws {@link ModelError} If AgentCore or the Harness stream fails
   */
  async *stream(
    args: InvokeArgs,
    options?: InvokeOptions
  ): AsyncGenerator<AgentCoreHarnessStreamEvent, AgentResult, undefined> {
    this._acquireLock()
    try {
      return yield* this._stream(args, options)
    } finally {
      this._isInvoking = false
    }
  }

  private async *_stream(
    args: InvokeArgs,
    options?: InvokeOptions
  ): AsyncGenerator<AgentCoreHarnessStreamEvent, AgentResult, undefined> {
    const text = textFromInvokeArgs(args)
    assertSupportedOptions(options)

    const invocationState = options?.invocationState ?? {}
    const cancelSignal = options?.cancelSignal
    const decoder = new HarnessStreamDecoder()

    if (cancelSignal?.aborted) {
      const result = createCancelledResult(decoder, invocationState)
      yield new AgentCoreHarnessResultEvent({ result })
      return result
    }

    try {
      const response = await this._client.send(
        new InvokeHarnessCommand({
          harnessArn: this.harnessArn,
          runtimeSessionId: this.runtimeSessionId,
          messages: [{ role: 'user', content: [{ text }] }],
        }),
        cancelSignal ? { abortSignal: cancelSignal } : {}
      )
      if (response.stream) {
        for await (const chunk of response.stream) {
          if (cancelSignal?.aborted) break

          const streamError = harnessStreamError(chunk)
          if (streamError) throw streamError

          const event = chunk as AgentCoreHarnessEventData
          decoder.accept(event)
          yield new AgentCoreHarnessStreamUpdateEvent(event)
        }
      }
    } catch (error) {
      if (cancelSignal?.aborted) {
        const result = createCancelledResult(decoder, invocationState)
        yield new AgentCoreHarnessResultEvent({ result })
        return result
      }
      throw normalizeHarnessError(error)
    }

    if (cancelSignal?.aborted) {
      const result = createCancelledResult(decoder, invocationState)
      yield new AgentCoreHarnessResultEvent({ result })
      return result
    }

    const decoded = decoder.complete()
    throwIfUnrecoverable(decoded)
    const result = new AgentResult({
      stopReason: decoded.stopReason,
      lastMessage: decoded.message,
      invocationState,
    })
    yield new AgentCoreHarnessResultEvent({ result })
    return result
  }

  private _acquireLock(): void {
    if (this._isInvoking) {
      throw new ConcurrentInvocationError(
        'AgentCoreHarnessAgent is already processing an invocation. Wait for the current invoke() or stream() call to complete before invoking again.'
      )
    }
    this._isInvoking = true
  }
}

function textFromInvokeArgs(args: InvokeArgs): string {
  if (typeof args === 'string') {
    return requireNonEmptyText(args)
  }

  if (!Array.isArray(args)) {
    throw new TypeError('AgentCoreHarnessAgent does not support checkpoint resume input.')
  }
  if (args.length === 0) {
    throw new TypeError('AgentCoreHarnessAgent input must contain non-empty text.')
  }

  const first = args[0]!
  if (isInterruptResponseContent(first)) {
    throw new TypeError('AgentCoreHarnessAgent does not support interrupt response input.')
  }

  if ('role' in first) {
    const messages = (args as (Message | MessageData)[]).map((message) =>
      message instanceof Message ? message : Message.fromMessageData(message)
    )
    const latestUserMessage = messages
      .slice()
      .reverse()
      .find((message) => message.role === 'user')
    if (latestUserMessage === undefined) {
      throw new TypeError('AgentCoreHarnessAgent message input must include at least one user message.')
    }
    return textFromContentBlocks(latestUserMessage.content)
  }

  const blocks = (args as (ContentBlock | ContentBlockData)[]).map((block) =>
    'type' in block ? block : contentBlockFromData(block)
  )
  return textFromContentBlocks(blocks)
}

function textFromContentBlocks(blocks: ContentBlock[]): string {
  const textBlocks = blocks.filter((block): block is TextBlock => block.type === 'textBlock')
  const unsupportedTypes = blocks.filter((block) => block.type !== 'textBlock').map((block) => block.type)
  if (unsupportedTypes.length > 0) {
    throw new TypeError(
      `AgentCoreHarnessAgent accepts only text content blocks; received ${unsupportedTypes.join(', ')}.`
    )
  }
  return requireNonEmptyText(textBlocks.map((block) => block.text).join('\n'))
}

function requireNonEmptyText(text: string): string {
  if (text.length === 0) {
    throw new TypeError('AgentCoreHarnessAgent input must contain non-empty text.')
  }
  return text
}

function assertSupportedOptions(options: InvokeOptions | undefined): void {
  if (options?.structuredOutputSchema !== undefined) {
    throw new TypeError('InvokeOptions.structuredOutputSchema is not supported by AgentCoreHarnessAgent.')
  }
  if (options?.limits !== undefined) {
    throw new TypeError('InvokeOptions.limits is not supported by AgentCoreHarnessAgent.')
  }
}

function createCancelledResult(
  decoder: HarnessStreamDecoder,
  invocationState: NonNullable<InvokeOptions['invocationState']>
): AgentResult {
  return new AgentResult({
    stopReason: 'cancelled',
    lastMessage: decoder.partialMessage(),
    invocationState,
  })
}

function throwIfUnrecoverable(decoded: DecodedHarnessInvocation): void {
  // A context-window stop reason is a completed Harness turn; request and stream failures are normalized separately.
  switch (decoded.stopReason) {
    case 'maxTokens':
      throw new MaxTokensError(
        'Model reached maximum token limit. This is an unrecoverable state that requires intervention.',
        decoded.message
      )
    case 'interrupted':
    case 'malformedModelOutput':
    case 'malformedToolUse':
      throw new ModelError(`Harness ended the turn with an unrecoverable stop reason: ${decoded.stopReason}`)
  }
  if (decoded.toolInputParseError) {
    throw new ModelError('Unable to parse Harness tool input JSON.', { cause: decoded.toolInputParseError })
  }
}

function harnessStreamError(chunk: InvokeHarnessStreamOutput): ModelError | undefined {
  if ('internalServerException' in chunk && chunk.internalServerException) {
    return new ModelError(chunk.internalServerException.message ?? 'AgentCore Harness internal server error', {
      cause: chunk.internalServerException,
    })
  }
  if ('validationException' in chunk && chunk.validationException) {
    const message = chunk.validationException.message ?? 'AgentCore Harness validation error'
    if (isContextWindowOverflow(message)) {
      return new ContextWindowOverflowError(message, { cause: chunk.validationException })
    }
    return new ModelError(message, { cause: chunk.validationException })
  }
  if ('runtimeClientError' in chunk && chunk.runtimeClientError) {
    return new ModelError(chunk.runtimeClientError.message ?? 'AgentCore Harness runtime client error', {
      cause: chunk.runtimeClientError,
    })
  }
  return undefined
}

function normalizeHarnessError(error: unknown): Error {
  if (error instanceof ModelError) return error

  const normalized = normalizeError(error)
  if (THROTTLING_ERROR_NAMES.has(normalized.name)) {
    return new ModelThrottledError(normalized.message, { cause: error })
  }
  if (isContextWindowOverflow(normalized.message)) {
    return new ContextWindowOverflowError(normalized.message, { cause: error })
  }
  return new ModelError(normalized.message, { cause: error })
}

function isContextWindowOverflow(message: string): boolean {
  const normalized = message.toLowerCase()
  return CONTEXT_WINDOW_OVERFLOW_ERRORS.some((phrase) => normalized.includes(phrase))
}
