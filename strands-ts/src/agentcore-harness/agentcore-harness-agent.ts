/** InvokableAgent wrapper for the AgentCore Harness API. */

import {
  type BedrockAgentCoreClient,
  type BedrockAgentCoreClientConfig,
  InvokeHarnessCommand,
  type InvokeHarnessCommandInput,
  type HarnessModelConfiguration,
  type HarnessSkill,
  type HarnessMessage,
  type HarnessTool,
  type InvokeHarnessStreamOutput,
} from '@aws-sdk/client-bedrock-agentcore'
import {
  type BedrockAgentCoreControlClient,
  type BedrockAgentCoreControlClientConfig,
  GetHarnessCommand,
  GetHarnessEndpointCommand,
} from '@aws-sdk/client-bedrock-agentcore-control'
import type { InvokableAgent, InvokeArgs, InvokeOptions } from '../types/agent.js'
import { AgentResult } from '../types/agent.js'
import type { ToolExecutorStrategy } from '../agent/agent.js'
import { AgentMetrics } from '../telemetry/meter.js'
import { accumulateUsage, createEmptyUsage, type Usage } from '../models/streaming.js'
import { Message, TextBlock, ToolResultBlock, type StopReason } from '../types/messages.js'
import { createErrorResult, createSuccessResult, type InvokableTool, type Tool } from '../tools/tool.js'
import type { ToolUse } from '../tools/types.js'
import {
  ConcurrentInvocationError,
  ContextWindowOverflowError,
  MaxTokensError,
  ModelError,
  ModelThrottledError,
  ToolValidationError,
  normalizeError,
} from '../errors.js'
import { logger } from '../logging/logger.js'
import { ToolRegistry } from '../registry/tool-registry.js'
import {
  AgentCoreHarnessResultEvent,
  AgentCoreHarnessStreamUpdateEvent,
  type AgentCoreHarnessEventData,
  type AgentCoreHarnessStreamEvent,
} from './events.js'
import {
  formatHarnessInput,
  formatHarnessMessage,
  formatHarnessSystemPrompt,
  formatHarnessTools,
} from './request-formatting.js'
import { HarnessStreamDecoder } from './stream-decoder.js'
import { assertHostToolsAllowed, isHostToolAllowed, mergeHarnessTools } from './tool-configuration.js'
import { createHarnessClient, createHarnessControlClient } from './clients.js'

/** Provider messages that the harness surfaces for context-window overflows. */
const CONTEXT_WINDOW_OVERFLOW_ERRORS = [
  'input is too long for requested model',
  'input length and `max_tokens` exceed context limit',
  'too many total text bytes',
  'prompt is too long',
]

const RUNTIME_SESSION_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_-]*$/

const DEFAULT_HARNESS_QUALIFIER = 'DEFAULT'

// Always-on harness tools and the namespace that contains them.
const RESERVED_HARNESS_TOOL_NAMES = new Set(['builtin', 'shell', 'file_operations'])

const THROTTLING_ERROR_NAMES = new Set(['ThrottlingException', 'ThrottledException'])

interface ResolvedToolConfiguration {
  tools: HarnessTool[]
  allowedTools?: string[]
}

interface CachedVersionedToolConfiguration {
  harnessVersion: string
  toolConfiguration: ResolvedToolConfiguration
}

type HarnessRequestConfiguration = Pick<
  InvokeHarnessCommandInput,
  | 'model'
  | 'systemPrompt'
  | 'skills'
  | 'allowedTools'
  | 'maxIterations'
  | 'maxTokens'
  | 'timeoutSeconds'
  | 'qualifier'
  | 'runtimeUserId'
  | 'actorId'
>

/**
 * Configuration for {@link AgentCoreHarnessAgent}.
 *
 * Harness request overrides are snapshotted during construction and apply to every invocation.
 * Create another agent instance when a conversation needs different request configuration.
 */
export interface AgentCoreHarnessAgentConfig {
  /** ARN of the harness to invoke. */
  harnessArn: string
  /**
   * Conversation session ID. Must be 33-100 characters and match `[A-Za-z0-9][A-Za-z0-9_-]*`.
   * Reuse it to continue a conversation; a UUID is valid.
   */
  runtimeSessionId: string
  /**
   * Local tools exposed to the harness as inline functions. Only {@link InvokableTool} is supported,
   * and tools run without a `ToolContext` or the local tool lifecycle. They cannot use invocation
   * state, hooks, interventions, middleware, progress, local tracing or metrics, observe
   * cancellation, or use local agent, sandbox, and app state. Requires
   * `bedrock-agentcore:GetHarness` so deployed tools can be preserved. Named endpoints also require
   * `bedrock-agentcore:GetHarnessEndpoint` on the Harness and endpoint. A local tool binds to a
   * same-name deployed inline function only when its description and input schema match.
   */
  tools?: InvokableTool<unknown, unknown>[]
  /**
   * Host tool execution strategy. Defaults to `'concurrent'`; use `'sequential'` for tools with
   * ordering dependencies or shared side effects.
   */
  toolExecutor?: ToolExecutorStrategy
  /**
   * Harness model override. This is wire configuration, not a local Strands `Model`.
   */
  modelConfig?: HarnessModelConfiguration
  /** Text-only system prompt override. Omit to use the harness's default. */
  systemPrompt?: string | TextBlock[]
  /** Skills override. Omit to use the harness's configured skills. */
  skills?: HarnessSkill[]
  /**
   * Tool allowlist override. Omit to preserve the deployed allowlist; an empty array disables all
   * tools. Host inline functions use the Harness namespace syntax: for a tool named `get_weather`,
   * use `@get_weather`, `@get_weather/get_weather`, a matching namespace glob, or `*`.
   */
  allowedTools?: string[]
  /** Maximum server-side loop iterations. */
  maxIterations?: number
  /** Maximum model output tokens per server-side iteration. */
  maxTokens?: number
  /** Maximum server-side loop duration in seconds. */
  timeoutSeconds?: number
  /**
   * Harness endpoint alias. Omit or use `DEFAULT` for the current Harness. Named endpoints allow
   * host-tool configuration to be bound to their serving version.
   */
  qualifier?: string
  /** End-user identity passed through to the runtime container. Omit if not needed. */
  runtimeUserId?: string
  /** Actor ID override for harness memory operations. */
  actorId?: string
  /** AWS region for constructed clients. Takes precedence over client-specific configuration. */
  region?: string
  /**
   * Data-plane client configuration. Constructed clients default to `maxAttempts: 1` because
   * `InvokeHarness` is stateful and has no idempotency token. Set `maxAttempts` to opt into retries.
   */
  clientConfig?: BedrockAgentCoreClientConfig
  /** Injected data-plane client, used with its existing retry configuration. */
  client?: BedrockAgentCoreClient
  /** Control-plane client configuration used when host tools are present. Reads retain AWS SDK retries. */
  controlClientConfig?: BedrockAgentCoreControlClientConfig
  /** Injected control-plane client. */
  controlClient?: BedrockAgentCoreControlClient
  /** Optional unique identifier. Defaults to the harness ARN. */
  id?: string
  /** Optional name. */
  name?: string
  /** Optional description. */
  description?: string
}

/**
 * Wraps a managed AgentCore Harness as an {@link InvokableAgent}.
 *
 * The harness owns the agent loop, model, memory, and deployed tools. Custom tools can run locally
 * through the Harness return-of-control flow.
 *
 * @example
 * ```typescript
 * import { randomUUID } from 'node:crypto'
 * import { AgentCoreHarnessAgent } from '@strands-agents/sdk/agentcore-harness'
 *
 * const harnessArn = process.env.AGENTCORE_HARNESS_ARN
 * if (!harnessArn) throw new Error('Set AGENTCORE_HARNESS_ARN')
 *
 * const agent = new AgentCoreHarnessAgent({
 *   harnessArn,
 *   runtimeSessionId: randomUUID(),
 * })
 * const result = await agent.invoke('Summarize the tools available in this harness.')
 * ```
 */
export class AgentCoreHarnessAgent implements InvokableAgent {
  private readonly _requestConfig: HarnessRequestConfiguration
  private readonly _region: string | undefined
  private readonly _clientConfig: BedrockAgentCoreClientConfig | undefined
  private readonly _controlClientConfig: BedrockAgentCoreControlClientConfig | undefined
  private _client: BedrockAgentCoreClient | undefined
  private _controlClient: BedrockAgentCoreControlClient | undefined
  private readonly _hostToolRegistry: ToolRegistry
  private readonly _toolExecutor: ToolExecutorStrategy
  private _isInvoking = false
  private _cachedVersionedToolConfiguration: CachedVersionedToolConfiguration | undefined

  /** The resource ARN for the managed harness. */
  readonly harnessArn: string

  /** The runtime session ID identifying the conversation. */
  readonly runtimeSessionId: string

  /** The unique identifier of the agent instance. */
  readonly id: string

  /** The name of the agent. */
  readonly name?: string

  /** Optional description of what the agent does. */
  readonly description?: string

  /**
   * Creates an agent for a managed harness.
   *
   * Later mutation of `config` or its nested request values does not change this agent.
   *
   * @param config - Harness, session, request, client, and host-tool configuration
   * @throws {@link ToolValidationError} If a host tool is invalid, duplicated, reserved, or explicitly disallowed
   * @throws TypeError If a request override is invalid or cannot be snapshotted
   * @throws Error If the runtime session ID does not satisfy the Harness contract
   */
  constructor(config: AgentCoreHarnessAgentConfig) {
    if (config.runtimeSessionId.length < 33 || config.runtimeSessionId.length > 100) {
      throw new Error(
        `runtimeSessionId must be 33-100 characters, got ${config.runtimeSessionId.length}. The harness requires this length to identify a session.`
      )
    }
    if (!RUNTIME_SESSION_ID_PATTERN.test(config.runtimeSessionId)) {
      throw new Error(
        'runtimeSessionId must start with an alphanumeric character and contain only alphanumeric characters, hyphens, and underscores.'
      )
    }
    this._toolExecutor = resolveToolExecutor(config.toolExecutor)
    this._requestConfig = snapshotRequestConfiguration(config)
    this._region = config.region
    this._clientConfig = config.clientConfig === undefined ? undefined : { ...config.clientConfig }
    this._controlClientConfig = config.controlClientConfig === undefined ? undefined : { ...config.controlClientConfig }
    this._client = config.client
    this._controlClient = config.controlClient
    this.harnessArn = config.harnessArn
    this.runtimeSessionId = config.runtimeSessionId
    this.id = config.id ?? config.harnessArn
    if (config.name !== undefined) this.name = config.name
    if (config.description !== undefined) this.description = config.description
    for (const hostTool of config.tools ?? []) {
      if (!isInvokableTool(hostTool)) {
        throw new ToolValidationError(
          `Host tool '${hostTool.name}' must implement invoke(). AgentCoreHarnessAgent does not support stream-only tools.`
        )
      }
    }
    this._hostToolRegistry = new ToolRegistry(config.tools)
    const reservedHostTool = this._hostToolRegistry
      .list()
      .find((hostTool) => RESERVED_HARNESS_TOOL_NAMES.has(hostTool.name))
    if (reservedHostTool !== undefined) {
      throw new ToolValidationError(`Host tool name '${reservedHostTool.name}' is reserved by AgentCore Harness.`)
    }
    assertHostToolsAllowed(this._hostToolRegistry.list(), this._requestConfig.allowedTools)
  }

  /**
   * Invokes the harness and returns the final result.
   * See {@link stream} for supported options and event behavior.
   *
   * @param args - New input for the Harness-owned conversation
   * @param options - Cancellation, invocation state, and other shared invocation options
   * @returns The final Harness response and metrics reported for this invocation
   * @throws {@link ConcurrentInvocationError} If this instance is already processing an invocation
   * @throws TypeError If the input or invocation options are unsupported
   * @throws {@link ToolValidationError} If deployed and host tool configuration is incompatible
   * @throws {@link ModelThrottledError} If the data or control plane reports throttling
   * @throws {@link ModelError} If the data plane, model, or Harness stream fails
   * @throws Error If a non-throttling control-plane read fails
   */
  async invoke(args: InvokeArgs, options?: InvokeOptions): Promise<AgentResult> {
    const gen = this.stream(args, options)
    let next = await gen.next()
    while (!next.done) {
      next = await gen.next()
    }
    return next.value
  }

  /**
   * Streams raw Harness events followed by a final {@link AgentCoreHarnessResultEvent}.
   *
   * Host tool calls run locally and resume the Harness with their results. `cancelSignal` and
   * `invocationState` are supported. `limits` is rejected because the managed Harness cannot
   * preserve the shared {@link InvokeOptions} budget semantics. Structured output options are not
   * supported.
   *
   * Cancellation before host tool execution prevents the tools from starting. After host work
   * starts, running tools finish and unstarted tools receive cancellation results. The agent sends
   * every result to the Harness before returning with `stopReason: 'cancelled'`.
   *
   * Input supports non-empty text, tool-use, tool-result, and reasoning content. Empty input and
   * unsupported content such as images, video, documents, cache points, guard content, and
   * citations are rejected before any AWS request; mixed content is never partially sent.
   *
   * When the Harness reports usage, `cycleCount` counts completed assistant messages rather than
   * HTTP requests or metadata records. User tool-result messages do not count. `totalDuration`
   * measures client-observed invocation time. Usage may omit host-tool turns without metadata.
   *
   * @param args - New input for the Harness-owned conversation
   * @param options - Cancellation, invocation state, and other shared invocation options
   * @returns Raw Harness events, followed by the final result event and generator return value
   * @throws {@link ConcurrentInvocationError} If this instance is already processing an invocation
   * @throws TypeError If the input or invocation options are unsupported
   * @throws {@link ToolValidationError} If deployed and host tool configuration is incompatible
   * @throws {@link ModelThrottledError} If the data or control plane reports throttling
   * @throws {@link ModelError} If the data plane, model, or Harness stream fails
   * @throws Error If a non-throttling control-plane read fails
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
    if (options?.structuredOutputSchema !== undefined) {
      throw new TypeError(
        'InvokeOptions.structuredOutputSchema is not supported by AgentCoreHarnessAgent. AgentCoreHarnessAgent cannot be used as a Swarm node.'
      )
    }
    if (options?.limits !== undefined) {
      throw new TypeError(
        'InvokeOptions.limits is not supported by AgentCoreHarnessAgent. Configure Harness-side maxIterations, maxTokens, or timeoutSeconds when constructing the agent.'
      )
    }

    const invocationState = options?.invocationState ?? {}
    const cancelSignal = options?.cancelSignal

    // The harness owns conversation history by session, so the first request carries only the new
    // turn; a follow-up request carries the assistant tool-use plus the host tool result.
    let messages: HarnessMessage[] = formatHarnessInput(args)
    const client = this._getClient()
    let toolConfiguration: ResolvedToolConfiguration | undefined
    let lastMessage = new Message({ role: 'assistant', content: [] })
    let stopReason: StopReason
    let mustCommitHostResults = false
    const accumulatedUsage = createEmptyUsage()
    let accumulatedLatencyMs = 0
    let remoteCycleCount = 0
    const invocationStartedAt = Date.now()
    // Context-size fields describe the most recent reporting model turn, not the invocation total.
    let lastUsage: Usage | undefined

    while (true) {
      const isHostResultContinuation = mustCommitHostResults

      if (!isHostResultContinuation && cancelSignal?.aborted) {
        stopReason = 'cancelled'
        break
      }

      toolConfiguration = await this._resolveToolConfiguration(
        isHostResultContinuation ? undefined : cancelSignal,
        isHostResultContinuation ? toolConfiguration : undefined
      )
      if (!isHostResultContinuation && cancelSignal?.aborted) {
        stopReason = 'cancelled'
        break
      }
      const request = this._buildRequest(messages, toolConfiguration)
      const decoder = new HarnessStreamDecoder()

      try {
        const response = await client.send(
          new InvokeHarnessCommand(request),
          cancelSignal && !isHostResultContinuation ? { abortSignal: cancelSignal } : {}
        )
        if (response.stream) {
          for await (const chunk of response.stream) {
            const event = this._streamEventOrThrow(chunk)
            yield new AgentCoreHarnessStreamUpdateEvent(event)
            decoder.accept(event)
          }
        }
      } catch (unknownError) {
        // Cancellation returns the partial result with stopReason 'cancelled' rather than throwing.
        if (!isHostResultContinuation && cancelSignal?.aborted) {
          lastMessage = decoder.partialMessage()
          stopReason = 'cancelled'
          break
        }
        throw this._normalizeError(unknownError)
      }

      const turnResult = decoder.complete()
      remoteCycleCount += turnResult.assistantMessageCount
      if (turnResult.usage !== undefined) {
        accumulateUsage(accumulatedUsage, turnResult.usage)
        accumulatedLatencyMs += turnResult.latencyMs ?? 0
        lastUsage = turnResult.latestUsage ?? turnResult.usage
      }

      lastMessage = turnResult.message
      stopReason = turnResult.stopReason

      const hostToolUses = this._hostToolUses(lastMessage)
      const cancellationRequested = cancelSignal?.aborted ?? false
      if (stopReason !== 'toolUse' || hostToolUses.length === 0) {
        if (cancellationRequested) {
          stopReason = 'cancelled'
        } else if (stopReason === 'maxTokens') {
          throw new MaxTokensError(
            'Model reached maximum token limit. This is an unrecoverable state that requires intervention.',
            lastMessage
          )
        } else if (stopReason === 'malformedModelOutput' || stopReason === 'malformedToolUse') {
          throw new ModelError(`Harness ended the turn with an unrecoverable stop reason: ${stopReason}`)
        } else if (turnResult.toolInputParseError) {
          throw new ModelError('unable to parse tool input JSON', { cause: turnResult.toolInputParseError })
        }
        break
      }
      if (turnResult.toolInputParseError && !cancellationRequested) {
        throw new ModelError('unable to parse tool input JSON', { cause: turnResult.toolInputParseError })
      }

      const toolResults = await this._runHostTools(hostToolUses, toolConfiguration.allowedTools, cancelSignal)
      messages = [
        formatHarnessMessage(lastMessage),
        formatHarnessMessage(new Message({ role: 'user', content: toolResults })),
      ]
      mustCommitHostResults = true
    }

    const result = new AgentResult({
      stopReason,
      lastMessage,
      invocationState,
      ...(lastUsage !== undefined && {
        metrics: new AgentMetrics({
          cycleCount: remoteCycleCount,
          accumulatedUsage,
          accumulatedMetrics: { latencyMs: accumulatedLatencyMs },
          latestContextSize: lastUsage.inputTokens,
          projectedContextSize: lastUsage.inputTokens + lastUsage.outputTokens,
          totalDuration: Date.now() - invocationStartedAt,
        }),
      }),
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

  private _getClient(): BedrockAgentCoreClient {
    this._client ??= createHarnessClient(this._clientConfig, this._region)
    return this._client
  }

  private _getControlClient(): BedrockAgentCoreControlClient {
    this._controlClient ??= createHarnessControlClient(this._controlClientConfig, this._region)
    return this._controlClient
  }

  private async _resolveToolConfiguration(
    cancelSignal?: AbortSignal,
    continuationFallback?: ResolvedToolConfiguration
  ): Promise<ResolvedToolConfiguration> {
    if (this._hostToolRegistry.list().length === 0) {
      return {
        tools: [],
        ...(this._requestConfig.allowedTools !== undefined && {
          allowedTools: [...this._requestConfig.allowedTools],
        }),
      }
    }

    const hostTools = formatHarnessTools(this._hostToolRegistry.list())
    try {
      return await this._loadMergedToolConfiguration(hostTools, cancelSignal)
    } catch (error) {
      if (continuationFallback !== undefined) {
        logger.warn(
          `harness_arn=<${this.harnessArn}>, qualifier=<${this._requestConfig.qualifier ?? DEFAULT_HARNESS_QUALIFIER}>, error=<${normalizeError(error).message}> | unable to refresh harness tool configuration, using previous snapshot for mandatory host result continuation`
        )
        return continuationFallback
      }
      // The caller observes cancellation as a result; _stream checks the signal before invoking.
      if (cancelSignal?.aborted) {
        return {
          tools: hostTools,
          ...(this._requestConfig.allowedTools !== undefined && {
            allowedTools: [...this._requestConfig.allowedTools],
          }),
        }
      }
      if (error instanceof ToolValidationError) throw error
      const normalizedError = normalizeError(error)
      if (normalizedError instanceof ModelThrottledError) throw normalizedError
      if (THROTTLING_ERROR_NAMES.has(normalizedError.name)) {
        throw new ModelThrottledError(normalizedError.message, { cause: error })
      }
      throw normalizedError
    }
  }

  private async _loadMergedToolConfiguration(
    hostTools: HarnessTool[],
    cancelSignal?: AbortSignal
  ): Promise<ResolvedToolConfiguration> {
    const harnessId = harnessIdFromArn(this.harnessArn)
    const qualifier = this._requestConfig.qualifier
    const requestOptions = cancelSignal ? { abortSignal: cancelSignal } : {}

    if (qualifier !== undefined && qualifier !== '' && qualifier !== DEFAULT_HARNESS_QUALIFIER) {
      const endpointResponse = await this._getControlClient().send(
        new GetHarnessEndpointCommand({ harnessId, endpointName: qualifier }),
        requestOptions
      )
      const harnessVersion = endpointResponse.endpoint?.liveVersion
      if (!harnessVersion) {
        throw new Error(`Harness endpoint '${qualifier}' has no live version`)
      }
      if (this._cachedVersionedToolConfiguration?.harnessVersion === harnessVersion) {
        return this._cachedVersionedToolConfiguration.toolConfiguration
      }

      const response = await this._getControlClient().send(
        new GetHarnessCommand({ harnessId, harnessVersion }),
        requestOptions
      )
      const toolConfiguration = this._mergeToolConfiguration(response.harness, hostTools)
      this._cachedVersionedToolConfiguration = { harnessVersion, toolConfiguration }
      return toolConfiguration
    }

    const response = await this._getControlClient().send(new GetHarnessCommand({ harnessId }), requestOptions)
    if (response.harness?.status === 'UPDATING') {
      throw new Error(
        'Default Harness configuration cannot be resolved safely while the Harness is UPDATING. Retry after the update completes or use a named endpoint.'
      )
    }
    return this._mergeToolConfiguration(response.harness, hostTools)
  }

  private _mergeToolConfiguration(
    deployedHarness:
      | {
          tools?: HarnessTool[] | undefined
          allowedTools?: string[] | undefined
        }
      | undefined,
    hostTools: HarnessTool[]
  ): ResolvedToolConfiguration {
    if (deployedHarness === undefined) {
      throw new Error('GetHarness returned no Harness configuration')
    }
    const deployedTools = deployedHarness.tools ?? []
    const allowedTools = this._requestConfig.allowedTools ?? deployedHarness.allowedTools
    assertHostToolsAllowed(this._hostToolRegistry.list(), allowedTools)
    return {
      tools: mergeHarnessTools(deployedTools, hostTools),
      ...(allowedTools !== undefined && { allowedTools: [...allowedTools] }),
    }
  }

  private _buildRequest(
    messages: HarnessMessage[],
    toolConfiguration: ResolvedToolConfiguration
  ): InvokeHarnessCommandInput {
    const { tools, allowedTools } = toolConfiguration
    const requestConfiguration = cloneRequestValue(this._requestConfig, 'request configuration')
    delete requestConfiguration.allowedTools
    return {
      harnessArn: this.harnessArn,
      runtimeSessionId: this.runtimeSessionId,
      messages,
      ...requestConfiguration,
      ...(tools.length > 0 && { tools: cloneRequestValue(tools, 'tools') }),
      ...(allowedTools !== undefined && { allowedTools: [...allowedTools] }),
    }
  }

  private _hostToolUses(message: Message): ToolUse[] {
    const toolUses: ToolUse[] = []
    for (const block of message.content) {
      if (block.type === 'toolUseBlock' && this._hostToolRegistry.get(block.name) !== undefined) {
        toolUses.push({ name: block.name, toolUseId: block.toolUseId, input: block.input })
      }
    }
    return toolUses
  }

  private async _runHostTools(
    toolUses: ToolUse[],
    allowedTools: string[] | undefined,
    cancelSignal?: AbortSignal
  ): Promise<ToolResultBlock[]> {
    switch (this._toolExecutor) {
      case 'sequential': {
        const results: ToolResultBlock[] = []
        for (const toolUse of toolUses) {
          results.push(
            cancelSignal?.aborted
              ? this._createCancelledHostToolResult(toolUse)
              : await this._runHostTool(toolUse, allowedTools)
          )
        }
        return results
      }
      case 'concurrent': {
        if (cancelSignal?.aborted) {
          return toolUses.map((toolUse) => this._createCancelledHostToolResult(toolUse))
        }
        return Promise.all(toolUses.map((toolUse) => this._runHostTool(toolUse, allowedTools)))
      }
      default: {
        const _exhaustive: never = this._toolExecutor
        throw new TypeError(`Unknown toolExecutor: ${_exhaustive as string}`)
      }
    }
  }

  private async _runHostTool(toolUse: ToolUse, allowedTools: string[] | undefined): Promise<ToolResultBlock> {
    if (!isHostToolAllowed(toolUse.name, allowedTools)) {
      throw new ModelError(
        `Harness requested host tool '${toolUse.name}' even though it is excluded by effective allowedTools. The callback was not executed.`
      )
    }
    // The constructor accepts only invokable tools, and the registry is immutable after construction.
    const tool = this._hostToolRegistry.get(toolUse.name) as InvokableTool<unknown, unknown>
    try {
      const value = await tool.invoke(toolUse.input)
      const result = createSuccessResult(value, toolUse.toolUseId)
      // Validate inside the tool boundary so an unsupported return becomes a committed error result.
      formatHarnessMessage(new Message({ role: 'user', content: [result] }))
      return result
    } catch (error) {
      logger.warn(`tool=<${toolUse.name}>, tool_use_id=<${toolUse.toolUseId}> | host tool execution failed`)
      return createErrorResult(error, toolUse.toolUseId)
    }
  }

  private _createCancelledHostToolResult(toolUse: ToolUse): ToolResultBlock {
    return new ToolResultBlock({
      toolUseId: toolUse.toolUseId,
      status: 'error',
      content: [new TextBlock('Tool execution cancelled')],
    })
  }

  private _streamEventOrThrow(chunk: InvokeHarnessStreamOutput): AgentCoreHarnessEventData {
    if ('internalServerException' in chunk && chunk.internalServerException) {
      throw new ModelError(chunk.internalServerException.message ?? 'AgentCore Harness internal server error', {
        cause: chunk.internalServerException,
      })
    }
    if ('validationException' in chunk && chunk.validationException) {
      const message = chunk.validationException.message ?? 'AgentCore Harness validation error'
      if (CONTEXT_WINDOW_OVERFLOW_ERRORS.some((phrase) => message.toLowerCase().includes(phrase))) {
        throw new ContextWindowOverflowError(message, { cause: chunk.validationException })
      }
      throw new ModelError(message, { cause: chunk.validationException })
    }
    if ('runtimeClientError' in chunk && chunk.runtimeClientError) {
      throw new ModelError(chunk.runtimeClientError.message ?? 'AgentCore Harness runtime client error', {
        cause: chunk.runtimeClientError,
      })
    }
    return chunk as AgentCoreHarnessEventData
  }

  private _normalizeError(unknownError: unknown): Error {
    if (unknownError instanceof ModelError) return unknownError

    const error = normalizeError(unknownError)
    if (THROTTLING_ERROR_NAMES.has(error.name)) {
      return new ModelThrottledError(error.message, { cause: unknownError })
    }
    if (CONTEXT_WINDOW_OVERFLOW_ERRORS.some((phrase) => error.message.toLowerCase().includes(phrase))) {
      return new ContextWindowOverflowError(error.message, { cause: unknownError })
    }
    // Never let a raw vendor error escape the provider boundary.
    return new ModelError(error.message, { cause: unknownError })
  }
}

function isInvokableTool(tool: Tool): boolean {
  return typeof (tool as Partial<InvokableTool<unknown, unknown>>).invoke === 'function'
}

function snapshotRequestConfiguration(config: AgentCoreHarnessAgentConfig): HarnessRequestConfiguration {
  const systemPrompt = formatHarnessSystemPrompt(config.systemPrompt)
  return cloneRequestValue(
    {
      ...(config.modelConfig !== undefined && { model: config.modelConfig }),
      ...(systemPrompt !== undefined && { systemPrompt }),
      ...(config.skills !== undefined && { skills: config.skills }),
      ...(config.allowedTools !== undefined && { allowedTools: config.allowedTools }),
      ...(config.maxIterations !== undefined && { maxIterations: config.maxIterations }),
      ...(config.maxTokens !== undefined && { maxTokens: config.maxTokens }),
      ...(config.timeoutSeconds !== undefined && { timeoutSeconds: config.timeoutSeconds }),
      ...(config.qualifier !== undefined && { qualifier: config.qualifier }),
      ...(config.runtimeUserId !== undefined && { runtimeUserId: config.runtimeUserId }),
      ...(config.actorId !== undefined && { actorId: config.actorId }),
    },
    'request configuration'
  )
}

function cloneRequestValue<T>(value: T, name: string): T {
  try {
    return globalThis.structuredClone(value)
  } catch (error) {
    throw new TypeError(`AgentCoreHarnessAgent ${name} must be structured-cloneable.`, { cause: error })
  }
}

function resolveToolExecutor(toolExecutor: unknown): ToolExecutorStrategy {
  switch (toolExecutor) {
    case undefined:
    case 'concurrent':
      return 'concurrent'
    case 'sequential':
      return 'sequential'
    default:
      throw new TypeError(`toolExecutor must be 'concurrent' or 'sequential', got '${String(toolExecutor)}'`)
  }
}

function harnessIdFromArn(harnessArn: string): string {
  return harnessArn.slice(harnessArn.lastIndexOf('/') + 1)
}
