/**
 * WASM component exporting strands:agent/api.
 *
 * The Agent resource holds a TS SDK Agent instance across multiple
 * generate() calls. Each generate() returns a response-stream whose
 * events() method yields the typed WIT stream-event. Consumers drain
 * the ReadableStream to completion; componentize-js turns that into
 * the component-model `stream<T>` on the wire.
 */

/// <reference path="./generated/interfaces/strands-agent-api.d.ts" />
/// <reference path="./generated/interfaces/strands-agent-messages.d.ts" />
/// <reference path="./generated/interfaces/strands-agent-streaming.d.ts" />
/// <reference path="./generated/interfaces/strands-agent-models.d.ts" />
/// <reference path="./generated/interfaces/strands-agent-tools.d.ts" />
/// <reference path="./generated/interfaces/strands-agent-sessions.d.ts" />
/// <reference path="./generated/interfaces/strands-agent-conversation.d.ts" />
/// <reference path="./generated/interfaces/strands-agent-tool-provider.d.ts" />

import type { AgentConfig, InvokeArgs, RespondArgs, AgentError } from 'strands:agent/api@0.1.0'
import type { Message as WitMessage, PromptInput } from 'strands:agent/messages@0.1.0'
import type {
  StreamEvent as WitStreamEvent,
  StopEvent as WitStopEvent,
  StopReason as WitStopReason,
  AgentTrace as WitAgentTrace,
  AgentMetrics as WitAgentMetrics,
} from 'strands:agent/streaming@0.1.0'
import type { ModelConfig as WitModelConfig, ModelParams as WitModelParams } from 'strands:agent/models@0.1.0'
import type { ToolSpec, ToolChoice as WitToolChoice } from 'strands:agent/tools@0.1.0'

import { callTool } from 'strands:agent/tool-provider@0.1.0'
import { Agent, FunctionTool, SessionManager, FileStorage } from '@strands-agents/sdk'
import { S3Storage } from '@strands-agents/sdk/session/s3-storage'
import { AnthropicModel } from '@strands-agents/sdk/models/anthropic'
import { BedrockModel } from '@strands-agents/sdk/models/bedrock'
import { OpenAIModel } from '@strands-agents/sdk/models/openai'
import { GoogleModel } from '@strands-agents/sdk/models/google'
import type {
  StopReason,
  AgentStreamEvent,
  Model,
  BaseModelConfig,
  Usage,
  Metrics,
  AgentResult,
  ToolContext,
  SystemPrompt,
  InvokeArgs as SdkInvokeArgs,
  Message,
  StreamOptions,
  ToolChoice,
  ModelStreamEvent,
  ContentBlock,
  SaveLatestStrategy,
  JSONValue,
} from '@strands-agents/sdk'
import {
  ConversationManager,
  NullConversationManager,
  SlidingWindowConversationManager,
  SummarizingConversationManager,
} from '@strands-agents/sdk'
import { z } from 'zod'

//
// --- logging + error helpers --------------------------------------------
//

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}

/** Wrap a throwable promise as a typed `agent-error` result. */
async function asAgentResult<T>(fn: () => Promise<T>, storageErrorWrap = false): Promise<{ tag: 'ok'; val: T } | { tag: 'err'; val: AgentError }> {
  try {
    return { tag: 'ok', val: await fn() }
  } catch (err) {
    const msg = errorMessage(err)
    if (storageErrorWrap) {
      return { tag: 'err', val: { tag: 'storage', val: { tag: 'permanent', val: msg } } }
    }
    return { tag: 'err', val: { tag: 'internal', val: msg } }
  }
}

//
// --- small shape maps ---------------------------------------------------
//

const STOP_REASON_MAP: Record<StopReason, WitStopReason> = {
  endTurn: 'end-turn',
  toolUse: 'tool-use',
  maxTokens: 'max-tokens',
  contentFiltered: 'content-filtered',
  guardrailIntervened: 'guardrail-intervened',
  stopSequence: 'stop-sequence',
  modelContextWindowExceeded: 'model-context-window-exceeded',
  cancelled: 'cancelled',
} as unknown as Record<StopReason, WitStopReason>

function mapStopReason(reason: StopReason): WitStopReason {
  return STOP_REASON_MAP[reason] ?? 'error'
}

function mapUsage(src: Partial<Usage> | null | undefined): WitStopEvent['usage'] {
  if (src == null) return undefined
  return {
    inputTokens: src.inputTokens ?? 0,
    outputTokens: src.outputTokens ?? 0,
    totalTokens: src.totalTokens ?? (src.inputTokens ?? 0) + (src.outputTokens ?? 0),
    cacheReadInputTokens: src.cacheReadInputTokens,
    cacheWriteInputTokens: src.cacheWriteInputTokens,
  }
}

function mapMetrics(src: Partial<Metrics> | null | undefined): WitStopEvent['metrics'] {
  if (src == null) return undefined
  return { latencyMs: typeof src.latencyMs === 'number' ? src.latencyMs : 0 }
}

/** Serialize a TS SDK Message to the WIT shape. */
function mapMessage(message: Message): WitMessage {
  return {
    role: message.role,
    content: message.content.map(mapContentBlock),
    metadata: message.metadata
      ? (JSON.parse(JSON.stringify(message.metadata)) as WitMessage['metadata'])
      : undefined,
  } as WitMessage
}

/** Serialize a TS SDK ContentBlock to the WIT tagged-variant shape. */
function mapContentBlock(block: ContentBlock): import('strands:agent/messages@0.1.0').ContentBlock {
  type WitBlock = import('strands:agent/messages@0.1.0').ContentBlock
  // block.type is the SDK class discriminator; toJSON drops class identity but keeps fields.
  const payload = JSON.parse(JSON.stringify(block))
  switch (block.type) {
    case 'textBlock': return { tag: 'text', val: payload } as WitBlock
    case 'toolUseBlock': return { tag: 'tool-use', val: payload } as WitBlock
    case 'toolResultBlock': return { tag: 'tool-result', val: payload } as WitBlock
    case 'reasoningBlock': return { tag: 'reasoning', val: payload } as WitBlock
    case 'cachePointBlock': return { tag: 'cache-point', val: payload } as WitBlock
    case 'imageBlock': return { tag: 'image', val: payload } as WitBlock
    case 'videoBlock': return { tag: 'video', val: payload } as WitBlock
    case 'documentBlock': return { tag: 'document', val: payload } as WitBlock
    case 'citationsBlock': return { tag: 'citations', val: payload } as WitBlock
    case 'guardContentBlock': return { tag: 'guard-content', val: payload } as WitBlock
    default: {
      block satisfies never
      throw new Error(`unknown content block: ${(block as { type: string }).type}`)
    }
  }
}

//
// --- stream event mapping ------------------------------------------------
//

/**
 * Translate a TS SDK `AgentStreamEvent` to its WIT counterpart. Returns
 * `null` for events whose data is available through another arm (e.g.
 * the terminal `AgentResultEvent`, which is surfaced via `stop`). See
 * docs/BRIDGE-COLLAPSE-PLAN.md for the plan to delete this function.
 */
function mapEvent(event: AgentStreamEvent): WitStreamEvent | null {
  switch (event.type) {
    case 'beforeInvocationEvent':
      return { tag: 'before-invocation', val: { invocationState: '{}' } }
    case 'afterInvocationEvent':
      return { tag: 'after-invocation', val: { invocationState: '{}' } }
    case 'messageAddedEvent':
      return { tag: 'message-added', val: { message: mapMessage(event.message) } }
    case 'beforeModelCallEvent':
      return { tag: 'before-model-call', val: { projectedInputTokens: undefined } }
    case 'afterModelCallEvent':
      return {
        tag: 'after-model-call',
        val: {
          attemptCount: 1,
          stopData: event.stopData
            ? {
                message: mapMessage(event.stopData.message),
                stopReason: mapStopReason(event.stopData.stopReason),
                redaction: event.stopData.redaction ? { userMessage: event.stopData.redaction.userMessage } : undefined,
              }
            : undefined,
          error: event.error ? { tag: 'internal', val: event.error.message } : undefined,
        },
      }
    case 'beforeToolsEvent':
      return { tag: 'before-tools', val: { message: mapMessage(event.message) } }
    case 'afterToolsEvent':
      return { tag: 'after-tools', val: { message: mapMessage(event.message) } }
    case 'beforeToolCallEvent':
      return {
        tag: 'before-tool-call',
        val: {
          toolUse: {
            name: event.toolUse.name,
            toolUseId: event.toolUse.toolUseId,
            input: JSON.stringify(event.toolUse.input ?? {}),
          },
        },
      }
    case 'afterToolCallEvent':
      return {
        tag: 'after-tool-call',
        val: {
          toolUse: {
            name: event.toolUse.name,
            toolUseId: event.toolUse.toolUseId,
            input: JSON.stringify(event.toolUse.input ?? {}),
          },
          toolResult: mapContentBlock(event.result) as unknown as import('strands:agent/messages@0.1.0').ToolResultBlock,
          error: event.error ? { tag: 'execution-failed', val: event.error.message } : undefined,
        },
      }
    case 'contentBlockEvent':
      return { tag: 'content-block', val: { contentBlock: mapContentBlock(event.contentBlock) } }
    case 'modelMessageEvent':
      return {
        tag: 'model-message',
        val: { message: mapMessage(event.message), stopReason: mapStopReason(event.stopReason) },
      }
    case 'toolResultEvent':
      return {
        tag: 'tool-result-hook',
        val: { toolResult: mapContentBlock(event.result) as unknown as import('strands:agent/messages@0.1.0').ToolResultBlock },
      }
    case 'toolStreamUpdateEvent':
      return { tag: 'tool-update', val: { data: JSON.stringify(event.event.data ?? null) } }
    case 'modelStreamUpdateEvent':
      return { tag: 'model-update', val: { event: JSON.stringify(event.event) } }
    case 'agentResultEvent':
      // The terminal `stop` arm carries this data instead.
      return null
    case 'interruptEvent':
      return {
        tag: 'interrupt',
        val: {
          id: event.interrupt.id,
          name: event.interrupt.name,
          reason:
            event.interrupt.reason !== undefined
              ? typeof event.interrupt.reason === 'string'
                ? event.interrupt.reason
                : JSON.stringify(event.interrupt.reason)
              : undefined,
        },
      }
    default: {
      event satisfies never
      return null
    }
  }
}

function mapStopEvent(result: AgentResult): WitStreamEvent {
  return {
    tag: 'stop',
    val: {
      reason: mapStopReason(result.stopReason),
      usage: mapUsage(result.metrics?.accumulatedUsage),
      metrics: mapMetrics(result.metrics?.accumulatedMetrics),
      structuredOutput: result.structuredOutput !== undefined ? JSON.stringify(result.structuredOutput) : undefined,
    },
  }
}

//
// --- config builders -----------------------------------------------------
//

function modelParamsConfig(params?: WitModelParams): Record<string, unknown> {
  if (!params) return {}
  return {
    ...(params.maxTokens != null ? { maxTokens: params.maxTokens } : {}),
    ...(params.temperature != null ? { temperature: params.temperature } : {}),
    ...(params.topP != null ? { topP: params.topP } : {}),
  }
}

function createModel(config?: WitModelConfig, params?: WitModelParams): Model<BaseModelConfig> {
  const base = modelParamsConfig(params)
  if (!config) return new BedrockModel(base)

  switch (config.tag) {
    case 'anthropic': {
      const extra = config.val.additionalConfig ? JSON.parse(config.val.additionalConfig) : {}
      return new AnthropicModel({
        ...base,
        ...(config.val.modelId ? { modelId: config.val.modelId } : {}),
        ...(config.val.apiKey ? { apiKey: config.val.apiKey } : {}),
        ...extra,
      })
    }
    case 'bedrock': {
      const extra = config.val.additionalConfig ? JSON.parse(config.val.additionalConfig) : {}
      const clientConfig: Record<string, unknown> = extra.clientConfig ?? {}
      if (config.val.accessKeyId && config.val.secretAccessKey) {
        clientConfig.credentials = {
          accessKeyId: config.val.accessKeyId,
          secretAccessKey: config.val.secretAccessKey,
          ...(config.val.sessionToken ? { sessionToken: config.val.sessionToken } : {}),
        }
      }
      return new BedrockModel({
        ...base,
        ...(config.val.modelId ? { modelId: config.val.modelId } : {}),
        ...(config.val.region ? { region: config.val.region } : {}),
        clientConfig,
        ...extra,
      })
    }
    case 'openai': {
      const extra = config.val.additionalConfig ? JSON.parse(config.val.additionalConfig) : {}
      return new OpenAIModel({
        ...base,
        ...(config.val.modelId ? { modelId: config.val.modelId } : {}),
        ...(config.val.apiKey ? { apiKey: config.val.apiKey } : {}),
        ...extra,
      })
    }
    case 'gemini': {
      const extra = config.val.additionalConfig ? JSON.parse(config.val.additionalConfig) : {}
      return new GoogleModel({
        ...base,
        ...(config.val.modelId ? { modelId: config.val.modelId } : {}),
        ...(config.val.apiKey ? { apiKey: config.val.apiKey } : {}),
        ...extra,
      })
    }
    case 'custom':
      // Phase 2: wire `model-provider` host interface.
      throw new Error(`model-config.custom is not implemented yet (provider-id: ${config.val.providerId})`)
    default: {
      config satisfies never
      throw new Error(`Unknown model-config arm`)
    }
  }
}

/** Convert WIT ToolSpecs into TS FunctionTools that call back to the host. */
function createTools(specs: ToolSpec[] | undefined): FunctionTool[] | undefined {
  if (!specs || specs.length === 0) return undefined

  return specs.map(
    (spec) =>
      new FunctionTool({
        name: spec.name,
        description: spec.description,
        inputSchema: JSON.parse(spec.inputSchema),
        callback: async (input: unknown, toolContext: ToolContext) => {
          const stream = callTool({
            name: spec.name,
            input: JSON.stringify(input),
            toolUseId: toolContext.toolUse.toolUseId,
          })
          for (;;) {
            const value = stream.read()
            if (value === undefined) {
              throw new Error(`tool ${spec.name} stream ended without complete/error`)
            }
            switch (value.tag) {
              case 'data':
                // Streaming tool progress is not surfaced to the SDK caller today.
                continue
              case 'complete':
                return value.val as unknown as JSONValue
              case 'error':
                throw new Error(`tool ${spec.name} failed: ${value.val.tag}`)
            }
          }
        },
      })
  )
}

function buildSystemPrompt(config: AgentConfig): SystemPrompt | undefined {
  const sp = config.systemPrompt
  if (!sp) return undefined
  if (sp.tag === 'text') return sp.val
  return sp.val as unknown as SystemPrompt
}

function createToolChoiceProxy(baseModel: Model<BaseModelConfig>, toolChoice: ToolChoice): Model<BaseModelConfig> {
  return new Proxy(baseModel, {
    get(target, prop, receiver) {
      if (prop === 'stream') {
        return async function* (messages: Message[], options?: StreamOptions): AsyncIterable<ModelStreamEvent> {
          yield* target.stream(messages, { ...options, toolChoice })
        }
      }
      return Reflect.get(target, prop, receiver)
    },
  }) as Model<BaseModelConfig>
}

/** Project a WIT `tool-choice` variant onto the TS SDK shape. */
function toolChoiceFromWit(tc: WitToolChoice): ToolChoice {
  switch (tc.tag) {
    case 'auto':
      return { auto: {} }
    case 'any':
      return { any: {} }
    case 'named':
      return { tool: { name: tc.val } }
  }
}

function createSessionManager(config: AgentConfig): SessionManager | undefined {
  if (!config.session) return undefined
  const sc = config.session
  let storage
  switch (sc.storage.tag) {
    case 'file':
      storage = new FileStorage(sc.storage.val.baseDir)
      break
    case 's3': {
      const s3 = sc.storage.val
      storage = new S3Storage({
        bucket: s3.bucket,
        ...(s3.region ? { region: s3.region } : {}),
        ...(s3.prefix ? { prefix: s3.prefix } : {}),
      })
      break
    }
    case 'custom':
      // Phase 2: wire `snapshot-storage` host interface.
      throw new Error(`storage-config.custom is not implemented yet (backend-id: ${sc.storage.val.backendId})`)
  }

  const saveLatestOn: SaveLatestStrategy | undefined = sc.saveLatest
    ? sc.saveLatest.tag === 'trigger'
      ? 'trigger'
      : sc.saveLatest.tag
    : undefined
  return new SessionManager({
    sessionId: sc.sessionId,
    storage: { snapshot: storage },
    ...(saveLatestOn !== undefined ? { saveLatestOn } : {}),
  })
}

function createConversationManager(config: AgentConfig): ConversationManager | undefined {
  const cm = config.conversationManager
  if (!cm) return undefined
  switch (cm.tag) {
    case 'none':
      return new NullConversationManager()
    case 'sliding-window':
      return new SlidingWindowConversationManager({
        windowSize: cm.val.windowSize,
        shouldTruncateResults: cm.val.shouldTruncateResults,
      })
    case 'summarizing': {
      const summaryModel = cm.val.summarizationModel ? createModel(cm.val.summarizationModel) : undefined
      return new SummarizingConversationManager({
        model: summaryModel,
        summaryRatio: cm.val.summaryRatio,
        preserveRecentMessages: cm.val.preserveRecentMessages,
        summarizationSystemPrompt: cm.val.summarizationSystemPrompt,
      })
    }
  }
}

function parseStructuredOutputSchema(jsonStr: string | undefined): z.ZodSchema | undefined {
  if (!jsonStr) return undefined
  try {
    return z.fromJSONSchema(JSON.parse(jsonStr)) as z.ZodSchema
  } catch (e) {
    throw new Error(`Invalid structured output schema: ${errorMessage(e)}`)
  }
}

function invokeInputFromWit(input: PromptInput): SdkInvokeArgs {
  return input.tag === 'text' ? input.val : (input.val as unknown as SdkInvokeArgs)
}

//
// --- resources -----------------------------------------------------------
//

class AgentImpl {
  private agent: Agent
  private defaultTools: FunctionTool[] | undefined
  private sessionManager: SessionManager | undefined

  constructor(config: AgentConfig) {
    const model = createModel(config.model, config.modelParams)
    this.defaultTools = createTools(config.tools)
    this.sessionManager = createSessionManager(config)

    this.agent = new Agent({
      model,
      systemPrompt: buildSystemPrompt(config),
      tools: this.defaultTools,
      sessionManager: this.sessionManager,
      conversationManager: createConversationManager(config),
      structuredOutputSchema: parseStructuredOutputSchema(config.structuredOutputSchema),
      printer: config.displayOutput ?? true,
    })
  }

  generate(args: InvokeArgs): ResponseStreamImpl {
    if (args.tools) {
      const requestTools = createTools(args.tools)
      this.agent.toolRegistry.clear()
      if (requestTools) this.agent.toolRegistry.add(requestTools)
    }

    let originalModel: Model<BaseModelConfig> | undefined
    if (args.toolChoice) {
      originalModel = this.agent.model
      this.agent.model = createToolChoiceProxy(originalModel, toolChoiceFromWit(args.toolChoice))
    }

    const structuredOutputSchema = parseStructuredOutputSchema(args.structuredOutputSchema)
    return new ResponseStreamImpl(
      this.agent,
      args.input,
      this.defaultTools,
      originalModel,
      structuredOutputSchema
    )
  }

  getMessages(): WitMessage[] {
    return this.agent.messages.map(mapMessage)
  }

  setMessages(messages: WitMessage[]): { tag: 'ok'; val: void } | { tag: 'err'; val: AgentError } {
    try {
      const parsed = messages.map((m) => JSON.parse(JSON.stringify(m)) as Message)
      this.agent.messages.splice(0, this.agent.messages.length, ...parsed)
      return { tag: 'ok', val: undefined }
    } catch (err) {
      return { tag: 'err', val: { tag: 'invalid-input', val: errorMessage(err) } }
    }
  }

  getAppState(): string {
    return JSON.stringify(this.agent.appState.getAll())
  }

  setAppState(json: string): { tag: 'ok'; val: void } | { tag: 'err'; val: AgentError } {
    try {
      const parsed = JSON.parse(json) as Record<string, JSONValue>
      this.agent.appState.clear()
      for (const [k, v] of Object.entries(parsed)) this.agent.appState.set(k, v)
      return { tag: 'ok', val: undefined }
    } catch (err) {
      return { tag: 'err', val: { tag: 'invalid-input', val: errorMessage(err) } }
    }
  }

  getModelState(): string {
    return JSON.stringify(this.agent.modelState.getAll())
  }

  setModelState(json: string): { tag: 'ok'; val: void } | { tag: 'err'; val: AgentError } {
    try {
      const parsed = JSON.parse(json) as Record<string, JSONValue>
      this.agent.modelState.clear()
      for (const [k, v] of Object.entries(parsed)) this.agent.modelState.set(k, v)
      return { tag: 'ok', val: undefined }
    } catch (err) {
      return { tag: 'err', val: { tag: 'invalid-input', val: errorMessage(err) } }
    }
  }

  getTraces(): WitAgentTrace[] {
    // Phase 2: surface the SDK's traces here. For now return empty.
    return []
  }

  getMetrics(): WitAgentMetrics {
    // Phase 2: surface the SDK's metrics here. For now return zeroes.
    return {
      cycleCount: 0,
      accumulatedUsage: { inputTokens: 0, outputTokens: 0, totalTokens: 0, cacheReadInputTokens: undefined, cacheWriteInputTokens: undefined },
      accumulatedMetrics: { latencyMs: 0 },
      invocations: [],
      cycles: [],
      toolMetrics: [],
      latestContextSize: undefined,
      projectedContextSize: undefined,
    }
  }

  async saveSession(): Promise<{ tag: 'ok'; val: void } | { tag: 'err'; val: AgentError }> {
    if (!this.sessionManager) return { tag: 'err', val: { tag: 'no-session-configured' } }
    return asAgentResult(async () => {
      await this.sessionManager!.saveSnapshot({ target: this.agent, isLatest: true })
    }, true)
  }

  async listSnapshots(): Promise<{ tag: 'ok'; val: string[] } | { tag: 'err'; val: AgentError }> {
    if (!this.sessionManager) return { tag: 'err', val: { tag: 'no-session-configured' } }
    return asAgentResult(() => this.sessionManager!.listSnapshotIds({ target: this.agent }), true)
  }

  async deleteSession(): Promise<{ tag: 'ok'; val: void } | { tag: 'err'; val: AgentError }> {
    if (!this.sessionManager) return { tag: 'err', val: { tag: 'no-session-configured' } }
    return { tag: 'err', val: { tag: 'internal', val: 'deleteSession not yet implemented' } }
  }
}

class EventStreamImpl {
  private parent: ResponseStreamImpl

  constructor(parent: ResponseStreamImpl) {
    this.parent = parent
  }

  read(): Promise<WitStreamEvent | undefined> {
    return this.parent._pullNext()
  }
}

class ResponseStreamImpl {
  private done = false
  private generator: AsyncGenerator<AgentStreamEvent, AgentResult | undefined, undefined>
  private interruptResolve: ((payload: string) => void) | null = null
  private agent: Agent
  private defaultTools: FunctionTool[] | undefined
  private originalModel: Model<BaseModelConfig> | undefined
  private pendingStop: WitStreamEvent | undefined

  constructor(
    agent: Agent,
    input: PromptInput,
    defaultTools?: FunctionTool[],
    originalModel?: Model<BaseModelConfig>,
    structuredOutputSchema?: z.ZodSchema
  ) {
    this.agent = agent
    this.defaultTools = defaultTools
    this.originalModel = originalModel
    this.generator = agent.stream(invokeInputFromWit(input), { structuredOutputSchema })
  }

  private restoreDefaults(): void {
    if (this.originalModel) this.agent.model = this.originalModel
    this.agent.toolRegistry.clear()
    if (this.defaultTools) this.agent.toolRegistry.add(this.defaultTools)
  }

  /** @internal Drains both the SDK iterator and any pending terminal stop. */
  async _pullNext(): Promise<WitStreamEvent | undefined> {
    if (this.pendingStop) {
      const stop = this.pendingStop
      this.pendingStop = undefined
      return stop
    }
    if (this.done) return undefined
    while (true) {
      try {
        const result = await this.generator.next()
        if (result.done) {
          this.done = true
          this.restoreDefaults()
          return result.value ? mapStopEvent(result.value) : undefined
        }
        const mapped = mapEvent(result.value)
        if (mapped) return mapped
        // null means the SDK event has no on-stream representation; loop.
      } catch (err) {
        this.done = true
        this.restoreDefaults()
        return { tag: 'error', val: { tag: 'internal', val: errorMessage(err) } }
      }
    }
  }

  events(): EventStreamImpl {
    return new EventStreamImpl(this)
  }

  respond(args: RespondArgs): { tag: 'ok'; val: void } | { tag: 'err'; val: AgentError } {
    if (!this.interruptResolve) {
      return { tag: 'err', val: { tag: 'unknown-interrupt', val: args.interruptId } }
    }
    // Phase 2: look up the interrupt by id and resolve the matching promise.
    this.interruptResolve(args.response)
    this.interruptResolve = null
    return { tag: 'ok', val: undefined }
  }

  cancel(): void {
    this.done = true
    this.restoreDefaults()
    void this.generator.return(undefined)
  }
}

export const api = {
  Agent: AgentImpl,
  ResponseStream: ResponseStreamImpl,
  EventStream: EventStreamImpl,
}

// Exported for contract testing. Not used by the WASM component build.
export { mapEvent, mapStopEvent, mapStopReason, mapUsage, mapMetrics, mapMessage, mapContentBlock, createTools, createToolChoiceProxy, toolChoiceFromWit }
