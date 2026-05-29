import {
  AgentResult,
  type AgentStreamEvent,
  type InvocationState,
  type InvokableAgent,
  type InvokeArgs,
  type InvokeOptions,
  type LocalAgent,
  type localAgentSymbol,
} from '../types/agent.js'
import { BedrockModel } from '../models/bedrock.js'
import {
  contentBlockFromData,
  type ContentBlock,
  type ContentBlockData,
  Message,
  type MessageData,
  type SystemPrompt,
  type SystemPromptData,
  TextBlock,
  ToolResultBlock,
  type ToolResultBlockData,
  ToolUseBlock,
} from '../types/messages.js'
import type { JSONValue } from '../types/json.js'
import { McpClient } from '../mcp.js'
import { isValidToolName, type Tool, type ToolContext } from '../tools/tool.js'
import type { ToolChoice } from '../tools/types.js'
import { systemPromptFromData } from '../types/messages.js'
import { normalizeError, ConcurrentInvocationError, StructuredOutputError } from '../errors.js'
import { Model } from '../models/model.js'
import type { BaseModelConfig, StreamAggregatedResult, StreamOptions } from '../models/model.js'
import { ModelPlugin } from '../plugins/model-plugin.js'
import { isModelStreamEvent } from '../models/streaming.js'
import { ToolRegistry } from '../registry/tool-registry.js'
import { StateStore } from '../state-store.js'
import { AgentPrinter, getDefaultAppender, type Printer } from './printer.js'
import type { Plugin } from '../plugins/plugin.js'
import type { InterventionHandler } from '../interventions/handler.js'
import { InterventionRegistry } from '../interventions/registry.js'
import type { LifecycleObserver } from '../types/lifecycle-observer.js'
import { PluginRegistry } from '../plugins/registry.js'
import { SlidingWindowConversationManager } from '../conversation-manager/sliding-window-conversation-manager.js'
import { NullConversationManager } from '../conversation-manager/null-conversation-manager.js'
import { ConversationManager } from '../conversation-manager/conversation-manager.js'
import { HookRegistryImplementation } from '../hooks/registry.js'
import type { HookableEventConstructor, HookCallback, HookCallbackOptions, HookCleanup } from '../hooks/types.js'
import {
  InitializedEvent,
  AfterInvocationEvent,
  AfterModelCallEvent,
  AfterToolCallEvent,
  AfterToolsEvent,
  BeforeInvocationEvent,
  BeforeModelCallEvent,
  BeforeToolCallEvent,
  BeforeToolsEvent,
  HookableEvent,
  MessageAddedEvent,
  ModelStreamUpdateEvent,
  ContentBlockEvent,
  ModelMessageEvent,
  ToolResultEvent,
  AgentResultEvent,
  ToolStreamUpdateEvent,
  InterruptEvent,
  type ModelStopData,
} from '../hooks/events.js'
import { StructuredOutputTool, STRUCTURED_OUTPUT_TOOL_NAME } from '../tools/structured-output-tool.js'
import { AgentAsTool } from './agent-as-tool.js'
import type { AgentAsToolOptions } from './agent-as-tool.js'
import { ToolCaller } from './tool-caller.js'
import type { ToolCallerProxy } from './tool-caller.js'

import type { z } from 'zod'
import { SessionManager } from '../session/session-manager.js'
import { Tracer } from '../telemetry/tracer.js'
import { Meter } from '../telemetry/meter.js'
import type { AttributeValue } from '@opentelemetry/api'
import { logger } from '../logging/logger.js'
import { CancelledError } from '../errors.js'
import { DefaultModelRetryStrategy } from '../retry/default-model-retry-strategy.js'
import type { RetryStrategy } from '../retry/retry-strategy.js'
import { warnOnDuplicateRetryStrategyTypes } from '../retry/retry-strategy.js'
import { InterruptError, InterruptState, interruptFromAgent } from '../interrupt.js'
import type { InterruptParams } from '../types/interrupt.js'
import { isInterruptResponseContent, type InterruptResponseContent } from '../types/interrupt.js'
import { takeSnapshot as takeSnapshotInternal, loadSnapshot as loadSnapshotInternal } from './snapshot.js'
import type { TakeSnapshotOptions } from './snapshot.js'
import type { Snapshot } from '../types/snapshot.js'

/**
 * Recursive type definition for nested tool arrays.
 * Allows tools to be organized in nested arrays of any depth.
 *
 * {@link Agent} instances in the array are automatically wrapped via
 * {@link Agent.asTool}, so they can be passed directly without calling
 * `.asTool()` explicitly.
 */
export type ToolList = (Tool | McpClient | Agent | ToolList)[]

/**
 * Strategy for executing tool calls that the model emits in a single assistant turn.
 *
 * - `'concurrent'` (default) — runs all tool calls from a single turn in parallel. Per-tool event
 *   order (`BeforeToolCallEvent` → `ToolStreamUpdateEvent*` → `AfterToolCallEvent` →
 *   `ToolResultEvent`) is preserved, while cross-tool events may interleave.
 * - `'sequential'` — runs tool calls one at a time
 *
 * Cancellation works identically in both modes: {@link Agent.cancel} flips
 * {@link Agent.cancelSignal} and tools must observe it cooperatively to stop early.
 * In concurrent mode, prompt batch-wide cancellation requires every in-flight tool
 * to honor the signal.
 */
export type ToolExecutorStrategy = 'sequential' | 'concurrent'

/**
 * Configuration object for creating a new Agent.
 */
export type AgentConfig = {
  /**
   * The model instance that the agent will use to make decisions.
   * Accepts either a Model instance or a string representing a Bedrock model ID.
   * When a string is provided, it will be used to create a BedrockModel instance.
   *
   * @example
   * ```typescript
   * // Using a string model ID (creates BedrockModel)
   * const agent = new Agent({
   *   model: 'anthropic.claude-3-5-sonnet-20240620-v1:0'
   * })
   *
   * // Using an explicit BedrockModel instance with configuration
   * const agent = new Agent({
   *   model: new BedrockModel({
   *     modelId: 'anthropic.claude-3-5-sonnet-20240620-v1:0',
   *     temperature: 0.7,
   *     maxTokens: 2048
   *   })
   * })
   * ```
   */
  model?: Model<BaseModelConfig> | string
  /** An initial set of messages to seed the agent's conversation history. */
  messages?: Message[] | MessageData[]
  /**
   * An initial set of tools to register with the agent.
   * Accepts nested arrays of tools at any depth, which will be flattened automatically.
   * {@link Agent} instances are automatically wrapped as tools via {@link Agent.asTool}.
   */
  tools?: ToolList
  /**
   * A system prompt which guides model behavior.
   */
  systemPrompt?: SystemPrompt | SystemPromptData
  /** Optional initial state values for the agent. */
  appState?: Record<string, JSONValue>
  /**
   * Optional initial model-provider state (e.g., restoring `responseId` from a
   * prior session). Typically only set when hydrating from a snapshot.
   */
  modelState?: Record<string, JSONValue>
  /**
   * Enable automatic printing of agent output to console.
   * When true, prints text generation, reasoning, and tool usage as they occur.
   * Defaults to true.
   */
  printer?: boolean
  /**
   * Conversation manager for handling message history and context overflow.
   * Defaults to SlidingWindowConversationManager with windowSize of 40.
   */
  conversationManager?: ConversationManager
  /**
   * Plugins to register with the agent.
   */
  plugins?: Plugin[]
  /**
   * Retry strategy (or strategies) for failed model/tool calls.
   *
   * - Omitted: a sensible default {@link DefaultModelRetryStrategy} with exponential backoff is used.
   * - Single strategy: the given strategy is used.
   * - Array of strategies: all are registered, in the given order. Passing two
   *   instances of the same concrete class logs a warning — they will collide
   *   on `plugin.name` when the plugin registry initializes.
   * - `null` or `[]`: retries are explicitly disabled; failures propagate to the caller.
   */
  retryStrategy?: RetryStrategy | RetryStrategy[] | null
  /**
   * Intervention handlers evaluated in registration order at each lifecycle point.
   */
  interventions?: InterventionHandler[]
  /**
   * Zod schema for structured output validation.
   */
  structuredOutputSchema?: z.ZodSchema
  /**
   * Session manager for saving and restoring agent sessions
   */
  sessionManager?: SessionManager
  /**
   * Custom trace attributes to include in all spans.
   * These attributes are merged with standard attributes in telemetry spans.
   * Telemetry must be enabled globally via telemetry.setupTracer() for these to take effect.
   */
  traceAttributes?: Record<string, AttributeValue>
  /**
   * Optional name for the agent. Defaults to "Strands Agent".
   */
  name?: string
  /**
   * Optional description of what the agent does.
   */
  description?: string
  /**
   * Optional unique identifier for the agent. Defaults to "agent".
   */
  id?: string
  /**
   * Strategy for executing tool calls from a single assistant turn.
   * Defaults to `'concurrent'`. See {@link ToolExecutorStrategy} for details.
   */
  toolExecutor?: ToolExecutorStrategy
}

/** Default name assigned to agents when none is provided. */
const DEFAULT_AGENT_NAME = 'Strands Agent'

/** Default identifier assigned to agents when none is provided. */
const DEFAULT_AGENT_ID = 'agent'

/** Result returned by tool-execution generators, threading the AfterToolsEvent back to the main loop. */
type ToolsExecutionResult = { message: Message; afterToolsEvent: AfterToolsEvent }

/**
 * Orchestrates the interaction between a model, a set of tools, and MCP clients.
 * The Agent is responsible for managing the lifecycle of tools and clients
 * and invoking the core decision-making loop.
 */
export class Agent implements LocalAgent, InvokableAgent {
  /** @internal */
  declare readonly [localAgentSymbol]: true

  /**
   * The conversation history of messages between user and assistant.
   */
  public messages: Message[]
  /**
   * App state storage accessible to tools and application logic.
   * State is not passed to the model during inference.
   */
  public readonly appState: StateStore
  /**
   * Runtime state for the model provider. Used by stateful models to persist
   * provider-specific data (e.g., response IDs for conversation chaining)
   * across invocations.
   */
  public readonly modelState: StateStore
  private readonly _conversationManager: ConversationManager

  /**
   * The model provider used by the agent for inference.
   */
  public model: Model

  /**
   * The system prompt to pass to the model provider.
   */
  public systemPrompt?: SystemPrompt

  /**
   * The name of the agent.
   */
  public readonly name: string

  /**
   * The unique identifier of the agent instance.
   */
  public readonly id: string

  /**
   * Optional description of what the agent does.
   */
  public readonly description?: string

  /**
   * The session manager for saving and restoring agent sessions, if configured.
   */
  public readonly sessionManager?: SessionManager | undefined

  private readonly _hooksRegistry: HookRegistryImplementation
  private readonly _pluginRegistry: PluginRegistry
  private readonly _interventionRegistry: InterventionRegistry
  private _toolRegistry: ToolRegistry
  private _mcpClients: McpClient[]
  private _initialized: boolean
  private _isInvoking: boolean = false
  private _abortController = new AbortController()
  private _abortSignal: AbortSignal = this._abortController.signal
  private _printer?: Printer
  private _structuredOutputSchema?: z.ZodSchema | undefined
  /** Tracer instance for creating and managing OpenTelemetry spans. */
  private _tracer: Tracer
  /** Meter instance for accumulating loop metrics during invocation. */
  private _meter: Meter
  /** Interrupt state for human-in-the-loop workflows. */
  _interruptState: InterruptState
  /** Strategy for executing tool calls from a single assistant turn. */
  private readonly _toolExecutor: ToolExecutorStrategy
  /** Direct tool caller — created via {@link ToolCaller.create} factory. */
  private readonly _toolCaller: ToolCallerProxy

  /**
   * Creates an instance of the Agent.
   * @param config - The configuration for the agent.
   */
  constructor(config?: AgentConfig) {
    // Initialize public fields
    this.messages = (config?.messages ?? []).map((msg) => (msg instanceof Message ? msg : Message.fromMessageData(msg)))
    this.appState = new StateStore(config?.appState)
    this.modelState = new StateStore(config?.modelState)
    this.name = config?.name ?? DEFAULT_AGENT_NAME
    this.id = config?.id ?? DEFAULT_AGENT_ID
    if (config?.description !== undefined) this.description = config.description
    this.sessionManager = config?.sessionManager

    if (typeof config?.model === 'string') {
      this.model = new BedrockModel({ modelId: config.model })
    } else {
      this.model = config?.model ?? new BedrockModel()
    }

    // Validate and assign conversation manager
    if (this.model.stateful) {
      if (config?.conversationManager) {
        throw new Error(
          'Cannot use a conversationManager with a stateful model. The model manages conversation state server-side.'
        )
      }
      this._conversationManager = new NullConversationManager()
    } else {
      this._conversationManager =
        config?.conversationManager ?? new SlidingWindowConversationManager({ windowSize: 40 })
    }

    const { tools, mcpClients } = flattenTools(config?.tools ?? [])
    this._toolRegistry = new ToolRegistry(tools)
    this._mcpClients = mcpClients

    // Initialize hooks registry
    this._hooksRegistry = new HookRegistryImplementation()

    this._interventionRegistry = new InterventionRegistry(config?.interventions ?? [], this._hooksRegistry)

    // `undefined` (omitted) → install the default; `null`/`[]` → explicit opt-out.
    const retryStrategies: RetryStrategy[] =
      config?.retryStrategy === null
        ? []
        : config?.retryStrategy === undefined
          ? [new DefaultModelRetryStrategy()]
          : Array.isArray(config.retryStrategy)
            ? config.retryStrategy
            : [config.retryStrategy]
    warnOnDuplicateRetryStrategyTypes(retryStrategies)

    // Initialize plugin registry with all plugins to be initialized during initialize().
    // Ordering notes:
    // - ModelPlugin is registered last so that on AfterInvocationEvent (which uses
    //   reverse callback ordering), it runs first — clearing messages before
    //   SessionManager saves.
    // - Retry-strategy ordering is not load-bearing for correctness: `DefaultModelRetryStrategy`
    //   guards on `event.retry`, so a user hook that already set it short-circuits
    //   the strategy regardless of registration order.
    this._pluginRegistry = new PluginRegistry([
      this._conversationManager,
      ...retryStrategies,
      ...(config?.plugins ?? []),
      ...(config?.sessionManager ? [config.sessionManager] : []),
      new ModelPlugin(this.model),
    ])

    if (config?.systemPrompt !== undefined) {
      this.systemPrompt = systemPromptFromData(config.systemPrompt)
    }

    // Create printer if printer is enabled (default: true)
    const printer = config?.printer ?? true
    if (printer) {
      this._printer = new AgentPrinter(getDefaultAppender())
    }

    // Store structured output schema
    this._structuredOutputSchema = config?.structuredOutputSchema

    // Initialize tracer - OTEL returns no-op tracer if not configured
    this._tracer = new Tracer(config?.traceAttributes)

    // Initialize meter for local metrics accumulation
    this._meter = new Meter()

    // Initialize interrupt state for human-in-the-loop workflows
    this._interruptState = new InterruptState()

    this._toolExecutor = config?.toolExecutor ?? 'concurrent'
    // Pass a private helper into ToolCaller so message append + hook firing
    // remains an internal concern of Agent (not exposed as a public method).
    this._toolCaller = ToolCaller.create(this, (message, invocationState) =>
      this._appendMessageAndFireHooks(message, invocationState)
    )

    this._initialized = false
  }

  /**
   * Register a hook callback for a specific event type.
   *
   * @param eventType - The event class constructor to register the callback for
   * @param callback - The callback function to invoke when the event occurs
   * @param options - Optional configuration including execution order
   * @returns Cleanup function that removes the callback when invoked
   *
   * @example
   * ```typescript
   * const agent = new Agent({ model })
   *
   * const cleanup = agent.addHook(BeforeInvocationEvent, (event) => {
   *   console.log('Invocation started')
   * })
   *
   * // Later, to remove the hook:
   * cleanup()
   * ```
   */
  addHook<T extends HookableEvent>(
    eventType: HookableEventConstructor<T>,
    callback: HookCallback<T>,
    options?: HookCallbackOptions
  ): HookCleanup {
    return this._hooksRegistry.addCallback(eventType, callback, options)
  }

  public async initialize(): Promise<void> {
    if (this._initialized) {
      return
    }

    // Initialize MCP clients and register their tools
    await Promise.all(
      this._mcpClients.map(async (client) => {
        const tools = await client.listTools()
        this._toolRegistry.add(tools)
        client.onToolsChanged = (oldTools, newTools): void => {
          oldTools.forEach((name) => this._toolRegistry.remove(name))
          this._toolRegistry.addOrReplace(newTools)
        }
      })
    )

    await this._pluginRegistry.initialize(this)

    for (const handler of this._interventionRegistry.handlers) {
      const observer = handler as Partial<LifecycleObserver>
      if (typeof observer.observeAgent === 'function') {
        await observer.observeAgent(this)
      }
    }

    await this._hooksRegistry.invokeCallbacks(new InitializedEvent({ agent: this }))

    this._initialized = true
  }

  /**
   * Acquires the invocation lock. Throws if an invocation is already in progress.
   * Callers must release via try/finally with `this._isInvoking = false`.
   */
  private acquireLock(): void {
    if (this._isInvoking) {
      throw new ConcurrentInvocationError(
        'Agent is already processing an invocation. Wait for the current invoke() or stream() call to complete before invoking again.'
      )
    }
    this._isInvoking = true
  }

  /**
   * Throws {@link CancelledError} if cancellation has been requested.
   * Called at cancellation checkpoints within the agent loop.
   */
  private _throwIfCancelled(): void {
    if (this.isCancelled) {
      throw new CancelledError()
    }
  }

  /**
   * The tools this agent can use.
   */
  get tools(): Tool[] {
    return this._toolRegistry.list()
  }

  /**
   * The tool registry for managing the agent's tools.
   */
  get toolRegistry(): ToolRegistry {
    return this._toolRegistry
  }

  /**
   * Whether the agent is currently processing an invocation.
   */
  get isInvoking(): boolean {
    return this._isInvoking
  }

  /**
   * Direct tool calling accessor.
   *
   * Returns a proxy where each property is a {@link ToolHandle} with
   * `.invoke()` and `.stream()` methods:
   * ```typescript
   * const result = await agent.tool.calculator!.invoke({ a: 5, b: 3 })
   *
   * for await (const event of agent.tool.calculator!.stream({ a: 5, b: 3 })) {
   *   console.log('progress:', event)
   * }
   * ```
   *
   * Supports underscore-to-hyphen and case-insensitive name resolution.
   * Results are recorded in message history by default (pass
   * `{ recordDirectToolCall: false }` to skip).
   */
  get tool(): ToolCallerProxy {
    return this._toolCaller
  }

  /**
   * The cancellation signal for the current invocation.
   *
   * Tools can pass this to cancellable operations (e.g., `fetch(url, { signal: agent.cancelSignal })`).
   * Hooks can check `event.agent.cancelSignal.aborted` to detect cancellation.
   */
  get cancelSignal(): AbortSignal {
    return this._abortSignal
  }

  /**
   * Cancels the current agent invocation cooperatively.
   *
   * The agent will stop at the next cancellation checkpoint:
   * - During model response streaming
   * - Before tool execution
   * - Between sequential tool executions
   * - At the top of each agent loop cycle
   *
   * If a tool is already executing, it will run to completion unless
   * the tool checks {@link LocalAgent.cancelSignal | cancelSignal} internally.
   *
   * Hook callbacks can check `event.agent.cancelSignal.aborted` to detect
   * cancellation and adjust their behavior accordingly.
   *
   * The stream/invoke call will return an AgentResult with `stopReason: 'cancelled'`.
   * If the agent is not currently invoking, this is a no-op.
   *
   * @example
   * ```typescript
   * const agent = new Agent({ model, tools })
   *
   * // Cancel after 5 seconds
   * setTimeout(() => agent.cancel(), 5000)
   * const result = await agent.invoke('Do something')
   * console.log(result.stopReason) // 'cancelled'
   * ```
   */
  public cancel(): void {
    if (this._isInvoking) {
      this._abortController.abort()
    }
  }

  /**
   * Whether the current invocation has been cancelled.
   * Returns `false` when the agent is idle.
   */
  private get isCancelled(): boolean {
    return this._abortSignal.aborted
  }

  /**
   * Invokes the agent and returns the final result.
   *
   * This is a convenience method that consumes the stream() method and returns
   * only the final AgentResult. Use stream() if you need access to intermediate
   * streaming events.
   *
   * @param args - Arguments for invoking the agent
   * @param options - Optional per-invocation options
   * @returns Promise that resolves to the final AgentResult
   *
   * @example
   * ```typescript
   * const agent = new Agent({ model, tools })
   * const result = await agent.invoke('What is 2 + 2?')
   * console.log(result.lastMessage) // Agent's response
   * ```
   */
  public async invoke(args: InvokeArgs, options?: InvokeOptions): Promise<AgentResult> {
    const gen = this.stream(args, options)
    let result = await gen.next()
    while (!result.done) {
      result = await gen.next()
    }
    return result.value
  }

  /**
   * Streams the agent execution, yielding events and returning the final result.
   *
   * The agent loop manages the conversation flow by:
   * 1. Streaming model responses and yielding all events
   * 2. Executing tools when the model requests them
   * 3. Continuing the loop until the model completes without tool use
   *
   * Use this method when you need access to intermediate streaming events.
   * For simple request/response without streaming, use invoke() instead.
   *
   * An explicit goal of this method is to always leave the message array in a way that
   * the agent can be reinvoked with a user prompt after this method completes. To that end
   * assistant messages containing tool uses are only added after tool execution succeeds
   * with valid toolResponses
   *
   * @param args - Arguments for invoking the agent
   * @param options - Optional per-invocation options
   * @returns Async generator that yields AgentStreamEvent objects and returns AgentResult
   *
   * @example
   * ```typescript
   * const agent = new Agent({ model, tools })
   *
   * for await (const event of agent.stream('Hello')) {
   *   console.log('Event:', event.type)
   * }
   * // Messages array is mutated in place and contains the full conversation
   * ```
   */
  public async *stream(
    args: InvokeArgs,
    options?: InvokeOptions
  ): AsyncGenerator<AgentStreamEvent, AgentResult, undefined> {
    this.acquireLock()
    try {
      await this.initialize()

      let currentArgs: InvokeArgs = args

      // Outer loop: re-enters _stream when a hook sets AfterInvocationEvent.resume.
      // One invocation lock spans the whole resume chain.
      while (true) {
        // Fresh AbortController per invocation iteration, composed with any external signal.
        this._abortController = new AbortController()
        this._abortSignal = options?.cancelSignal
          ? AbortSignal.any([this._abortController.signal, options.cancelSignal])
          : this._abortController.signal

        const streamGenerator = this._stream(currentArgs, options)
        let caughtError: Error | undefined
        let lastAfterInvocation: AfterInvocationEvent | undefined
        let iterationResult: IteratorResult<AgentStreamEvent, AgentResult>
        try {
          iterationResult = await streamGenerator.next()

          while (!iterationResult.done) {
            try {
              const processed = await this._invokeCallbacks(iterationResult.value)
              if (processed instanceof AfterInvocationEvent) {
                lastAfterInvocation = processed
              }
              yield processed
              iterationResult = await streamGenerator.next()
            } catch (error) {
              // Throw interrupt errors back into _stream so executeTools can store the
              // assistant message as pending execution state for resume.
              if (error instanceof InterruptError) {
                iterationResult = await streamGenerator.throw(error)
              } else {
                throw error
              }
            }
          }

          // Suppress AgentResultEvent for resumed iterations — only the final
          // invocation in a resume chain reports an agent result.
          if (lastAfterInvocation?.resume === undefined) {
            yield await this._invokeCallbacks(
              new AgentResultEvent({
                agent: this,
                result: iterationResult.value,
                invocationState: iterationResult.value.invocationState,
              })
            )
          }
        } catch (error) {
          caughtError = error as Error
          throw error
        } finally {
          // Drain _stream() so cleanup hooks and printer still fire.
          // Yield only on error (consumer may still be iterating); on a consumer
          // break, yielding would suspend the generator and leak the lock.
          let drainResult = await streamGenerator.return(undefined as never)
          while (!drainResult.done) {
            try {
              if (caughtError) {
                yield await this._invokeCallbacks(drainResult.value)
              } else {
                await this._invokeCallbacks(drainResult.value)
              }
            } catch (error) {
              logger.warn(
                `event_type=<${drainResult.value.type}>, error=<${error}> | error invoking callbacks during cleanup`
              )
            }
            drainResult = await streamGenerator.next()
          }

          // Reset controller and signal for next iteration / invocation
          this._abortController = new AbortController()
          this._abortSignal = this._abortController.signal
        }

        // Resume only on a clean invocation — errors propagate above.
        if (lastAfterInvocation?.resume !== undefined) {
          currentArgs = lastAfterInvocation.resume
          continue
        }

        return iterationResult.value
      }
    } finally {
      this._isInvoking = false
    }
  }

  /**
   * Returns a {@link Tool} that wraps this agent, allowing it to be used
   * as a tool by another agent.
   *
   * The returned tool accepts a single `input` string parameter, invokes
   * this agent, and returns the text response as a tool result.
   *
   * **Note:** You can also pass an Agent directly in another agent's
   * {@link AgentConfig.tools | tools} array — it will be wrapped
   * automatically via this method.
   *
   * @param options - Optional configuration for the tool name, description, and context preservation
   * @returns A Tool wrapping this agent
   *
   * @example
   * ```typescript
   * const researcher = new Agent({ name: 'researcher', description: 'Finds info', printer: false })
   *
   * // Explicit wrapping
   * const writer = new Agent({ tools: [researcher.asTool()] })
   *
   * // Automatic wrapping (equivalent)
   * const writer = new Agent({ tools: [researcher] })
   * ```
   */
  public asTool(options?: AgentAsToolOptions): Tool {
    return new AgentAsTool({ agent: this, ...options })
  }

  /**
   * Captures a point-in-time snapshot of the agent's current state.
   *
   * Use snapshots to checkpoint agent state for later restoration, enabling
   * use cases like undo/redo, branching conversations, and session persistence.
   *
   * Fields are selected via a preset/include/exclude model:
   * 1. Start with preset fields (e.g. `'session'` captures all fields)
   * 2. Add any `include` fields
   * 3. Remove any `exclude` fields
   *
   * @param options - Controls which fields to capture and optional app data to store
   * @returns A {@link Snapshot} containing the captured agent state
   * @throws Error if no fields would be included after applying options
   *
   * @example
   * ```typescript
   * // Capture all session-relevant state
   * const snapshot = agent.takeSnapshot({ preset: 'session' })
   *
   * // Capture only messages and state
   * const partial = agent.takeSnapshot({ include: ['messages', 'state'] })
   *
   * // Capture session state but exclude interrupts
   * const noInterrupts = agent.takeSnapshot({ preset: 'session', exclude: ['interrupts'] })
   *
   * // Attach application-owned metadata
   * const withMeta = agent.takeSnapshot({ preset: 'session', appData: { userId: 'u-123' } })
   * ```
   */
  public takeSnapshot(options: TakeSnapshotOptions): Snapshot {
    return takeSnapshotInternal(this, options)
  }

  /**
   * Restores agent state from a previously captured snapshot.
   *
   * Only fields present in `snapshot.data` are restored; absent fields are left
   * unchanged. This allows partial snapshots to update specific aspects of state
   * without affecting others.
   *
   * @param snapshot - The snapshot to restore from
   * @throws Error if `snapshot.schemaVersion` is incompatible or scope is wrong
   *
   * @example
   * ```typescript
   * // Save and restore a conversation checkpoint
   * const checkpoint = agent.takeSnapshot({ preset: 'session' })
   *
   * // ... agent continues processing ...
   *
   * // Restore to the checkpoint
   * agent.loadSnapshot(checkpoint)
   *
   * // Restore from a JSON-serialized snapshot (e.g. from storage)
   * const stored = JSON.parse(savedSnapshotJson)
   * agent.loadSnapshot(stored)
   * ```
   */
  public loadSnapshot(snapshot: Snapshot): void {
    loadSnapshotInternal(this, snapshot)
  }

  /**
   * Invokes hook callbacks and printer for a stream event.
   *
   * @param event - The event to process
   * @returns The event after processing
   */
  private async _invokeCallbacks(event: AgentStreamEvent): Promise<AgentStreamEvent> {
    if (event instanceof HookableEvent) {
      await this._hooksRegistry.invokeCallbacks(event)
    }
    this._printer?.processEvent(event)
    return event
  }

  /**
   * Internal implementation of the agent streaming logic.
   * Separated to centralize printer event processing in the public stream method.
   *
   * @param args - Arguments for invoking the agent
   * @param options - Optional per-invocation options
   * @returns Async generator that yields AgentStreamEvent objects and returns AgentResult
   */
  private async *_stream(
    args: InvokeArgs,
    options?: InvokeOptions
  ): AsyncGenerator<AgentStreamEvent, AgentResult, undefined> {
    let currentArgs: InvokeArgs | undefined = args
    let result: AgentResult | undefined

    // Resolve structured output schema from per-invocation options or constructor config
    const structuredOutputSchema = options?.structuredOutputSchema ?? this._structuredOutputSchema
    const structuredOutputTool = structuredOutputSchema ? new StructuredOutputTool(structuredOutputSchema) : undefined
    let structuredOutputChoice: ToolChoice | undefined

    // Resolve per-invocation state once. The same object is threaded through
    // every lifecycle hook event, every tool context, and is surfaced on the
    // AgentResult. Mutations by hooks/tools are visible across all recursive
    // agent loop cycles within this invocation.
    const invocationState: InvocationState = options?.invocationState ?? {}

    // Handle interrupt responses if present in input
    const interruptResponses = this._extractInterruptResponses(args)
    if (interruptResponses.length > 0) {
      this._interruptState.resume(interruptResponses)
    }

    // Reject non-interrupt input while in interrupted state
    if (this._interruptState.activated && interruptResponses.length === 0) {
      throw new TypeError('Agent is in an interrupted state. Resume by invoking with interruptResponse content blocks.')
    }

    const beforeInvocationEvent = new BeforeInvocationEvent({ agent: this, invocationState })
    yield beforeInvocationEvent

    if (beforeInvocationEvent.cancel) {
      const cancelText =
        typeof beforeInvocationEvent.cancel === 'string' ? beforeInvocationEvent.cancel : 'invocation denied by hook'
      const message = new Message({ role: 'assistant', content: [new TextBlock(cancelText)] })
      yield this._appendMessage(message, invocationState)
      yield new AfterInvocationEvent({ agent: this, invocationState })
      return new AgentResult({
        stopReason: 'endTurn',
        lastMessage: message,
        traces: this._tracer.localTraces,
        metrics: this._meter.metrics,
        invocationState,
      })
    }

    // Normalize input to get the user messages for telemetry
    const inputMessages = this._normalizeInput(args)

    // Start agent trace span
    this._meter.startNewInvocation()
    const agentModelId = this.model.modelId
    const agentSpanOptions: Parameters<Tracer['startAgentSpan']>[0] = {
      messages: inputMessages,
      agentName: this.name,
      agentId: this.id,
      tools: this.tools,
    }
    if (agentModelId) agentSpanOptions.modelId = agentModelId
    if (this.systemPrompt !== undefined) agentSpanOptions.systemPrompt = this.systemPrompt
    const agentSpan = this._tracer.startAgentSpan(agentSpanOptions)

    let caughtError: Error | undefined
    try {
      // Register structured output tool if schema provided
      if (structuredOutputTool) {
        this._toolRegistry.add(structuredOutputTool)
      }

      // Main agent loop - continues until model stops without requesting tools
      while (true) {
        this._throwIfCancelled()

        // Start metrics cycle tracking
        const { cycleId, startTime: cycleStartTime } = this._meter.startCycle()

        // Create agent loop cycle span within agent span context
        const cycleSpan = this._tracer.startAgentLoopSpan({
          cycleId,
          messages: this.messages,
        })

        try {
          // Normalize input and append user messages on first invocation only
          if (currentArgs !== undefined) {
            const messagesToAppend = this._normalizeInput(currentArgs)
            for (const message of messagesToAppend) {
              yield this._appendMessage(message, invocationState)
            }
            currentArgs = undefined
          }

          // Check if we're resuming from a tool interrupt
          const pendingExecution = this._interruptState.getPendingExecution()
          let assistantMessage: Message
          let completedToolResults: Map<string, ToolResultBlock> | undefined

          if (pendingExecution) {
            // Resume from stored state - skip model call
            assistantMessage = pendingExecution.assistantMessage
            completedToolResults = pendingExecution.completedToolResults
            this._interruptState.clearPendingToolExecution()
          } else {
            const modelResult = yield* this._invokeModel(invocationState, structuredOutputChoice)

            if (modelResult.stopReason !== 'toolUse') {
              // Schema set, we already forced, and the model still refused.
              // Throw before closing the span so the cycle span records the error.
              if (structuredOutputTool && structuredOutputChoice) {
                throw new StructuredOutputError(
                  'The model failed to invoke the structured output tool even after it was forced.'
                )
              }

              this._meter.endCycle(cycleStartTime)
              this._tracer.endAgentLoopSpan(cycleSpan)

              // Schema set, model ignored the tool — drop the response and force the tool next cycle.
              // Appending the plain-text turn here would leave the conversation ending on an
              // assistant message, which providers like Bedrock reject as assistant prefill.
              if (structuredOutputTool) {
                structuredOutputChoice = { tool: { name: STRUCTURED_OUTPUT_TOOL_NAME } }
                logger.debug(
                  'structured output schema set but model responded with plain text; forcing tool use on next cycle'
                )
                continue
              }

              // Normal end of turn.
              yield this._appendMessage(modelResult.message, invocationState)
              result = new AgentResult({
                stopReason: modelResult.stopReason,
                lastMessage: modelResult.message,
                traces: this._tracer.localTraces,
                metrics: this._meter.metrics,
                invocationState,
              })
              return result
            }

            // Cancel before tool execution: create error results for all pending tools
            if (this.isCancelled) {
              const toolUseBlocks = modelResult.message.content.filter(
                (block): block is ToolUseBlock => block.type === 'toolUseBlock'
              )
              const cancelBlocks = toolUseBlocks.map(
                (block) =>
                  new ToolResultBlock({
                    toolUseId: block.toolUseId,
                    status: 'error',
                    content: [new TextBlock('Tool execution cancelled')],
                  })
              )
              const toolResultMessage = new Message({ role: 'user', content: cancelBlocks })

              yield this._appendMessage(modelResult.message, invocationState)
              yield this._appendMessage(toolResultMessage, invocationState)

              this._meter.endCycle(cycleStartTime)
              this._tracer.endAgentLoopSpan(cycleSpan)

              result = new AgentResult({
                stopReason: 'cancelled',
                lastMessage: modelResult.message,
                traces: this._tracer.localTraces,
                metrics: this._meter.metrics,
                invocationState,
              })
              return result
            }

            assistantMessage = modelResult.message
          }

          // Execute tools
          const toolsResult = yield* this.executeTools(
            assistantMessage,
            this._toolRegistry,
            invocationState,
            completedToolResults
          )

          // When the consumer breaks the stream (e.g. agent.cancel() + break),
          // yield* returns undefined because the inner generator was closed.
          if (!toolsResult) {
            this._meter.endCycle(cycleStartTime)
            this._tracer.endAgentLoopSpan(cycleSpan)
            continue
          }
          const toolResultMessage = toolsResult.message

          /**
           * Deferred append: both messages are added AFTER tool execution completes.
           * This keeps agent.messages in a valid, reinvokable state at all times.
           * If interrupted during tool execution, messages has no dangling toolUse
           * without a matching toolResult, so the agent can be reinvoked cleanly.
           */
          yield this._appendMessage(assistantMessage, invocationState)
          yield this._appendMessage(toolResultMessage, invocationState)

          // Deactivate interrupt state after successful tool execution so the next
          // cycle starts with a clean slate (new interrupts can be raised again).
          if (this._interruptState.activated) {
            this._interruptState.deactivate()
          }

          this._meter.endCycle(cycleStartTime)
          this._tracer.endAgentLoopSpan(cycleSpan)

          // Hook requested halt: exit without calling the model again
          const { afterToolsEvent } = toolsResult
          if (afterToolsEvent.endTurn) {
            const endTurnText =
              typeof afterToolsEvent.endTurn === 'string'
                ? afterToolsEvent.endTurn
                : 'Turn ended early by hook after tool execution'
            const lastMessage = new Message({ role: 'assistant', content: [new TextBlock(endTurnText)] })
            yield this._appendMessage(lastMessage, invocationState)

            result = new AgentResult({
              stopReason: 'endTurn',
              lastMessage,
              traces: this._tracer.localTraces,
              metrics: this._meter.metrics,
              invocationState,
            })
            return result
          }

          // Structured output captured: exit
          const structuredOutput = structuredOutputTool
            ? this._extractStructuredOutput(assistantMessage, toolResultMessage)
            : undefined
          if (structuredOutput !== undefined) {
            result = new AgentResult({
              stopReason: 'toolUse',
              lastMessage: assistantMessage,
              traces: this._tracer.localTraces,
              structuredOutput,
              metrics: this._meter.metrics,
              invocationState,
            })
            return result
          }
        } catch (error) {
          this._meter.endCycle(cycleStartTime)
          this._tracer.endAgentLoopSpan(cycleSpan, { error: error as Error })
          throw error
        }
      }
    } catch (error) {
      if (error instanceof CancelledError) {
        // Cancelled during model streaming or at the top of a cycle.
        // No partial messages have been appended (deferred append pattern).
        const cancelMessage = new Message({
          role: 'assistant',
          content: [new TextBlock('Cancelled by user')],
        })
        yield this._appendMessage(cancelMessage, invocationState)

        result = new AgentResult({
          stopReason: 'cancelled',
          lastMessage: cancelMessage,
          traces: this._tracer.localTraces,
          metrics: this._meter.metrics,
          invocationState,
        })
        return result
      }
      if (error instanceof InterruptError) {
        // Fan out one event per interrupt. Each event exposes `interrupt.source` so
        // consumers can filter by origin (tool callback vs hook callback) without
        // subscribing to separate event types.
        for (const interrupt of error.interrupts) {
          yield new InterruptEvent({ agent: this, interrupt, invocationState })
        }
        result = this._createInterruptResult(invocationState)
        return result
      }
      caughtError = error as Error
      throw error
    } finally {
      // If cancelled but the catch block was bypassed (generator terminated
      // via .return() when the consumer breaks out of for-await), append an
      // assistant message so the agent can be reinvoked with a new user prompt.
      if (!caughtError && !result && this.isCancelled) {
        const cancelMessage = new Message({
          role: 'assistant',
          content: [new TextBlock('Cancelled by user')],
        })
        yield this._appendMessage(cancelMessage, invocationState)
      }

      this._tracer.endAgentSpan(agentSpan, {
        ...(caughtError && { error: caughtError }),
        ...(result?.lastMessage && { response: result.lastMessage }),
        accumulatedUsage: this._meter.metrics.accumulatedUsage,
        ...(result?.stopReason && { stopReason: result.stopReason }),
      })

      // Cleanup structured output tool
      if (structuredOutputTool) {
        this._toolRegistry.remove(STRUCTURED_OUTPUT_TOOL_NAME)
      }

      // Always emit final event
      yield new AfterInvocationEvent({ agent: this, invocationState })
    }
  }

  /**
   * Extracts the validated structured output result from tool execution.
   *
   * @param toolUseMessage - The assistant message containing tool use blocks
   * @param toolResultMessage - The message containing tool results
   * @returns The parsed structured output, or undefined if not found
   */
  private _extractStructuredOutput(toolUseMessage: Message, toolResultMessage: Message): unknown | undefined {
    const toolUse = toolUseMessage.content.find(
      (block): block is ToolUseBlock => block.type === 'toolUseBlock' && block.name === STRUCTURED_OUTPUT_TOOL_NAME
    )
    if (!toolUse) return undefined

    const toolResult = toolResultMessage.content.find(
      (block): block is ToolResultBlock =>
        block.type === 'toolResultBlock' && block.toolUseId === toolUse.toolUseId && block.status === 'success'
    )
    if (!toolResult) return undefined

    const firstContent = toolResult.content[0]
    return firstContent?.type === 'jsonBlock' ? firstContent.json : undefined
  }

  /**
   * Creates an AgentResult for an interrupt stop.
   *
   * @param invocationState - The current invocation state
   * @returns AgentResult with stopReason 'interrupt'
   */
  private _createInterruptResult(invocationState: InvocationState): AgentResult {
    this._interruptState.activate()
    return new AgentResult({
      stopReason: 'interrupt',
      lastMessage:
        this.messages.length > 0
          ? this.messages[this.messages.length - 1]!
          : new Message({ role: 'assistant', content: [new TextBlock('Interrupted')] }),
      traces: this._tracer.localTraces,
      metrics: this._meter.metrics,
      interrupts: this._interruptState.getUnansweredInterrupts(),
      invocationState,
    })
  }

  /**
   * Extracts interrupt response content blocks from invocation args.
   *
   * @param args - The invocation arguments
   * @returns Array of InterruptResponseContent blocks, empty if none found
   * @throws TypeError if args mix interrupt responses with other content
   */
  private _extractInterruptResponses(args: InvokeArgs): InterruptResponseContent[] {
    if (!Array.isArray(args) || args.length === 0) {
      return []
    }

    const responses: InterruptResponseContent[] = []
    let hasNonInterrupt = false

    for (const item of args) {
      if (isInterruptResponseContent(item)) {
        responses.push(item)
      } else {
        hasNonInterrupt = true
      }
    }

    if (responses.length > 0 && hasNonInterrupt) {
      throw new TypeError('Must resume from interrupt with a list of interruptResponse content blocks only')
    }

    return responses
  }

  /**
   * Normalizes agent invocation input into an array of messages to append.
   *
   * @param args - Optional arguments for invoking the model
   * @returns Array of messages to append to the conversation
   */
  private _normalizeInput(args?: InvokeArgs): Message[] {
    if (args !== undefined) {
      if (typeof args === 'string') {
        // String input: wrap in TextBlock and create user Message
        return [
          new Message({
            role: 'user',
            content: [new TextBlock(args)],
          }),
        ]
      } else if (Array.isArray(args) && args.length > 0) {
        const firstElement = args[0]!

        // Check if it's interrupt responses - skip creating messages for these
        if (isInterruptResponseContent(firstElement)) {
          // Pure interrupt responses: no messages to add
          return []
        }

        // Check if it's Message[] or MessageData[]
        if ('role' in firstElement && typeof firstElement.role === 'string') {
          // Check if it's a Message instance or MessageData
          if (firstElement instanceof Message) {
            // Message[] input: return all messages
            return args as Message[]
          } else {
            // MessageData[] input: convert to Message[]
            return (args as MessageData[]).map((data) => Message.fromMessageData(data))
          }
        } else {
          // It's ContentBlock[] or ContentBlockData[]
          // Check if it's ContentBlock instances or ContentBlockData
          let contentBlocks: ContentBlock[]
          if ('type' in firstElement && typeof firstElement.type === 'string') {
            // ContentBlock[] input: use as-is
            contentBlocks = args as ContentBlock[]
          } else {
            // ContentBlockData[] input: convert using helper function
            contentBlocks = (args as ContentBlockData[]).map(contentBlockFromData)
          }

          return [
            new Message({
              role: 'user',
              content: contentBlocks,
            }),
          ]
        }
      }
    }
    // undefined or empty array: no messages to append
    return []
  }

  /**
   * Invokes the model provider and streams all events.
   *
   * @param args - Optional arguments for invoking the model
   * @param toolChoice - Optional tool choice to force specific tool usage
   * @returns Object containing the assistant message, stop reason, and optional redaction message
   */
  private async *_invokeModel(
    invocationState: InvocationState,
    toolChoice?: ToolChoice
  ): AsyncGenerator<AgentStreamEvent, StreamAggregatedResult, undefined> {
    const toolSpecs = this._toolRegistry.list().map((tool) => tool.toolSpec)
    const streamOptions: StreamOptions = { toolSpecs, modelState: this.modelState }
    if (this.systemPrompt !== undefined) {
      streamOptions.systemPrompt = this.systemPrompt
    }

    // Add tool choice if provided
    if (toolChoice) {
      streamOptions.toolChoice = toolChoice
    }

    let attemptCount = 1
    while (true) {
      // Estimate input tokens for the upcoming model call (non-fatal if estimation fails)
      let projectedInputTokens: number | undefined
      try {
        projectedInputTokens = await this._estimateInputTokens(streamOptions)
      } catch (e) {
        logger.debug(`error=<${e}> | token estimation failed, proceeding without estimate`)
      }

      const beforeModelCallEvent = new BeforeModelCallEvent({
        agent: this,
        model: this.model,
        invocationState,
        ...(projectedInputTokens !== undefined && { projectedInputTokens }),
      })
      yield beforeModelCallEvent

      if (beforeModelCallEvent.cancel) {
        const cancelText =
          typeof beforeModelCallEvent.cancel === 'string' ? beforeModelCallEvent.cancel : 'model call denied by hook'
        const message = new Message({ role: 'assistant', content: [new TextBlock(cancelText)] })
        const stopData: ModelStopData = { message, stopReason: 'endTurn' }
        const afterModelCallEvent = new AfterModelCallEvent({
          agent: this,
          model: this.model,
          attemptCount,
          stopData,
          invocationState,
        })
        yield afterModelCallEvent

        if (afterModelCallEvent.retry) {
          attemptCount += 1
          continue
        }

        return { message, stopReason: 'endTurn' }
      }

      // Start model span within loop span context
      const modelId = this.model.modelId
      const modelSpan = this._tracer.startModelInvokeSpan({
        messages: this.messages,
        ...(modelId && { modelId }),
        ...(this.systemPrompt !== undefined && { systemPrompt: this.systemPrompt }),
      })

      try {
        const result = yield* this._streamFromModel(this.messages, streamOptions, invocationState)

        // Accumulate token usage and model latency metrics
        this._meter.updateCycle(result.metadata)

        // End model span with usage
        const usage = result.metadata?.usage
        const metrics = result.metadata?.metrics
        this._tracer.endModelInvokeSpan(modelSpan, {
          output: result.message,
          stopReason: result.stopReason,
          ...(usage && { usage }),
          ...(metrics && { metrics }),
        })

        yield new ModelMessageEvent({
          agent: this,
          message: result.message,
          stopReason: result.stopReason,
          invocationState,
        })

        // Handle user content redaction if guardrails blocked input
        if (result.redaction?.userMessage) {
          this._redactLastMessage(result.redaction.userMessage)
        }

        const stopData: ModelStopData = {
          message: result.message,
          stopReason: result.stopReason,
          ...(result.redaction && { redaction: result.redaction }),
        }

        const afterModelCallEvent = new AfterModelCallEvent({
          agent: this,
          model: this.model,
          attemptCount,
          stopData,
          invocationState,
        })
        yield afterModelCallEvent

        if (afterModelCallEvent.retry) {
          attemptCount += 1
          continue
        }

        return result
      } catch (error) {
        const modelError = normalizeError(error)

        // End model span with error
        this._tracer.endModelInvokeSpan(modelSpan, { error: modelError })

        // Create error event
        const errorEvent = new AfterModelCallEvent({
          agent: this,
          model: this.model,
          attemptCount,
          error: modelError,
          invocationState,
        })

        // Yield error event - stream will invoke hooks
        yield errorEvent

        // Let CancelledError propagate directly — no retry
        // (we emit the AfterModelCall because we already emitted Before and we guarentee the pair)
        if (error instanceof CancelledError) {
          throw error
        }

        // After yielding, hooks have been invoked and may have set retry
        if (errorEvent.retry) {
          attemptCount += 1
          continue
        }

        // Re-throw error
        throw error
      }
    }
  }

  /**
   * Streams events from the model and dispatches appropriate events for each.
   *
   * The model's `streamAggregated()` yields two kinds of output:
   * - **ModelStreamEvent**: Transient streaming deltas (partial data while generating).
   *   Wrapped in {@link ModelStreamUpdateEvent} before yielding.
   * - **ContentBlock**: Fully assembled results (after all deltas accumulate).
   *   Wrapped in {@link ContentBlockEvent} before yielding.
   *
   * These are separate event classes because they represent different granularities
   * (partial deltas vs finished blocks). Both are yielded in the stream and hookable.
   *
   * @param messages - Messages to send to the model
   * @param streamOptions - Options for streaming
   * @returns StreamAggregatedResult containing message, stop reason, and optional redaction message
   */
  private async *_streamFromModel(
    messages: Message[],
    streamOptions: StreamOptions,
    invocationState: InvocationState
  ): AsyncGenerator<AgentStreamEvent, StreamAggregatedResult, undefined> {
    messages = normalizeToolUseNames(messages)
    const streamGenerator = this.model.streamAggregated(messages, streamOptions)
    let result = await streamGenerator.next()

    while (!result.done) {
      this._throwIfCancelled()

      const event = result.value

      if (isModelStreamEvent(event)) {
        // ModelStreamEvent: wrap in ModelStreamUpdateEvent
        yield new ModelStreamUpdateEvent({ agent: this, event, invocationState })
      } else {
        // ContentBlock: wrap in ContentBlockEvent
        yield new ContentBlockEvent({ agent: this, contentBlock: event, invocationState })
      }
      result = await streamGenerator.next()
    }

    // result.done is true, result.value contains the return value
    return result.value
  }

  /**
   * Emits `BeforeToolsEvent`, handles the pre-launch cancel paths, then
   * delegates per-tool execution to the configured {@link ToolExecutorStrategy}.
   * Always pairs `BeforeToolsEvent` with a terminal `AfterToolsEvent`, even on
   * the invariant-violation throw path.
   *
   * @param assistantMessage - The assistant message containing tool use blocks
   * @param toolRegistry - Registry containing available tools
   * @returns Tool-result message and the dispatched AfterToolsEvent
   */
  private async *executeTools(
    assistantMessage: Message,
    toolRegistry: ToolRegistry,
    invocationState: InvocationState,
    completedToolResults?: Map<string, ToolResultBlock>
  ): AsyncGenerator<AgentStreamEvent, ToolsExecutionResult, undefined> {
    const beforeToolsEvent = new BeforeToolsEvent({ agent: this, message: assistantMessage, invocationState })
    try {
      yield beforeToolsEvent
    } catch (error) {
      // Store pending state before re-throwing so the agent can resume from this point.
      // The error must still propagate to _stream which handles the interrupt stop.
      if (error instanceof InterruptError) {
        this._interruptState.setPendingToolExecution({
          assistantMessageData: assistantMessage.toJSON(),
          completedToolResults: {},
        })
      }
      throw error
    }

    const toolUseBlocks = assistantMessage.content.filter(
      (block): block is ToolUseBlock => block.type === 'toolUseBlock'
    )
    if (toolUseBlocks.length === 0) {
      // Preserve BeforeToolsEvent/AfterToolsEvent bracket symmetry even on
      // this invariant-violation branch.
      yield new AfterToolsEvent({
        agent: this,
        message: new Message({ role: 'user', content: [] }),
        invocationState,
      })
      throw new Error('Model indicated toolUse but no tool use blocks found in message')
    }

    // Pre-launch cancel paths are strategy-independent.
    if (beforeToolsEvent.cancel) {
      const message = typeof beforeToolsEvent.cancel === 'string' ? beforeToolsEvent.cancel : 'Tool cancelled by hook'
      return yield* this._yieldCancelledToolResults(toolUseBlocks, message, invocationState)
    }
    if (this.isCancelled) {
      return yield* this._yieldCancelledToolResults(toolUseBlocks, 'Tool execution cancelled', invocationState)
    }

    switch (this._toolExecutor) {
      case 'sequential':
        return yield* this._executeToolsSequential(
          toolUseBlocks,
          toolRegistry,
          invocationState,
          completedToolResults,
          assistantMessage
        )
      case 'concurrent':
        return yield* this._executeToolsConcurrent(
          toolUseBlocks,
          toolRegistry,
          invocationState,
          completedToolResults,
          assistantMessage
        )
      default: {
        const _exhaustive: never = this._toolExecutor
        throw new Error(`Unknown toolExecutor: ${_exhaustive as string}`)
      }
    }
  }

  /**
   * Emits a `ToolResultEvent` for every block plus an `AfterToolsEvent`, and
   * returns the resulting tool-result message and dispatched event. Used by the pre-launch cancel
   * paths shared across executors.
   */
  private async *_yieldCancelledToolResults(
    toolUseBlocks: ToolUseBlock[],
    message: string,
    invocationState: InvocationState
  ): AsyncGenerator<AgentStreamEvent, ToolsExecutionResult, undefined> {
    const cancelBlocks = this._cancelAllAsResults(toolUseBlocks, message)
    for (const result of cancelBlocks) {
      yield new ToolResultEvent({ agent: this, result, invocationState })
    }
    const toolResultMessage = new Message({ role: 'user', content: cancelBlocks })
    const afterToolsEvent = new AfterToolsEvent({ agent: this, message: toolResultMessage, invocationState })
    yield afterToolsEvent
    return { message: toolResultMessage, afterToolsEvent }
  }

  /**
   * Executes tools one at a time, honoring `agent.cancelSignal` between
   * iterations to short-circuit not-yet-started tools.
   */
  private async *_executeToolsSequential(
    toolUseBlocks: ToolUseBlock[],
    toolRegistry: ToolRegistry,
    invocationState: InvocationState,
    completedToolResults?: Map<string, ToolResultBlock>,
    assistantMessage?: Message
  ): AsyncGenerator<AgentStreamEvent, ToolsExecutionResult, undefined> {
    const toolResultBlocks: ToolResultBlock[] = []
    let toolResultMessage: Message
    let afterToolsEvent: AfterToolsEvent

    try {
      for (const toolUseBlock of toolUseBlocks) {
        // Skip tools that were already completed before the interrupt
        if (completedToolResults?.has(toolUseBlock.toolUseId)) {
          const completedResult = completedToolResults.get(toolUseBlock.toolUseId)!
          // No events emitted for already-completed tools.
          // The result is included in the final tool result message.
          toolResultBlocks.push(completedResult)
          continue
        }

        if (this.isCancelled) {
          const cancelBlock = new ToolResultBlock({
            toolUseId: toolUseBlock.toolUseId,
            status: 'error',
            content: [new TextBlock('Tool execution cancelled')],
          })
          toolResultBlocks.push(cancelBlock)
          yield new ToolResultEvent({ agent: this, result: cancelBlock, invocationState })
          continue
        }

        try {
          const toolResultBlock = yield* this.executeTool(toolUseBlock, toolRegistry, invocationState)
          toolResultBlocks.push(toolResultBlock)
          yield new ToolResultEvent({ agent: this, result: toolResultBlock, invocationState })
        } catch (error) {
          if (error instanceof InterruptError) {
            // Store pending state with completed results so far
            const completedSoFar: Record<string, { toolResult: ToolResultBlockData }> = {}
            for (const block of toolResultBlocks) {
              completedSoFar[block.toolUseId] = block.toJSON()
            }
            // Also include any previously completed results
            if (completedToolResults) {
              for (const [id, block] of completedToolResults) {
                completedSoFar[id] = block.toJSON()
              }
            }
            this._interruptState.setPendingToolExecution({
              assistantMessageData: assistantMessage!.toJSON(),
              completedToolResults: completedSoFar,
            })
            throw error
          }
          throw error
        }
      }
    } finally {
      toolResultMessage = new Message({ role: 'user', content: toolResultBlocks })
      afterToolsEvent = new AfterToolsEvent({ agent: this, message: toolResultMessage, invocationState })
      yield afterToolsEvent
    }

    return { message: toolResultMessage, afterToolsEvent }
  }

  /**
   * Produces one error ToolResultBlock per tool use block, each carrying
   * `message` as its error text. Shared by pre-launch cancel paths.
   */
  private _cancelAllAsResults(toolUseBlocks: ToolUseBlock[], message: string): ToolResultBlock[] {
    return toolUseBlocks.map(
      (block) =>
        new ToolResultBlock({
          toolUseId: block.toolUseId,
          status: 'error',
          content: [new TextBlock(message)],
        })
    )
  }

  /**
   * Executes tools concurrently by merging N per-tool {@link executeTool}
   * async generators via `Promise.race`. Per-tool event order is preserved
   * (because each generator is iterated serially); cross-tool events may
   * interleave at race resolution boundaries.
   *
   * Per-tool retry (`AfterToolCallEvent.retry`) is isolated — it lives inside
   * `executeTool`'s own `while(true)` loop, so one tool retrying does not
   * disturb its siblings.
   */
  private async *_executeToolsConcurrent(
    toolUseBlocks: ToolUseBlock[],
    toolRegistry: ToolRegistry,
    invocationState: InvocationState,
    completedToolResults?: Map<string, ToolResultBlock>,
    assistantMessage?: Message
  ): AsyncGenerator<AgentStreamEvent, ToolsExecutionResult, undefined> {
    let toolResultMessage: Message
    let afterToolsEvent: AfterToolsEvent

    // Wrap each in-flight `.next()` so the raced promise always resolves to a
    // tagged Step. That prevents one generator rejection from rejecting the
    // whole race and lets us convert per-tool failures into ToolResultBlocks
    // without orphaning other generators.
    type Step =
      | { idx: number; kind: 'next'; res: IteratorResult<AgentStreamEvent, ToolResultBlock> }
      | { idx: number; kind: 'throw'; error: unknown }

    const gens = toolUseBlocks.map((block) => ({
      block,
      gen: completedToolResults?.has(block.toolUseId)
        ? undefined // Skip already-completed tools
        : this.executeTool(block, toolRegistry, invocationState),
    }))

    const step = (idx: number): Promise<Step> =>
      gens[idx]!.gen!.next().then(
        (res): Step => ({ idx, kind: 'next', res }),
        (error: unknown): Step => ({ idx, kind: 'throw', error })
      )

    // Seed completed results from resume state
    const resultsByToolUseId = new Map<string, ToolResultBlock>()
    if (completedToolResults) {
      for (const [id, result] of completedToolResults) {
        resultsByToolUseId.set(id, result)
      }
    }

    // Only race tools that need execution
    const pendingNext = new Map<number, Promise<Step>>()
    for (let idx = 0; idx < gens.length; idx++) {
      if (gens[idx]!.gen) {
        pendingNext.set(idx, step(idx))
      }
    }

    // Track interrupts — let all other tools finish before propagating
    let interruptError: InterruptError | undefined

    try {
      while (pendingNext.size > 0) {
        const winner = await Promise.race(pendingNext.values())
        const { idx } = winner
        const block = gens[idx]!.block

        if (winner.kind === 'throw') {
          pendingNext.delete(idx)

          // Detect InterruptError — don't convert to error result, track it
          if (winner.error instanceof InterruptError) {
            interruptError = winner.error
            continue
          }

          const err = normalizeError(winner.error)
          const result = new ToolResultBlock({
            toolUseId: block.toolUseId,
            status: 'error',
            content: [new TextBlock(err.message)],
            error: err,
          })
          resultsByToolUseId.set(block.toolUseId, result)
          yield new ToolResultEvent({ agent: this, result, invocationState })
          continue
        }

        if (winner.res.done) {
          pendingNext.delete(idx)
          resultsByToolUseId.set(block.toolUseId, winner.res.value)
          yield new ToolResultEvent({ agent: this, result: winner.res.value, invocationState })
        } else {
          try {
            yield winner.res.value
          } catch (e) {
            // InterruptError thrown back into generator from stream() error injection
            if (e instanceof InterruptError) {
              interruptError = e
              pendingNext.delete(idx)
              continue
            }
            throw e
          }
          pendingNext.set(idx, step(idx))
        }
      }

      // After all tools finish, propagate interrupt if one was raised
      if (interruptError) {
        const completedSoFar: Record<string, { toolResult: ToolResultBlockData }> = {}
        for (const [id, result] of resultsByToolUseId) {
          completedSoFar[id] = result.toJSON()
        }
        this._interruptState.setPendingToolExecution({
          assistantMessageData: assistantMessage!.toJSON(),
          completedToolResults: completedSoFar,
        })
        throw interruptError
      }
    } finally {
      // Close any generators still in-flight (e.g. consumer broke out of stream).
      await Promise.allSettled(
        Array.from(pendingNext.keys(), (idx) => gens[idx]!.gen!.return(undefined as unknown as ToolResultBlock))
      )

      // Build the result message from whatever completed, in source order.
      // Missing entries get a fallback error block so the message always
      // accounts for every toolUseBlock the model emitted.
      const toolResultBlocks: ToolResultBlock[] = []
      for (const block of toolUseBlocks) {
        const result = resultsByToolUseId.get(block.toolUseId)
        if (result) {
          toolResultBlocks.push(result)
        } else {
          toolResultBlocks.push(
            new ToolResultBlock({
              toolUseId: block.toolUseId,
              status: 'error',
              content: [new TextBlock('Tool execution interrupted')],
            })
          )
        }
      }

      toolResultMessage = new Message({ role: 'user', content: toolResultBlocks })
      afterToolsEvent = new AfterToolsEvent({ agent: this, message: toolResultMessage, invocationState })
      yield afterToolsEvent
    }

    return { message: toolResultMessage, afterToolsEvent }
  }

  /**
   * Executes a single tool and returns the result.
   * If the tool is not found or fails to return a result, returns an error ToolResult
   * instead of throwing an exception. This allows the agent loop to continue and
   * let the model handle the error gracefully.
   *
   * @param toolUseBlock - Tool use block to execute
   * @param toolRegistry - Registry containing available tools
   * @returns Tool result block
   */
  private async *executeTool(
    toolUseBlock: ToolUseBlock,
    toolRegistry: ToolRegistry,
    invocationState: InvocationState
  ): AsyncGenerator<AgentStreamEvent, ToolResultBlock, undefined> {
    const registryTool = toolRegistry.get(toolUseBlock.name)

    // Create toolUse object for hook events and telemetry. Callbacks may mutate
    // this object's fields (input/name/toolUseId) inside BeforeToolCallEvent.
    const toolUse = {
      name: toolUseBlock.name,
      toolUseId: toolUseBlock.toolUseId,
      input: toolUseBlock.input,
    }

    // Retry loop for tool execution
    while (true) {
      const beforeToolCallEvent = new BeforeToolCallEvent({
        agent: this,
        toolUse,
        tool: registryTool,
        invocationState,
      })
      yield beforeToolCallEvent

      // Resolve the tool that would actually execute. selectedTool wins;
      // otherwise if the hook renamed toolUse.name, re-resolve from the
      // registry under the new name; otherwise use the original registry
      // lookup. Resolved before the cancel check so AfterToolCallEvent.tool
      // is consistent whether the cancel or execution branch runs.
      const effectiveTool =
        beforeToolCallEvent.selectedTool ??
        (toolUse.name !== toolUseBlock.name ? toolRegistry.get(toolUse.name) : registryTool)

      // Cancel individual tool if hook requested it
      if (beforeToolCallEvent.cancel) {
        const cancelMessage =
          typeof beforeToolCallEvent.cancel === 'string' ? beforeToolCallEvent.cancel : 'Tool cancelled by hook'
        const cancelResult = new ToolResultBlock({
          toolUseId: toolUse.toolUseId,
          status: 'error',
          content: [new TextBlock(cancelMessage)],
        })
        const afterToolCallEvent = new AfterToolCallEvent({
          agent: this,
          toolUse,
          tool: effectiveTool,
          result: cancelResult,
          invocationState,
        })
        yield afterToolCallEvent
        if (afterToolCallEvent.retry) {
          continue
        }
        return afterToolCallEvent.result
      }

      // Start tool span within loop span context
      const toolSpan = this._tracer.startToolCallSpan({
        tool: toolUse,
      })

      // Track tool execution time for metrics
      const toolStartTime = Date.now()

      let toolResult: ToolResultBlock
      let error: Error | undefined

      if (!effectiveTool) {
        // Tool not found
        toolResult = new ToolResultBlock({
          toolUseId: toolUse.toolUseId,
          status: 'error',
          content: [new TextBlock(`Tool '${toolUse.name}' not found in registry`)],
        })
      } else {
        // Execute tool within the tool span context
        const toolContext: ToolContext = {
          toolUse: {
            name: toolUse.name,
            toolUseId: toolUse.toolUseId,
            input: toolUse.input,
          },
          agent: this,
          invocationState,
          interrupt: <T = JSONValue>(params: InterruptParams): T => {
            return interruptFromAgent<T>(this, `tool:${toolUseBlock.toolUseId}:${params.name}`, params, 'tool')
          },
        }

        try {
          // Manually iterate tool stream to wrap each ToolStreamEvent in ToolStreamUpdateEvent.
          // This keeps the tool authoring interface unchanged — tools construct ToolStreamEvent
          // without knowledge of agents or hooks, and we wrap at the boundary.
          // Tool execution is ran within the tool span's context so that
          // downstream calls (e.g., MCP clients) can propagate trace context
          const toolGenerator = this._tracer.withSpanContext(toolSpan, () => effectiveTool.stream(toolContext))
          let toolNext = await this._tracer.withSpanContext(toolSpan, () => toolGenerator.next())
          while (!toolNext.done) {
            yield new ToolStreamUpdateEvent({ agent: this, event: toolNext.value, invocationState })
            toolNext = await this._tracer.withSpanContext(toolSpan, () => toolGenerator.next())
          }
          const result = toolNext.value

          if (!result) {
            // Tool didn't return a result
            toolResult = new ToolResultBlock({
              toolUseId: toolUse.toolUseId,
              status: 'error',
              content: [new TextBlock(`Tool '${toolUse.name}' did not return a result`)],
            })
          } else {
            toolResult = result
            error = result.error
          }
        } catch (e) {
          // Re-throw InterruptError to allow interrupt handling
          if (e instanceof InterruptError) {
            throw e
          }
          // Tool execution failed with error
          error = normalizeError(e)
          toolResult = new ToolResultBlock({
            toolUseId: toolUse.toolUseId,
            status: 'error',
            content: [new TextBlock(error.message)],
            error,
          })
        }
      }

      // End tool span with the raw tool result — telemetry reflects what the
      // tool actually returned, independent of AfterToolCallEvent mutations.
      this._tracer.endToolCallSpan(toolSpan, { toolResult, ...(error && { error }) })

      // End tool metrics tracking
      this._meter.endToolCall({
        tool: toolUse,
        duration: Date.now() - toolStartTime,
        success: toolResult.status === 'success',
      })

      // Single point for AfterToolCallEvent
      const afterToolCallEvent = new AfterToolCallEvent({
        agent: this,
        toolUse,
        tool: effectiveTool,
        result: toolResult,
        invocationState,
        ...(error !== undefined && { error }),
      })
      yield afterToolCallEvent

      if (afterToolCallEvent.retry) {
        continue
      }

      // Return the (possibly mutated) result so hook transformations propagate
      // to ToolResultEvent and the conversation message the model will see.
      return afterToolCallEvent.result
    }
  }

  /**
   * Redacts the last message in the conversation history.
   * Called when guardrails block user input and redaction is enabled.
   *
   * Follows the redaction strategy:
   * - If the message contains at least one toolResult block, all toolResult blocks
   *   are kept with redacted content, and all other blocks are discarded.
   * - Otherwise, the entire content is replaced with a single text block containing
   *   the redaction message.
   *
   * @param redactMessage - The redaction message to replace the content with
   */
  private _redactLastMessage(redactMessage: string): void {
    // Find and redact the last message
    const lastIndex = this.messages.length - 1
    if (lastIndex >= 0) {
      const lastMessage = this.messages[lastIndex]
      if (lastMessage && lastMessage.role === 'user') {
        // Collect only tool result blocks with redacted content
        const redactedContent: ContentBlock[] = []
        for (const block of lastMessage.content) {
          if (block.type === 'toolResultBlock') {
            // Preserve tool result block structure, only redact its content
            redactedContent.push(
              new ToolResultBlock({
                toolUseId: block.toolUseId,
                status: block.status,
                content: [new TextBlock(redactMessage)],
              })
            )
          }
        }

        // If no tool result blocks were found, replace entire content with redaction message
        if (redactedContent.length === 0) {
          redactedContent.push(new TextBlock(redactMessage))
        }

        this.messages[lastIndex] = new Message({
          role: 'user',
          content: redactedContent,
        })
      } else if (lastMessage) {
        // Unexpected state: redaction requested but last message is not from user
        logger.warn(
          `role=<${lastMessage.role}> | received input redaction but last message is not from user | redaction skipped`
        )
      }
    }
  }

  /**
   * Estimate the input token count for the next model call.
   *
   * Uses the token counting strategy: reads inputTokens + outputTokens
   * from the last assistant message's metadata as a known baseline, then estimates
   * only new messages added after it. Falls back to full estimation when no metadata
   * is available (cold start or first call).
   *
   * @param streamOptions - The stream options containing system prompt and tool specs
   * @returns Estimated input token count
   */
  private async _estimateInputTokens(streamOptions: StreamOptions): Promise<number> {
    // Find the last assistant message with usage metadata
    let lastAssistantIdx = -1
    for (let i = this.messages.length - 1; i >= 0; i--) {
      if (this.messages[i]!.role === 'assistant' && this.messages[i]!.metadata?.usage) {
        lastAssistantIdx = i
        break
      }
    }

    let estimate: number
    if (lastAssistantIdx >= 0) {
      const usage = this.messages[lastAssistantIdx]!.metadata!.usage!
      const knownBaseline = usage.inputTokens + usage.outputTokens
      const newMessages = this.messages.slice(lastAssistantIdx + 1)
      if (newMessages.length === 0) {
        estimate = knownBaseline
      } else {
        // System prompt and tool spec tokens are already included in the baseline from the prior model call
        estimate = knownBaseline + (await this.model.countTokens(newMessages))
      }
    } else {
      estimate = await this.model.countTokens(this.messages, {
        ...(streamOptions.systemPrompt !== undefined && { systemPrompt: streamOptions.systemPrompt }),
        ...(streamOptions.toolSpecs !== undefined && { toolSpecs: streamOptions.toolSpecs }),
      })
    }

    return estimate
  }

  /**
   * Appends a message to the conversation history and fires MessageAddedEvent hooks.
   *
   * Used by {@link ToolCaller} (via the helper passed to `ToolCaller.create`) for
   * direct tool calls that cannot yield events into the agent stream. This stays
   * private — callers outside the agent should never directly mutate messages.
   */
  private async _appendMessageAndFireHooks(message: Message, invocationState: InvocationState = {}): Promise<void> {
    this.messages.push(message)
    await this._hooksRegistry.invokeCallbacks(new MessageAddedEvent({ agent: this, message, invocationState }))
  }

  /**
   * Appends a message to the conversation history and returns the event for yielding.
   *
   * @param message - The message to append
   * @returns MessageAddedEvent to be yielded
   */
  private _appendMessage(message: Message, invocationState: InvocationState): MessageAddedEvent {
    this.messages.push(message)
    return new MessageAddedEvent({ agent: this, message, invocationState })
  }
}

const INVALID_TOOL_NAME_PLACEHOLDER = 'INVALID_TOOL_NAME'

/**
 * Replaces invalid tool-use names on assistant messages with `INVALID_TOOL_NAME`
 * so providers that reject malformed names don't fail the whole request.
 * Returns the input unchanged (same reference) when nothing needs replacing.
 */
function normalizeToolUseNames(messages: Message[]): Message[] {
  let replaced = false
  const next = messages.map((message) => {
    if (!message || message.role !== 'assistant') return message

    let messageReplaced = false
    const content = message.content.map((block) => {
      if (block.type !== 'toolUseBlock') return block
      if (isValidToolName(block.name)) return block
      messageReplaced = true
      logger.debug(`tool_name=<${block.name}> | replacing invalid tool name with ${INVALID_TOOL_NAME_PLACEHOLDER}`)
      return new ToolUseBlock({
        name: INVALID_TOOL_NAME_PLACEHOLDER,
        toolUseId: block.toolUseId,
        input: block.input,
        ...(block.reasoningSignature !== undefined && { reasoningSignature: block.reasoningSignature }),
      })
    })

    if (!messageReplaced) return message
    replaced = true
    return new Message({
      role: message.role,
      content,
      ...(message.metadata !== undefined && { metadata: message.metadata }),
    })
  })

  return replaced ? next : messages
}

/**
 * Recursively flattens nested arrays of tools into a single flat array.
 * @param tools - Tools or nested arrays of tools
 * @returns Flat array of tools and MCP clients
 */
function flattenTools(toolList: ToolList): { tools: Tool[]; mcpClients: McpClient[] } {
  const tools: Tool[] = []
  const mcpClients: McpClient[] = []

  for (const item of toolList) {
    if (Array.isArray(item)) {
      const { tools: nestedTools, mcpClients: nestedMcpClients } = flattenTools(item)
      tools.push(...nestedTools)
      mcpClients.push(...nestedMcpClients)
    } else if (item instanceof Agent) {
      tools.push(item.asTool())
    } else if (item instanceof McpClient) {
      mcpClients.push(item)
    } else {
      tools.push(item)
    }
  }

  return { tools, mcpClients }
}
