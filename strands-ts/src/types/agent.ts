import type { StateStore } from '../state-store.js'
import type { ContentBlock, ContentBlockData, Message, MessageData, StopReason, SystemPrompt } from './messages.js'
import type { Interrupt } from '../interrupt.js'
import type { InterruptResponseContent, InterruptResponseContentData } from './interrupt.js'
import type { AgentTrace } from '../telemetry/tracer.js'
import type { Snapshot } from './snapshot.js'
import type { TakeSnapshotOptions } from '../agent/snapshot.js'
import type {
  BeforeInvocationEvent,
  AfterInvocationEvent,
  BeforeModelCallEvent,
  AfterModelCallEvent,
  BeforeToolsEvent,
  AfterToolsEvent,
  BeforeToolCallEvent,
  AfterToolCallEvent,
  MessageAddedEvent,
  ModelStreamUpdateEvent,
  ContentBlockEvent,
  ModelMessageEvent,
  ToolResultEvent,
  ToolStreamUpdateEvent,
  AgentResultEvent,
  InterruptEvent,
  HookableEvent,
  StreamEvent,
} from '../hooks/events.js'
import type { HookCallback, HookableEventConstructor, HookCallbackOptions, HookCleanup } from '../hooks/types.js'
import type { ToolRegistry } from '../registry/tool-registry.js'
import type { Model } from '../models/model.js'
import type { z } from 'zod'
import { AgentMetrics } from '../telemetry/meter.js'

/**
 * Arguments for invoking an agent.
 *
 * Supports multiple input formats:
 * - `string` - User text input (wrapped in TextBlock, creates user Message)
 * - `ContentBlock[]` | `ContentBlockData[]` - Array of content blocks (creates single user Message)
 * - `Message[]` | `MessageData[]` - Array of messages (appends all to conversation)
 * - `InterruptResponseContent[]` - Array of interrupt responses (resumes from interrupted state)
 */
export type InvokeArgs =
  | string
  | ContentBlock[]
  | ContentBlockData[]
  | Message[]
  | MessageData[]
  | InterruptResponseContent[]
  | InterruptResponseContentData[]

/**
 * Per-invocation state threaded through hooks and tools for a single agent
 * invocation, and returned on {@link AgentResult.invocationState}. One object
 * per invocation, shared by reference; mutations by hooks or tools are visible
 * to subsequent hooks, tools, and recursive loop cycles.
 *
 * Typically used for request-scoped context (`userId`, `requestId`, `traceId`)
 * or cross-hook counters. The core agent loop writes no keys into it — the
 * key space is the caller's. Transport bridges may populate reserved keys
 * (e.g. `A2AExecutor` sets `a2aRequestContext`); those bridges document their
 * own reserved keys.
 *
 * Distinct from {@link LocalAgent.appState}: `appState` is durable across
 * invocations, JSON-serializable, and deep-copied. `invocationState` is
 * ephemeral and accepts arbitrary values.
 *
 * Excluded from `toJSON()` on {@link AgentResult} and all hook events because
 * values may not be serializable; callers produce a serialized form explicitly
 * if needed.
 */
export type InvocationState = Record<string, unknown>

/**
 * Options for a single agent invocation.
 */
export interface InvokeOptions {
  /**
   * Zod schema for structured output validation, overriding the constructor-provided schema for this invocation only.
   */
  structuredOutputSchema?: z.ZodSchema

  /**
   * Per-invocation state. Passed to lifecycle hook events and tools, and
   * returned on {@link AgentResult.invocationState}. Mutable — hooks and tools
   * may read and write. See {@link InvocationState} for details.
   *
   * Defaults to an empty object when omitted.
   */
  invocationState?: InvocationState

  /**
   * External AbortSignal for cancelling the agent invocation.
   *
   * Use this when cancellation is driven by something outside the agent — for example,
   * a client disconnect, a framework-managed request lifecycle, or a declarative timeout.
   * The agent composes this signal with its own internal controller, so both
   * `agent.cancel()` and this signal can trigger cancellation independently.
   *
   * When the signal fires, the agent stops at the next cancellation checkpoint and
   * returns an AgentResult with `stopReason: 'cancelled'`. See
   * {@link LocalAgent.cancelSignal} for how tools can participate in cancellation.
   *
   * @example
   * ```typescript
   * // Timeout-based cancellation
   * const result = await agent.invoke('Hello', {
   *   cancelSignal: AbortSignal.timeout(5000),
   * })
   *
   * // Framework-driven cancellation (e.g., client disconnect)
   * app.post('/chat', async (req, res) => {
   *   const result = await agent.invoke(req.body.message, {
   *     cancelSignal: req.signal,
   *   })
   *   res.json(result)
   * })
   * ```
   */
  cancelSignal?: AbortSignal
}

/**
 * Interface for agents that support request-response invocation.
 *
 * Both `Agent` (full orchestration agent) and `A2AAgent` (remote agent proxy)
 * implement this interface, enabling polymorphic usage across the SDK.
 */
export interface InvokableAgent {
  /**
   * The unique identifier of the agent instance.
   */
  readonly id: string

  /**
   * The name of the agent.
   */
  readonly name?: string

  /**
   * Optional description of what the agent does.
   */
  readonly description?: string

  /**
   * Invokes the agent and returns the final result.
   *
   * @param args - Arguments for invoking the agent
   * @param options - Optional invocation options (e.g. structured output schema)
   * @returns Promise that resolves to the final AgentResult
   */
  invoke(args: InvokeArgs, options?: InvokeOptions): Promise<AgentResult>

  /**
   * Streams the agent execution, yielding events and returning the final result.
   *
   * @param args - Arguments for invoking the agent
   * @param options - Optional invocation options (e.g. structured output schema)
   * @returns Async generator that yields stream events and returns AgentResult
   */
  stream(args: InvokeArgs, options?: InvokeOptions): AsyncGenerator<StreamEvent, AgentResult, undefined>
}

/**
 * Branded symbol that prevents external implementations of {@link LocalAgent}.
 *
 * @internal
 */
export declare const localAgentSymbol: unique symbol

/**
 * Interface for agents with locally accessible state, messages, tools, and hooks.
 *
 * This interface is exported for typing purposes only (e.g. in {@link ToolContext},
 * hook events, and {@link Plugin.initAgent}). The Strands SDK is responsible for
 * providing all implementations. External code should not implement this interface.
 *
 * @internal Not for external implementation. Use the {@link Agent} class instead.
 */
export interface LocalAgent {
  /** @internal Prevents external implementations of this interface. */
  readonly [localAgentSymbol]: true

  /**
   * The unique identifier of the agent instance.
   */
  readonly id: string

  /**
   * App state storage accessible to tools and application logic.
   */
  appState: StateStore

  /**
   * The conversation history of messages between user and assistant.
   */
  messages: Message[]

  /**
   * Runtime state for the model provider. Used by stateful models to persist
   * provider-specific data (e.g., response IDs for server-side conversation chaining)
   * across invocations.
   */
  modelState: StateStore

  /**
   * The tool registry for registering tools with the agent.
   */
  readonly toolRegistry: ToolRegistry

  /**
   * The model provider used by the agent for inference.
   */
  readonly model: Model

  /**
   * The system prompt to pass to the model provider.
   */
  systemPrompt?: SystemPrompt

  /**
   * The cancellation signal for the current invocation.
   *
   * Cancellation in the SDK is **cooperative**. The agent checks for cancellation at
   * built-in checkpoints (between loop cycles, during model streaming, and between
   * sequential tool executions), but once a tool callback is running, only the tool
   * itself can respond to cancellation. There are two patterns:
   *
   * **Polling** — check `cancelSignal.aborted` between steps in a loop:
   * ```ts
   * callback: async ({ items }, context) => {
   *   const results = []
   *   for (const item of items) {
   *     if (context.agent.cancelSignal.aborted) return results
   *     results.push(await process(item))
   *   }
   *   return results
   * }
   * ```
   *
   * **Signal forwarding** — pass to APIs that accept `AbortSignal`:
   * ```ts
   * callback: async ({ url }, context) => {
   *   const res = await fetch(url, { signal: context.agent.cancelSignal })
   *   return res.text()
   * }
   * ```
   *
   * If a tool does neither, it will run to completion even after cancellation is
   * requested. The agent will resume cancellation handling after the tool returns.
   *
   * The cancelSignal can also be utilized in hook callbacks.
   */
  readonly cancelSignal: AbortSignal

  /**
   * Register a hook callback for a specific event type.
   *
   * Hooks execute in order from lowest to highest. Lower values always run
   * first, on both Before* and After* events. Within the same order, After*
   * events reverse registration order for cleanup symmetry.
   *
   * @param eventType - The event class constructor to register the callback for
   * @param callback - The callback function to invoke when the event occurs
   * @param options - Optional configuration including execution order
   * @returns Cleanup function that removes the callback when invoked
   */
  addHook<T extends HookableEvent>(
    eventType: HookableEventConstructor<T>,
    callback: HookCallback<T>,
    options?: HookCallbackOptions
  ): HookCleanup

  /**
   * Captures a point-in-time snapshot of the agent's current state.
   *
   * @param options - Controls which fields to capture and optional app data to store
   * @returns A Snapshot containing the captured agent state
   */
  takeSnapshot(options: TakeSnapshotOptions): Snapshot

  /**
   * Restores agent state from a previously captured snapshot.
   *
   * Only fields present in `snapshot.data` are restored; absent fields are left unchanged.
   *
   * @param snapshot - The snapshot to restore from
   */
  loadSnapshot(snapshot: Snapshot): void
}

/**
 * Result returned by the agent loop.
 */
export class AgentResult {
  readonly type = 'agentResult' as const

  /**
   * The stop reason from the final model response.
   */
  readonly stopReason: StopReason

  /**
   * The last message added to the messages array.
   */
  readonly lastMessage: Message

  /**
   * Local execution traces collected during the agent invocation.
   * Contains timing and hierarchy of operations within the agent loop.
   */
  readonly traces?: AgentTrace[]

  /**
   * The validated structured output from the LLM, if a schema was provided.
   * Type represents any validated Zod schema output.
   */
  readonly structuredOutput?: z.output<z.ZodType>

  /**
   * Aggregated metrics for the agent's loop execution.
   * Tracks cycle counts, token usage, tool execution stats, and model latency.
   */
  readonly metrics?: AgentMetrics

  /**
   * Per-invocation state passed into the agent, threaded through hooks and
   * tools, and surfaced here at the end of the invocation. See
   * {@link InvocationState} for details. Always defined — defaults to `{}` when
   * no `invocationState` was provided in {@link InvokeOptions}.
   */
  readonly invocationState: InvocationState

  /**
   * Interrupts that caused the agent to stop, when `stopReason` is `'interrupt'`.
   * Contains the unanswered interrupts that require human input to resume.
   */
  readonly interrupts?: Interrupt[]

  constructor(data: {
    stopReason: StopReason
    lastMessage: Message
    invocationState: InvocationState
    traces?: AgentTrace[]
    metrics?: AgentMetrics
    structuredOutput?: z.output<z.ZodType>
    interrupts?: Interrupt[]
  }) {
    this.stopReason = data.stopReason
    this.lastMessage = data.lastMessage
    this.invocationState = data.invocationState
    if (data.traces !== undefined) {
      this.traces = data.traces
    }
    if (data.metrics !== undefined) {
      this.metrics = data.metrics
    }
    if (data.structuredOutput !== undefined) {
      this.structuredOutput = data.structuredOutput
    }
    if (data.interrupts !== undefined) {
      this.interrupts = data.interrupts
    }
  }

  /**
   * The most recent input token count from the last model invocation.
   * Convenience accessor that delegates to `metrics.latestContextSize`.
   * Returns `undefined` when no metrics or invocations are available.
   */
  get contextSize(): number | undefined {
    return this.metrics?.latestContextSize
  }

  /**
   * Projected context size for the next model call (inputTokens + outputTokens from the last call).
   * Convenience accessor that delegates to `metrics.projectedContextSize`.
   * Returns `undefined` when no metrics or invocations are available.
   */
  get projectedContextSize(): number | undefined {
    return this.metrics?.projectedContextSize
  }

  /**
   * Custom JSON serialization that excludes traces, metrics, and invocationState.
   * Traces and metrics are excluded to avoid sending large payloads over the wire
   * in API responses; `invocationState` is excluded because its values are
   * caller-owned and may not be serializable (see {@link InvocationState}).
   *
   * All three remain accessible via their properties for debugging.
   *
   * @returns Object representation for safe serialization
   */
  public toJSON(): object {
    return {
      type: this.type,
      stopReason: this.stopReason,
      lastMessage: this.lastMessage,
      ...(this.structuredOutput !== undefined && { structuredOutput: this.structuredOutput }),
    }
  }

  /**
   * Extracts a string representation of the result.
   *
   * Priority order:
   * 1. `interrupts` serialized as JSON, if any are present
   * 2. `structuredOutput` serialized as JSON
   * 3. Text from `textBlock`, `reasoningBlock`, and `citationsBlock` content blocks
   *
   * @returns String representation of the result: JSON for interrupts/structuredOutput, or text content joined by newlines.
   */
  public toString(): string {
    if (this.interrupts && this.interrupts.length > 0) {
      return JSON.stringify(this.interrupts)
    }

    if (this.structuredOutput !== undefined) {
      return JSON.stringify(this.structuredOutput)
    }

    const textParts: string[] = []

    for (const block of this.lastMessage.content) {
      switch (block.type) {
        case 'textBlock':
          textParts.push(block.text)
          break
        case 'reasoningBlock':
          if (block.text) {
            // Add indentation to reasoning content
            const indentedText = block.text.replace(/\n/g, '\n   ')
            textParts.push(`💭 Reasoning:\n   ${indentedText}`)
          }
          break
        case 'citationsBlock':
          for (const c of block.content) {
            if ('text' in c) {
              textParts.push(c.text)
            }
          }
          break
        default:
          console.debug(`Skipping content block type: ${block.type}`)
          break
      }
    }

    return textParts.join('\n')
  }
}

/**
 * Union type representing all possible streaming events from an agent.
 * This includes model events, tool events, and agent-specific lifecycle events.
 *
 * This is a discriminated union where each event has a unique type field,
 * allowing for type-safe event handling using switch statements.
 *
 * Every member extends {@link HookableEvent} (which extends {@link StreamEvent}),
 * making all events both streamable and subscribable via hook callbacks.
 * Raw data objects from lower layers (model, tools) should be wrapped
 * in a StreamEvent subclass at the agent boundary rather than added directly.
 */
export type AgentStreamEvent =
  | ModelStreamUpdateEvent
  | ContentBlockEvent
  | ModelMessageEvent
  | ToolStreamUpdateEvent
  | ToolResultEvent
  | BeforeInvocationEvent
  | AfterInvocationEvent
  | BeforeModelCallEvent
  | AfterModelCallEvent
  | BeforeToolsEvent
  | AfterToolsEvent
  | BeforeToolCallEvent
  | AfterToolCallEvent
  | MessageAddedEvent
  | InterruptEvent
  | AgentResultEvent
