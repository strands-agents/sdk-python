import { normalizeError } from '../../errors.js'
import { AfterToolCallEvent, BeforeToolCallEvent, ToolStreamUpdateEvent } from '../../hooks/events.js'
import { InterruptError, interruptFromState } from '../../interrupt.js'
import { createMiddlewareInterrupt } from '../../middleware/interrupt.js'
import { ExecuteToolStage } from '../../middleware/index.js'
import { deepCopy } from '../../types/json.js'
import { TextBlock, ToolResultBlock } from '../../types/messages.js'

import type { Agent } from '../../agent/agent.js'
import type { BackgroundTasks } from '../../background-tasks/background-tasks.js'
import type { ToolUseData } from '../../hooks/events.js'
import type { ExecuteToolContext, ExecuteToolResult, MiddlewareRegistry } from '../../middleware/index.js'
import type { Meter } from '../../telemetry/meter.js'
import type { Tracer } from '../../telemetry/tracer.js'
import type { AgentStreamEvent, InvocationState } from '../../types/agent.js'
import type { InterruptParams } from '../../types/interrupt.js'
import type { JSONValue } from '../../types/json.js'
import type { Message, ToolResultBlockData, ToolUseBlock } from '../../types/messages.js'
import type { Tool, ToolContext } from '../tool.js'
import type { InterruptState } from '../../interrupt.js'
import type { SpanContext } from '@opentelemetry/api'

/**
 * Dependencies supplied for one tool-executor invocation.
 *
 * @internal
 */
export interface ToolExecutorOptions {
  /** Agent whose tools are being executed. */
  readonly agent: Agent
  /** Registry containing middleware for the agent. */
  readonly middlewareRegistry: MiddlewareRegistry
  /** Tracer used to record tool calls. */
  readonly tracer: Tracer
  /** Meter used to record tool-call metrics. */
  readonly meter: Meter
  /** Background Tasks integration configured on the agent. */
  readonly backgroundTasks?: BackgroundTasks
  readonly passId: string
  readonly cancelSignal: AbortSignal
  readonly interruptState: InterruptState
  readonly backgroundTask: boolean
  /** Whether `BeforeToolCallEvent` already completed before this execution was scheduled. */
  readonly beforeToolCallCompleted: boolean
  readonly originSpanContext?: SpanContext
}

/**
 * Input for executing the tool calls from one model turn.
 *
 * @internal
 */
export interface ToolExecutionInput {
  /** Tool calls to execute. */
  readonly toolUseBlocks: readonly ToolUseBlock[]
  /** Accumulates completed tool results and may remain partial when execution exits early. */
  readonly toolResultBlocks: ToolResultBlock[]
  /** State shared across the current agent invocation. */
  readonly invocationState: InvocationState
  /** Assistant message that requested the tool calls. */
  readonly assistantMessage: Message
  /** Results restored when resuming interrupted execution. */
  readonly completedToolResults?: ReadonlyMap<string, ToolResultBlock>
}

/**
 * Shared pipeline for executing tool calls.
 *
 * @internal
 */
export abstract class ToolExecutor {
  /**
   * Executes the tool calls from one model turn.
   *
   * @param options - Agent dependencies used to execute tools
   * @param input - Tool calls and invocation state
   * @returns Stream of tool lifecycle events
   * @internal
   */
  abstract execute(
    options: ToolExecutorOptions,
    input: ToolExecutionInput
  ): AsyncGenerator<AgentStreamEvent, void, undefined>

  // Tool lookup and tool-body failures become ToolResultBlocks so the model can
  // respond to them; lifecycle and middleware failures may still propagate.
  protected async *executeTool(
    options: ToolExecutorOptions,
    toolUseBlock: ToolUseBlock,
    invocationState: InvocationState
  ): AsyncGenerator<AgentStreamEvent, ToolResultBlock, undefined> {
    const registryTool = options.agent.toolRegistry.get(toolUseBlock.name)
    const routed =
      options.backgroundTasks && !options.backgroundTask
        ? options.backgroundTasks._routeToolCall({
            toolUseBlock,
            tool: registryTool,
          })
        : { kind: 'execute' as const, input: toolUseBlock.input }

    if (routed.kind === 'result') {
      return this._normalizeToolResultId(routed.result, toolUseBlock.toolUseId)
    }

    // Callbacks may replace or mutate this value inside BeforeToolCallEvent.
    let toolUse = {
      name: toolUseBlock.name,
      toolUseId: toolUseBlock.toolUseId,
      input: routed.input,
    }
    let approvedToolUse = toolUse
    let effectiveTool = registryTool

    while (true) {
      if (!options.beforeToolCallCompleted) {
        const beforeToolCallEvent = new BeforeToolCallEvent({
          agent: options.agent,
          toolUse,
          tool: registryTool,
          invocationState,
        })
        beforeToolCallEvent._setInterruptState(options.interruptState)
        yield beforeToolCallEvent

        const hookToolUse = beforeToolCallEvent.toolUse
        // selectedTool takes precedence over resolving a hook-renamed toolUse;
        // otherwise retain the original registry lookup. Resolve before cancellation
        // so AfterToolCallEvent reports the same effective tool on every path.
        effectiveTool =
          beforeToolCallEvent.selectedTool ??
          (hookToolUse.name !== toolUseBlock.name ? options.agent.toolRegistry.get(hookToolUse.name) : registryTool)
        approvedToolUse = hookToolUse
        toolUse = {
          ...hookToolUse,
          toolUseId: toolUseBlock.toolUseId,
        }

        if (beforeToolCallEvent.cancel) {
          const cancelMessage =
            typeof beforeToolCallEvent.cancel === 'string' ? beforeToolCallEvent.cancel : 'Tool cancelled by hook'
          const cancelResult = new ToolResultBlock({
            toolUseId: toolUse.toolUseId,
            status: 'error',
            content: [new TextBlock(cancelMessage)],
          })
          const afterToolCallEvent = new AfterToolCallEvent({
            agent: options.agent,
            toolUse,
            tool: effectiveTool,
            result: cancelResult,
            invocationState,
          })
          yield afterToolCallEvent
          if (this._shouldRetry(afterToolCallEvent, options.cancelSignal)) {
            continue
          }
          return this._normalizeToolResultId(afterToolCallEvent.result, toolUseBlock.toolUseId)
        }
      }

      if (routed.kind === 'background') {
        return options.backgroundTasks!._submitApprovedToolCall(options.agent, {
          originalToolUseBlock: toolUseBlock,
          toolUse: approvedToolUse,
          effectiveTool,
          invocationState,
          passId: options.passId,
          ...(options.originSpanContext && { originSpanContext: options.originSpanContext }),
        })
      }

      const toolResult = this._normalizeToolResultId(
        yield* this._executeToolWithMiddleware(options, effectiveTool, toolUse, invocationState),
        toolUseBlock.toolUseId
      )
      const error = toolResult.error
      const afterToolCallEvent = new AfterToolCallEvent({
        agent: options.agent,
        toolUse,
        tool: effectiveTool,
        result: toolResult,
        invocationState,
        ...(error !== undefined && { error }),
      })
      yield afterToolCallEvent

      if (this._shouldRetry(afterToolCallEvent, options.cancelSignal)) {
        continue
      }

      // Return the hook-transformed result so downstream events and the model
      // observe the same value.
      return this._normalizeToolResultId(afterToolCallEvent.result, toolUseBlock.toolUseId)
    }
  }

  private _shouldRetry(afterToolCallEvent: AfterToolCallEvent, cancelSignal: AbortSignal): boolean {
    return afterToolCallEvent.retry === true && !cancelSignal.aborted
  }

  private _normalizeToolResultId(result: ToolResultBlock, toolUseId: string): ToolResultBlock {
    if (result.toolUseId === toolUseId) {
      return result
    }

    return new ToolResultBlock({
      toolUseId,
      status: result.status,
      content: result.content,
      ...(result.error !== undefined && { error: result.error }),
    })
  }

  // Keys and serialized results use model-issued toolUseIds so resume and
  // provider correlation match the assistant's tool-use blocks.
  protected _storePendingToolExecution(
    options: ToolExecutorOptions,
    assistantMessage: Message,
    completedToolResults: ReadonlyMap<string, ToolResultBlock>
  ): void {
    const serializedResults: Record<string, { toolResult: ToolResultBlockData }> = {}
    for (const [toolUseId, result] of completedToolResults) {
      serializedResults[toolUseId] = result.toJSON()
    }
    options.interruptState.setPendingToolExecution({
      assistantMessageData: assistantMessage.toJSON(),
      completedToolResults: serializedResults,
    })
  }

  private async *_executeToolWithMiddleware(
    options: ToolExecutorOptions,
    tool: Tool | undefined,
    toolUse: ToolUseData,
    invocationState: InvocationState
  ): AsyncGenerator<AgentStreamEvent, ToolResultBlock, undefined> {
    const context: ExecuteToolContext = {
      agent: options.agent,
      tool,
      toolUse: deepCopy(toolUse) as unknown as ToolUseData,
      invocationState,
      cancelSignal: options.cancelSignal,
      interrupt: createMiddlewareInterrupt(options.interruptState, `middleware:executeTool:${toolUse.toolUseId}`),
    }

    // async function* does not bind lexical `this`; capture the executor for the terminal callback.
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const executor = this
    const middlewareResult = yield* options.middlewareRegistry.invoke(
      ExecuteToolStage,
      context,
      async function* (
        middlewareContext: ExecuteToolContext
      ): AsyncGenerator<AgentStreamEvent, ExecuteToolResult, undefined> {
        const result = yield* executor._executeToolCore(
          options,
          middlewareContext.tool,
          middlewareContext.toolUse,
          middlewareContext.invocationState
        )
        return { result }
      }
    )
    return middlewareResult.result
  }

  private async *_executeToolCore(
    options: ToolExecutorOptions,
    effectiveTool: Tool | undefined,
    toolUse: ToolUseData,
    invocationState: InvocationState
  ): AsyncGenerator<AgentStreamEvent, ToolResultBlock, undefined> {
    const toolSpan = options.tracer.startToolCallSpan({ tool: toolUse })
    const toolStartTime = Date.now()

    let toolResult: ToolResultBlock | undefined
    let error: Error | undefined

    try {
      if (!effectiveTool) {
        toolResult = new ToolResultBlock({
          toolUseId: toolUse.toolUseId,
          status: 'error',
          content: [new TextBlock(`Tool '${toolUse.name}' not found in registry`)],
        })
      } else {
        const toolContext: ToolContext = {
          toolUse: {
            name: toolUse.name,
            toolUseId: toolUse.toolUseId,
            input: toolUse.input,
          },
          agent: options.agent,
          invocationState,
          cancelSignal: options.cancelSignal,
          interrupt: <T = JSONValue>(params: InterruptParams): T =>
            interruptFromState<T>(options.interruptState, `tool:${toolUse.toolUseId}:${params.name}`, params, 'tool'),
        }

        // Iterate manually to wrap raw tool events at the agent boundary and
        // re-enter the tool span for every asynchronous step.
        const toolGenerator = options.tracer.withSpanContext(toolSpan, () => effectiveTool.stream(toolContext))
        let toolNext = await options.tracer.withSpanContext(toolSpan, () => toolGenerator.next())
        while (!toolNext.done) {
          yield new ToolStreamUpdateEvent({ agent: options.agent, event: toolNext.value, invocationState })
          toolNext = await options.tracer.withSpanContext(toolSpan, () => toolGenerator.next())
        }

        toolResult =
          toolNext.value ??
          new ToolResultBlock({
            toolUseId: toolUse.toolUseId,
            status: 'error',
            content: [new TextBlock(`Tool '${toolUse.name}' did not return a result`)],
          })
        error = toolNext.value?.error
      }

      return toolResult
    } catch (caughtError) {
      if (caughtError instanceof InterruptError) {
        throw caughtError
      }

      error = normalizeError(caughtError)
      toolResult = new ToolResultBlock({
        toolUseId: toolUse.toolUseId,
        status: 'error',
        content: [new TextBlock(error.message)],
        error,
      })
      return toolResult
    } finally {
      // Record the raw execution outcome before AfterToolCallEvent can transform it.
      options.tracer.endToolCallSpan(toolSpan, {
        ...(toolResult !== undefined && { toolResult }),
        ...(error !== undefined && { error }),
      })
      options.meter.endToolCall({
        tool: toolUse,
        duration: Date.now() - toolStartTime,
        success: toolResult?.status === 'success',
      })
    }
  }
}
