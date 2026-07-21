/**
 * AgentDelegation — enforces delegation semantics for tool routing.
 *
 * When a tool is configured with `delegate: true`, this plugin ensures:
 * 1. The delegation tool is the only tool called in the turn (single-call constraint)
 * 2. The agent loop exits immediately after a successful delegation (via stopEventLoop)
 * 3. The AgentResult is transformed with `stopReason: 'toolUse'` and the tool's content
 * 4. Streaming events from the delegate agent are surfaced natively in the parent stream
 */

import type { Plugin } from '../../plugins/plugin.js'
import { AgentResult } from '../../types/agent.js'
import type { LocalAgent, AgentStreamEvent } from '../../types/agent.js'
import type { ContentBlock } from '../../types/messages.js'
import {
  AfterToolCallEvent,
  AfterToolsEvent,
  BeforeToolCallEvent,
  BeforeToolsEvent,
  StreamEvent,
  ToolStreamUpdateEvent,
} from '../../hooks/events.js'
import { HookOrder } from '../../hooks/types.js'
import { AgentStreamStage, ExecuteToolStage } from '../../middleware/index.js'
import type {
  AgentStreamContext,
  AgentStreamResult,
  ExecuteToolContext,
  ExecuteToolResult,
  MiddlewareNext,
} from '../../middleware/index.js'
import { Message, TextBlock, ToolResultBlock, ToolUseBlock } from '../../types/messages.js'
import { AgentAsTool } from '../../agent/agent-as-tool.js'

/**
 * Checks whether a tool registered on the agent is a delegation AgentAsTool.
 */
function isDelegationTool(agent: LocalAgent, toolName: string): boolean {
  const tool = agent.toolRegistry.get(toolName)
  return tool instanceof AgentAsTool && tool.delegate
}

/**
 * Converts ToolResultContent blocks into valid ContentBlocks for an assistant message.
 *
 * AgentAsTool only produces TextBlock or JsonBlock results. JsonBlock is not part
 * of the ContentBlock union, so it is serialized to a TextBlock.
 */
function toContentBlocks(block: ToolResultBlock): ContentBlock[] {
  return block.content.map((contentBlock): ContentBlock => {
    if (contentBlock.type === 'jsonBlock') {
      return new TextBlock(JSON.stringify(contentBlock.json))
    }
    return contentBlock as ContentBlock
  })
}

/**
 * Plugin that enforces delegation semantics for tool routing.
 *
 * Automatically registered when any tool in the agent's tool list has `delegate: true`.
 * Implements single-call constraint, early loop exit, and result transformation.
 *
 * @example
 * ```typescript
 * import { Agent } from '@strands-agents/sdk'
 *
 * const specialist = new Agent({ name: 'Specialist' })
 * const orchestrator = new Agent({
 *   tools: [specialist.asTool({ delegate: true })],
 *   // AgentDelegation is auto-registered — no manual setup needed
 * })
 * ```
 */
export class AgentDelegation implements Plugin {
  readonly name = 'strands:agent-delegation'

  /** Stores the delegation tool result per agent, consumed by the stream middleware. */
  private readonly _delegationResult = new WeakMap<LocalAgent, ToolResultBlock>()

  /** Tracks the number of tool use blocks in the current batch per agent. */
  private readonly _toolUseCount = new WeakMap<LocalAgent, number>()

  initAgent(agent: LocalAgent): void {
    // Fail fast: delegation is incompatible with stateful models
    // Stateful models manage conversation state server-side. Delegation's early
    // exit would leave an unclosed function call on the server, corrupting the
    // next request.
    if (agent.model.stateful) {
      const hasDelegationTool = agent.toolRegistry.list().some((tool) => tool instanceof AgentAsTool && tool.delegate)
      if (hasDelegationTool) {
        throw new Error(
          'Delegation tools (delegate: true) are not supported with stateful models. ' +
            "Stateful models manage conversation state server-side, and delegation's early loop exit " +
            'would leave unclosed function calls on the server.'
        )
      }
    }

    agent.addHook(BeforeToolsEvent, (event) => this._onBeforeTools(event))
    agent.addHook(BeforeToolCallEvent, (event) => this._onBeforeToolCall(event), { order: HookOrder.SDK_LAST })
    agent.addHook(AfterToolCallEvent, (event) => this._onAfterToolCall(event))
    agent.addHook(AfterToolsEvent, (event) => this._onAfterTools(event))

    // async function* doesn't bind lexical `this`; capture for the terminal callback.
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const self = this
    agent.addMiddleware(
      AgentStreamStage,
      async function* (
        context: AgentStreamContext,
        next: MiddlewareNext<AgentStreamContext, AgentStreamResult, AgentStreamEvent>
      ): AsyncGenerator<AgentStreamEvent, AgentStreamResult, undefined> {
        return yield* self._handleStream(context, next)
      }
    )

    // ExecuteToolStage middleware: unwraps inner agent streaming events for delegation tools
    // so they appear as native events in the parent agent's stream.
    agent.addMiddleware(
      ExecuteToolStage,
      async function* (
        context: ExecuteToolContext,
        next: MiddlewareNext<ExecuteToolContext, ExecuteToolResult, AgentStreamEvent>
      ): AsyncGenerator<AgentStreamEvent, ExecuteToolResult, undefined> {
        return yield* self._handleToolExecution(context, next, agent)
      }
    )
  }

  /**
   * BeforeToolsEvent hook: enforces single-call constraint against registry names
   * and tracks batch size for post-hook enforcement.
   */
  private _onBeforeTools(event: BeforeToolsEvent): void {
    // Skip for stateful models — delegation semantics are disabled.
    // This runtime guard covers delegation tools added after initialization
    // (e.g., via toolRegistry.add()), complementing the init-time check.
    if (event.agent.model.stateful) return

    const toolUseBlocks = event.message.content.filter((block): block is ToolUseBlock => block.type === 'toolUseBlock')

    // Track batch size so _onAfterToolCall can enforce the single-call
    // constraint even when BeforeToolCallEvent hooks replace the effective tool.
    this._toolUseCount.set(event.agent, toolUseBlocks.length)

    // Cancel the batch if a delegation tool is present alongside other tools.
    const hasDelegation = toolUseBlocks.some((block) => isDelegationTool(event.agent, block.name))
    if (hasDelegation && toolUseBlocks.length > 1) {
      event.cancel =
        'This tool call was not executed. A delegation tool must be the only ' +
        'tool called in a turn. Retry with a single delegation tool call or ' +
        'use only non-delegation tools.'
    }
  }

  /**
   * BeforeToolCallEvent hook (SDK_LAST): cancels delegation tools that were
   * injected via selectedTool replacement in a multi-tool batch.
   *
   * Runs after all user hooks have resolved, so it sees the final effective
   * tool. If a hook replaced a non-delegation tool with a delegation AgentAsTool
   * and the batch has multiple tools, cancels this tool before execution —
   * preventing the delegate from running rather than rejecting it after the fact.
   */
  private _onBeforeToolCall(event: BeforeToolCallEvent): void {
    if (event.agent.model.stateful) return

    const toolUseCount = this._toolUseCount.get(event.agent) ?? 1
    if (toolUseCount <= 1) return

    // Resolve the effective tool: selectedTool wins, otherwise re-resolve
    // from registry if the name was rewritten, otherwise use the original.
    const effectiveTool =
      event.selectedTool ?? (event.toolUse.name !== event.tool?.name ? event.agent.toolRegistry.get(event.toolUse.name) : event.tool)

    if (!(effectiveTool instanceof AgentAsTool) || !effectiveTool.delegate) return

    event.cancel =
      'Delegation failed: a delegation tool must be the only tool called in a turn. ' +
      'The tool was redirected to a delegation agent in a multi-tool batch, which is not allowed. ' +
      'Retry with a single delegation tool call.'
  }

  /**
   * AfterToolCallEvent hook: tracks successful delegation results.
   *
   * Checks the *effective* tool (after BeforeToolCallEvent hooks may have
   * rewritten selectedTool, toolUse.name, or toolUse.toolUseId) to determine
   * whether this is a delegation call. If the effective tool has `delegate: true`
   * and the result is successful, provisionally stashes the result. The actual
   * stopEventLoop signal is deferred to AfterToolsEvent so that hook-requested
   * retries can settle first.
   */
  private _onAfterToolCall(event: AfterToolCallEvent): void {
    if (event.agent.model.stateful || !(event.tool instanceof AgentAsTool) || !event.tool.delegate) return

    // If the delegation tool errored, clear any prior stash (e.g., from a
    // retried attempt that succeeded previously) — let the model recover.
    if (event.result.status === 'error') {
      this._delegationResult.delete(event.agent)
      return
    }

    // Provisionally stash the result. If a retry is requested, executeTool
    // loops back and this hook fires again with the new result, overwriting
    // the stash. Only the final post-retry result survives to AfterToolsEvent.
    this._delegationResult.set(event.agent, event.result)
  }

  /**
   * AfterToolsEvent hook: finalizes delegation by signaling the agent loop to stop.
   *
   * Fires after all tools in the batch have completed.
   * If a delegation result was stashed by _onAfterToolCall, this is where
   * stopEventLoop is set, guaranteeing that retries have fully settled
   * before the loop-exit decision is made.
   */
  private _onAfterTools(event: AfterToolsEvent): void {
    if (!this._delegationResult.has(event.agent)) return

    event.invocationState.stopEventLoop = true
  }

  /**
   * ExecuteToolStage middleware: surfaces delegate agent streaming events natively.
   *
   * Only activates for delegation tools (AgentAsTool with `delegate: true`).
   * Non-delegation agent tools yield their events as normal ToolStreamUpdateEvents —
   * their results are standard tool results and should not be translated.
   *
   * For delegation specifically, the inner agent's events (model streaming, content
   * blocks, tool calls, etc.) are unwrapped from `ToolStreamEvent.data` and yielded
   * directly as native AgentStreamEvents, making the delegation transparent to
   * stream consumers.
   */
  private async *_handleToolExecution(
    context: ExecuteToolContext,
    next: MiddlewareNext<ExecuteToolContext, ExecuteToolResult, AgentStreamEvent>,
    agent: LocalAgent
  ): AsyncGenerator<AgentStreamEvent, ExecuteToolResult, undefined> {
    // Only translate streaming events for delegation tools on non-stateful models.
    // Non-delegation AgentAsTools produce normal tool results, and stateful models
    // skip delegation semantics entirely (runtime guard for late-registered tools).
    if (!(context.tool instanceof AgentAsTool) || !context.tool.delegate || agent.model.stateful) {
      return yield* next(context)
    }

    // Iterate the inner pipeline manually so we can transform events
    const gen = next(context)
    let result = await gen.next()
    while (!result.done) {
      const event = result.value

      // ToolStreamUpdateEvents from a delegation tool may contain wrapped inner agent events.
      // Unwrap them: if the data is a StreamEvent instance, yield it as a native event.
      if (event.type === 'toolStreamUpdateEvent') {
        const innerData = (event as ToolStreamUpdateEvent).event.data
        if (innerData instanceof StreamEvent) {
          yield innerData as AgentStreamEvent
        } else {
          // Regular tool stream events (e.g., from the inner agent's own tools) pass through
          yield event
        }
      } else {
        yield event
      }

      result = await gen.next()
    }
    return result.value
  }

  /**
   * AgentStreamStage middleware: transforms the AgentResult on delegation.
   *
   * When a delegation result was stashed (indicating successful delegation),
   * consumes it and replaces the AgentResult with `stopReason: 'endTurn'`
   * and the tool's content as `lastMessage`. Also appends the delegation
   * message to `agent.messages` so that downstream hooks (e.g. GoalLoop)
   * reading `lastAssistantMessage(agent.messages)` see the delegated content
   * rather than the raw toolUseBlock.
   */
  private async *_handleStream(
    context: AgentStreamContext,
    next: MiddlewareNext<AgentStreamContext, AgentStreamResult, AgentStreamEvent>
  ): AsyncGenerator<AgentStreamEvent, AgentStreamResult, undefined> {
    // Clear any stale entry from a prior failed invocation before the loop runs.
    this._delegationResult.delete(context.agent)

    const streamResult = yield* next(context)

    // Consume the result stashed during this invocation by _onAfterToolCall.
    // The presence of a stashed result is the authoritative signal that
    // delegation occurred — stopEventLoop is consumed (cleared) by the agent
    // loop to prevent leaking across nested agent frames.
    const delegationBlock = this._delegationResult.get(context.agent)
    this._delegationResult.delete(context.agent)

    // No delegation result stashed — pass through unchanged
    if (!delegationBlock) return streamResult

    const delegationMessage = new Message({
      role: 'assistant',
      content: toContentBlocks(delegationBlock),
    })

    // Append the delegation message to agent.messages so downstream consumers
    // (GoalLoop, session managers, etc.) see the delegated answer as the last
    // assistant message rather than the raw toolUseBlock.
    context.agent.messages.push(delegationMessage)

    // Replace AgentResult with the delegation tool's content
    return {
      result: new AgentResult({
        stopReason: 'endTurn',
        lastMessage: delegationMessage,
        invocationState: streamResult.result.invocationState,
        ...(streamResult.result.metrics !== undefined && { metrics: streamResult.result.metrics }),
        ...(streamResult.result.traces !== undefined && { traces: streamResult.result.traces }),
      }),
    }
  }
}
