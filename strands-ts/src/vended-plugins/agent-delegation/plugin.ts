/**
 * AgentDelegation — enforces delegation semantics for tool routing.
 *
 * When a tool is configured with `delegate: true`, this plugin ensures:
 * 1. The delegation tool is the only tool called in the turn (single-call constraint)
 * 2. The agent loop exits immediately after a successful delegation (via stopEventLoop)
 * 3. The AgentResult is transformed with `stopReason: 'delegated'` and the tool's content
 */

import type { Plugin } from '../../plugins/plugin.js'
import { AgentResult } from '../../types/agent.js'
import type { LocalAgent, AgentStreamEvent } from '../../types/agent.js'
import type { ContentBlock } from '../../types/messages.js'
import { AfterToolCallEvent, BeforeToolsEvent } from '../../hooks/events.js'
import { AgentStreamStage } from '../../middleware/index.js'
import type { AgentStreamContext, AgentStreamResult, MiddlewareNext } from '../../middleware/index.js'
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

  initAgent(agent: LocalAgent): void {
    agent.addHook(BeforeToolsEvent, (event) => this._onBeforeTools(event))
    agent.addHook(AfterToolCallEvent, (event) => this._onAfterToolCall(event))

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
  }

  /**
   * BeforeToolsEvent hook: enforces single-call constraint.
   *
   * If a delegation tool is present alongside other tools, cancel all.
   */
  private _onBeforeTools(event: BeforeToolsEvent): void {
    // Stateful models manage conversation state server-side. Delegation's early
    // exit would leave an unclosed function call on the server, corrupting the
    // next request. Skip all delegation logic and let the tool run normally.
    if (event.agent.model.stateful) return

    const toolUseBlocks = event.message.content.filter((block): block is ToolUseBlock => block.type === 'toolUseBlock')

    const delegationBlocks = toolUseBlocks.filter((block) => isDelegationTool(event.agent, block.name))

    // No delegation tools in this batch — let normal execution proceed
    if (delegationBlocks.length === 0) return

    // Delegation tool(s) present alongside other tools — cancel all
    if (toolUseBlocks.length > 1) {
      event.cancel =
        'This tool call was not executed. A delegation tool must be the only ' +
        'tool called in a turn. Retry with a single delegation tool call or ' +
        'use only non-delegation tools.'
      return
    }
  }

  /**
   * AfterToolCallEvent hook: signals the agent loop to stop on successful delegation.
   *
   * Checks the *effective* tool (after BeforeToolCallEvent hooks may have
   * rewritten selectedTool, toolUse.name, or toolUse.toolUseId) to determine
   * whether this is a delegation call. If the effective tool has `delegate: true`
   * and the result is successful, sets `invocationState.stopEventLoop = true`
   * so the agent loop exits without calling the model again, and stashes the
   * result for the stream middleware to consume.
   */
  private _onAfterToolCall(event: AfterToolCallEvent): void {
    // Skip for stateful models — delegation semantics are disabled.
    if (event.agent.model.stateful) return

    // Only trigger if the effective tool is a delegation AgentAsTool
    if (!(event.tool instanceof AgentAsTool) || !event.tool.delegate) return

    // If the delegation tool errored, don't trigger — let the model recover
    if (event.result.status === 'error') return

    // Signal the agent loop to stop after this tool batch completes
    event.invocationState.stopEventLoop = true

    // Stash the result for the stream middleware to transform into the AgentResult.
    this._delegationResult.set(event.agent, event.result)
  }

  /**
   * AgentStreamStage middleware: transforms the AgentResult on delegation.
   *
   * When stopEventLoop was triggered by a delegation tool, consumes the stashed
   * tool result and replaces the AgentResult with `stopReason: 'delegated'`
   * and the tool's content as `lastMessage`.
   */
  private async *_handleStream(
    context: AgentStreamContext,
    next: MiddlewareNext<AgentStreamContext, AgentStreamResult, AgentStreamEvent>
  ): AsyncGenerator<AgentStreamEvent, AgentStreamResult, undefined> {
    // Clear any stale entry from a prior failed invocation before the loop runs.
    this._delegationResult.delete(context.agent)

    const streamResult = yield* next(context)

    // Only transform when stopEventLoop was set (delegation or otherwise)
    if (streamResult.result.invocationState.stopEventLoop !== true) return streamResult

    // Consume the result stashed during this invocation by _onAfterToolCall.
    const delegationBlock = this._delegationResult.get(context.agent)
    this._delegationResult.delete(context.agent)
    if (!delegationBlock) return streamResult

    // Replace AgentResult with the delegation tool's content
    return {
      result: new AgentResult({
        stopReason: 'delegated',
        lastMessage: new Message({
          role: 'assistant',
          content: toContentBlocks(delegationBlock),
        }),
        invocationState: streamResult.result.invocationState,
        ...(streamResult.result.metrics !== undefined && { metrics: streamResult.result.metrics }),
        ...(streamResult.result.traces !== undefined && { traces: streamResult.result.traces }),
      }),
    }
  }
}
