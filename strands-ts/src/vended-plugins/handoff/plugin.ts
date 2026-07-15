/**
 * HandoffPlugin — enforces handoff semantics for agent-as-tool routing.
 *
 * When a sub-agent is wrapped with `handoff: true`, this plugin ensures:
 * 1. The handoff tool is the only tool called in the turn (single-call constraint)
 * 2. The agent loop exits immediately after a successful handoff (early exit)
 * 3. The AgentResult is transformed with `stopReason: 'handoff'` and the sub-agent's content
 */

import type { Plugin } from '../../plugins/plugin.js'
import { AgentResult } from '../../types/agent.js'
import type { LocalAgent, AgentStreamEvent } from '../../types/agent.js'
import type { ContentBlock, ToolResultContent } from '../../types/messages.js'
import { BeforeToolsEvent, AfterToolsEvent } from '../../hooks/events.js'
import { AgentStreamStage } from '../../middleware/index.js'
import type { AgentStreamContext, AgentStreamResult, MiddlewareNext } from '../../middleware/index.js'
import { AgentAsTool } from '../../agent/agent-as-tool.js'
import { Message, ToolResultBlock, ToolUseBlock } from '../../types/messages.js'

/**
 * Checks whether a tool registered on the agent is a handoff tool.
 *
 * A handoff tool is an AgentAsTool instance with `handoff === true`.
 */
function isHandoffTool(agent: LocalAgent, toolName: string): boolean {
  const tool = agent.toolRegistry.get(toolName)
  return tool instanceof AgentAsTool && tool.handoff
}

/**
 * Extracts a text representation from a ToolResultBlock's content array.
 *
 * Concatenates text blocks and JSON-stringified JSON blocks, separated by newlines.
 */
function extractText(block: ToolResultBlock): string {
  return block.content
    .map((contentBlock: ToolResultContent) => {
      if (contentBlock.type === 'textBlock') return contentBlock.text
      if (contentBlock.type === 'jsonBlock') return JSON.stringify(contentBlock.json)
      return ''
    })
    .filter(Boolean)
    .join('\n')
}

/**
 * Plugin that enforces handoff semantics for agent-as-tool routing.
 *
 * Automatically registered when any tool in the agent's tool list has `handoff: true`.
 * Implements single-call constraint, early loop exit, and result transformation.
 *
 * @example
 * ```typescript
 * import { Agent } from '@strands-agents/sdk'
 *
 * const specialist = new Agent({ name: 'Specialist' })
 * const orchestrator = new Agent({
 *   tools: [specialist.asTool({ handoff: true })],
 *   // HandoffPlugin is auto-registered — no manual setup needed
 * })
 * ```
 */
export class HandoffPlugin implements Plugin {
  readonly name = 'strands:handoff'

  private _handoffTriggered = false
  private _handoffToolUseId: string | undefined

  initAgent(agent: LocalAgent): void {
    agent.addHook(BeforeToolsEvent, (event) => this._onBeforeTools(event))
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
  }

  /**
   * BeforeToolsEvent hook: enforces single-call constraint.
   *
   * If a handoff tool is present alongside other tools, cancel all.
   * If a single handoff tool is alone, record its toolUseId for later.
   */
  private _onBeforeTools(event: BeforeToolsEvent): void {
    const toolUseBlocks = event.message.content.filter((block): block is ToolUseBlock => block.type === 'toolUseBlock')

    const handoffBlocks = toolUseBlocks.filter((block) => isHandoffTool(event.agent, block.name))

    // No handoff tools in this batch — let normal execution proceed
    if (handoffBlocks.length === 0) return

    // Handoff tool(s) present alongside other tools — cancel all
    if (toolUseBlocks.length > 1) {
      event.cancel =
        'This tool call was not executed. A handoff tool must be the only ' +
        'tool called in a turn. Retry with a single handoff tool call or ' +
        'use only non-handoff tools.'
      return
    }

    // Single handoff tool — allow execution, record the toolUseId
    this._handoffToolUseId = handoffBlocks[0]!.toolUseId
  }

  /**
   * AfterToolsEvent hook: triggers early exit on successful handoff.
   *
   * If the recorded handoff tool completed successfully, set endTurn.
   * If it errored, reset state and let the model recover.
   */
  private _onAfterTools(event: AfterToolsEvent): void {
    if (!this._handoffToolUseId) return

    // Find the tool result for the handoff tool
    const handoffResult = event.message.content.find(
      (block): block is ToolResultBlock =>
        block.type === 'toolResultBlock' && block.toolUseId === this._handoffToolUseId
    )

    // If the handoff tool errored or wasn't found, don't trigger — let the model recover
    if (!handoffResult || handoffResult.status === 'error') {
      this._handoffToolUseId = undefined
      return
    }

    // Extract text representation and end the turn
    this._handoffTriggered = true
    event.endTurn = extractText(handoffResult)
    this._handoffToolUseId = undefined
  }

  /**
   * AgentStreamStage middleware: transforms the AgentResult on handoff.
   *
   * When a handoff was triggered, replaces the result with a new AgentResult
   * that has `stopReason: 'handoff'` and the handoff tool's content blocks
   * as `lastMessage`.
   */
  private async *_handleStream(
    context: AgentStreamContext,
    next: MiddlewareNext<AgentStreamContext, AgentStreamResult, AgentStreamEvent>
  ): AsyncGenerator<AgentStreamEvent, AgentStreamResult, undefined> {
    const streamResult = yield* next(context)

    if (!this._handoffTriggered) return streamResult
    this._handoffTriggered = false

    const handoffBlock = this._findHandoffResult(context.agent.messages, context.agent)
    if (!handoffBlock) return streamResult

    // Replace AgentResult with rich content from the handoff tool.
    // ToolResultContent includes JsonBlock which isn't in the ContentBlock union,
    // but the design requires preserving all content types verbatim.
    return {
      result: new AgentResult({
        stopReason: 'handoff',
        lastMessage: new Message({
          role: 'assistant',
          content: handoffBlock.content as ContentBlock[],
        }),
        invocationState: streamResult.result.invocationState,
        ...(streamResult.result.metrics !== undefined && { metrics: streamResult.result.metrics }),
        ...(streamResult.result.traces !== undefined && { traces: streamResult.result.traces }),
      }),
    }
  }

  /**
   * Finds the most recent successful handoff tool result by walking messages
   * from the end, matching assistant tool-use blocks to their user-side results.
   */
  private _findHandoffResult(messages: Message[], agent: LocalAgent): ToolResultBlock | undefined {
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i]!
      if (msg.role !== 'user') continue

      const prevMsg = messages[i - 1]
      if (!prevMsg || prevMsg.role !== 'assistant') continue

      for (const block of msg.content) {
        if (block.type !== 'toolResultBlock' || block.status !== 'success') continue

        const toolUse = prevMsg.content.find(
          (b): b is ToolUseBlock => b.type === 'toolUseBlock' && b.toolUseId === block.toolUseId
        )
        if (toolUse && isHandoffTool(agent, toolUse.name)) return block
      }
    }

    return undefined
  }
}
