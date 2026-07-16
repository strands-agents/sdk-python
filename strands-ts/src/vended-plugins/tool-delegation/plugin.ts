/**
 * ToolDelegationPlugin — enforces tool-delegation semantics for tool routing.
 *
 * When a tool is configured with `delegate: true`, this plugin ensures:
 * 1. The delegation tool is the only tool called in the turn (single-call constraint)
 * 2. The agent loop exits immediately after a successful delegation (early exit)
 * 3. The AgentResult is transformed with `stopReason: 'delegated'` and the tool's content
 */

import type { Plugin } from '../../plugins/plugin.js'
import { AgentResult } from '../../types/agent.js'
import type { LocalAgent, AgentStreamEvent } from '../../types/agent.js'
import type { ContentBlock } from '../../types/messages.js'
import { BeforeToolsEvent, AfterToolsEvent } from '../../hooks/events.js'
import { AgentStreamStage } from '../../middleware/index.js'
import type { AgentStreamContext, AgentStreamResult, MiddlewareNext } from '../../middleware/index.js'
import { Message, TextBlock, ToolResultBlock, ToolUseBlock } from '../../types/messages.js'

/**
 * Checks whether a tool registered on the agent has delegate enabled.
 */
function isDelegationTool(agent: LocalAgent, toolName: string): boolean {
  const tool = agent.toolRegistry.get(toolName)
  return tool?.delegate ?? false
}

/**
 * Extracts a text representation from a ToolResultBlock's content array.
 *
 * Concatenates text blocks and JSON-stringified JSON blocks, separated by newlines.
 */
function extractText(block: ToolResultBlock): string {
  return block.content
    .map((contentBlock) => {
      if (contentBlock.type === 'textBlock') return contentBlock.text
      if (contentBlock.type === 'jsonBlock') return JSON.stringify(contentBlock.json)
      return ''
    })
    .filter(Boolean)
    .join('\n')
}

/**
 * Converts ToolResultContent blocks into valid ContentBlocks for an assistant message.
 *
 * JsonBlock is not part of the ContentBlock union, so it is serialized to a TextBlock.
 * ImageBlock, VideoBlock, and DocumentBlock pass through unchanged since they exist
 * in both unions.
 */
function toContentBlocks(block: ToolResultBlock): ContentBlock[] {
  return block.content.map((contentBlock): ContentBlock => {
    switch (contentBlock.type) {
      case 'jsonBlock':
        return new TextBlock(JSON.stringify(contentBlock.json))
      case 'textBlock':
      case 'imageBlock':
      case 'videoBlock':
      case 'documentBlock':
        return contentBlock
      default: {
        const _exhaustive: never = contentBlock
        return _exhaustive
      }
    }
  })
}

/**
 * Per-agent mutable state for the tool-delegation plugin.
 *
 * Scoped via WeakMap so concurrent invocations on different agents don't interfere.
 */
interface ToolDelegationState {
  /** Whether a successful delegation was detected in the current turn. */
  triggered: boolean
  /** The toolUseId of the pending delegation tool, set in BeforeTools and consumed in AfterTools. */
  toolUseId?: string
  /** The resolved delegation ToolResultBlock, captured in AfterTools and consumed by the middleware. */
  toolResult?: ToolResultBlock
}

/**
 * Plugin that enforces tool-delegation semantics for tool routing.
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
 *   // ToolDelegationPlugin is auto-registered — no manual setup needed
 * })
 * ```
 */
export class ToolDelegationPlugin implements Plugin {
  readonly name = 'strands:tool-delegation'

  /** Per-agent delegation state. Keyed by agent so one plugin instance can serve many. */
  private readonly _state = new WeakMap<LocalAgent, ToolDelegationState>()

  private _getState(agent: LocalAgent): ToolDelegationState {
    let state = this._state.get(agent)
    if (!state) {
      state = { triggered: false }
      this._state.set(agent, state)
    }
    return state
  }

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
   * If a delegation tool is present alongside other tools, cancel all.
   * If a single delegation tool is alone, record its toolUseId for later.
   */
  private _onBeforeTools(event: BeforeToolsEvent): void {
    // Reset per-turn state to prevent stale values from a prior turn leaking
    // forward (e.g., if AfterToolsEvent never fired due to an exception).
    const state = this._getState(event.agent)
    state.triggered = false
    delete state.toolUseId
    delete state.toolResult

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

    // Single delegation tool — allow execution, record the toolUseId
    state.toolUseId = delegationBlocks[0]!.toolUseId
  }

  /**
   * AfterToolsEvent hook: triggers early exit on successful delegation.
   *
   * If the recorded delegation tool completed successfully, set endTurn.
   * If it errored, reset state and let the model recover.
   *
   * Note: `event.endTurn` accepts only `boolean | string`, so the agent loop
   * appends a text-only assistant message to `agent.messages`. The full typed
   * content blocks are delivered to the caller via `result.lastMessage` in the
   * AgentStreamStage middleware below. This means conversation history contains
   * a text summary while the returned AgentResult preserves rich content
   * (images, JSON, documents) verbatim.
   */
  private _onAfterTools(event: AfterToolsEvent): void {
    const state = this._getState(event.agent)
    if (!state.toolUseId) return

    // Find the tool result for the delegation tool
    const delegationResult = event.message.content.find(
      (block): block is ToolResultBlock => block.type === 'toolResultBlock' && block.toolUseId === state.toolUseId
    )

    // If the delegation tool errored or wasn't found, don't trigger — let the model recover
    if (!delegationResult || delegationResult.status === 'error') {
      delete state.toolUseId
      return
    }

    // Extract text representation and end the turn
    state.triggered = true
    state.toolResult = delegationResult
    event.endTurn = extractText(delegationResult)
    delete state.toolUseId
  }

  /**
   * AgentStreamStage middleware: transforms the AgentResult on delegation.
   *
   * When a delegation was triggered, replaces the result with a new AgentResult
   * that has `stopReason: 'delegated'` and the tool's content blocks as
   * `lastMessage`. This `lastMessage` may differ from the text-only message
   * appended to `agent.messages` by the endTurn path. See the AfterToolsEvent
   * hook comment for details on why this divergence exists.
   */
  private async *_handleStream(
    context: AgentStreamContext,
    next: MiddlewareNext<AgentStreamContext, AgentStreamResult, AgentStreamEvent>
  ): AsyncGenerator<AgentStreamEvent, AgentStreamResult, undefined> {
    const streamResult = yield* next(context)

    const state = this._getState(context.agent)
    if (!state.triggered) return streamResult
    state.triggered = false

    const delegationBlock = state.toolResult
    delete state.toolResult
    if (!delegationBlock) return streamResult

    // Replace AgentResult with rich content from the delegation tool.
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
