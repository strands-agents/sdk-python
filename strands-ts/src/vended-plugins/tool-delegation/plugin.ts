/**
 * ToolDelegation — enforces tool-delegation semantics for tool routing.
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
import { AfterToolCallEvent, BeforeToolsEvent, AfterToolsEvent } from '../../hooks/events.js'
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
  /** The resolved delegation ToolResultBlock, captured in AfterToolCallEvent and consumed by the middleware. */
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
 *   // ToolDelegation is auto-registered — no manual setup needed
 * })
 * ```
 */
export class ToolDelegation implements Plugin {
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
  }

  /**
   * BeforeToolsEvent hook: enforces single-call constraint.
   *
   * If a delegation tool is present alongside other tools, cancel all.
   */
  private _onBeforeTools(event: BeforeToolsEvent): void {
    // Reset per-turn state to prevent stale values from a prior turn leaking
    // forward (e.g., if AfterToolsEvent never fired due to an exception).
    const state = this._getState(event.agent)
    state.triggered = false
    delete state.toolResult

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
   * AfterToolCallEvent hook: captures delegation result at execution time.
   *
   * Checks the *effective* tool (after BeforeToolCallEvent hooks may have
   * rewritten selectedTool, toolUse.name, or toolUse.toolUseId) to determine
   * whether this is a delegation call. If the effective tool has `delegate: true`
   * and the result is successful, captures the result for the middleware.
   */
  private _onAfterToolCall(event: AfterToolCallEvent): void {
    // Skip for stateful models — delegation semantics are disabled.
    if (event.agent.model.stateful) return

    // Only trigger if the effective tool is a delegation tool
    if (!event.tool?.delegate) return

    // If the delegation tool errored, don't trigger — let the model recover
    if (event.result.status === 'error') return

    const state = this._getState(event.agent)
    state.triggered = true
    state.toolResult = event.result
  }

  /**
   * AfterToolsEvent hook: triggers early exit on successful delegation.
   *
   * If a delegation was detected in AfterToolCallEvent (state.triggered is true),
   * sets endTurn so the agent loop exits without calling the model again.
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
    if (!state.triggered || !state.toolResult) return

    // Extract text representation and end the turn.
    // Use `|| true` so that non-text-only results (e.g. image/document blocks)
    // still produce a truthy endTurn — extractText returns '' for those, which
    // would skip the agent loop's early-exit check.
    const textSummary = extractText(state.toolResult)
    event.endTurn = textSummary || true
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
    const state = this._getState(context.agent)

    // Unconditionally clear any stale delegation state from a prior invocation
    // that may not have completed cleanup (e.g., the stream threw after
    // _onAfterTools committed state).
    state.triggered = false
    delete state.toolResult

    const streamResult = yield* next(context)

    const triggered = state.triggered
    const delegationBlock = state.toolResult
    state.triggered = false
    delete state.toolResult

    if (!triggered) return streamResult
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
