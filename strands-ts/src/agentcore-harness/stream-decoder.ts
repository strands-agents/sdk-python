import type {
  HarnessContentBlockDelta,
  HarnessContentBlockStart,
  HarnessStopReason,
  HarnessToolResultContentBlock,
} from '@aws-sdk/client-bedrock-agentcore'
import { ModelError } from '../errors.js'
import { logger } from '../logging/logger.js'
import { accumulateUsage, type Usage } from '../models/streaming.js'
import { JsonBlock, Message, ReasoningBlock, TextBlock, ToolResultBlock, ToolUseBlock } from '../types/messages.js'
import type { JSONValue } from '../types/json.js'
import type { ContentBlock, Role, StopReason, ToolResultContent } from '../types/messages.js'
import type { ToolResultStatus } from '../tools/types.js'
import type { AgentCoreHarnessEventData } from './events.js'

/** Maps Harness wire stop reasons onto Strands stop reasons. */
const STOP_REASON_MAP = {
  end_turn: 'endTurn',
  tool_use: 'toolUse',
  tool_result: 'toolResult',
  max_tokens: 'maxTokens',
  stop_sequence: 'stopSequence',
  content_filtered: 'contentFiltered',
  model_context_window_exceeded: 'modelContextWindowExceeded',
  max_iterations_exceeded: 'limitTurns',
  max_output_tokens_exceeded: 'limitOutputTokens',
  timeout_exceeded: 'timeoutExceeded',
  interrupted: 'interrupt',
  partial_turn: 'pauseTurn',
  malformed_model_output: 'malformedModelOutput',
  malformed_tool_use: 'malformedToolUse',
} satisfies Record<HarnessStopReason, StopReason>

/**
 * A completed turn reconstructed from a Harness event stream.
 *
 * @internal
 */
export interface DecodedHarnessTurn {
  /** Most recent message completed by the Harness call. */
  message: Message
  /** Reason the most recent message stopped. */
  stopReason: StopReason
  /** Token usage accumulated across the Harness call. */
  usage?: Usage
  /** Most recent token usage reported during the Harness call. */
  latestUsage?: Usage
  /** Latency accumulated across the Harness call in milliseconds. */
  latencyMs?: number
  /** Tool-input parse error associated with the retained message. */
  toolInputParseError?: SyntaxError
  /** Number of assistant messages completed during this Harness call. */
  assistantMessageCount: number
}

/**
 * Reconstructs the most recent message and metadata from one InvokeHarness event stream.
 *
 * @internal
 */
export class HarnessStreamDecoder {
  private readonly _state = createStreamState()

  /**
   * Folds one non-error Harness stream event into the current turn.
   *
   * @param event - Harness event to process
   */
  accept(event: AgentCoreHarnessEventData): void {
    accumulate(this._state, event)
  }

  /**
   * Completes the turn after the Harness stream ends.
   *
   * @returns Reconstructed message, stop reason, usage, latency, and any deferred parse error
   * @throws \{ModelError\} When the stream did not complete a message
   */
  complete(): DecodedHarnessTurn {
    if (this._state.stopReason === undefined) {
      throw new ModelError('Stream ended without completing a message')
    }
    return {
      message: toMessage(this._state),
      stopReason: this._state.stopReason,
      ...(this._state.usage !== undefined && { usage: this._state.usage }),
      ...(this._state.latestUsage !== undefined && { latestUsage: this._state.latestUsage }),
      ...(this._state.latencyMs !== undefined && { latencyMs: this._state.latencyMs }),
      ...(this._state.toolInputParseError !== undefined && {
        toolInputParseError: this._state.toolInputParseError,
      }),
      assistantMessageCount: this._state.assistantMessageCount,
    }
  }

  /**
   * Returns the message assembled before a cancelled stream ended.
   *
   * @returns Partially reconstructed message
   */
  partialMessage(): Message {
    return toMessage(this._state)
  }
}

/** The content block currently being assembled from stream events. */
type PendingBlock =
  | { kind: 'text'; text: string }
  | { kind: 'toolUse'; toolUseId: string; name: string; input: string }
  | { kind: 'toolResult'; toolUseId: string; status: ToolResultStatus; content: ToolResultContent[] }
  | { kind: 'reasoning'; text: string; signature?: string; redactedContent?: Uint8Array }

/** Accumulated state for one InvokeHarness event stream. */
interface StreamState {
  role: Role
  content: ContentBlock[]
  pending: PendingBlock | undefined
  stopReason: StopReason | undefined
  toolInputParseError: SyntaxError | undefined
  usage: Usage | undefined
  latestUsage: Usage | undefined
  latencyMs: number | undefined
  assistantMessageCount: number
}

/** Creates fresh accumulation state for a single InvokeHarness turn. */
function createStreamState(): StreamState {
  return {
    role: 'assistant',
    content: [],
    pending: undefined,
    stopReason: undefined,
    toolInputParseError: undefined,
    usage: undefined,
    latestUsage: undefined,
    latencyMs: undefined,
    assistantMessageCount: 0,
  }
}

/** Folds one stream event into the accumulation state. */
function accumulate(state: StreamState, event: AgentCoreHarnessEventData): void {
  if ('messageStart' in event && event.messageStart) {
    // A single InvokeHarness call can stream multiple messages. Retain only the most recent message.
    flushPending(state)
    state.content.length = 0
    state.stopReason = undefined
    state.toolInputParseError = undefined
    state.role = roleFromHarness(event.messageStart.role)
  }
  if ('contentBlockStart' in event && event.contentBlockStart) {
    flushPending(state)
    state.pending = startBlock(event.contentBlockStart.start)
  }
  if ('contentBlockDelta' in event && event.contentBlockDelta) {
    applyDelta(state, event.contentBlockDelta.delta)
  }
  if ('contentBlockStop' in event && event.contentBlockStop) {
    flushPending(state)
  }
  if ('messageStop' in event && event.messageStop) {
    state.stopReason = stopReasonFromHarness(event.messageStop.stopReason)
    if (state.role === 'assistant') state.assistantMessageCount++
  }
  if ('metadata' in event && event.metadata) {
    const usage = event.metadata.usage
    if (usage) {
      const reportedUsage: Usage = {
        inputTokens: usage.inputTokens ?? 0,
        outputTokens: usage.outputTokens ?? 0,
        totalTokens: usage.totalTokens ?? 0,
        ...(usage.cacheReadInputTokens !== undefined && { cacheReadInputTokens: usage.cacheReadInputTokens }),
        ...(usage.cacheWriteInputTokens !== undefined && { cacheWriteInputTokens: usage.cacheWriteInputTokens }),
      }
      if (state.usage === undefined) state.usage = { ...reportedUsage }
      else accumulateUsage(state.usage, reportedUsage)
      state.latestUsage = reportedUsage
    }
    if (event.metadata.metrics?.latencyMs !== undefined) {
      state.latencyMs = (state.latencyMs ?? 0) + event.metadata.metrics.latencyMs
    }
  }
}

/** Finalizes the accumulated state into a message. */
function toMessage(state: StreamState): Message {
  flushPending(state)
  return new Message({ role: state.role, content: state.content })
}

/** Maps a Harness stop reason to SDK casing, preserving unknown future values. */
function stopReasonFromHarness(stopReason: string | undefined): StopReason {
  if (stopReason === undefined || stopReason.trim().length === 0) {
    throw new ModelError('Harness messageStop event is missing a non-empty stopReason')
  }
  const mappedStopReason = STOP_REASON_MAP[stopReason as HarnessStopReason]
  if (mappedStopReason !== undefined) return mappedStopReason

  const fallback = stopReason.replace(/_([a-z])/g, (_, letter: string) => letter.toUpperCase())
  logger.warn(`stop_reason=<${stopReason}>, fallback=<${fallback}> | unknown stop reason, converting to camelCase`)
  return fallback
}

/** Opens a pending block when a content-block start identifies its type. */
function startBlock(start: HarnessContentBlockStart | undefined): PendingBlock | undefined {
  if (start && 'toolUse' in start && start.toolUse) {
    return {
      kind: 'toolUse',
      toolUseId: requiredWireString(start.toolUse.toolUseId, 'tool-use start', 'toolUseId'),
      name: requiredWireString(start.toolUse.name, 'tool-use start', 'name'),
      input: '',
    }
  }
  if (start && 'toolResult' in start && start.toolResult) {
    return {
      kind: 'toolResult',
      toolUseId: requiredWireString(start.toolResult.toolUseId, 'tool-result start', 'toolUseId'),
      status: start.toolResult.status === 'error' ? 'error' : 'success',
      content: [],
    }
  }
  return undefined
}

/** Validates the message role carried by a stream event. */
function roleFromHarness(role: string | undefined): Role {
  if (role === 'assistant' || role === 'user') return role
  throw new ModelError(`Harness messageStart event has invalid role '${String(role)}'`)
}

/** Narrows optional Smithy strings that are required for a valid stream. */
function requiredWireString(value: string | undefined, event: string, field: string): string {
  if (value !== undefined && value.length > 0) return value
  throw new ModelError(`Harness ${event} event is missing a non-empty ${field}`)
}

/** Appends a delta to the pending block, matched to its kind. */
function applyDelta(state: StreamState, delta: HarnessContentBlockDelta | undefined): void {
  if (!delta) return
  const pending = state.pending
  if ('text' in delta && delta.text !== undefined) {
    if (pending?.kind === 'text') pending.text += delta.text
    else if (pending?.kind === 'reasoning') pending.text += delta.text
    else state.pending = { kind: 'text', text: delta.text }
  } else if ('toolUse' in delta && delta.toolUse && pending?.kind === 'toolUse') {
    pending.input += delta.toolUse.input ?? ''
  } else if ('toolResult' in delta && delta.toolResult && pending?.kind === 'toolResult') {
    for (const item of delta.toolResult) {
      pending.content.push(toolResultContentFromHarness(item))
    }
  } else if ('reasoningContent' in delta && delta.reasoningContent && pending?.kind === 'reasoning') {
    const reasoning = delta.reasoningContent
    if ('text' in reasoning && reasoning.text !== undefined) pending.text += reasoning.text
    if ('signature' in reasoning && reasoning.signature !== undefined) {
      pending.signature = (pending.signature ?? '') + reasoning.signature
    }
    if ('redactedContent' in reasoning && reasoning.redactedContent !== undefined) {
      pending.redactedContent = concatBytes(pending.redactedContent, reasoning.redactedContent)
    }
  } else if ('reasoningContent' in delta && delta.reasoningContent && pending === undefined) {
    const reasoning = delta.reasoningContent
    state.pending = {
      kind: 'reasoning',
      text: 'text' in reasoning && reasoning.text ? reasoning.text : '',
      ...('signature' in reasoning && reasoning.signature !== undefined && { signature: reasoning.signature }),
      ...('redactedContent' in reasoning &&
        reasoning.redactedContent !== undefined && { redactedContent: reasoning.redactedContent }),
    }
  }
}

/** Concatenates streamed binary reasoning content. */
function concatBytes(current: Uint8Array | undefined, chunk: Uint8Array): Uint8Array {
  if (current === undefined) return chunk
  const combined = new Uint8Array(current.length + chunk.length)
  combined.set(current)
  combined.set(chunk, current.length)
  return combined
}

/** Pushes the pending block onto the content list and clears it. */
function flushPending(state: StreamState): void {
  const pending = state.pending
  if (!pending) return
  switch (pending.kind) {
    case 'text':
      if (pending.text) state.content.push(new TextBlock(pending.text))
      break
    case 'toolUse': {
      let input: JSONValue = {}
      if (pending.input) {
        try {
          input = JSON.parse(pending.input) as JSONValue
        } catch (error) {
          if (error instanceof SyntaxError && !state.toolInputParseError) state.toolInputParseError = error
        }
      }
      state.content.push(new ToolUseBlock({ toolUseId: pending.toolUseId, name: pending.name, input }))
      break
    }
    case 'toolResult':
      state.content.push(
        new ToolResultBlock({ toolUseId: pending.toolUseId, status: pending.status, content: pending.content })
      )
      break
    case 'reasoning':
      state.content.push(
        new ReasoningBlock({
          ...(pending.text && { text: pending.text }),
          ...(pending.signature !== undefined && { signature: pending.signature }),
          ...(pending.redactedContent !== undefined && { redactedContent: pending.redactedContent }),
        })
      )
      break
  }
  state.pending = undefined
}

/** Converts Harness tool-result content into a Strands tool-result content block. */
function toolResultContentFromHarness(item: HarnessToolResultContentBlock): ToolResultContent {
  if ('json' in item && item.json !== undefined) return new JsonBlock({ json: item.json as JSONValue })
  if ('text' in item && item.text !== undefined) return new TextBlock(item.text)
  return new TextBlock(JSON.stringify(item))
}
