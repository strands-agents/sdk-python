/**
 * AgentCore Harness-specific stream events yielded by AgentCoreHarnessAgent.stream().
 */

import type { InvokeHarnessStreamOutput } from '@aws-sdk/client-bedrock-agentcore'
import { StreamEvent } from '../hooks/events.js'
import type { AgentResultEvent } from '../hooks/events.js'

/**
 * Union of non-error events received from an AgentCore Harness invocation.
 *
 * Harness error events are raised as exceptions instead of being yielded from the agent stream.
 */
export type AgentCoreHarnessEventData = Exclude<
  InvokeHarnessStreamOutput,
  | InvokeHarnessStreamOutput.InternalServerExceptionMember
  | InvokeHarnessStreamOutput.ValidationExceptionMember
  | InvokeHarnessStreamOutput.RuntimeClientErrorMember
>

/**
 * Event wrapping a raw InvokeHarness streaming event.
 *
 * Yielded by `AgentCoreHarnessAgent.stream()` for each non-error event received from the remote agent.
 * Harness error events are raised as exceptions instead.
 * The `event` property is the raw InvokeHarness streaming event.
 */
export class AgentCoreHarnessStreamUpdateEvent extends StreamEvent {
  readonly type = 'agentCoreHarnessStreamUpdateEvent' as const
  readonly event: AgentCoreHarnessEventData

  /**
   * Creates a stream update event.
   *
   * @param event - Raw non-error event received from InvokeHarness
   */
  constructor(event: AgentCoreHarnessEventData) {
    super()
    this.event = event
  }

  /**
   * Serializes the event.
   *
   * @returns The event type and raw Harness event
   */
  toJSON(): Pick<AgentCoreHarnessStreamUpdateEvent, 'type' | 'event'> {
    return { type: this.type, event: this.event }
  }
}

/**
 * Event triggered as the final event in the AgentCore Harness agent stream.
 * Wraps the agent result containing the stop reason and last message.
 */
export class AgentCoreHarnessResultEvent extends StreamEvent {
  readonly type = 'agentCoreHarnessResultEvent' as const
  readonly result: AgentResultEvent['result']

  /**
   * Creates the final result event.
   *
   * @param data - Final result from the Harness invocation
   */
  constructor(data: Pick<AgentResultEvent, 'result'>) {
    super()
    this.result = data.result
  }

  /**
   * Serializes the event.
   *
   * @returns The event type and final result
   */
  toJSON(): Pick<AgentCoreHarnessResultEvent, 'type' | 'result'> {
    return { type: this.type, result: this.result }
  }
}

/**
 * Union of all events yielded by `AgentCoreHarnessAgent.stream()`.
 *
 * Includes raw streaming events ({@link AgentCoreHarnessStreamUpdateEvent}) and the final
 * result event ({@link AgentCoreHarnessResultEvent}).
 */
export type AgentCoreHarnessStreamEvent = AgentCoreHarnessStreamUpdateEvent | AgentCoreHarnessResultEvent
