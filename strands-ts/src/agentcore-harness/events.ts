/** AgentCore Harness stream events yielded by {@link AgentCoreHarnessAgent}. */

import type { InvokeHarnessStreamOutput } from '@aws-sdk/client-bedrock-agentcore'
import { StreamEvent } from '../hooks/events.js'
import type { AgentResultEvent } from '../hooks/events.js'

/** Non-error events received from an AgentCore Harness invocation. */
export type AgentCoreHarnessEventData = Exclude<
  InvokeHarnessStreamOutput,
  | InvokeHarnessStreamOutput.InternalServerExceptionMember
  | InvokeHarnessStreamOutput.ValidationExceptionMember
  | InvokeHarnessStreamOutput.RuntimeClientErrorMember
>

/** Wraps one raw event from the `InvokeHarness` response stream. */
export class AgentCoreHarnessStreamUpdateEvent extends StreamEvent {
  readonly type = 'agentCoreHarnessStreamUpdateEvent' as const
  readonly event: AgentCoreHarnessEventData

  /**
   * Creates a Harness stream update.
   *
   * @param event - Raw non-error event received from `InvokeHarness`
   */
  constructor(event: AgentCoreHarnessEventData) {
    super()
    this.event = event
  }

  /**
   * Serializes the event.
   *
   * @returns Event type and raw Harness event
   */
  toJSON(): Pick<AgentCoreHarnessStreamUpdateEvent, 'type' | 'event'> {
    return { type: this.type, event: this.event }
  }
}

/** Wraps the final result yielded by an AgentCore Harness agent stream. */
export class AgentCoreHarnessResultEvent extends StreamEvent {
  readonly type = 'agentCoreHarnessResultEvent' as const
  readonly result: AgentResultEvent['result']

  /**
   * Creates a final Harness result event.
   *
   * @param data - Completed agent result
   */
  constructor(data: Pick<AgentResultEvent, 'result'>) {
    super()
    this.result = data.result
  }

  /**
   * Serializes the event.
   *
   * @returns Event type and completed result
   */
  toJSON(): Pick<AgentCoreHarnessResultEvent, 'type' | 'result'> {
    return { type: this.type, result: this.result }
  }
}

/** Events yielded by {@link AgentCoreHarnessAgent.stream}. */
export type AgentCoreHarnessStreamEvent = AgentCoreHarnessStreamUpdateEvent | AgentCoreHarnessResultEvent
