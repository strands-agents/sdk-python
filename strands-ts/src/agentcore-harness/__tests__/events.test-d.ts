import { describe, expectTypeOf, it } from 'vitest'
import type { InvokeHarnessStreamOutput } from '@aws-sdk/client-bedrock-agentcore'
import type { AgentCoreHarnessEventData, AgentCoreHarnessStreamUpdateEvent } from '../events.js'

describe('AgentCoreHarnessEventData', () => {
  it('includes non-error stream events', () => {
    expectTypeOf<InvokeHarnessStreamOutput.MessageStartMember>().toExtend<AgentCoreHarnessEventData>()
    expectTypeOf<AgentCoreHarnessStreamUpdateEvent['event']>().toEqualTypeOf<AgentCoreHarnessEventData>()
  })

  it('excludes harness error events', () => {
    expectTypeOf<InvokeHarnessStreamOutput.InternalServerExceptionMember>().not.toExtend<AgentCoreHarnessEventData>()
    expectTypeOf<InvokeHarnessStreamOutput.ValidationExceptionMember>().not.toExtend<AgentCoreHarnessEventData>()
    expectTypeOf<InvokeHarnessStreamOutput.RuntimeClientErrorMember>().not.toExtend<AgentCoreHarnessEventData>()
  })
})
