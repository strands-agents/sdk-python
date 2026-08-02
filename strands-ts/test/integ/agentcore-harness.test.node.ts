import { randomUUID } from 'node:crypto'
import { describe, expect, it } from 'vitest'
import { AgentCoreHarnessAgent } from '@strands-agents/sdk/agentcore-harness'

const harnessArn = process.env.AGENTCORE_HARNESS_ARN

describe.skipIf(!harnessArn)('AgentCoreHarnessAgent integration', () => {
  it('invokes a deployed Harness and continues its session', async () => {
    const agent = new AgentCoreHarnessAgent({
      harnessArn: harnessArn!,
      runtimeSessionId: randomUUID(),
    })

    const first = await agent.invoke('Reply with a short greeting.')
    const second = await agent.invoke('Reply with a different short greeting.')

    expect(first).toMatchObject({ stopReason: 'endTurn' })
    expect(second).toMatchObject({ stopReason: 'endTurn' })
    expect(first.toString().length).toBeGreaterThan(0)
    expect(second.toString().length).toBeGreaterThan(0)
  }, 120_000)
})
