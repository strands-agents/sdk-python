import { describe, expect, it, vi } from 'vitest'
import { z } from 'zod'
import { Graph } from '../../multiagent/graph.js'
import { Status } from '../../multiagent/state.js'
import { Swarm } from '../../multiagent/swarm.js'
import { tool } from '../../tools/tool-factory.js'
import { AgentCoreHarnessAgent } from '../agentcore-harness-agent.js'
import {
  HARNESS_ARN,
  SESSION_ID,
  chunk,
  harnessStream,
  mockClient,
  mockControlClient,
} from './harness-test-fixtures.js'

describe('AgentCoreHarnessAgent composition', () => {
  it('rejects structured output before resolving tools or invoking the harness', async () => {
    const hostTool = tool({
      name: 'get_weather',
      description: 'Get weather',
      inputSchema: z.object({}),
      callback: vi.fn(() => 'sunny'),
    })
    const { client, send } = mockClient()
    const { controlClient, send: controlSend } = mockControlClient()
    const agent = new AgentCoreHarnessAgent({
      harnessArn: HARNESS_ARN,
      runtimeSessionId: SESSION_ID,
      client,
      controlClient,
      tools: [hostTool],
      allowedTools: ['@get_weather'],
    })

    await expect(
      agent.invoke('Hi', {
        structuredOutputSchema: z.object({ answer: z.string() }),
      })
    ).rejects.toThrow(
      'InvokeOptions.structuredOutputSchema is not supported by AgentCoreHarnessAgent. AgentCoreHarnessAgent cannot be used as a Swarm node.'
    )
    expect(controlSend).not.toHaveBeenCalled()
    expect(send).not.toHaveBeenCalled()
  })

  it('runs as a Graph node and reuses its configured remote session across Graph invocations', async () => {
    const { client, send } = mockClient(
      harnessStream(chunk.messageStop('end_turn')),
      harnessStream(chunk.messageStop('end_turn'))
    )
    const agent = new AgentCoreHarnessAgent({
      harnessArn: HARNESS_ARN,
      runtimeSessionId: SESSION_ID,
      client,
    })
    const graph = new Graph({ nodes: [agent], edges: [], maxSteps: 2 })

    const first = await graph.invoke('First')
    const second = await graph.invoke('Second')

    expect(first.status).toBe(Status.COMPLETED)
    expect(second.status).toBe(Status.COMPLETED)
    expect(
      send.mock.calls.map((call) => ({
        runtimeSessionId: call[0].input.runtimeSessionId,
        messages: call[0].input.messages,
      }))
    ).toStrictEqual([
      { runtimeSessionId: SESSION_ID, messages: [{ role: 'user', content: [{ text: 'First' }] }] },
      { runtimeSessionId: SESSION_ID, messages: [{ role: 'user', content: [{ text: 'Second' }] }] },
    ])
  })

  it('fails a Swarm node without making a Harness request', async () => {
    const { client, send } = mockClient()
    const agent = new AgentCoreHarnessAgent({
      harnessArn: HARNESS_ARN,
      runtimeSessionId: SESSION_ID,
      client,
    })
    const swarm = new Swarm({ nodes: [agent], start: agent.id, maxSteps: 1 })

    const result = await swarm.invoke('Hi')

    expect(result.status).toBe(Status.FAILED)
    expect(result.results).toHaveLength(1)
    expect(result.results[0]).toMatchObject({
      nodeId: agent.id,
      status: Status.FAILED,
      error: {
        message:
          'InvokeOptions.structuredOutputSchema is not supported by AgentCoreHarnessAgent. AgentCoreHarnessAgent cannot be used as a Swarm node.',
      },
    })
    expect(send).not.toHaveBeenCalled()
  })
})
