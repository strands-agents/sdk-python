import { randomUUID } from 'node:crypto'
import { Agent, Graph } from '@strands-agents/sdk'
import {
  AgentCoreHarnessAgent,
  AgentCoreHarnessResultEvent,
  AgentCoreHarnessStreamUpdateEvent,
} from '@strands-agents/sdk/agentcore-harness'

async function invokeHarnessSession() {
  // --8<-- [start:session]
  const harnessArn = process.env.AGENTCORE_HARNESS_ARN
  if (!harnessArn) throw new Error('Set AGENTCORE_HARNESS_ARN')

  const harnessAgent = new AgentCoreHarnessAgent({
    harnessArn,
    runtimeSessionId: randomUUID(),
  })

  const first = await harnessAgent.invoke(
    'Remember that the project codename is Aurora. Reply with "ready".'
  )
  console.log(first.toString())
  // Typical output:
  // ready

  const followUp = await harnessAgent.invoke('What project codename did I give you?')
  console.log(followUp.toString())
  // Typical output:
  // Aurora
  // --8<-- [end:session]
}

async function streamHarness() {
  // --8<-- [start:stream]
  const harnessArn = process.env.AGENTCORE_HARNESS_ARN
  if (!harnessArn) throw new Error('Set AGENTCORE_HARNESS_ARN')

  const harnessAgent = new AgentCoreHarnessAgent({
    harnessArn,
    runtimeSessionId: randomUUID(),
  })

  for await (const event of harnessAgent.stream('Summarize the latest project status.')) {
    if (event instanceof AgentCoreHarnessStreamUpdateEvent) {
      const rawEvent = event.event

      if ('contentBlockDelta' in rawEvent && rawEvent.contentBlockDelta) {
        const delta = rawEvent.contentBlockDelta.delta
        if (delta && 'text' in delta) process.stdout.write(delta.text ?? '')
      }

      if ('metadata' in rawEvent && rawEvent.metadata) {
        console.log('\nUsage:', rawEvent.metadata.usage)
        console.log('Latency:', rawEvent.metadata.metrics)
      }
    }

    if (event instanceof AgentCoreHarnessResultEvent) {
      console.log('\nStop reason:', event.result.stopReason)
    }
  }
  // --8<-- [end:stream]
}

async function useHarnessInGraph() {
  // --8<-- [start:graph]
  const harnessArn = process.env.AGENTCORE_HARNESS_ARN
  if (!harnessArn) throw new Error('Set AGENTCORE_HARNESS_ARN')

  const researcher = new AgentCoreHarnessAgent({
    harnessArn,
    runtimeSessionId: randomUUID(),
    id: 'remote-researcher',
  })
  const editor = new Agent({
    id: 'local-editor',
    systemPrompt: 'Turn the research into a concise engineering brief.',
  })

  const graph = new Graph({
    nodes: [researcher, editor],
    edges: [[researcher.id, editor.id]],
    maxSteps: 2,
  })

  const result = await graph.invoke('Compare three approaches to request caching.')
  console.log(result.status)
  // completed
  // --8<-- [end:graph]
}
