import { randomUUID } from 'node:crypto'
import { describe, expect, it } from 'vitest'
import { z } from 'zod'
import { collectGenerator } from '$/sdk/__fixtures__/model-test-helpers.js'
import {
  AgentCoreHarnessAgent,
  AgentCoreHarnessStreamUpdateEvent,
  type AgentCoreHarnessStreamEvent,
} from '$/sdk/agentcore-harness/index.js'
import { ModelError } from '$/sdk/errors.js'
import { tool } from '$/sdk/index.js'

const harnessArn = process.env.AGENTCORE_HARNESS_ARN
const region = process.env.AGENTCORE_HARNESS_REGION ?? process.env.AWS_REGION ?? 'us-east-1'
const qualifier = process.env.AGENTCORE_HARNESS_QUALIFIER
const reasoningModelId = process.env.AGENTCORE_HARNESS_REASONING_MODEL_ID

function requiredHarnessArn(): string {
  if (!harnessArn) throw new Error('AGENTCORE_HARNESS_ARN is required')
  return harnessArn
}

function sessionId(label: string): string {
  return `strands-integ-${label}-${randomUUID()}`.slice(0, 100)
}

function completedMessageRoles(events: AgentCoreHarnessStreamEvent[]): string[] {
  const roles: string[] = []
  let currentRole = 'assistant'
  for (const event of events) {
    if (!(event instanceof AgentCoreHarnessStreamUpdateEvent)) continue
    const chunk = event.event
    if ('messageStart' in chunk && chunk.messageStart?.role) currentRole = chunk.messageStart.role
    if ('messageStop' in chunk && chunk.messageStop) roles.push(currentRole)
  }
  return roles
}

describe.skipIf(!harnessArn)('AgentCoreHarnessAgent integration', () => {
  it('continues a session and reports stream cycles independently of request metadata', async () => {
    const runtimeSessionId = sessionId('memory')
    const memoryToken = `cobalt-${randomUUID().slice(0, 8)}`
    const agent = new AgentCoreHarnessAgent({
      harnessArn: requiredHarnessArn(),
      runtimeSessionId,
      region,
      ...(qualifier && { qualifier }),
      systemPrompt: 'Follow the user instructions exactly and answer concisely.',
    })

    await agent.invoke(`Remember the exact token ${memoryToken}. Reply only with ACK.`)
    const { items, result } = await collectGenerator(
      agent.stream('Return the exact token I asked you to remember. Include nothing else.')
    )
    const completedRoles = completedMessageRoles(items)

    expect(result.stopReason).toBe('endTurn')
    expect(result.toString()).toContain(memoryToken)
    expect(completedRoles).toContain('assistant')
    expect(result.metrics).toBeDefined()
    expect(result.metrics!.cycleCount).toBe(completedRoles.filter((role) => role === 'assistant').length)
    expect(result.metrics!.totalDuration).toBeGreaterThan(0)
  }, 120_000)

  it('runs an allowed host tool and commits its result before completing', async () => {
    const receipt = `receipt-${randomUUID().slice(0, 8)}`
    let observedSku: string | undefined
    let callCount = 0
    const lookupInventory = tool({
      name: 'lookup_inventory',
      description: 'Look up the available quantity and receipt for a product SKU',
      inputSchema: z.object({ sku: z.string() }),
      callback: ({ sku }) => {
        observedSku = sku
        callCount++
        return { sku, availableQuantity: 7, receipt }
      },
    })
    const agent = new AgentCoreHarnessAgent({
      harnessArn: requiredHarnessArn(),
      runtimeSessionId: sessionId('host-tool'),
      region,
      ...(qualifier && { qualifier }),
      tools: [lookupInventory],
      allowedTools: ['@lookup_inventory'],
      systemPrompt:
        'For inventory questions, call lookup_inventory exactly once and include its receipt unchanged in the answer.',
    })

    const { items, result } = await collectGenerator(agent.stream('Check inventory for SKU-1234.'))
    const completedRoles = completedMessageRoles(items)

    expect(observedSku).toBe('SKU-1234')
    expect(callCount).toBe(1)
    expect(result.stopReason).toBe('endTurn')
    expect(result.toString()).toContain(receipt)
    expect(completedRoles.filter((role) => role === 'assistant').length).toBeGreaterThanOrEqual(2)
    expect(result.metrics).toBeDefined()
    expect(result.metrics!.cycleCount).toBe(completedRoles.filter((role) => role === 'assistant').length)
  }, 120_000)

  it.skipIf(!reasoningModelId)(
    'preserves reasoning text and its signature',
    async () => {
      const agent = new AgentCoreHarnessAgent({
        harnessArn: requiredHarnessArn(),
        runtimeSessionId: sessionId('reasoning'),
        region,
        ...(qualifier && { qualifier }),
        modelConfig: {
          bedrockModelConfig: {
            modelId: reasoningModelId!,
            apiFormat: 'converse_stream',
            maxTokens: 2_000,
            additionalParams: {
              additionalModelRequestFields: {
                thinking: { type: 'enabled', budget_tokens: 1_024 },
              },
            },
          },
        },
      })

      const { items, result } = await collectGenerator(agent.stream('What is 17 multiplied by 19?'))
      const rawEvents = items
        .filter(
          (event): event is AgentCoreHarnessStreamUpdateEvent => event instanceof AgentCoreHarnessStreamUpdateEvent
        )
        .map((event) => event.event)
      const reasoningDeltas = rawEvents.filter(
        (event) =>
          'contentBlockDelta' in event &&
          event.contentBlockDelta?.delta &&
          'reasoningContent' in event.contentBlockDelta.delta
      )

      expect(reasoningDeltas.length).toBeGreaterThan(0)
      expect(
        reasoningDeltas.some(
          (event) =>
            'contentBlockDelta' in event &&
            event.contentBlockDelta?.delta &&
            'reasoningContent' in event.contentBlockDelta.delta &&
            event.contentBlockDelta.delta.reasoningContent?.signature
        )
      ).toBe(true)
      expect(
        result.lastMessage.content.some((block) => block.type === 'reasoningBlock' && block.text && block.signature)
      ).toBe(true)
    },
    120_000
  )

  it('preserves a non-throttling control-plane error', async () => {
    const hostTool = tool({
      name: 'lookup_inventory',
      description: 'Look up inventory for a product SKU',
      inputSchema: z.object({ sku: z.string() }),
      callback: () => 'unused',
    })
    const agent = new AgentCoreHarnessAgent({
      harnessArn: requiredHarnessArn(),
      runtimeSessionId: sessionId('missing-endpoint'),
      region,
      qualifier: `missing-${randomUUID().slice(0, 8)}`,
      tools: [hostTool],
    })

    try {
      await agent.invoke('Check inventory.')
      expect.unreachable('Expected the missing endpoint lookup to fail')
    } catch (error) {
      expect(error).toBeInstanceOf(Error)
      expect(error).not.toBeInstanceOf(ModelError)
      expect((error as Error).name).toMatch(/Exception$/)
      expect(error).toHaveProperty('$metadata.httpStatusCode', 400)
    }
  })
})
