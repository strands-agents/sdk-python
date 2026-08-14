import { describe, expect, it, vi } from 'vitest'
import { z } from 'zod'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { Agent } from '../../agent/agent.js'
import { SlidingWindowConversationManager } from '../../conversation-manager/sliding-window-conversation-manager.js'
import type { StreamOptions } from '../../models/model.js'
import { InMemoryStorage } from '../../storage/in-memory-storage.js'
import { tool } from '../../tools/tool-factory.js'
import { Message } from '../../types/messages.js'
import { ContextOffloader } from '../../vended-plugins/context-offloader/plugin.js'
import { GoalLoop } from '../../vended-plugins/goal/plugin.js'
import type { BackgroundTasksConfig } from '../types.js'

class RecordingModel extends MockMessageModel {
  readonly requests: Message[][] = []

  override async *stream(messages: Message[], options?: StreamOptions) {
    this.requests.push(messages.map((message) => message.clone()))
    yield* super.stream(messages, options)
  }
}

function backgroundTasks(policy: 'always' | 'agentic' = 'always'): BackgroundTasksConfig {
  return { timeout: 5_000, [policy]: ['*'] }
}

function workTool(result: string, callback = vi.fn(() => result)) {
  return {
    callback,
    tool: tool({
      name: 'work',
      description: 'Perform background work.',
      inputSchema: z.object({ value: z.string() }),
      callback,
    }),
  }
}

function continuationIndex(messages: readonly Message[]): number {
  return messages.findIndex((message) =>
    message.content.some((block) => block.type === 'toolUseBlock' && block.name === 'strands_background_task_result')
  )
}

describe('Background Tasks composition', () => {
  it('delivers results before a GoalLoop continuation', async () => {
    const background = backgroundTasks()
    let validation = 0
    const goal = new GoalLoop({
      name: 'composition',
      goal: () => {
        validation += 1
        return validation === 2 || { passed: false, feedback: 'use the completed background result' }
      },
      maxAttempts: 2,
    })
    const { tool: work } = workTool('background complete')
    const model = new RecordingModel()
      .addTurn({
        type: 'toolUseBlock',
        name: 'work',
        toolUseId: 'work-1',
        input: { value: 'x' },
      })
      .addTurn({ type: 'textBlock', text: 'draft' })
      .addTurn({ type: 'textBlock', text: 'background-aware draft' })
    const agent = new Agent({
      model,
      tools: [work],
      backgroundTasks: background,
      plugins: [goal],
      printer: false,
    })
    const result = await agent.invoke('start')

    expect(result.lastMessage.content).toEqual([
      expect.objectContaining({ type: 'textBlock', text: 'background-aware draft' }),
    ])
    expect(goal.lastResult(agent)).toEqual({
      passed: true,
      stopReason: 'satisfied',
      attempts: [
        { attempt: 1, passed: false, feedback: 'use the completed background result' },
        { attempt: 2, passed: true },
      ],
    })
    const deliveryRequest = model.requests.find((messages) => continuationIndex(messages) >= 0)!
    expect(deliveryRequest).toBeDefined()
  })

  it('retains structured output while delivering a co-running background task', async () => {
    const background = backgroundTasks()
    const { tool: work } = workTool('background complete')
    const model = new RecordingModel()
      .addTurn([
        {
          type: 'toolUseBlock',
          name: 'work',
          toolUseId: 'work-1',
          input: { value: 'x' },
        },
        {
          type: 'toolUseBlock',
          name: 'strands_structured_output',
          toolUseId: 'structured-1',
          input: { value: 1 },
        },
      ])
      .addTurn({
        type: 'toolUseBlock',
        name: 'strands_structured_output',
        toolUseId: 'structured-2',
        input: { value: 2 },
      })
    const agent = new Agent({
      model,
      tools: [work],
      backgroundTasks: background,
      structuredOutputSchema: z.object({ value: z.number() }),
      printer: false,
    })
    const result = await agent.invoke('start')

    expect(result.structuredOutput).toEqual({ value: 2 })
    await expect(agent.backgroundTasks!.list()).resolves.toEqual([])
    expect(model.requests.some((messages) => continuationIndex(messages) >= 0)).toBe(true)
  })

  it('delivers the ContextOffloader-transformed result rather than the oversized payload', async () => {
    const offloadStorage = new InMemoryStorage()
    const background = backgroundTasks()
    const offloader = new ContextOffloader({
      storage: offloadStorage,
      maxResultTokens: 10,
      previewTokens: 2,
      includeRetrievalTool: false,
    })
    const largeResult = 'x'.repeat(2_000)
    const { tool: work } = workTool(largeResult)
    const model = new RecordingModel()
      .addTurn({
        type: 'toolUseBlock',
        name: 'work',
        toolUseId: 'work-1',
        input: { value: 'x' },
      })
      .addTurn({ type: 'textBlock', text: 'admitted' })
      .addTurn({ type: 'textBlock', text: 'delivered' })
    const agent = new Agent({
      model,
      tools: [work],
      backgroundTasks: background,
      plugins: [offloader],
      printer: false,
    })
    await agent.invoke('start')

    const deliveryIndex = continuationIndex(agent.messages)
    const deliveredResult = agent.messages[deliveryIndex + 1]?.content[0]
    expect(deliveredResult).toEqual(expect.objectContaining({ type: 'toolResultBlock' }))
    expect(deliveredResult?.type === 'toolResultBlock' ? deliveredResult.content[1] : undefined).toEqual(
      expect.objectContaining({ text: expect.stringContaining('[Offloaded:') })
    )
    expect(JSON.stringify(deliveredResult)).not.toContain(largeResult)
  })

  it('protects the complete delivery pair through proactive compaction', async () => {
    const background = backgroundTasks()
    const { tool: work } = workTool('background complete')
    const model = new RecordingModel()
      .addTurn({
        type: 'toolUseBlock',
        name: 'work',
        toolUseId: 'work-1',
        input: { value: 'x' },
      })
      .addTurn({ type: 'textBlock', text: 'admitted' })
      .addTurn({ type: 'textBlock', text: 'delivery consumed' })
    model.updateConfig({ contextWindowLimit: 500, maxTokens: 10 })
    const agent = new Agent({
      model,
      tools: [work],
      backgroundTasks: background,
      conversationManager: new SlidingWindowConversationManager({
        windowSize: 0,
        proactiveCompression: { compressionThreshold: 0.1 },
      }),
      printer: false,
    })
    await agent.invoke('start')

    const deliveryRequest = model.requests.find((messages) => continuationIndex(messages) >= 0)!
    const deliveryIndex = continuationIndex(deliveryRequest)
    const deliveryMessages = deliveryRequest.slice(deliveryIndex, deliveryIndex + 2)
    expect(deliveryMessages.map((message) => message.metadata?.custom?.pinned)).toEqual([true, true])
    expect(deliveryMessages[0]!.content[0]).toEqual(
      expect.objectContaining({ type: 'toolUseBlock', name: 'strands_background_task_result' })
    )
    expect(deliveryMessages[1]!.content[0]).toEqual(expect.objectContaining({ type: 'toolResultBlock' }))
  })
})
