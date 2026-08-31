import { describe, expect, it } from 'vitest'
import { z } from 'zod'

import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { ExecuteToolStage } from '../../middleware/index.js'
import { tool } from '../../tools/tool-factory.js'
import { InterruptResponseContent } from '../../types/interrupt.js'
import { TextBlock } from '../../types/messages.js'
import { Agent } from '../agent.js'

import type { BackgroundTask } from '../../background-tasks/types.js'

const BACKGROUND_TASKS_STATE_KEY = 'strands.backgroundTasks'

/**
 * A tool whose callback suspends until `release()` is called. `started` resolves once
 * the callback is entered. The callback also resolves on `cancelSignal` abort so
 * cancellation can end it.
 */
function createGate(name: string) {
  let signalStarted!: () => void
  const started = new Promise<void>((resolve) => (signalStarted = resolve))
  let release!: () => void
  const released = new Promise<void>((resolve) => (release = resolve))

  const gateTool = tool({
    name,
    description: `Gated tool ${name}`,
    inputSchema: z.object({}),
    callback: async (_input, context) => {
      signalStarted()
      await new Promise<void>((resolve) => {
        void released.then(resolve)
        context?.cancelSignal.addEventListener('abort', () => resolve(), { once: true })
      })
      return `${name} done`
    },
  })

  return { tool: gateTool, started, release }
}

/** Waits for a same-loop condition with a bounded number of macrotask hops. */
async function until(condition: () => boolean, label: string): Promise<void> {
  for (let i = 0; i < 2000 && !condition(); i++) {
    await new Promise((resolve) => setTimeout(resolve, 0))
  }
  if (!condition()) throw new Error(`timed out waiting for: ${label}`)
}

/** The text of a result's last message. */
function resultText(result: { lastMessage: { content: readonly unknown[] } }): string {
  const block = result.lastMessage.content[0]
  return block instanceof TextBlock ? block.text : ''
}

/** Index of the first message whose content matches the predicate, or -1. */
function messageIndex(agent: Agent, predicate: (content: readonly unknown[]) => boolean): number {
  return agent.messages.findIndex((message) => predicate(message.content))
}

function hasText(content: readonly unknown[], text: string): boolean {
  return content.some((block) => block instanceof TextBlock && block.text === text)
}

function hasBackgroundDelivery(content: readonly unknown[]): boolean {
  return content.some(
    (block) =>
      typeof block === 'object' &&
      block !== null &&
      'name' in block &&
      (block as { name?: string }).name === 'strands_background_task_result'
  )
}

function persistedTasks(agent: Agent): BackgroundTask[] | undefined {
  return agent.appState.get(BACKGROUND_TASKS_STATE_KEY) as unknown as BackgroundTask[] | undefined
}

describe('concurrentInvocationMode enqueue × backgroundTasks', () => {
  it('runs a queued invocation strictly after background-task settlement and delivery', async () => {
    const work = createGate('work')
    const model = new MockMessageModel()
      .addTurn({ type: 'toolUseBlock', name: 'work', toolUseId: 'work-use', input: {} })
      .addTurn({ type: 'textBlock', text: 'first done' })
      .addTurn({ type: 'textBlock', text: 'delivered' })
      .addTurn({ type: 'textBlock', text: 'second done' })
    const agent = new Agent({
      model,
      tools: [work.tool],
      backgroundTasks: { always: [work.tool] },
      concurrentInvocationMode: 'enqueue',
      printer: false,
    })

    const first = agent.invoke('first')
    await work.started
    // Wait until the first invocation is past its final model turn — it is now blocked
    // in AfterInvocation awaiting background-task settlement, still holding the turn.
    await until(() => agent.messages.some((message) => hasText(message.content, 'first done')), 'first final turn')

    const second = agent.invoke('second')
    await until(() => agent.pendingInvocations.length === 1, 'second invocation queued')

    // The queued caller must NOT start while the first invocation awaits settlement.
    expect(model.callCount).toBe(2)

    work.release()
    const firstResult = await first
    const secondResult = await second

    // The first invocation absorbed the delivery continuation; the queued one ran after.
    expect(resultText(firstResult)).toBe('delivered')
    expect(resultText(secondResult)).toBe('second done')
    expect(agent.pendingInvocations).toHaveLength(0)

    // Durable history: delivery landed before the queued invocation's input.
    const deliveryIndex = messageIndex(agent, hasBackgroundDelivery)
    const secondInputIndex = messageIndex(agent, (content) => hasText(content, 'second'))
    expect(deliveryIndex).toBeGreaterThan(-1)
    expect(secondInputIndex).toBeGreaterThan(deliveryIndex)
    // The task is fully consumed — nothing left to re-deliver.
    expect(persistedTasks(agent)).toBeUndefined()
  })

  it('delivers a task that outlives its dispatching invocation inside the queued invocation', async () => {
    const work = createGate('work')
    const firstGate = createGate('first_gate')
    const secondGate = createGate('second_gate')
    const model = new MockMessageModel()
      .addTurn([
        { type: 'toolUseBlock', name: 'work', toolUseId: 'work-use', input: {} },
        { type: 'toolUseBlock', name: 'first_gate', toolUseId: 'first-gate-use', input: {} },
      ])
      .addTurn({ type: 'textBlock', text: 'first done' })
      .addTurn({ type: 'toolUseBlock', name: 'second_gate', toolUseId: 'second-gate-use', input: {} })
      .addTurn({ type: 'textBlock', text: 'second done' })
    const agent = new Agent({
      model,
      tools: [work.tool, firstGate.tool, secondGate.tool],
      backgroundTasks: { always: [work.tool], never: [firstGate.tool, secondGate.tool], waitForCompletion: false },
      concurrentInvocationMode: 'enqueue',
      printer: false,
    })

    const first = agent.invoke('first')
    await firstGate.started
    const second = agent.invoke('second')
    expect(agent.pendingInvocations).toHaveLength(1)

    // First invocation ends with the background task still working (waitForCompletion: false).
    firstGate.release()
    const firstResult = await first
    expect(resultText(firstResult)).toBe('first done')
    expect(persistedTasks(agent)?.map((task) => task.status)).toEqual(['working'])

    // The queued invocation now holds the turn; the task settles while it runs.
    await secondGate.started
    work.release()
    await until(() => persistedTasks(agent)?.[0]?.status === 'completed', 'background task settled')
    secondGate.release()

    // The queued invocation absorbs the delivery: the settled task dispatched by
    // invocation one is injected at the queued invocation's next model call.
    const secondResult = await second
    expect(resultText(secondResult)).toBe('second done')
    const deliveryIndex = messageIndex(agent, hasBackgroundDelivery)
    const secondInputIndex = messageIndex(agent, (content) => hasText(content, 'second'))
    expect(deliveryIndex).toBeGreaterThan(secondInputIndex)
    expect(persistedTasks(agent)).toBeUndefined()
  })

  it('rejects a queued caller loudly when a background task interrupt ends the running invocation', async () => {
    const approval = tool({
      name: 'approval',
      description: 'Wait for approval.',
      inputSchema: z.object({}),
      callback: () => 'approved',
    })
    const holdGate = createGate('hold_gate')
    const model = new MockMessageModel()
      .addTurn([
        { type: 'toolUseBlock', name: 'approval', toolUseId: 'approval-use', input: {} },
        { type: 'toolUseBlock', name: 'hold_gate', toolUseId: 'hold-gate-use', input: {} },
      ])
      .addTurn({ type: 'textBlock', text: 'resumed' })
      .addTurn({ type: 'textBlock', text: 'delivered' })
    const agent = new Agent({
      model,
      tools: [approval, holdGate.tool],
      backgroundTasks: { always: [approval], never: [holdGate.tool] },
      concurrentInvocationMode: 'enqueue',
      printer: false,
    })
    agent.addMiddleware(ExecuteToolStage, async function* (context, next) {
      if (context.toolUse.name === 'approval') {
        context.interrupt<string>({ name: 'approve', reason: 'Approve work?' })
      }
      return yield* next(context)
    })

    // The foreground gate holds the first invocation open while the background
    // approval task interrupts (input_required) and the second caller queues up.
    const first = agent.invoke('first')
    await holdGate.started
    const second = agent.invoke('second')
    expect(agent.pendingInvocations).toHaveLength(1)
    holdGate.release()

    const firstResult = await first
    expect(firstResult.stopReason).toBe('interrupt')

    // The queued caller cannot answer someone else's interrupt: it fails loudly rather
    // than silently absorbing or corrupting the interrupted state.
    await expect(second).rejects.toThrow(/interrupted state/)

    // The interrupt remains resumable after the queued caller's rejection.
    const resumed = await agent.invoke([
      new InterruptResponseContent({
        interruptId: firstResult.interrupts![0]!.id,
        response: 'yes',
      }),
    ])
    expect(resumed.stopReason).toBe('endTurn')
    expect(messageIndex(agent, hasBackgroundDelivery)).toBeGreaterThan(-1)
    expect(persistedTasks(agent)).toBeUndefined()
  })
})
