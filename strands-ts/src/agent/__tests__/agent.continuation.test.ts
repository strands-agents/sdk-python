import { describe, expect, it, vi } from 'vitest'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { createMockTool } from '../../__fixtures__/tool-helpers.js'
import {
  AfterInvocationEvent,
  BeforeInvocationEvent,
  BeforeModelCallEvent,
  MessageAddedEvent,
} from '../../hooks/events.js'
import { AgentStreamStage, InvokeModelStage } from '../../middleware/stages.js'
import { AgentResult } from '../../types/agent.js'
import { Message, TextBlock, ToolResultBlock, ToolUseBlock } from '../../types/messages.js'
import { logger } from '../../logging/logger.js'
import { Agent } from '../agent.js'
import { continuations as continuationManager } from '../continuation.js'

import type { InvokeArgs } from '../../types/agent.js'
import type { AgentStreamContext, AgentStreamResult } from '../../middleware/stages.js'
import type { StopReason } from '../../types/messages.js'
import type { ContinuationInput } from '../continuation.js'

function textOf(message: Message): string {
  return message.content.flatMap((block) => (block.type === 'textBlock' ? [block.text] : [])).join('')
}

function textModel(...turns: string[]): MockMessageModel {
  return turns.reduce((model, text) => model.addTurn({ type: 'textBlock', text }), new MockMessageModel())
}

function captureRequests(model: MockMessageModel): Message[][] {
  const requests: Message[][] = []
  const originalStream = model.stream.bind(model)
  vi.spyOn(model, 'stream').mockImplementation(async function* (messages, options) {
    requests.push(messages.map((message) => message.clone()))
    yield* originalStream(messages, options)
  })
  return requests
}

function toolExchange(toolUseId: string, text: string): [Message, Message] {
  return [
    new Message({
      role: 'assistant',
      content: [new ToolUseBlock({ name: 'deferred-result', toolUseId, input: {} })],
    }),
    new Message({
      role: 'user',
      content: [
        new ToolResultBlock({
          toolUseId,
          status: 'success',
          content: [new TextBlock(text)],
        }),
      ],
    }),
  ]
}

function continueAfterInvocation(agent: Agent, ...inputs: ContinuationInput[]): void {
  let armed = false
  agent.addHook(AfterInvocationEvent, (event) => {
    if (armed) return
    armed = true
    for (const input of inputs) {
      continuationManager.add(event, input)
    }
  })
}

function streamResult(
  context: AgentStreamContext,
  output: string | Message,
  stopReason: StopReason = 'endTurn'
): AgentStreamResult {
  return {
    result: new AgentResult({
      stopReason,
      lastMessage:
        typeof output === 'string' ? new Message({ role: 'assistant', content: [new TextBlock(output)] }) : output,
      invocationState: context.options?.invocationState ?? {},
    }),
  }
}

describe('Agent continuation input', () => {
  it('combines internal contributions before public resume input', async () => {
    const model = new MockMessageModel()
      .addTurn({ type: 'textBlock', text: 'initial' })
      .addTurn({ type: 'toolUseBlock', name: 'noop', toolUseId: 'tool-1', input: {} })
      .addTurn({ type: 'textBlock', text: 'final' })
    const requests = captureRequests(model)
    const consumed: string[] = []
    const projectedInputTokens: number[] = []
    const agent = new Agent({ model, tools: [createMockTool('noop', () => 'done')], printer: false })
    let armed = false
    let streamStageCalls = 0

    vi.spyOn(model, 'countTokens').mockImplementation(async (messages) => messages.length)
    agent.addHook(AfterInvocationEvent, (event) => {
      if (armed) return
      armed = true
      for (const args of ['first', 'second', 'third']) {
        continuationManager.add(event, {
          args,
          onConsumed: () => {
            consumed.push(args)
          },
        })
      }
      event.resume = 'public'
    })
    agent.addMiddleware(AgentStreamStage.Input, async (context) => {
      streamStageCalls += 1
      return streamStageCalls === 2 ? { ...context, args: 'rewritten public' } : context
    })
    agent.addHook(BeforeModelCallEvent, (event) => {
      projectedInputTokens.push(event.projectedInputTokens ?? 0)
    })
    await agent.invoke('start')

    expect(requests[1]?.map((message) => message.role)).toEqual(['user', 'assistant', 'user'])
    expect(requests[1]?.at(-1)?.content.map((block) => (block.type === 'textBlock' ? block.text : block.type))).toEqual(
      ['first', 'second', 'third', 'rewritten public']
    )
    expect(requests[2]?.map(textOf)).toEqual(['start', 'initial', 'firstsecondthirdrewritten public', '', ''])
    expect(agent.messages.map(textOf)).toEqual([
      'start',
      'initial',
      'first',
      'second',
      'third',
      'rewritten public',
      '',
      '',
      'final',
    ])
    expect(projectedInputTokens[1]).toBe(3)
    expect(consumed).toEqual(['first', 'second', 'third'])
  })

  it('consumes complete message input registered before the model call', async () => {
    const model = textModel('final')
    const requests = captureRequests(model)
    const consumed = vi.fn()
    const rejected = vi.fn()
    const [toolUse, toolResult] = toolExchange('delivery-1', 'complete')
    const agent = new Agent({ model, printer: false })
    let armed = false

    agent.addHook(BeforeModelCallEvent, (event) => {
      if (armed) return
      armed = true
      continuationManager.add(event, { args: [toolUse, toolResult], onConsumed: consumed, onRejected: rejected })
    })
    agent.addMiddleware(InvokeModelStage.Input, async (context) => ({
      ...context,
      messages: context.messages.map((message) => {
        const clone = message.clone()
        return new Message({ role: clone.role, content: clone.content })
      }),
    }))
    await agent.invoke('start')

    expect(requests[0]?.map((message) => message.role)).toEqual(['user', 'assistant', 'user'])
    expect(agent.messages).toEqual([expect.any(Message), toolUse, toolResult, expect.any(Message)])
    expect(consumed).toHaveBeenCalledTimes(1)
    expect(rejected).not.toHaveBeenCalled()
  })

  it('preserves public resume visibility and history when the provider fails', async () => {
    const model = new MockMessageModel()
      .addTurn({ type: 'textBlock', text: 'initial' })
      .addTurn(new Error('resumed call failed'))
    const beforeModelMessages: string[][] = []
    const agent = new Agent({ model, printer: false })
    let armed = false
    let streamStageCalls = 0

    agent.addHook(AfterInvocationEvent, (event) => {
      if (armed) return
      armed = true
      event.resume = 'follow-up'
    })
    agent.addMiddleware(AgentStreamStage.Input, async (context) => {
      streamStageCalls += 1
      return streamStageCalls === 2 ? { ...context, args: 'rewritten follow-up' } : context
    })
    agent.addHook(BeforeModelCallEvent, () => {
      beforeModelMessages.push(agent.messages.map(textOf))
    })

    await expect(agent.invoke('start')).rejects.toThrow('resumed call failed')

    expect(beforeModelMessages).toEqual([['start'], ['start', 'initial', 'rewritten follow-up']])
    expect(agent.messages.map(textOf)).toEqual(['start', 'initial', 'rewritten follow-up'])
  })

  it('leaves history unchanged when a resume-only pass is short-circuited', async () => {
    const model = textModel('initial')
    const agent = new Agent({ model, printer: false })
    let afterCalls = 0
    let streamStageCalls = 0

    agent.addHook(AfterInvocationEvent, (event) => {
      afterCalls += 1
      if (afterCalls === 1) event.resume = 'public'
    })
    agent.addMiddleware(AgentStreamStage, async function* (context, next) {
      streamStageCalls += 1
      if (streamStageCalls === 1) return yield* next(context)
      return streamResult(context, 'fabricated')
    })

    const result = await agent.invoke('start')

    expect(textOf(result.lastMessage)).toBe('fabricated')
    expect(agent.messages.map(textOf)).toEqual(['start', 'initial'])
  })

  it('keeps public resume history ordered when the follow-up model call is cancelled', async () => {
    const model = textModel('initial', 'unreachable')
    const agent = new Agent({ model, printer: false })
    let afterCalls = 0
    let modelCalls = 0

    agent.addHook(AfterInvocationEvent, (event) => {
      afterCalls += 1
      if (afterCalls === 1) event.resume = 'public follow-up'
    })
    agent.addHook(BeforeModelCallEvent, (event) => {
      modelCalls += 1
      if (modelCalls === 2) event.cancel = 'model denied'
    })

    await agent.invoke('start')

    expect(agent.messages.map(textOf)).toEqual(['start', 'initial', 'public follow-up', 'model denied'])
  })

  it('rejects input without masking the provider error when its rejection callback fails', async () => {
    const model = new MockMessageModel()
      .addTurn({ type: 'textBlock', text: 'initial' })
      .addTurn(new Error('resumed call failed'))
    const consumed = vi.fn()
    const rejected = vi.fn().mockRejectedValue(new Error('rejection callback failed'))
    const agent = new Agent({ model, printer: false })
    continueAfterInvocation(agent, { args: 'internal follow-up', onConsumed: consumed, onRejected: rejected })

    await expect(agent.invoke('start')).rejects.toThrow('resumed call failed')

    expect(consumed).not.toHaveBeenCalled()
    expect(rejected).toHaveBeenCalledTimes(1)
    expect(agent.messages.map(textOf)).toEqual(['start', 'initial'])
  })

  it.each(['model', 'agentMiddleware', 'modelMiddleware'] as const)(
    'keeps a consumed message batch when a later stage fails after %s completion',
    async (completion) => {
      const model = textModel('initial', 'unrecorded', 'discarded')
      const contribution = toolExchange('hook-delivery', 'internal contribution')
      const beforeModelContribution = new Message({
        role: 'user',
        content: [new TextBlock('before-model contribution')],
      })
      const consumed = vi.fn()
      const rejected = vi.fn()
      const added: Message[] = []
      const agent = new Agent({ model, printer: false })
      const continuation = {
        args: contribution,
        ...(completion !== 'agentMiddleware' && { onConsumed: consumed }),
        onRejected: rejected,
      }

      continueAfterInvocation(agent, continuation)
      let beforeModelCalls = 0
      agent.addHook(BeforeModelCallEvent, (event) => {
        beforeModelCalls += 1
        if (beforeModelCalls === 2) {
          continuationManager.add(event, { args: [beforeModelContribution] })
        }
      })
      agent.addHook(MessageAddedEvent, (event) => {
        if (!contribution.includes(event.message) && event.message !== beforeModelContribution) return
        added.push(event.message)
        if (event.message === contribution[0] && completion !== 'modelMiddleware') {
          throw new Error('message hook failed')
        }
      })
      if (completion === 'agentMiddleware') {
        let calls = 0
        agent.addMiddleware(AgentStreamStage, async function* (context, next) {
          calls += 1
          if (calls === 1) return yield* next(context)
          return streamResult(context, 'cached')
        })
      }
      if (completion === 'modelMiddleware') {
        const contributionIds = new Set(contribution.map((message) => message.trackingId))
        let calls = 0
        agent.addMiddleware(InvokeModelStage, async function* (context, next) {
          calls += 1
          if (calls !== 2) return yield* next(context)

          yield* next(context)
          yield* next({
            ...context,
            messages: context.messages.filter((message) => !contributionIds.has(message.trackingId)),
          })
          throw new Error('model middleware failed')
        })
      }

      await expect(agent.invoke('start')).rejects.toThrow(
        completion === 'modelMiddleware' ? 'model middleware failed' : 'message hook failed'
      )

      expect(rejected).not.toHaveBeenCalled()
      expect(consumed).toHaveBeenCalledTimes(completion === 'agentMiddleware' ? 0 : 1)
      const expectedMessages =
        completion === 'agentMiddleware' ? contribution : [...contribution, beforeModelContribution]
      expect(added).toEqual(expectedMessages)
      expect(agent.messages).toEqual(expect.arrayContaining(expectedMessages))
    }
  )

  it.each(['pendingPass', 'activePass', 'beforeModelCall'] as const)(
    'rejects unconsumed input when the stream closes with a %s continuation',
    async (stage) => {
      const model = textModel('initial', 'unreachable')
      const rejected = vi.fn()
      const agent = new Agent({ model, printer: false })

      if (stage === 'beforeModelCall') {
        agent.addHook(BeforeModelCallEvent, (event) => {
          continuationManager.add(event, { args: 'pending', onRejected: rejected })
        })
      } else {
        continueAfterInvocation(agent, { args: 'pending', onRejected: rejected })
      }

      let beforeInvocationCount = 0
      for await (const event of agent.stream('start')) {
        if (event instanceof BeforeInvocationEvent) beforeInvocationCount += 1
        const shouldClose =
          (stage === 'pendingPass' && event instanceof AfterInvocationEvent) ||
          (stage === 'activePass' && event instanceof BeforeInvocationEvent && beforeInvocationCount === 2) ||
          (stage === 'beforeModelCall' && event instanceof BeforeModelCallEvent)
        if (shouldClose) break
      }

      expect(rejected).toHaveBeenCalledTimes(1)
    }
  )

  it('invokes every committed message hook before yielding the batch', async () => {
    const model = textModel('initial', 'final')
    const first = new Message({ role: 'user', content: [new TextBlock('first')] })
    const second = new Message({ role: 'user', content: [new TextBlock('second')] })
    const added: Message[] = []
    const history: string[][] = []
    const agent = new Agent({ model, printer: false })

    continueAfterInvocation(agent, { args: [first, second] })
    agent.addHook(MessageAddedEvent, (event) => {
      if (event.message === first || event.message === second) {
        added.push(event.message)
        history.push(agent.messages.map(textOf))
      }
    })

    for await (const event of agent.stream('start')) {
      if (event instanceof MessageAddedEvent && event.message === first) break
    }

    expect(added).toEqual([first, second])
    expect(history).toEqual([
      ['start', 'initial', 'first'],
      ['start', 'initial', 'first', 'second'],
    ])
    expect(agent.messages).toEqual([expect.any(Message), expect.any(Message), first, second])
  })

  it('settles continuation input against the model middleware result that wins', async () => {
    const model = textModel('initial', 'first retry', 'final')
    const consumed = vi.fn()
    const rejected = vi.fn()
    const added: string[] = []
    const contribution = new Message({ role: 'user', content: [new TextBlock('start')] })
    const agent = new Agent({ model, printer: false })
    let modelStageCalls = 0

    continueAfterInvocation(agent, { args: [contribution], onConsumed: consumed, onRejected: rejected })
    agent.addHook(MessageAddedEvent, (event) => {
      if (event.message.trackingId === contribution.trackingId) added.push(event.message.trackingId)
    })
    agent.addMiddleware(InvokeModelStage, async function* (context, next) {
      modelStageCalls += 1
      if (modelStageCalls === 1) return yield* next(context)
      yield* next(context)
      const winningMessages = context.messages.filter((message) => message.trackingId !== contribution.trackingId)
      const winningResult = yield* next({
        ...context,
        messages: winningMessages,
      })
      winningMessages.push(contribution)
      return {
        result: {
          ...winningResult.result,
          message: winningResult.result.message.clone(),
        },
      }
    })

    await agent.invoke('start')

    expect(consumed).not.toHaveBeenCalled()
    expect(rejected).toHaveBeenCalledTimes(1)
    expect(added).toHaveLength(0)
    expect(agent.messages.map(textOf)).toEqual(['start', 'initial', 'final'])
  })

  it('rejects ambiguous model-confirmed input when an identical public resume is retained', async () => {
    const model = textModel('initial', 'final')
    const consumed = vi.fn()
    const rejected = vi.fn()
    const agent = new Agent({ model, printer: false })
    let afterInvocationCalls = 0
    let modelStageCalls = 0

    agent.addHook(AfterInvocationEvent, (event) => {
      afterInvocationCalls += 1
      if (afterInvocationCalls !== 1) return
      continuationManager.add(event, {
        args: 'duplicate',
        onConsumed: consumed,
        onRejected: rejected,
      })
      event.resume = 'duplicate'
    })
    agent.addMiddleware(InvokeModelStage.Input, async (context) => {
      modelStageCalls += 1
      if (modelStageCalls !== 2) return context
      const messages = context.messages.slice(0, -1)
      const merged = context.messages.at(-1)!
      messages.push(new Message({ role: merged.role, content: [merged.content.at(-1)!] }))
      return { ...context, messages }
    })

    await agent.invoke('start')

    expect(consumed).not.toHaveBeenCalled()
    expect(rejected).toHaveBeenCalledTimes(1)
    expect(agent.messages.map(textOf)).toEqual(['start', 'initial', 'duplicate', 'final'])
  })

  it('does not turn consumption callback failures into model failures', async () => {
    const model = textModel('initial', 'unrecorded')
    const laterConsumed = vi.fn()
    const warn = vi.spyOn(logger, 'warn').mockImplementation(() => {})
    const agent = new Agent({ model, printer: false })

    continueAfterInvocation(
      agent,
      {
        args: 'first',
        onConsumed: () => {
          throw new Error('consumption callback failed')
        },
      },
      { args: 'second', onConsumed: laterConsumed }
    )

    const result = await agent.invoke('start')

    expect(result.stopReason).toBe('endTurn')
    expect(laterConsumed).toHaveBeenCalledTimes(1)
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('continuation consumption callback failed'))
    warn.mockRestore()
  })

  it('redacts every history message represented by a continuation request', async () => {
    const model = textModel('blocked')
    const originalStream = model.stream.bind(model)
    vi.spyOn(model, 'stream').mockImplementation(async function* (messages, options) {
      yield* originalStream(messages, options)
      yield {
        type: 'modelRedactionEvent',
        inputRedaction: { replaceContent: '[redacted]' },
      }
    })
    const agent = new Agent({ model, printer: false })
    agent.addHook(BeforeModelCallEvent, (event) => {
      continuationManager.add(event, { args: 'internal secret' })
    })

    await agent.invoke('original secret')

    expect(agent.messages.map(textOf)).toEqual(['[redacted]', 'blocked'])
  })

  it.each(['interrupt', 'cancelled', 'checkpoint'] as const)(
    'rejects pending input instead of following a terminal %s result',
    async (stopReason) => {
      const model = textModel('unreachable')
      const rejected = vi.fn()
      const agent = new Agent({ model, printer: false })

      // eslint-disable-next-line require-yield
      agent.addMiddleware(AgentStreamStage, async function* (context) {
        return streamResult(context, stopReason, stopReason)
      })
      agent.addHook(AfterInvocationEvent, (event) => {
        continuationManager.add(event, { args: 'continue', onRejected: rejected })
      })

      const result = await agent.invoke('start')

      expect(result.stopReason).toBe(stopReason)
      expect(rejected).toHaveBeenCalledWith(
        expect.objectContaining({ message: `Continuation rejected by ${stopReason}` })
      )
    }
  )

  it('rejects an invalid contribution without discarding valid siblings or the completed result', async () => {
    const model = textModel('initial', 'final')
    const requests = captureRequests(model)
    const invalidRejected = vi.fn()
    const orphanRejected = vi.fn()
    const validConsumed = vi.fn()
    const [toolUse, toolResult] = toolExchange('delivery-1', 'complete')
    const agent = new Agent({ model, printer: false })

    continueAfterInvocation(
      agent,
      { args: [], onRejected: invalidRejected },
      {
        args: [
          new Message({
            role: 'user',
            content: [
              new ToolResultBlock({
                toolUseId: 'missing',
                status: 'success',
                content: [new TextBlock('orphan')],
              }),
            ],
          }),
        ],
        onRejected: orphanRejected,
      },
      { args: [toolUse, toolResult], onConsumed: validConsumed }
    )

    const result = await agent.invoke('start')

    expect(result.stopReason).toBe('endTurn')
    expect(invalidRejected).toHaveBeenCalledWith(
      expect.objectContaining({ message: 'Continuation input must contain a complete message sequence' })
    )
    expect(orphanRejected).toHaveBeenCalledTimes(1)
    expect(validConsumed).toHaveBeenCalledTimes(1)
    expect(requests[1]?.slice(-2).map((message) => message.trackingId)).toEqual([
      toolUse.trackingId,
      toolResult.trackingId,
    ])
    expect(agent.messages).toEqual([expect.any(Message), expect.any(Message), toolUse, toolResult, expect.any(Message)])
  })

  it('rejects prepared contributions when public resume normalization fails', async () => {
    const model = textModel('initial')
    const rejected = vi.fn()
    const agent = new Agent({ model, printer: false })
    let armed = false

    agent.addHook(AfterInvocationEvent, (event) => {
      if (armed) return
      armed = true
      continuationManager.add(event, { args: 'internal', onRejected: rejected })
      event.resume = [{ unsupported: true }] as unknown as InvokeArgs
    })

    await expect(agent.invoke('start')).rejects.toThrow()

    expect(rejected).toHaveBeenCalledTimes(1)
    expect(agent.messages.map(textOf)).toEqual(['start', 'initial'])
  })

  it('rejects active and newly registered input when BeforeInvocationEvent cancels the pass', async () => {
    const model = textModel('initial')
    const activeRejected = vi.fn()
    const lateRejected = vi.fn()
    const agent = new Agent({ model, printer: false })
    let beforeCalls = 0
    let afterCalls = 0

    agent.addHook(BeforeInvocationEvent, (event) => {
      beforeCalls += 1
      if (beforeCalls === 2) event.cancel = 'invocation denied'
    })
    agent.addHook(AfterInvocationEvent, (event) => {
      afterCalls += 1
      if (afterCalls === 1) {
        continuationManager.add(event, { args: 'active', onRejected: activeRejected })
      } else if (afterCalls === 2) {
        continuationManager.add(event, { args: 'late', onRejected: lateRejected })
      }
    })

    const result = await agent.invoke('start')

    expect(result.stopReason).toBe('endTurn')
    expect(activeRejected).toHaveBeenCalledWith(
      expect.objectContaining({ message: 'Continuation invocation cancelled by BeforeInvocationEvent' })
    )
    expect(lateRejected).toHaveBeenCalledWith(
      expect.objectContaining({ message: 'Invocation cancelled by BeforeInvocationEvent' })
    )
    expect(agent.messages.map(textOf)).toEqual(['start', 'initial', 'invocation denied'])
  })

  it.each([
    [{ turns: 3 }, 'limitTurns'],
    [{ outputTokens: 3 }, 'limitOutputTokens'],
  ] as const)('shares %s across middleware and continuation passes', async (limits, stopReason) => {
    const usage = { inputTokens: 2, outputTokens: 1, totalTokens: 3 }
    const model = new MockMessageModel()
      .addTurn({ type: 'textBlock', text: 'initial' }, { usage })
      .addTurn({ type: 'textBlock', text: 'second' }, { usage })
      .addTurn({ type: 'textBlock', text: 'third' }, { usage })
      .addTurn({ type: 'textBlock', text: 'unreachable' }, { usage })
    const consumed = vi.fn()
    const rejected = vi.fn()
    const agent = new Agent({ model, printer: false })
    let streamCalls = 0
    agent.addMiddleware(AgentStreamStage, async function* (context, next) {
      streamCalls += 1
      if (streamCalls !== 1) return yield* next(context)
      yield* next(context)
      return yield* next(context)
    })
    agent.addHook(AfterInvocationEvent, (event) => {
      continuationManager.add(event, { args: 'continue', onConsumed: consumed, onRejected: rejected })
    })

    const result = await agent.invoke('start', { limits })

    expect(result.stopReason).toBe('endTurn')
    expect(model.callCount).toBe(3)
    expect(consumed).toHaveBeenCalledTimes(1)
    expect(rejected).toHaveBeenCalledWith(
      expect.objectContaining({ message: `Continuation rejected by ${stopReason}` })
    )
    expect(agent.messages.map(textOf)).toEqual(['start', 'initial', 'start', 'second', 'continue', 'third'])
  })

  it('uses fresh limits and cancellation for a public resume mixed with internal input', async () => {
    const model = textModel('initial', 'final')
    const consumed = vi.fn()
    const rejected = vi.fn()
    const agent = new Agent({ model, printer: false })
    let armed = false

    agent.addHook(AfterInvocationEvent, (event) => {
      if (armed) return
      armed = true
      agent.cancel()
      continuationManager.add(event, { args: 'internal', onConsumed: consumed, onRejected: rejected })
      event.resume = 'public'
    })

    const result = await agent.invoke('start', { limits: { turns: 1 } })

    expect(result.stopReason).toBe('endTurn')
    expect(model.callCount).toBe(2)
    expect(consumed).toHaveBeenCalledTimes(1)
    expect(rejected).not.toHaveBeenCalled()
    expect(agent.messages.map(textOf)).toEqual(['start', 'initial', 'internal', 'public', 'final'])
  })

  it('preserves cancellation across continuation passes', async () => {
    const model = textModel('initial', 'unreachable')
    const consumed = vi.fn()
    const rejected = vi.fn()
    const agent = new Agent({ model, printer: false })
    continueAfterInvocation(agent, { args: 'continue', onConsumed: consumed, onRejected: rejected })
    agent.addHook(AfterInvocationEvent, () => {
      agent.cancel()
    })

    const result = await agent.invoke('start')

    expect(result.stopReason).toBe('cancelled')
    expect(model.callCount).toBe(1)
    expect(consumed).not.toHaveBeenCalled()
    expect(rejected).toHaveBeenCalledTimes(1)
    expect(agent.messages.map(textOf)).toEqual(['start', 'initial'])
  })
})
