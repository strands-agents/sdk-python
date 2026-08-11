import { describe, expect, it } from 'vitest'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { collectGenerator } from '../../__fixtures__/model-test-helpers.js'
import { AfterInvocationEvent, AgentResultEvent, MessageAddedEvent } from '../../hooks/events.js'
import { Message, TextBlock, ToolResultBlock, ToolUseBlock } from '../../types/messages.js'
import { Agent } from '../agent.js'

describe('Agent continuations', () => {
  it('orders independent intents and emits one final result', async () => {
    const model = new MockMessageModel()
      .addTurn({ type: 'textBlock', text: 'initial' })
      .addTurn({ type: 'textBlock', text: 'final' })
    const agent = new Agent({ model, printer: false })
    const lifecycle: string[] = []
    let armed = false
    let resultEvents = 0

    const deferredAssistant = new Message({
      role: 'assistant',
      content: [
        new ToolUseBlock({
          name: 'deferred_result',
          toolUseId: 'delivery-1',
          input: { taskId: 'task-1' },
        }),
      ],
    })
    const deferredResult = new Message({
      role: 'user',
      content: [
        new ToolResultBlock({
          toolUseId: 'delivery-1',
          status: 'success',
          content: [new TextBlock('background result')],
        }),
      ],
    })

    agent.addHook(AfterInvocationEvent, (event) => {
      if (armed) return
      armed = true
      event._continueWith({
        phase: 'guidance',
        args: 'review the completed work',
        onAccepted: () => {
          lifecycle.push('accepted:guidance')
        },
        onCommitted: () => {
          lifecycle.push('committed:guidance')
        },
      })
      event._continueWith({
        phase: 'deferredResult',
        args: [deferredAssistant, deferredResult],
        onAccepted: () => {
          lifecycle.push('accepted:deferred')
        },
        onCommitted: () => {
          lifecycle.push('committed:deferred')
        },
      })
    })
    agent.addHook(MessageAddedEvent, (event) => {
      if (event.message === deferredAssistant) lifecycle.push('message:deferred')
    })
    agent.addHook(AfterInvocationEvent, () => {
      if (agent.messages.includes(deferredAssistant)) lifecycle.push('after:continuation')
    })
    agent.addHook(AgentResultEvent, () => {
      resultEvents += 1
    })

    const { result } = await collectGenerator(agent.stream('start'))

    expect(result.lastMessage).toEqual(
      expect.objectContaining({
        role: 'assistant',
        content: [expect.objectContaining({ type: 'textBlock', text: 'final' })],
      })
    )
    expect(resultEvents).toBe(1)
    expect(lifecycle).toEqual([
      'accepted:deferred',
      'accepted:guidance',
      'message:deferred',
      'after:continuation',
      'committed:deferred',
      'committed:guidance',
    ])
    expect(
      agent.messages.map((message) => ({
        role: message.role,
        contentTypes: message.content.map((block) => block.type),
      }))
    ).toEqual([
      { role: 'user', contentTypes: ['textBlock'] },
      { role: 'assistant', contentTypes: ['textBlock'] },
      { role: 'assistant', contentTypes: ['toolUseBlock'] },
      { role: 'user', contentTypes: ['toolResultBlock'] },
      { role: 'user', contentTypes: ['textBlock'] },
      { role: 'assistant', contentTypes: ['textBlock'] },
    ])
  })

  it('commits the complete message batch before message hooks run', async () => {
    const agent = new Agent({
      model: new MockMessageModel()
        .addTurn({ type: 'textBlock', text: 'initial' })
        .addTurn({ type: 'textBlock', text: 'unreachable' }),
      printer: false,
    })
    const appended: string[] = []
    const rejected: string[] = []
    let armed = false
    const first = new Message({ role: 'user', content: [new TextBlock('first')] })
    const second = new Message({ role: 'user', content: [new TextBlock('second')] })

    agent.addHook(AfterInvocationEvent, (event) => {
      if (armed) return
      armed = true
      event._continueWith({
        phase: 'deferredResult',
        args: [first, second],
        onCommitted: () => {
          appended.push('appended')
        },
        onRejected: () => {
          rejected.push('rejected')
        },
      })
    })
    agent.addHook(MessageAddedEvent, (event) => {
      if (event.message === first) {
        expect(agent.messages.slice(-3, -1)).toEqual([first, second])
        throw new Error('message hook failed')
      }
    })

    await expect(agent.invoke('start')).rejects.toThrow('message hook failed')
    expect(appended).toEqual([])
    expect(rejected).toEqual(['rejected'])
    expect(agent.messages.slice(-3, -1)).toEqual([first, second])
  })

  it('does not integrate a continuation when an invocation hook fails', async () => {
    const agent = new Agent({
      model: new MockMessageModel()
        .addTurn({ type: 'textBlock', text: 'initial' })
        .addTurn({ type: 'textBlock', text: 'final' }),
      printer: false,
    })
    const deferred = new Message({ role: 'user', content: [new TextBlock('deferred')] })
    const lifecycle: string[] = []
    let armed = false

    agent.addHook(AfterInvocationEvent, (event) => {
      if (armed) return
      armed = true
      event._continueWith({
        phase: 'deferredResult',
        args: [deferred],
        onCommitted: () => {
          lifecycle.push('integrated')
        },
        onRejected: () => {
          lifecycle.push('rejected')
        },
      })
    })
    agent.addHook(AfterInvocationEvent, () => {
      if (agent.messages.includes(deferred)) throw new Error('snapshot write failed')
    })

    await expect(agent.invoke('start')).rejects.toThrow('snapshot write failed')
    expect(lifecycle).toEqual(['rejected'])
  })
})
