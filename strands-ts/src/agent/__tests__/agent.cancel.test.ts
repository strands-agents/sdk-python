import { describe, expect, it } from 'vitest'
import { Agent } from '../agent.js'
import { AfterInvocationEvent, AfterModelCallEvent, BeforeModelCallEvent } from '../../hooks/index.js'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { createMockTool } from '../../__fixtures__/tool-helpers.js'
import { TextBlock, ToolResultBlock } from '../../types/messages.js'
import { tool } from '../../tools/tool-factory.js'

describe('Agent Cancellation', () => {
  describe('cancel() when idle', () => {
    it('is a no-op and cancelSignal is not aborted', () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Hello' })
      const agent = new Agent({ model })

      expect(agent.cancelSignal.aborted).toBe(false)
      agent.cancel() // Should not throw
      expect(agent.cancelSignal.aborted).toBe(false)
      expect(agent.cancelSignal.aborted).toBe(false)
    })
  })

  describe('cancel at top of loop (checkpoint A)', () => {
    it('cancels immediately with already-aborted signal', async () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Hello' })
      const agent = new Agent({ model, printer: false })

      const controller = new AbortController()
      controller.abort()

      const result = await agent.invoke('Hi', { cancelSignal: controller.signal })

      expect(result.stopReason).toBe('cancelled')
      expect(result.lastMessage.content[0]).toEqual(new TextBlock('Cancelled by user'))
      // User message is not appended — cancel fires before message append in the loop
      expect(agent.messages).toHaveLength(1)
      expect(agent.messages[0]!.role).toBe('assistant')
    })

    it('cancels at top of second cycle when tool calls cancel()', async () => {
      const executedTools: string[] = []

      let agent: Agent
      const tool = createMockTool('cancelTool', () => {
        executedTools.push('cancelTool')
        agent.cancel()
        return new ToolResultBlock({
          toolUseId: 'tool-1',
          status: 'success',
          content: [new TextBlock('Done')],
        })
      })

      const model = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'cancelTool', toolUseId: 'tool-1', input: {} })
        .addTurn({ type: 'textBlock', text: 'Should not reach' })

      agent = new Agent({ model, tools: [tool], printer: false })
      const result = await agent.invoke('Go')

      expect(result.stopReason).toBe('cancelled')
      expect(executedTools).toEqual(['cancelTool'])
      // messages: user, assistant(toolUse), user(toolResult), assistant(synthetic cancel)
      expect(agent.messages).toHaveLength(4)
      expect(agent.messages[3]!.content[0]).toEqual(new TextBlock('Cancelled by user'))
    })
  })

  describe('cancel during model streaming (checkpoint B)', () => {
    it('cancels when signal is aborted before model processes events', async () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Hello' })
      const agent = new Agent({ model, printer: false })

      agent.addHook(BeforeModelCallEvent, () => {
        agent.cancel()
      })

      const result = await agent.invoke('Hi')

      expect(result.stopReason).toBe('cancelled')
      expect(result.lastMessage.content[0]).toEqual(new TextBlock('Cancelled by user'))
    })
  })

  describe('cancel before tool execution (checkpoint C)', () => {
    it('creates error results for all pending tools without executing them', async () => {
      let toolExecuted = false
      const tool = createMockTool('myTool', () => {
        toolExecuted = true
        return new ToolResultBlock({
          toolUseId: 'tool-1',
          status: 'success',
          content: [new TextBlock('Success')],
        })
      })

      const model = new MockMessageModel().addTurn({
        type: 'toolUseBlock',
        name: 'myTool',
        toolUseId: 'tool-1',
        input: {},
      })

      const agent = new Agent({ model, tools: [tool], printer: false })
      agent.addHook(AfterModelCallEvent, (event) => {
        if (event.stopData?.stopReason === 'toolUse') {
          agent.cancel()
        }
      })

      const result = await agent.invoke('Do it')

      expect(result.stopReason).toBe('cancelled')
      expect(toolExecuted).toBe(false)

      // Messages: user, assistant(toolUse), user(cancelled toolResult)
      expect(agent.messages).toHaveLength(3)
      const toolResultMsg = agent.messages[2]!
      expect(toolResultMsg.content[0]).toEqual(
        new ToolResultBlock({
          toolUseId: 'tool-1',
          status: 'error',
          content: [new TextBlock('Tool execution cancelled')],
        })
      )
    })

    it('creates error results for multiple pending tools', async () => {
      const tool1 = createMockTool('tool1', () => {
        return new ToolResultBlock({ toolUseId: 't1', status: 'success', content: [new TextBlock('R1')] })
      })
      const tool2 = createMockTool('tool2', () => {
        return new ToolResultBlock({ toolUseId: 't2', status: 'success', content: [new TextBlock('R2')] })
      })

      const model = new MockMessageModel().addTurn([
        { type: 'toolUseBlock', name: 'tool1', toolUseId: 't1', input: {} },
        { type: 'toolUseBlock', name: 'tool2', toolUseId: 't2', input: {} },
      ])

      const agent = new Agent({ model, tools: [tool1, tool2], printer: false })
      agent.addHook(AfterModelCallEvent, (event) => {
        if (event.stopData?.stopReason === 'toolUse') {
          agent.cancel()
        }
      })

      const result = await agent.invoke('Do both')

      expect(result.stopReason).toBe('cancelled')

      const toolResultMsg = agent.messages[2]!
      expect(toolResultMsg.content).toHaveLength(2)
      expect(toolResultMsg.content[0]).toEqual(
        new ToolResultBlock({ toolUseId: 't1', status: 'error', content: [new TextBlock('Tool execution cancelled')] })
      )
      expect(toolResultMsg.content[1]).toEqual(
        new ToolResultBlock({ toolUseId: 't2', status: 'error', content: [new TextBlock('Tool execution cancelled')] })
      )
    })
  })

  describe('cancel between sequential tool executions', () => {
    it('skips remaining tools after first tool calls cancel()', async () => {
      const executedTools: string[] = []

      let agent: Agent
      const tool1 = createMockTool('firstTool', () => {
        executedTools.push('firstTool')
        agent.cancel()
        return new ToolResultBlock({ toolUseId: 't1', status: 'success', content: [new TextBlock('Done')] })
      })
      const tool2 = createMockTool('secondTool', () => {
        executedTools.push('secondTool')
        return new ToolResultBlock({ toolUseId: 't2', status: 'success', content: [new TextBlock('Done')] })
      })

      const model = new MockMessageModel()
        .addTurn([
          { type: 'toolUseBlock', name: 'firstTool', toolUseId: 't1', input: {} },
          { type: 'toolUseBlock', name: 'secondTool', toolUseId: 't2', input: {} },
        ])
        .addTurn({ type: 'textBlock', text: 'Should not reach' })

      agent = new Agent({ model, tools: [tool1, tool2], toolExecutor: 'sequential', printer: false })
      const result = await agent.invoke('Go')

      expect(result.stopReason).toBe('cancelled')
      expect(executedTools).toEqual(['firstTool'])

      // First tool succeeded, second was cancelled
      // messages: user, assistant(toolUse), user(toolResults), assistant(synthetic cancel)
      expect(agent.messages).toHaveLength(4)
      const toolResultMsg = agent.messages[2]!
      expect(toolResultMsg.content[0]).toEqual(
        new ToolResultBlock({ toolUseId: 't1', status: 'success', content: [new TextBlock('Done')] })
      )
      expect(toolResultMsg.content[1]).toEqual(
        new ToolResultBlock({ toolUseId: 't2', status: 'error', content: [new TextBlock('Tool execution cancelled')] })
      )
    })
  })

  describe('InvokeOptions.cancelSignal', () => {
    it('cancels via external AbortSignal', async () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Hello' })
      const agent = new Agent({ model, printer: false })

      const controller = new AbortController()
      agent.addHook(BeforeModelCallEvent, () => {
        controller.abort()
      })

      const result = await agent.invoke('Hi', { cancelSignal: controller.signal })

      expect(result.stopReason).toBe('cancelled')
    })
  })

  describe('agent reuse after cancel', () => {
    it('allows a second invocation after cancellation', async () => {
      const model = new MockMessageModel()
        .addTurn({ type: 'textBlock', text: 'First' })
        .addTurn({ type: 'textBlock', text: 'Second' })

      const agent = new Agent({ model, printer: false })

      let hookCallCount = 0
      agent.addHook(BeforeModelCallEvent, () => {
        hookCallCount++
        if (hookCallCount === 1) {
          agent.cancel()
        }
      })

      // First invocation: cancelled
      const result1 = await agent.invoke('Hello')
      expect(result1.stopReason).toBe('cancelled')

      // Second invocation: succeeds normally
      const result2 = await agent.invoke('Hello again')
      expect(result2.stopReason).toBe('endTurn')
      expect(result2.lastMessage.content[0]).toEqual(new TextBlock('Second'))
    })
  })

  describe('cancel via stream break (for-await + break)', () => {
    it('appends assistant message when stream is broken out of after cancel()', async () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'A long story...' })
      const agent = new Agent({ model, printer: false })

      for await (const event of agent.stream('Write a story')) {
        if (event.type === 'modelStreamUpdateEvent') {
          agent.cancel()
          break
        }
      }

      const lastMessage = agent.messages[agent.messages.length - 1]!
      expect(lastMessage.role).toBe('assistant')
      expect(lastMessage.content[0]).toEqual(new TextBlock('Cancelled by user'))
    })

    it('allows reuse after cancellation via stream break', async () => {
      const model = new MockMessageModel()
        .addTurn({ type: 'textBlock', text: 'A long story...' })
        .addTurn({ type: 'textBlock', text: 'pineapple' })

      const agent = new Agent({ model, printer: false })

      // First invocation: cancel during streaming via break
      for await (const event of agent.stream('Write a story')) {
        if (event.type === 'modelStreamUpdateEvent') {
          agent.cancel()
          break
        }
      }

      // Second invocation: should succeed normally
      const result = await agent.invoke('Say the word "pineapple"')
      expect(result.stopReason).toBe('endTurn')
      expect(result.lastMessage.content[0]).toEqual(new TextBlock('pineapple'))
    })
  })

  describe('AfterInvocationEvent', () => {
    it('still fires when invocation is cancelled', async () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Hello' })
      const agent = new Agent({ model, printer: false })

      let afterInvocationFired = false
      agent.addHook(AfterInvocationEvent, () => {
        afterInvocationFired = true
      })

      agent.addHook(BeforeModelCallEvent, () => {
        agent.cancel()
      })

      await agent.invoke('Hi')
      expect(afterInvocationFired).toBe(true)
    })
  })

  describe('messages state invariants', () => {
    it('has no orphaned toolUse blocks after cancel during streaming', async () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Hello' })
      const agent = new Agent({ model, printer: false })

      agent.addHook(BeforeModelCallEvent, () => {
        agent.cancel()
      })

      await agent.invoke('Hi')

      // Every assistant message with toolUse blocks must be followed by a user message with matching toolResults
      for (let i = 0; i < agent.messages.length; i++) {
        const msg = agent.messages[i]!
        if (msg.role === 'assistant') {
          const toolUseBlocks = msg.content.filter((b) => b.type === 'toolUseBlock')
          if (toolUseBlocks.length > 0) {
            const nextMsg = agent.messages[i + 1]
            expect(nextMsg).toBeDefined()
            expect(nextMsg!.role).toBe('user')
            const toolResultBlocks = nextMsg!.content.filter((b) => b.type === 'toolResultBlock')
            expect(toolResultBlocks).toHaveLength(toolUseBlocks.length)
          }
        }
      }
    })

    it('has no orphaned toolUse blocks after cancel before tools', async () => {
      const tool = createMockTool('myTool', () => {
        return new ToolResultBlock({ toolUseId: 'tool-1', status: 'success', content: [new TextBlock('Done')] })
      })

      const model = new MockMessageModel().addTurn({
        type: 'toolUseBlock',
        name: 'myTool',
        toolUseId: 'tool-1',
        input: {},
      })

      const agent = new Agent({ model, tools: [tool], printer: false })
      agent.addHook(AfterModelCallEvent, (event) => {
        if (event.stopData?.stopReason === 'toolUse') {
          agent.cancel()
        }
      })

      await agent.invoke('Do it')

      // Verify every toolUse has a matching toolResult
      for (let i = 0; i < agent.messages.length; i++) {
        const msg = agent.messages[i]!
        if (msg.role === 'assistant') {
          const toolUseBlocks = msg.content.filter((b) => b.type === 'toolUseBlock')
          if (toolUseBlocks.length > 0) {
            const nextMsg = agent.messages[i + 1]
            expect(nextMsg).toBeDefined()
            expect(nextMsg!.role).toBe('user')
            const toolResultBlocks = nextMsg!.content.filter((b) => b.type === 'toolResultBlock')
            expect(toolResultBlocks).toHaveLength(toolUseBlocks.length)
          }
        }
      }
    })
  })

  describe('tool-level cancellation cooperation', () => {
    it('exposes cancelSignal to tools via context.agent', async () => {
      let signalSeen: AbortSignal | undefined

      const signalTool = tool({
        name: 'signalTool',
        description: 'Tool that reads the cancellation signal',
        callback: (_input, context) => {
          signalSeen = context?.agent.cancelSignal
          return 'done'
        },
      })

      const model = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'signalTool', toolUseId: 't1', input: {} })
        .addTurn({ type: 'textBlock', text: 'Done' })

      const agent = new Agent({ model, tools: [signalTool], printer: false })
      await agent.invoke('Go')

      expect(signalSeen).toBeInstanceOf(AbortSignal)
      expect(signalSeen!.aborted).toBe(false)
    })

    it('signal is aborted when tool checks it after cancel()', async () => {
      let signalAborted: boolean | undefined

      let agent: Agent
      const checkTool = tool({
        name: 'checkTool',
        description: 'Tool that cancels then checks the signal',
        callback: (_input, context) => {
          agent.cancel()
          signalAborted = context?.agent.cancelSignal.aborted
          return 'done'
        },
      })

      const model = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'checkTool', toolUseId: 't1', input: {} })
        .addTurn({ type: 'textBlock', text: 'Should not reach' })

      agent = new Agent({ model, tools: [checkTool], printer: false })
      const result = await agent.invoke('Go')

      expect(signalAborted).toBe(true)
      expect(result.stopReason).toBe('cancelled')
    })
  })
})
