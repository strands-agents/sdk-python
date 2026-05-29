import { describe, expect, it } from 'vitest'
import { Agent } from '../agent.js'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { createMockTool } from '../../__fixtures__/tool-helpers.js'
import { ToolResultBlock } from '../../types/messages.js'
import { AfterToolCallEvent, BeforeToolCallEvent, BeforeToolsEvent, InterruptEvent } from '../../hooks/events.js'
import { FunctionTool } from '../../tools/function-tool.js'
import { InterruptResponseContent } from '../../types/interrupt.js'
import type { InterruptState, PendingToolExecution } from '../../interrupt.js'

/** Access the agent's internal interrupt state for test assertions. */
function getPendingToolExecution(agent: Agent): PendingToolExecution | undefined {
  // yes it's dirty, but we don't want to expose this publicly
  return (agent as unknown as { _interruptState: InterruptState })._interruptState.pendingToolExecution
}

describe('Agent interrupt system', () => {
  describe('interrupt from tool callback', () => {
    it('returns stopReason interrupt when tool calls interrupt()', async () => {
      // Model returns tool use first, then text block (following standard test pattern)
      const model = new MockMessageModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'confirmTool',
          toolUseId: 'tool-1',
          input: {},
        })
        .addTurn({ type: 'textBlock', text: 'Should not reach this' })

      const tool = createMockTool('confirmTool', (context) => {
        context.interrupt({ name: 'confirm', reason: 'Please confirm' })
      })

      const agent = new Agent({ model, tools: [tool], printer: false })
      const result = await agent.invoke('Test')

      expect(result).toMatchObject({
        stopReason: 'interrupt',
        interrupts: [{ name: 'confirm', reason: 'Please confirm' }],
      })
    })
  })

  describe('interrupt from BeforeToolCallEvent hook', () => {
    it('returns stopReason interrupt when hook calls interrupt()', async () => {
      // Model returns tool use first, then text block (following standard test pattern)
      const model = new MockMessageModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'testTool',
          toolUseId: 'tool-1',
          input: {},
        })
        .addTurn({ type: 'textBlock', text: 'Should not reach this' })

      const tool = createMockTool('testTool', () => 'Success')

      const agent = new Agent({ model, tools: [tool], printer: false })

      agent.addHook(BeforeToolCallEvent, (event) => {
        if (event.toolUse.name === 'testTool') {
          event.interrupt({ name: 'confirm_tool', reason: 'Confirm tool execution?' })
        }
      })

      const result = await agent.invoke('Test')

      expect(result).toMatchObject({
        stopReason: 'interrupt',
        interrupts: [{ name: 'confirm_tool', reason: 'Confirm tool execution?' }],
      })
    })

    it('stores pending state and resumes correctly after interrupt', async () => {
      const model = new MockMessageModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'deleteTool',
          toolUseId: 'tool-1',
          input: { key: 'X' },
        })
        .addTurn({ type: 'textBlock', text: 'Deleted' })

      let toolExecuted = false
      const tool = createMockTool('deleteTool', () => {
        toolExecuted = true
        return 'deleted'
      })

      const agent = new Agent({ model, tools: [tool], printer: false })

      agent.addHook(BeforeToolCallEvent, (event) => {
        if (event.toolUse.name === 'deleteTool') {
          const approval = event.interrupt<string>({ name: 'approve_delete', reason: 'Confirm delete?' })
          if (approval !== 'yes') {
            event.cancel = 'not approved'
          }
        }
      })

      // First invocation — hook interrupts before tool runs
      const interruptResult = await agent.invoke('Delete X')

      expect(interruptResult.stopReason).toBe('interrupt')
      expect(interruptResult.interrupts).toMatchObject([
        { id: expect.any(String), name: 'approve_delete', reason: 'Confirm delete?', source: 'hook' },
      ])
      expect(toolExecuted).toBe(false)
      expect(model.callCount).toBe(1)

      // Verify pending execution state was stored (the core of pgrayy's concern:
      // the InterruptError thrown back into the generator at `yield beforeToolCallEvent`
      // must propagate to executeTools' catch block which stores this state)
      const pendingExecution = getPendingToolExecution(agent)
      expect(pendingExecution).toEqual({
        assistantMessageData: {
          role: 'assistant',
          content: [{ toolUse: { name: 'deleteTool', toolUseId: 'tool-1', input: { key: 'X' } } }],
        },
        completedToolResults: {},
      })

      // Resume with approval — tool should now execute
      const finalResult = await agent.invoke([
        new InterruptResponseContent({
          interruptId: interruptResult.interrupts![0]!.id,
          response: 'yes',
        }),
      ])

      expect(finalResult.stopReason).toBe('endTurn')
      expect(toolExecuted).toBe(true)
      expect(model.callCount).toBe(2)
    })

    it('preserves completed tool results when interrupt fires on a later tool', async () => {
      // Tools A, B, C — hook interrupts on B's BeforeToolCallEvent
      // A should complete, B and C should not execute
      // On resume, A is skipped, B and C execute
      const model = new MockMessageModel()
        .addTurn([
          { type: 'toolUseBlock', name: 'toolA', toolUseId: 'tool-a', input: {} },
          { type: 'toolUseBlock', name: 'toolB', toolUseId: 'tool-b', input: {} },
          { type: 'toolUseBlock', name: 'toolC', toolUseId: 'tool-c', input: {} },
        ])
        .addTurn({ type: 'textBlock', text: 'All done' })

      const executionLog: string[] = []

      const toolA = createMockTool('toolA', () => {
        executionLog.push('A')
        return 'A result'
      })
      const toolB = createMockTool('toolB', () => {
        executionLog.push('B')
        return 'B result'
      })
      const toolC = createMockTool('toolC', () => {
        executionLog.push('C')
        return 'C result'
      })

      const agent = new Agent({ model, tools: [toolA, toolB, toolC], toolExecutor: 'sequential', printer: false })

      agent.addHook(BeforeToolCallEvent, (event) => {
        if (event.toolUse.name === 'toolB') {
          event.interrupt({ name: 'approve_b', reason: 'Approve B?' })
        }
      })

      const interruptResult = await agent.invoke('Run all')

      expect(interruptResult.stopReason).toBe('interrupt')
      expect(executionLog).toEqual(['A'])

      // Verify pending state includes A's completed result
      const pendingExecution = getPendingToolExecution(agent)
      expect(Object.keys(pendingExecution!.completedToolResults)).toEqual(['tool-a'])
      expect(pendingExecution!.completedToolResults['tool-a']!.toolResult.toolUseId).toBe('tool-a')

      // Resume — A should be skipped, B and C should execute
      const finalResult = await agent.invoke([
        new InterruptResponseContent({
          interruptId: interruptResult.interrupts![0]!.id,
          response: 'approved',
        }),
      ])

      expect(finalResult.stopReason).toBe('endTurn')
      expect(executionLog).toEqual(['A', 'B', 'C'])
      expect(model.callCount).toBe(2)
    })
  })

  describe('interrupt from BeforeToolsEvent hook', () => {
    it('returns stopReason interrupt when hook calls interrupt()', async () => {
      // Model returns tool use first, then text block (following standard test pattern)
      const model = new MockMessageModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'testTool',
          toolUseId: 'tool-1',
          input: {},
        })
        .addTurn({ type: 'textBlock', text: 'Should not reach this' })

      const tool = createMockTool('testTool', () => 'Success')

      const agent = new Agent({ model, tools: [tool], printer: false })

      agent.addHook(BeforeToolsEvent, (event) => {
        event.interrupt({ name: 'batch_approval', reason: 'Approve all tools?' })
      })

      const result = await agent.invoke('Test')

      expect(result).toMatchObject({
        stopReason: 'interrupt',
        interrupts: [{ name: 'batch_approval', reason: 'Approve all tools?' }],
      })
    })
  })

  describe('resume flow - interrupt → response → continue', () => {
    it('resumes tool callback execution without re-calling model', async () => {
      // Turn 0: Model returns tool use (will be interrupted)
      // Turn 1: Model returns final response (after tool completes on resume)
      // Note: Resume skips model call and uses stored message
      const model = new MockMessageModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'confirmTool',
          toolUseId: 'tool-1',
          input: { amount: 5000 },
        })
        .addTurn({ type: 'textBlock', text: 'Transfer completed' })

      let callCount = 0
      let receivedResponse: unknown
      const tool = new FunctionTool({
        name: 'confirmTool',
        description: 'Tool that requires confirmation',
        inputSchema: {
          type: 'object',
          properties: { amount: { type: 'number' } },
        },
        callback: (rawInput, context) => {
          callCount++
          const input = rawInput as { amount: number }
          const response = context.interrupt({
            name: 'confirm_transfer',
            reason: `Confirm transfer of $${input.amount}?`,
          })
          receivedResponse = response
          return (response as { approved: boolean })?.approved ? 'Transfer approved' : 'Transfer denied'
        },
      })

      const agent = new Agent({ model, tools: [tool], printer: false })

      // First invocation - triggers interrupt
      const interruptResult = await agent.invoke('Transfer $5000')

      expect(interruptResult).toMatchObject({
        stopReason: 'interrupt',
        interrupts: [{ name: 'confirm_transfer', reason: 'Confirm transfer of $5000?' }],
      })
      expect(callCount).toBe(1) // Tool was called once before interrupt
      expect(model.callCount).toBe(1) // Model was called once

      // Resume with user response
      const finalResult = await agent.invoke([
        new InterruptResponseContent({
          interruptId: interruptResult.interrupts![0]!.id,
          response: { approved: true },
        }),
      ])

      expect(finalResult.stopReason).toBe('endTurn')
      expect(receivedResponse).toEqual({ approved: true })
      expect(callCount).toBe(2)
      expect(model.callCount).toBe(2)

      // Verify tool result was added to messages
      const toolResultMessage = agent.messages.find(
        (m) => m.role === 'user' && m.content.some((b) => b.type === 'toolResultBlock')
      )
      expect(toolResultMessage).toBeDefined()
      const toolResult = toolResultMessage?.content.find((b) => b.type === 'toolResultBlock') as
        | ToolResultBlock
        | undefined
      expect(toolResult?.content[0]).toMatchObject({ type: 'textBlock', text: 'Transfer approved' })
    })

    it('skips already-completed tools when resuming from partial execution', async () => {
      // Scenario: Tools A, B, C where A & B succeed but C interrupts
      // On resume: A & B should NOT re-execute, only C should execute
      const model = new MockMessageModel()
        .addTurn([
          { type: 'toolUseBlock', name: 'toolA', toolUseId: 'tool-a', input: {} },
          { type: 'toolUseBlock', name: 'toolB', toolUseId: 'tool-b', input: {} },
          { type: 'toolUseBlock', name: 'toolC', toolUseId: 'tool-c', input: {} },
        ])
        .addTurn({ type: 'textBlock', text: 'All tools completed' })

      const executionLog: string[] = []

      const toolA = createMockTool('toolA', () => {
        executionLog.push('A')
        return 'A result'
      })

      const toolB = createMockTool('toolB', () => {
        executionLog.push('B')
        return 'B result'
      })

      const toolC = createMockTool('toolC', (context) => {
        const response = context.interrupt({
          name: 'confirm_c',
          reason: 'Confirm tool C?',
        })
        executionLog.push('C')
        return (response as { approved: boolean })?.approved ? 'C approved' : 'C denied'
      })

      const agent = new Agent({ model, tools: [toolA, toolB, toolC], printer: false })

      // First invocation - A & B execute, C interrupts
      const interruptResult = await agent.invoke('Run all tools')

      expect(interruptResult).toMatchObject({
        stopReason: 'interrupt',
        interrupts: [{ name: 'confirm_c', reason: 'Confirm tool C?' }],
      })
      expect(executionLog).toEqual(['A', 'B'])
      expect(model.callCount).toBe(1)

      // Resume with response for C
      const finalResult = await agent.invoke([
        new InterruptResponseContent({
          interruptId: interruptResult.interrupts![0]!.id,
          response: { approved: true },
        }),
      ])

      expect(finalResult.stopReason).toBe('endTurn')
      expect(executionLog).toEqual(['A', 'B', 'C'])
      expect(model.callCount).toBe(2)

      // Verify all tool results are present in messages
      const toolResultMessage = agent.messages.find(
        (m) => m.role === 'user' && m.content.filter((b) => b.type === 'toolResultBlock').length === 3
      )
      expect(toolResultMessage).toBeDefined()
    })

    it('throws TypeError when sending a new message while in interrupted state', async () => {
      const model = new MockMessageModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'confirmTool',
          toolUseId: 'tool-1',
          input: {},
        })
        .addTurn({ type: 'textBlock', text: 'Different response' })

      const tool = createMockTool('confirmTool', (context) => {
        context.interrupt({ name: 'confirm', reason: 'Confirm?' })
      })

      const agent = new Agent({ model, tools: [tool], printer: false })

      // First invocation - triggers interrupt
      const interruptResult = await agent.invoke('First message')
      expect(interruptResult).toMatchObject({ stopReason: 'interrupt' })

      // Sending a new message instead of interrupt responses should throw
      await expect(agent.invoke('Different question')).rejects.toThrow(TypeError)
      await expect(agent.invoke('Different question')).rejects.toThrow('Agent is in an interrupted state')
    })
  })

  describe('error handling', () => {
    it('throws error when interrupt() called on event with non-Agent implementation', async () => {
      const mockLocalAgent = { id: 'mock' } as unknown as Agent
      const event = new BeforeToolCallEvent({
        agent: mockLocalAgent,
        toolUse: { name: 'test', toolUseId: 'id', input: {} },
        tool: undefined,
        invocationState: {},
      })

      expect(() => {
        event.interrupt({ name: 'test', reason: 'test' })
      }).toThrow('Interrupt state not available')
    })

    it('throws TypeError when interrupt responses are mixed with other content blocks', async () => {
      const model = new MockMessageModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'confirmTool',
          toolUseId: 'tool-1',
          input: {},
        })
        .addTurn({ type: 'textBlock', text: 'Done' })

      const tool = createMockTool('confirmTool', (context) => {
        context.interrupt({ name: 'confirm', reason: 'Confirm?' })
      })

      const agent = new Agent({ model, tools: [tool], printer: false })

      // First invocation - triggers interrupt
      const interruptResult = await agent.invoke('Test')
      expect(interruptResult.stopReason).toBe('interrupt')

      // Resume with mixed content: interrupt response + text block
      await expect(
        agent.invoke([
          new InterruptResponseContent({
            interruptId: interruptResult.interrupts![0]!.id,
            response: 'yes',
          }),
          { type: 'textBlock', text: 'extra text' },
        ] as any)
      ).rejects.toThrow(TypeError)

      await expect(
        agent.invoke([
          new InterruptResponseContent({
            interruptId: interruptResult.interrupts![0]!.id,
            response: 'yes',
          }),
          { type: 'textBlock', text: 'extra text' },
        ] as any)
      ).rejects.toThrow('Must resume from interrupt with a list of interruptResponse content blocks only')
    })

    it('allows pure interrupt response arrays without error', async () => {
      const model = new MockMessageModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'confirmTool',
          toolUseId: 'tool-1',
          input: {},
        })
        .addTurn({ type: 'textBlock', text: 'Done' })

      const tool = createMockTool('confirmTool', (context) => {
        const response = context.interrupt({ name: 'confirm', reason: 'Confirm?' })
        return `Got: ${response}`
      })

      const agent = new Agent({ model, tools: [tool], printer: false })

      const interruptResult = await agent.invoke('Test')
      expect(interruptResult.stopReason).toBe('interrupt')

      // Resume with pure interrupt responses — should succeed
      const finalResult = await agent.invoke([
        new InterruptResponseContent({
          interruptId: interruptResult.interrupts![0]!.id,
          response: 'approved',
        }),
      ])

      expect(finalResult.stopReason).toBe('endTurn')
    })
  })

  describe('multiple hook interrupts', () => {
    it('collects interrupts from multiple BeforeToolCallEvent hooks', async () => {
      const model = new MockMessageModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'testTool',
          toolUseId: 'tool-1',
          input: {},
        })
        .addTurn({ type: 'textBlock', text: 'Should not reach this' })

      const tool = createMockTool('testTool', () => 'Success')

      const agent = new Agent({ model, tools: [tool], printer: false })

      agent.addHook(BeforeToolCallEvent, (event) => {
        event.interrupt({ name: 'security_check', reason: 'Security review required' })
      })
      agent.addHook(BeforeToolCallEvent, (event) => {
        event.interrupt({ name: 'budget_check', reason: 'Budget approval required' })
      })

      const result = await agent.invoke('Test')

      expect(result).toMatchObject({
        stopReason: 'interrupt',
        interrupts: expect.arrayContaining([
          expect.objectContaining({ name: 'security_check', reason: 'Security review required' }),
          expect.objectContaining({ name: 'budget_check', reason: 'Budget approval required' }),
        ]),
      })
    })

    it('collects interrupts from multiple BeforeToolsEvent hooks', async () => {
      const model = new MockMessageModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'testTool',
          toolUseId: 'tool-1',
          input: {},
        })
        .addTurn({ type: 'textBlock', text: 'Should not reach this' })

      const tool = createMockTool('testTool', () => 'Success')

      const agent = new Agent({ model, tools: [tool], printer: false })

      agent.addHook(BeforeToolsEvent, (event) => {
        event.interrupt({ name: 'approval_a', reason: 'First approval' })
      })
      agent.addHook(BeforeToolsEvent, (event) => {
        event.interrupt({ name: 'approval_b', reason: 'Second approval' })
      })

      const result = await agent.invoke('Test')

      expect(result).toMatchObject({
        stopReason: 'interrupt',
        interrupts: expect.arrayContaining([
          expect.objectContaining({ name: 'approval_a', reason: 'First approval' }),
          expect.objectContaining({ name: 'approval_b', reason: 'Second approval' }),
        ]),
      })
    })

    it('resumes correctly after multiple interrupts are answered', async () => {
      const model = new MockMessageModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'testTool',
          toolUseId: 'tool-1',
          input: {},
        })
        .addTurn({ type: 'textBlock', text: 'All approved' })

      let securityResponse: unknown
      let budgetResponse: unknown
      let hookCallCount = 0

      const tool = createMockTool('testTool', () => 'Success')

      const agent = new Agent({ model, tools: [tool], printer: false })

      agent.addHook(BeforeToolCallEvent, (event) => {
        hookCallCount++
        securityResponse = event.interrupt({ name: 'security_check', reason: 'Security review' })
      })
      agent.addHook(BeforeToolCallEvent, (event) => {
        hookCallCount++
        budgetResponse = event.interrupt({ name: 'budget_check', reason: 'Budget review' })
      })

      // First invocation — both hooks interrupt
      const interruptResult = await agent.invoke('Test')
      expect(interruptResult).toMatchObject({
        stopReason: 'interrupt',
        interrupts: expect.arrayContaining([
          expect.objectContaining({ name: 'security_check' }),
          expect.objectContaining({ name: 'budget_check' }),
        ]),
      })
      expect(interruptResult.interrupts).toHaveLength(2)
      expect(hookCallCount).toBe(2)
      expect(model.callCount).toBe(1)

      // Resume with responses for both interrupts
      const finalResult = await agent.invoke(
        interruptResult.interrupts!.map(
          (interrupt) =>
            new InterruptResponseContent({
              interruptId: interrupt.id,
              response: `approved:${interrupt.name}`,
            })
        )
      )

      expect(finalResult.stopReason).toBe('endTurn')
      expect(model.callCount).toBe(2)
      expect(securityResponse).toBe('approved:security_check')
      expect(budgetResponse).toBe('approved:budget_check')
    })
  })

  describe('multi-cycle interrupts', () => {
    it('interrupts again on cycle 2 after resuming from cycle 1 (BeforeToolsEvent)', async () => {
      // Cycle 1: model returns tool use → hook interrupts → user resumes → tool executes
      // Cycle 2: model returns another tool use → same hook should interrupt again
      const model = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'testTool', toolUseId: 'tool-1', input: {} })
        .addTurn({ type: 'toolUseBlock', name: 'testTool', toolUseId: 'tool-2', input: {} })
        .addTurn({ type: 'textBlock', text: 'Done' })

      const tool = createMockTool('testTool', () => 'ok')

      let interruptCount = 0
      const agent = new Agent({ model, tools: [tool], printer: false })

      agent.addHook(BeforeToolsEvent, (event) => {
        interruptCount++
        event.interrupt({ name: 'approval', reason: 'Approve?' })
      })

      // Cycle 1: interrupt
      const result1 = await agent.invoke('Go')
      expect(result1).toMatchObject({
        stopReason: 'interrupt',
        interrupts: [{ name: 'approval', reason: 'Approve?' }],
      })
      expect(interruptCount).toBe(1)

      // Resume cycle 1
      const result2 = await agent.invoke(
        result1.interrupts!.map(
          (i) =>
            new InterruptResponseContent({
              interruptId: i.id,
              response: 'yes',
            })
        )
      )

      // Cycle 2: should interrupt again, not silently pass through
      expect(result2).toMatchObject({ stopReason: 'interrupt' })
      expect(interruptCount).toBe(3)
    })
  })

  describe('event contract during interrupt', () => {
    it('does not fire AfterToolCallEvent when BeforeToolCallEvent interrupt triggers', async () => {
      const model = new MockMessageModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'testTool',
          toolUseId: 'tool-1',
          input: {},
        })
        .addTurn({ type: 'textBlock', text: 'Done' })

      const tool = createMockTool('testTool', () => 'Success')
      const agent = new Agent({ model, tools: [tool], printer: false })

      const firedEvents: string[] = []

      agent.addHook(BeforeToolCallEvent, (event) => {
        firedEvents.push('BeforeToolCallEvent')
        event.interrupt({ name: 'confirm', reason: 'Confirm?' })
      })
      agent.addHook(AfterToolCallEvent, () => {
        firedEvents.push('AfterToolCallEvent')
      })

      const result = await agent.invoke('Test')

      expect(result.stopReason).toBe('interrupt')
      expect(firedEvents).toContain('BeforeToolCallEvent')
      expect(firedEvents).not.toContain('AfterToolCallEvent')
    })

    it('does not fire AfterToolCallEvent when tool callback interrupts', async () => {
      const model = new MockMessageModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'confirmTool',
          toolUseId: 'tool-1',
          input: {},
        })
        .addTurn({ type: 'textBlock', text: 'Done' })

      const tool = createMockTool('confirmTool', (context) => {
        context.interrupt({ name: 'confirm', reason: 'Confirm?' })
      })
      const agent = new Agent({ model, tools: [tool], printer: false })

      const firedEvents: string[] = []

      agent.addHook(BeforeToolCallEvent, () => {
        firedEvents.push('BeforeToolCallEvent')
      })
      agent.addHook(AfterToolCallEvent, () => {
        firedEvents.push('AfterToolCallEvent')
      })

      const result = await agent.invoke('Test')

      expect(result.stopReason).toBe('interrupt')
      expect(firedEvents).toContain('BeforeToolCallEvent')
      expect(firedEvents).not.toContain('AfterToolCallEvent')
    })
  })

  describe('concurrent tool execution with interrupts', () => {
    it('allows in-flight tool to complete when sibling interrupts', async () => {
      // Use gated tools to prove concurrency: A completes AFTER B interrupts,
      // demonstrating that the executor waits for in-flight tools.
      const model = new MockMessageModel()
        .addTurn([
          { type: 'toolUseBlock', name: 'toolA', toolUseId: 'tool-a', input: {} },
          { type: 'toolUseBlock', name: 'toolB', toolUseId: 'tool-b', input: {} },
        ])
        .addTurn({ type: 'textBlock', text: 'Done' })

      let toolACompleted = false
      let toolAResolve: () => void
      const toolAGate = new Promise<void>((resolve) => (toolAResolve = resolve))
      let toolAStartedResolve: () => void
      const toolAStarted = new Promise<void>((resolve) => (toolAStartedResolve = resolve))

      const toolA = new FunctionTool({
        name: 'toolA',
        description: 'Gated tool A',
        inputSchema: { type: 'object', properties: {} },
        callback: async () => {
          toolAStartedResolve()
          await toolAGate
          toolACompleted = true
          return 'A done'
        },
      })

      const toolB = new FunctionTool({
        name: 'toolB',
        description: 'Interrupting tool B',
        inputSchema: { type: 'object', properties: {} },
        callback: (_input, context) => {
          // Interrupt immediately — A is still in-flight
          context!.interrupt({ name: 'confirm_b', reason: 'Approve B?' })
          return 'B done'
        },
      })

      const agent = new Agent({
        model,
        tools: [toolA, toolB],
        toolExecutor: 'concurrent',
        printer: false,
      })

      const invocation = agent.invoke('Go')

      // Wait for A to start (proves both tools launched concurrently)
      await toolAStarted

      // B has already interrupted, but A is still in-flight
      expect(toolACompleted).toBe(false)

      // Release A — executor should let it finish
      toolAResolve!()
      const result = await invocation

      expect(result.stopReason).toBe('interrupt')
      expect(toolACompleted).toBe(true)
      expect(result.interrupts).toMatchObject([
        { id: expect.any(String), name: 'confirm_b', reason: 'Approve B?', source: 'tool' },
      ])

      // Verify A's result was captured in pending state
      const pendingExecution = getPendingToolExecution(agent)
      expect(pendingExecution!.completedToolResults['tool-a']).toEqual({
        toolResult: { toolUseId: 'tool-a', status: 'success', content: [{ text: 'A done' }] },
      })
    })

    it('stores completed tool results and resumes only the interrupted tool', async () => {
      const model = new MockMessageModel()
        .addTurn([
          { type: 'toolUseBlock', name: 'toolA', toolUseId: 'tool-a', input: {} },
          { type: 'toolUseBlock', name: 'toolB', toolUseId: 'tool-b', input: {} },
        ])
        .addTurn({ type: 'textBlock', text: 'Done' })

      let toolAResolve: () => void
      const toolAGate = new Promise<void>((resolve) => (toolAResolve = resolve))
      const executionLog: string[] = []

      const toolA = new FunctionTool({
        name: 'toolA',
        description: 'Gated tool A',
        inputSchema: { type: 'object', properties: {} },
        callback: async () => {
          executionLog.push('A')
          await toolAGate
          return 'A result'
        },
      })

      const toolB = new FunctionTool({
        name: 'toolB',
        description: 'Interrupting tool B',
        inputSchema: { type: 'object', properties: {} },
        callback: (_input, context) => {
          executionLog.push('B')
          const response = context!.interrupt<string>({ name: 'confirm_b', reason: 'Approve?' })
          return `B: ${response}`
        },
      })

      const agent = new Agent({
        model,
        tools: [toolA, toolB],
        toolExecutor: 'concurrent',
        printer: false,
      })

      // Release A immediately so it completes
      toolAResolve!()
      const interruptResult = await agent.invoke('Go')

      expect(interruptResult.stopReason).toBe('interrupt')
      expect(executionLog).toEqual(['A', 'B'])

      // Verify pending state has A's result
      const pendingExecution = getPendingToolExecution(agent)
      expect(Object.keys(pendingExecution!.completedToolResults)).toEqual(['tool-a'])

      // Resume — only B should re-execute
      executionLog.length = 0
      const finalResult = await agent.invoke([
        {
          interruptResponse: {
            interruptId: interruptResult.interrupts![0]!.id,
            response: 'approved',
          },
        },
      ])

      expect(finalResult.stopReason).toBe('endTurn')
      expect(executionLog).toEqual(['B'])
    })

    it('handles BeforeToolCallEvent interrupt in concurrent mode', async () => {
      const model = new MockMessageModel()
        .addTurn([
          { type: 'toolUseBlock', name: 'toolA', toolUseId: 'tool-a', input: {} },
          { type: 'toolUseBlock', name: 'toolB', toolUseId: 'tool-b', input: {} },
        ])
        .addTurn({ type: 'textBlock', text: 'Done' })

      const executionLog: string[] = []

      const toolA = new FunctionTool({
        name: 'toolA',
        description: 'Tool A',
        inputSchema: { type: 'object', properties: {} },
        callback: async () => {
          executionLog.push('A')
          return 'A result'
        },
      })
      const toolB = new FunctionTool({
        name: 'toolB',
        description: 'Tool B',
        inputSchema: { type: 'object', properties: {} },
        callback: async () => {
          executionLog.push('B')
          return 'B result'
        },
      })

      const agent = new Agent({
        model,
        tools: [toolA, toolB],
        toolExecutor: 'concurrent',
        printer: false,
      })

      agent.addHook(BeforeToolCallEvent, (event) => {
        if (event.toolUse.name === 'toolB') {
          event.interrupt({ name: 'approve_b', reason: 'Approve B?' })
        }
      })

      const interruptResult = await agent.invoke('Go')

      expect(interruptResult.stopReason).toBe('interrupt')
      expect(interruptResult.interrupts).toMatchObject([
        { id: expect.any(String), name: 'approve_b', reason: 'Approve B?', source: 'hook' },
      ])
      // A should have executed, B should not (interrupted before execution)
      expect(executionLog).toContain('A')
      expect(executionLog).not.toContain('B')
    })
  })

  describe('InterruptEvent emission', () => {
    it('yields one InterruptEvent per unanswered interrupt at stop, tagged with source', async () => {
      const model = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'toolA', toolUseId: 'tool-a', input: {} })
        .addTurn({ type: 'textBlock', text: 'done' })

      const toolA = createMockTool('toolA', (context) => {
        context.interrupt({ name: 'confirm_tool', reason: 'ok?' })
      })

      const agent = new Agent({ model, tools: [toolA], printer: false })

      // Hook-raised interrupt on a different identifier, via BeforeToolCallEvent.
      agent.addHook(BeforeToolCallEvent, (event) => {
        if (event.toolUse.name === 'toolA') {
          event.interrupt({ name: 'confirm_hook', reason: 'hook ok?' })
        }
      })

      const emittedEvents: InterruptEvent[] = []
      agent.addHook(InterruptEvent, (event) => {
        emittedEvents.push(event)
      })

      const result = await agent.invoke('go')

      expect(result.stopReason).toBe('interrupt')
      expect(result.interrupts).toHaveLength(emittedEvents.length)
      // One event per interrupt, each tagged by its origin. Hook interrupts fire
      // before tool callbacks, so the hook interrupt is the only one in this run.
      for (const event of emittedEvents) {
        expect(event.interrupt.source).toBe('hook')
      }
    })

    it('InterruptEvent is available on the stream', async () => {
      const model = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'approveMe', toolUseId: 'tool-1', input: {} })
        .addTurn({ type: 'textBlock', text: 'done' })

      const tool = createMockTool('approveMe', (context) => {
        context.interrupt({ name: 'approve', reason: 'please' })
      })

      const agent = new Agent({ model, tools: [tool], printer: false })

      const events: InterruptEvent[] = []
      for await (const event of agent.stream('go')) {
        if (event instanceof InterruptEvent) events.push(event)
      }

      expect(events).toHaveLength(1)
      expect(events[0]!.interrupt).toMatchObject({ name: 'approve', source: 'tool' })
    })
  })
})
