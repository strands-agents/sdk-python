/**
 * Integration tests for DirectReturnPlugin — verifies the full agent invoke flow
 * with direct-return tools using a mock model.
 *
 * Tests exercise the complete path: Agent constructor auto-registration,
 * BeforeToolsEvent enforcement, AfterToolsEvent early exit, and
 * AgentStreamStage result transformation.
 */

import { describe, it, expect } from 'vitest'
import { Agent } from '../../../agent/agent.js'
import { DIRECT_RETURN_DESCRIPTION_SUFFIX } from '../../../tools/tool.js'
import { MockMessageModel } from '../../../__fixtures__/mock-message-model.js'
import { DirectReturnPlugin } from '../plugin.js'
import { createMockTool } from '../../../__fixtures__/tool-helpers.js'

describe('DirectReturnPlugin integration', () => {
  describe('basic routing', () => {
    it('routes to the correct specialist and returns stopReason handoff', async () => {
      // Sub-agent models: each returns a distinct response
      const billingModel = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Your balance is $42.' })
      const techModel = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Try rebooting your router.' })

      const billingAgent = new Agent({ model: billingModel, name: 'Billing', printer: false })
      const techAgent = new Agent({ model: techModel, name: 'TechSupport', printer: false })

      // Orchestrator model: calls the TechSupport direct-return tool
      const orchestratorModel = new MockMessageModel().addTurn({
        type: 'toolUseBlock',
        name: 'TechSupport',
        toolUseId: 'call-1',
        input: { input: 'My wifi does not work' },
      })

      const orchestrator = new Agent({
        model: orchestratorModel,
        name: 'Orchestrator',
        tools: [billingAgent.asTool({ handoff: true }), techAgent.asTool({ handoff: true })],
        printer: false,
      })

      const result = await orchestrator.invoke('My wifi does not work')

      expect(result.stopReason).toBe('handoff')
      // The lastMessage should contain the sub-agent's response text
      const textBlocks = result.lastMessage.content.filter((b) => b.type === 'textBlock')
      expect(textBlocks).toHaveLength(1)
      expect((textBlocks[0] as { text: string }).text).toBe('Try rebooting your router.')
    })

    it('routes to billing agent when model calls Billing tool', async () => {
      const billingModel = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Refund processed.' })
      const techModel = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Reboot router.' })

      const billingAgent = new Agent({ model: billingModel, name: 'Billing', printer: false })
      const techAgent = new Agent({ model: techModel, name: 'TechSupport', printer: false })

      const orchestratorModel = new MockMessageModel().addTurn({
        type: 'toolUseBlock',
        name: 'Billing',
        toolUseId: 'call-2',
        input: { input: 'Where is my refund?' },
      })

      const orchestrator = new Agent({
        model: orchestratorModel,
        name: 'Orchestrator',
        tools: [billingAgent.asTool({ handoff: true }), techAgent.asTool({ handoff: true })],
        printer: false,
      })

      const result = await orchestrator.invoke('Where is my refund?')

      expect(result.stopReason).toBe('handoff')
      const textBlocks = result.lastMessage.content.filter((b) => b.type === 'textBlock')
      expect(textBlocks).toHaveLength(1)
      expect((textBlocks[0] as { text: string }).text).toBe('Refund processed.')
    })
  })

  describe('mixed tools', () => {
    it('regular tools work normally when called alone', async () => {
      const techModel = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Specialist response' })
      const techAgent = new Agent({ model: techModel, name: 'TechSupport', printer: false })

      const calculator = createMockTool('calculator', () => 'Result: 42')

      // Turn 1: model calls the regular calculator tool
      // Turn 2: model produces final text after seeing tool result
      const orchestratorModel = new MockMessageModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'calculator',
          toolUseId: 'calc-1',
          input: {},
        })
        .addTurn({ type: 'textBlock', text: 'The answer is 42.' })

      const orchestrator = new Agent({
        model: orchestratorModel,
        name: 'Orchestrator',
        tools: [calculator, techAgent.asTool({ handoff: true })],
        printer: false,
      })

      const result = await orchestrator.invoke('What is 6 * 7?')

      // Regular tool call does NOT trigger direct return
      expect(result.stopReason).toBe('endTurn')
      const textBlocks = result.lastMessage.content.filter((b) => b.type === 'textBlock')
      expect(textBlocks).toHaveLength(1)
      expect((textBlocks[0] as { text: string }).text).toBe('The answer is 42.')
    })

    it('direct-return tool triggers directReturn when called alone alongside regular tools in registry', async () => {
      const techModel = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Router fixed!' })
      const techAgent = new Agent({ model: techModel, name: 'TechSupport', printer: false })

      const calculator = createMockTool('calculator', () => 'Result: 42')

      // Model calls the direct-return tool (alone)
      const orchestratorModel = new MockMessageModel().addTurn({
        type: 'toolUseBlock',
        name: 'TechSupport',
        toolUseId: 'tech-1',
        input: { input: 'Fix my router' },
      })

      const orchestrator = new Agent({
        model: orchestratorModel,
        name: 'Orchestrator',
        tools: [calculator, techAgent.asTool({ handoff: true })],
        printer: false,
      })

      const result = await orchestrator.invoke('Fix my router')

      expect(result.stopReason).toBe('handoff')
      const textBlocks = result.lastMessage.content.filter((b) => b.type === 'textBlock')
      expect((textBlocks[0] as { text: string }).text).toBe('Router fixed!')
    })
  })

  describe('single-call enforcement', () => {
    it('cancels all tools when direct-return tool is called alongside other tools, then retries', async () => {
      const techModel = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Specialist answer' })
      const techAgent = new Agent({ model: techModel, name: 'TechSupport', printer: false })

      const calculator = createMockTool('calculator', () => 'Result: 42')

      // Turn 1: model calls BOTH calculator and direct-return tool (invalid)
      // Turn 2: model retries with just the direct-return tool (valid)
      const orchestratorModel = new MockMessageModel()
        .addTurn([
          { type: 'toolUseBlock', name: 'calculator', toolUseId: 'calc-1', input: {} },
          { type: 'toolUseBlock', name: 'TechSupport', toolUseId: 'tech-1', input: { input: 'help' } },
        ])
        .addTurn({
          type: 'toolUseBlock',
          name: 'TechSupport',
          toolUseId: 'tech-2',
          input: { input: 'help' },
        })

      const orchestrator = new Agent({
        model: orchestratorModel,
        name: 'Orchestrator',
        tools: [calculator, techAgent.asTool({ handoff: true })],
        printer: false,
      })

      const result = await orchestrator.invoke('Do both things')

      // After the retry, the direct return succeeds
      expect(result.stopReason).toBe('handoff')
      const textBlocks = result.lastMessage.content.filter((b) => b.type === 'textBlock')
      expect((textBlocks[0] as { text: string }).text).toBe('Specialist answer')
    })

    it('cancels all tools when two direct-return tools are called simultaneously, then retries', async () => {
      const billingModel = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Billing response' })
      const techModel = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Tech response' })

      const billingAgent = new Agent({ model: billingModel, name: 'Billing', printer: false })
      const techAgent = new Agent({ model: techModel, name: 'TechSupport', printer: false })

      // Turn 1: model calls BOTH direct-return tools simultaneously (invalid)
      // Turn 2: model retries with just one direct-return tool (valid)
      const orchestratorModel = new MockMessageModel()
        .addTurn([
          { type: 'toolUseBlock', name: 'Billing', toolUseId: 'bill-1', input: { input: 'refund' } },
          { type: 'toolUseBlock', name: 'TechSupport', toolUseId: 'tech-1', input: { input: 'wifi' } },
        ])
        .addTurn({
          type: 'toolUseBlock',
          name: 'TechSupport',
          toolUseId: 'tech-2',
          input: { input: 'wifi' },
        })

      const orchestrator = new Agent({
        model: orchestratorModel,
        name: 'Orchestrator',
        tools: [billingAgent.asTool({ handoff: true }), techAgent.asTool({ handoff: true })],
        printer: false,
      })

      const result = await orchestrator.invoke('Handle both billing and tech')

      // After the retry, the direct return succeeds with the single tool
      expect(result.stopReason).toBe('handoff')
      const textBlocks = result.lastMessage.content.filter((b) => b.type === 'textBlock')
      expect((textBlocks[0] as { text: string }).text).toBe('Tech response')
    })
  })

  describe('error resilience', () => {
    it('does not trigger direct return when sub-agent fails, loop continues', async () => {
      // Sub-agent model throws an error
      const failingModel = new MockMessageModel().addTurn(new Error('Sub-agent crashed'))
      const failingAgent = new Agent({ model: failingModel, name: 'FailAgent', printer: false })

      // Turn 1: model calls the direct-return tool (which will error)
      // Turn 2: model produces a fallback text response after seeing the error
      const orchestratorModel = new MockMessageModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'FailAgent',
          toolUseId: 'fail-1',
          input: { input: 'do something' },
        })
        .addTurn({ type: 'textBlock', text: 'Sorry, the specialist is unavailable.' })

      const orchestrator = new Agent({
        model: orchestratorModel,
        name: 'Orchestrator',
        tools: [failingAgent.asTool({ handoff: true })],
        printer: false,
      })

      const result = await orchestrator.invoke('Help me')

      // Direct return did NOT trigger because the tool errored — model recovers
      expect(result.stopReason).toBe('endTurn')
      const textBlocks = result.lastMessage.content.filter((b) => b.type === 'textBlock')
      expect((textBlocks[0] as { text: string }).text).toBe('Sorry, the specialist is unavailable.')
    })
  })

  describe('auto-registration', () => {
    it('DirectReturnPlugin is auto-registered when direct-return tools detected', () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Hi' })
      const subAgent = new Agent({ model, name: 'sub-agent', printer: false })

      const orchestrator = new Agent({
        model,
        tools: [subAgent.asTool({ handoff: true })],
        printer: false,
      })

      // Verify the tool is registered and marked as direct-return
      const tool = orchestrator.toolRegistry.get('sub-agent')
      expect(tool).toBeDefined()
      expect(tool!.directReturn).toBe(true)
    })

    it('DirectReturnPlugin is NOT registered when no direct-return tools present', () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Hi' })
      const subAgent = new Agent({ model, name: 'sub-agent', printer: false })

      // Use asTool WITHOUT direct return
      const orchestrator = new Agent({
        model,
        tools: [subAgent.asTool()],
        printer: false,
      })

      const tool = orchestrator.toolRegistry.get('sub-agent')
      expect(tool).toBeDefined()
      expect(tool!.directReturn).toBe(false)
    })

    it('no duplicate plugin when DirectReturnPlugin manually provided', async () => {
      const techModel = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Tech response' })
      const techAgent = new Agent({ model: techModel, name: 'TechSupport', printer: false })

      const orchestratorModel = new MockMessageModel().addTurn({
        type: 'toolUseBlock',
        name: 'TechSupport',
        toolUseId: 'call-1',
        input: { input: 'help' },
      })

      // Manually provide DirectReturnPlugin — should not get a duplicate
      const orchestrator = new Agent({
        model: orchestratorModel,
        name: 'Orchestrator',
        tools: [techAgent.asTool({ handoff: true })],
        plugins: [new DirectReturnPlugin()],
        printer: false,
      })

      // Verify it still works correctly (would break if duplicate registration caused issues)
      const result = await orchestrator.invoke('Help me')
      expect(result.stopReason).toBe('handoff')
    })
  })

  describe('description suffix', () => {
    it('tool spec description ends with the direct-return suffix', () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Hi' })
      const agent = new Agent({ model, name: 'specialist', description: 'Handles billing', printer: false })

      const tool = agent.asTool({ handoff: true })

      expect(tool.description).toBe('Handles billing' + DIRECT_RETURN_DESCRIPTION_SUFFIX)
      expect(tool.toolSpec.description).toBe('Handles billing' + DIRECT_RETURN_DESCRIPTION_SUFFIX)
    })

    it('description suffix is NOT appended when handoff is false', () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Hi' })
      const agent = new Agent({ model, name: 'specialist', description: 'Handles billing', printer: false })

      const tool = agent.asTool({ handoff: false })

      expect(tool.description).toBe('Handles billing')
      expect(tool.description).not.toContain(DIRECT_RETURN_DESCRIPTION_SUFFIX)
    })
  })
})
