/**
 * Integration tests for DirectReturnPlugin — verifies the full agent invoke flow
 * with direct-return tools using a mock model.
 *
 * Tests exercise the complete path: Agent constructor auto-registration,
 * BeforeToolsEvent enforcement, AfterToolsEvent early exit, and
 * AgentStreamStage result transformation.
 */

import { describe, it, expect } from 'vitest'
import { z } from 'zod'
import { Agent } from '../../../agent/agent.js'
import { MockMessageModel } from '../../../__fixtures__/mock-message-model.js'
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

  describe('structured output preservation', () => {
    it('preserves JSON content from sub-agent with structuredOutputSchema', async () => {
      const schema = z.object({ status: z.string(), total: z.number() })

      // Sub-agent calls the structured output tool to produce JSON
      const subModel = new MockMessageModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'strands_structured_output',
          toolUseId: 'so-1',
          input: { status: 'refunded', total: 42 },
        })
        .addTurn({ type: 'textBlock', text: 'Done' })

      const subAgent = new Agent({
        model: subModel,
        name: 'SchemaAgent',
        structuredOutputSchema: schema,
        printer: false,
      })

      // Orchestrator calls the direct-return sub-agent
      const orchestratorModel = new MockMessageModel().addTurn({
        type: 'toolUseBlock',
        name: 'SchemaAgent',
        toolUseId: 'call-1',
        input: { input: 'Generate the schema' },
      })

      const orchestrator = new Agent({
        model: orchestratorModel,
        name: 'Orchestrator',
        tools: [subAgent.asTool({ handoff: true })],
        printer: false,
      })

      const result = await orchestrator.invoke('Generate schema')

      expect(result.stopReason).toBe('handoff')
      // JsonBlock is converted to TextBlock with JSON-stringified content
      const textBlocks = result.lastMessage.content.filter((b) => b.type === 'textBlock')
      expect(textBlocks).toHaveLength(1)
      expect(JSON.parse((textBlocks[0] as { text: string }).text)).toEqual({ status: 'refunded', total: 42 })
    })
  })
})
