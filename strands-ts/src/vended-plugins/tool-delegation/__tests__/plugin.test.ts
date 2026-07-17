/**
 * Integration tests for ToolDelegation — verifies the full agent invoke flow
 * with delegation tools using a mock model.
 *
 * Tests exercise the complete path: Agent constructor auto-registration,
 * BeforeToolsEvent enforcement, AfterToolsEvent early exit, and
 * AgentStreamStage result transformation.
 */

import { describe, it, expect } from 'vitest'
import { z } from 'zod'
import { Agent } from '../../../agent/agent.js'
import { AfterToolsEvent } from '../../../hooks/events.js'
import { MockMessageModel } from '../../../__fixtures__/mock-message-model.js'
import { createMockTool } from '../../../__fixtures__/tool-helpers.js'

describe('ToolDelegation integration', () => {
  describe('basic routing', () => {
    it('routes to the correct specialist and returns stopReason delegated', async () => {
      // Sub-agent models: each returns a distinct response
      const billingModel = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Your balance is $42.' })
      const techModel = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Try rebooting your router.' })

      const billingAgent = new Agent({ model: billingModel, name: 'Billing', printer: false })
      const techAgent = new Agent({ model: techModel, name: 'TechSupport', printer: false })

      // Orchestrator model: calls the TechSupport delegation tool
      const orchestratorModel = new MockMessageModel().addTurn({
        type: 'toolUseBlock',
        name: 'TechSupport',
        toolUseId: 'call-1',
        input: { input: 'My wifi does not work' },
      })

      const orchestrator = new Agent({
        model: orchestratorModel,
        name: 'Orchestrator',
        tools: [billingAgent.asTool({ delegate: true }), techAgent.asTool({ delegate: true })],
        printer: false,
      })

      const result = await orchestrator.invoke('My wifi does not work')

      expect(result.stopReason).toBe('delegated')
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
        tools: [calculator, techAgent.asTool({ delegate: true })],
        printer: false,
      })

      const result = await orchestrator.invoke('What is 6 * 7?')

      // Regular tool call does NOT trigger delegation
      expect(result.stopReason).toBe('endTurn')
      const textBlocks = result.lastMessage.content.filter((b) => b.type === 'textBlock')
      expect(textBlocks).toHaveLength(1)
      expect((textBlocks[0] as { text: string }).text).toBe('The answer is 42.')
    })

    it('delegation tool triggers handoff when called alone alongside regular tools in registry', async () => {
      const techModel = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Router fixed!' })
      const techAgent = new Agent({ model: techModel, name: 'TechSupport', printer: false })

      const calculator = createMockTool('calculator', () => 'Result: 42')

      // Model calls the delegation tool (alone)
      const orchestratorModel = new MockMessageModel().addTurn({
        type: 'toolUseBlock',
        name: 'TechSupport',
        toolUseId: 'tech-1',
        input: { input: 'Fix my router' },
      })

      const orchestrator = new Agent({
        model: orchestratorModel,
        name: 'Orchestrator',
        tools: [calculator, techAgent.asTool({ delegate: true })],
        printer: false,
      })

      const result = await orchestrator.invoke('Fix my router')

      expect(result.stopReason).toBe('delegated')
      const textBlocks = result.lastMessage.content.filter((b) => b.type === 'textBlock')
      expect((textBlocks[0] as { text: string }).text).toBe('Router fixed!')
    })
  })

  describe('single-call enforcement', () => {
    it('cancels all tools when delegation tool is called alongside other tools, then retries', async () => {
      const techModel = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Specialist answer' })
      const techAgent = new Agent({ model: techModel, name: 'TechSupport', printer: false })

      const calculator = createMockTool('calculator', () => 'Result: 42')

      // Turn 1: model calls BOTH calculator and delegation tool (invalid)
      // Turn 2: model retries with just the delegation tool (valid)
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
        tools: [calculator, techAgent.asTool({ delegate: true })],
        printer: false,
      })

      const result = await orchestrator.invoke('Do both things')

      // After the retry, the delegation succeeds
      expect(result.stopReason).toBe('delegated')
      const textBlocks = result.lastMessage.content.filter((b) => b.type === 'textBlock')
      expect((textBlocks[0] as { text: string }).text).toBe('Specialist answer')
    })

    it('cancels all tools when two delegation tools are called simultaneously, then retries', async () => {
      const billingModel = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Billing response' })
      const techModel = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Tech response' })

      const billingAgent = new Agent({ model: billingModel, name: 'Billing', printer: false })
      const techAgent = new Agent({ model: techModel, name: 'TechSupport', printer: false })

      // Turn 1: model calls BOTH delegation tools simultaneously (invalid)
      // Turn 2: model retries with just one delegation tool (valid)
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
        tools: [billingAgent.asTool({ delegate: true }), techAgent.asTool({ delegate: true })],
        printer: false,
      })

      const result = await orchestrator.invoke('Handle both billing and tech')

      // After the retry, the delegation succeeds with the single tool
      expect(result.stopReason).toBe('delegated')
      const textBlocks = result.lastMessage.content.filter((b) => b.type === 'textBlock')
      expect((textBlocks[0] as { text: string }).text).toBe('Tech response')
    })
  })

  describe('error resilience', () => {
    it('does not trigger delegation when sub-agent fails, loop continues', async () => {
      // Sub-agent model throws an error
      const failingModel = new MockMessageModel().addTurn(new Error('Sub-agent crashed'))
      const failingAgent = new Agent({ model: failingModel, name: 'FailAgent', printer: false })

      // Turn 1: model calls the delegation tool (which will error)
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
        tools: [failingAgent.asTool({ delegate: true })],
        printer: false,
      })

      const result = await orchestrator.invoke('Help me')

      // Delegation did NOT trigger because the tool errored — model recovers
      expect(result.stopReason).toBe('endTurn')
      const textBlocks = result.lastMessage.content.filter((b) => b.type === 'textBlock')
      expect((textBlocks[0] as { text: string }).text).toBe('Sorry, the specialist is unavailable.')
    })
  })

  describe('cross-request state isolation', () => {
    it('does not leak delegation state when a later AfterTools hook throws', async () => {
      // Repro: first request — delegation tool succeeds (state committed), but
      // a later AfterToolsEvent hook throws. Second request should NOT see
      // stale stopReason: 'delegated' from the first request.
      const subModel = new MockMessageModel().addTurn({ type: 'textBlock', text: 'STALE_FIRST_RESULT' })
      const subAgent = new Agent({ model: subModel, name: 'Sub', printer: false })

      // First request: model calls the delegation tool
      // Second request: model produces a fresh text response
      const orchestratorModel = new MockMessageModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'Sub',
          toolUseId: 'del-1',
          input: { input: 'first request' },
        })
        .addTurn({ type: 'textBlock', text: 'FRESH_SECOND_RESULT' })

      const orchestrator = new Agent({
        model: orchestratorModel,
        name: 'Orchestrator',
        tools: [subAgent.asTool({ delegate: true })],
        printer: false,
      })

      // Register a hook AFTER the plugin's hook so it fires after _onAfterTools
      // commits delegation state.
      let throwOnNext = true
      orchestrator.addHook(AfterToolsEvent, () => {
        if (throwOnNext) {
          throw new Error('AFTER_TOOLS_BOOM')
        }
      })

      // First invocation should throw because of the later hook
      await expect(orchestrator.invoke('first')).rejects.toThrow('AFTER_TOOLS_BOOM')

      // Disable the throwing hook for the second invocation
      throwOnNext = false

      // Second invocation must NOT produce stale delegation result
      const result = await orchestrator.invoke('second')
      expect(result.stopReason).toBe('endTurn')
      const textBlocks = result.lastMessage.content.filter((b) => b.type === 'textBlock')
      expect((textBlocks[0] as { text: string }).text).toBe('FRESH_SECOND_RESULT')
    })
  })

  describe('non-text delegation', () => {
    it('delegates successfully when sub-agent returns empty text (e.g. image-only response)', async () => {
      // Simulate a sub-agent whose toString() returns '' — this happens when the
      // sub-agent's lastMessage contains only non-text content (image, document, video).
      // AgentAsTool wraps it in TextBlock(''), so the ToolResultBlock has a single
      // empty TextBlock. extractText() returns '' for this, which previously caused
      // endTurn to be falsy and the early-exit check to fail.
      const emptyTextModel = new MockMessageModel().addTurn({ type: 'textBlock', text: '' })
      const imageAgent = new Agent({ model: emptyTextModel, name: 'ImageAgent', printer: false })

      // Orchestrator model has ONLY one turn (the delegation tool call).
      // If the early-exit fails and the orchestrator tries a second model call,
      // MockMessageModel throws "All turns have been consumed" — so the test
      // implicitly verifies no extra model call occurs.
      const orchestratorModel = new MockMessageModel().addTurn({
        type: 'toolUseBlock',
        name: 'ImageAgent',
        toolUseId: 'img-1',
        input: { input: 'Generate an image' },
      })

      const orchestrator = new Agent({
        model: orchestratorModel,
        name: 'Orchestrator',
        tools: [imageAgent.asTool({ delegate: true })],
        printer: false,
      })

      const result = await orchestrator.invoke('Generate an image')

      // Delegation MUST trigger even when the sub-agent returns empty text
      expect(result.stopReason).toBe('delegated')
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

      // Orchestrator calls the delegation sub-agent
      const orchestratorModel = new MockMessageModel().addTurn({
        type: 'toolUseBlock',
        name: 'SchemaAgent',
        toolUseId: 'call-1',
        input: { input: 'Generate the schema' },
      })

      const orchestrator = new Agent({
        model: orchestratorModel,
        name: 'Orchestrator',
        tools: [subAgent.asTool({ delegate: true })],
        printer: false,
      })

      const result = await orchestrator.invoke('Generate schema')

      expect(result.stopReason).toBe('delegated')
      // JsonBlock is converted to TextBlock with JSON-stringified content
      const textBlocks = result.lastMessage.content.filter((b) => b.type === 'textBlock')
      expect(textBlocks).toHaveLength(1)
      expect(JSON.parse((textBlocks[0] as { text: string }).text)).toEqual({ status: 'refunded', total: 42 })
    })
  })

  describe('stateful model bypass', () => {
    it('skips delegation logic when model is stateful — tool runs as a normal tool', async () => {
      class StatefulModel extends MockMessageModel {
        override get stateful(): boolean {
          return true
        }
      }

      const subModel = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Sub response' })
      const subAgent = new Agent({ model: subModel, name: 'Sub', printer: false })

      // Turn 1: model calls the delegation tool (which won't trigger delegation)
      // Turn 2: model produces final text after seeing tool result
      const statefulModel = new StatefulModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'Sub',
          toolUseId: 'del-1',
          input: { input: 'do something' },
        })
        .addTurn({ type: 'textBlock', text: 'Got the sub-agent response, done.' })

      const orchestrator = new Agent({
        model: statefulModel,
        name: 'Orchestrator',
        tools: [subAgent.asTool({ delegate: true })],
        printer: false,
      })

      const result = await orchestrator.invoke('Do something')

      // Delegation did NOT trigger — model consumed the tool result normally
      expect(result.stopReason).toBe('endTurn')
      const textBlocks = result.lastMessage.content.filter((b) => b.type === 'textBlock')
      expect((textBlocks[0] as { text: string }).text).toBe('Got the sub-agent response, done.')
    })

    it('does not enforce single-call constraint for stateful models', async () => {
      class StatefulModel extends MockMessageModel {
        override get stateful(): boolean {
          return true
        }
      }

      const subModel = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Sub response' })
      const subAgent = new Agent({ model: subModel, name: 'Sub', printer: false })

      const calculator = createMockTool('calculator', () => 'Result: 42')

      // Model calls BOTH calculator and delegation tool simultaneously.
      // With a stateful model, this should NOT be cancelled.
      const statefulModel = new StatefulModel()
        .addTurn([
          { type: 'toolUseBlock', name: 'calculator', toolUseId: 'calc-1', input: {} },
          { type: 'toolUseBlock', name: 'Sub', toolUseId: 'del-1', input: { input: 'help' } },
        ])
        .addTurn({ type: 'textBlock', text: 'Both tools ran successfully.' })

      const orchestrator = new Agent({
        model: statefulModel,
        name: 'Orchestrator',
        tools: [calculator, subAgent.asTool({ delegate: true })],
        printer: false,
      })

      const result = await orchestrator.invoke('Do both')

      // Both tools ran normally — no cancellation, no delegation
      expect(result.stopReason).toBe('endTurn')
      const textBlocks = result.lastMessage.content.filter((b) => b.type === 'textBlock')
      expect((textBlocks[0] as { text: string }).text).toBe('Both tools ran successfully.')
    })
  })
})
