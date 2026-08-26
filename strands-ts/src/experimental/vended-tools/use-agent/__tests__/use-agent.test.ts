import { describe, expect, it, vi } from 'vitest'

import { Agent, inheritPreToolHooksSymbol } from '../../../../agent/agent.js'
import { MockMessageModel } from '../../../../__fixtures__/mock-message-model.js'
import { collectGenerator, TestModelProvider } from '../../../../__fixtures__/model-test-helpers.js'
import { createMockContext, createMockTool } from '../../../../__fixtures__/tool-helpers.js'
import { BeforeToolCallEvent, BeforeToolsEvent } from '../../../../hooks/events.js'
import { InterventionActions, InterventionHandler } from '../../../../interventions/index.js'
import type { Tool, ToolContext } from '../../../../tools/tool.js'
import { InterruptResponseContent } from '../../../../types/interrupt.js'
import { JsonBlock, type ToolResultBlock } from '../../../../types/messages.js'
import type { JSONValue } from '../../../../types/json.js'
import { makeUseAgent } from '../index.js'

function context(
  parent: Agent,
  useAgent: Tool,
  input: Record<string, JSONValue>,
  cancelSignal = new AbortController().signal
): ToolContext {
  return {
    ...createMockContext({ name: useAgent.name, toolUseId: 'use-agent-call', input }),
    agent: parent,
    invocationState: {},
    cancelSignal,
  }
}

function textModel(text: string): MockMessageModel {
  return new MockMessageModel().addTurn({ type: 'textBlock', text })
}

function resultJson(result: ToolResultBlock): Record<string, JSONValue> {
  return (result.content[0] as JsonBlock).json as Record<string, JSONValue>
}

function gatedModel(): { model: TestModelProvider; release: () => void } {
  let release!: () => void
  const gate = new Promise<void>((resolve) => {
    release = resolve
  })
  const model = new TestModelProvider(async function* () {
    await gate
    yield { type: 'modelMessageStartEvent', role: 'assistant' }
    yield { type: 'modelMessageStopEvent', stopReason: 'endTurn' }
  })
  return { model, release }
}

class ConfirmDangerous extends InterventionHandler {
  readonly name = 'confirm-dangerous'

  override beforeToolCall(event: Parameters<InterventionHandler['beforeToolCall']>[0]) {
    return event.toolUse.name.startsWith('dangerous')
      ? InterventionActions.confirm('approve?')
      : InterventionActions.proceed()
  }
}

describe('makeUseAgent', () => {
  it('exposes the narrow input and bounded execution options', () => {
    const spec = makeUseAgent().toolSpec

    expect(spec.name).toBe('use_agent')
    expect(Object.keys(spec.inputSchema?.properties ?? {})).toEqual(['task', 'instructions', 'tools'])
    expect(spec.inputSchema?.required).toEqual(['task'])
    expect(spec.inputSchema?.additionalProperties).toBe(false)
    expect(() => makeUseAgent({ limits: { turns: 51 } })).toThrow(
      'limits.turns must be a safe integer between 1 and 50'
    )
  })

  it('grants only named parent tools and preserves inherited pre-tool policy', async () => {
    const extra = vi.fn(() => 'extra')
    const blocked = vi.fn(() => 'blocked')
    const extraTool = createMockTool('extra', extra)
    const blockedTool = createMockTool('blocked', blocked)

    const omittedUseAgent = makeUseAgent()
    const omittedParent = new Agent({
      model: new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'extra', toolUseId: 'extra-call', input: {} })
        .addTurn({ type: 'textBlock', text: 'handled' }),
      tools: [extraTool, omittedUseAgent],
    })
    const omitted = await collectGenerator(
      omittedUseAgent.stream(context(omittedParent, omittedUseAgent, { task: 'work' }))
    )

    const beforeTools = vi.fn()
    const policyOrder: string[] = []
    const governedUseAgent = makeUseAgent()
    const ancestor = new Agent({ model: textModel('unused') })
    ancestor.addHook(
      BeforeToolCallEvent,
      () => {
        policyOrder.push('ancestor')
      },
      { order: 100 }
    )
    const governedParent = new Agent({
      model: new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'extra', toolUseId: 'extra-call', input: {} })
        .addTurn({ type: 'textBlock', text: 'handled' }),
      tools: [extraTool, blockedTool, governedUseAgent],
    })
    governedParent[inheritPreToolHooksSymbol](ancestor)
    governedParent.addHook(BeforeToolsEvent, beforeTools)
    governedParent.addHook(BeforeToolCallEvent, (event) => {
      policyOrder.push('parent')
      event.selectedTool = blockedTool
    })
    const governed = await collectGenerator(
      governedUseAgent.stream(
        context(governedParent, governedUseAgent, {
          task: 'work',
          tools: ['extra'],
        })
      )
    )

    expect(extra).not.toHaveBeenCalled()
    expect(blocked).not.toHaveBeenCalled()
    expect(beforeTools).toHaveBeenCalledTimes(1)
    expect(policyOrder).toEqual(['parent', 'ancestor'])
    expect(resultJson(omitted.result)).toMatchObject({ status: 'completed', output: 'handled' })
    expect(governed.result.status).toBe('success')
  })

  it('rejects grants or models that bypass the governed tool registry', async () => {
    const registered = makeUseAgent()
    const parent = new Agent({ model: textModel('unused'), tools: [registered] })

    const missing = await collectGenerator(
      registered.stream(context(parent, registered, { task: 'work', tools: ['missing'] }))
    )
    const different = await collectGenerator(
      makeUseAgent().stream(context(parent, makeUseAgent(), { task: 'work', tools: ['use_agent'] }))
    )

    const nativeModel = textModel('unused')
    Object.assign(nativeModel.getConfig(), { params: { tools: [{ type: 'web_search' }] } })
    const nativeTool = makeUseAgent()
    const native = await collectGenerator(
      nativeTool.stream(context(new Agent({ model: nativeModel }), nativeTool, { task: 'work' }))
    )

    const recursiveTool = makeUseAgent({ limits: { depth: 1 } })
    const recursiveParent = new Agent({
      model: new MockMessageModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'use_agent',
          toolUseId: 'nested-call',
          input: { task: 'nested' },
        })
        .addTurn({ type: 'textBlock', text: 'done' }),
      tools: [recursiveTool],
    })
    const recursive = await collectGenerator(
      recursiveTool.stream(context(recursiveParent, recursiveTool, { task: 'work', tools: ['use_agent'] }))
    )

    expect(resultJson(missing.result).error).toBe("Tool 'missing' was not found on the parent agent")
    expect(resultJson(different.result).error).toBe('A child can receive only the currently executing use_agent tool')
    expect(resultJson(native.result).error).toContain('provider-native model tools')
    expect(resultJson(recursive.result)).toMatchObject({ status: 'completed', output: 'done' })
  })

  it('scopes repeated interrupts and resumes the same child state', async () => {
    const dangerous = vi.fn(() => 'deleted')
    const dangerousTool = createMockTool('dangerous', dangerous)
    const model = new MockMessageModel()
      .addTurn({
        type: 'toolUseBlock',
        name: 'use_agent',
        toolUseId: 'use-agent-call',
        input: { task: 'delete', tools: ['dangerous'] },
      })
      .addTurn({ type: 'toolUseBlock', name: 'dangerous', toolUseId: 'dangerous-call', input: {} })
      .addTurn({ type: 'toolUseBlock', name: 'dangerous', toolUseId: 'dangerous-call', input: {} })
      .addTurn({ type: 'textBlock', text: 'child done' })
      .addTurn({ type: 'textBlock', text: 'outer done' })
    const useAgent = makeUseAgent()
    const parent = new Agent({
      model,
      tools: [dangerousTool, useAgent],
      interventions: [new ConfirmDangerous()],
      printer: false,
    })

    const first = await parent.invoke('go')
    const firstInterrupt = first.interrupts![0]!

    const restoredTool = makeUseAgent()
    const restoredParent = new Agent({
      model: textModel('unused'),
      tools: [dangerousTool, restoredTool],
      interventions: [new ConfirmDangerous()],
      printer: false,
    })
    restoredParent.loadSnapshot(parent.takeSnapshot({ preset: 'session' }))
    const { result: restored } = await collectGenerator(
      restoredTool.stream(context(restoredParent, restoredTool, { task: 'delete', tools: ['dangerous'] }))
    )

    const second = await parent.invoke([
      new InterruptResponseContent({ interruptId: firstInterrupt.id, response: 'yes' }),
    ])
    const secondInterrupt = second.interrupts![0]!
    const completed = await parent.invoke([
      new InterruptResponseContent({ interruptId: secondInterrupt.id, response: 'yes' }),
    ])

    expect(secondInterrupt.id).not.toBe(firstInterrupt.id)
    expect(completed.toString()).toBe('outer done')
    expect(dangerous).toHaveBeenCalledTimes(2)
    expect(resultJson(restored).error).toBe(
      'use_agent cannot resume an interrupted child after the parent or tool instance was restored'
    )
  })

  it('preserves turn and token budgets across an interrupt', async () => {
    for (const { limits, usage, stopReason } of [
      { limits: { turns: 1 }, usage: undefined, stopReason: 'limitTurns' },
      {
        limits: { totalTokens: 1 },
        usage: { inputTokens: 5, outputTokens: 5, totalTokens: 10 },
        stopReason: 'limitTotalTokens',
      },
    ]) {
      const dangerous = vi.fn(() => 'deleted')
      const model = new MockMessageModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'use_agent',
          toolUseId: 'use-agent-call',
          input: { task: 'delete', tools: ['dangerous'] },
        })
        .addTurn(
          { type: 'toolUseBlock', name: 'dangerous', toolUseId: 'dangerous-call', input: {} },
          usage ? { usage } : undefined
        )
        .addTurn({ type: 'textBlock', text: 'budget exhausted' })
      const useAgent = makeUseAgent({ limits })
      const parent = new Agent({
        model,
        tools: [createMockTool('dangerous', dangerous), useAgent],
        interventions: [new ConfirmDangerous()],
        printer: false,
      })

      const interrupted = await parent.invoke('go')
      await parent.invoke([
        new InterruptResponseContent({ interruptId: interrupted.interrupts![0]!.id, response: 'yes' }),
      ])
      const toolResult = parent.messages
        .flatMap((message) => message.content)
        .find((block) => block.type === 'toolResultBlock' && block.toolUseId === 'use-agent-call')

      expect(dangerous).not.toHaveBeenCalled()
      expect(toolResult?.toJSON()).toMatchObject({
        toolResult: {
          content: [{ json: { status: 'failed', error: `use_agent child stopped before completion: ${stopReason}` } }],
        },
      })
    }
  })

  it('returns promptly on parent cancellation and timeout', async () => {
    const cancelledRun = gatedModel()
    const cancelledTool = makeUseAgent()
    const cancelledParent = new Agent({ model: cancelledRun.model, tools: [cancelledTool] })
    const controller = new AbortController()
    const cancelledPromise = collectGenerator(
      cancelledTool.stream(context(cancelledParent, cancelledTool, { task: 'work' }, controller.signal))
    )
    await Promise.resolve()
    controller.abort(new DOMException('use_agent child was cancelled', 'AbortError'))
    const cancelled = await cancelledPromise
    cancelledRun.release()

    const timedRun = gatedModel()
    const timedTool = makeUseAgent({ limits: { timeoutSeconds: 1 } })
    const timedParent = new Agent({ model: timedRun.model, tools: [timedTool] })
    const timed = await collectGenerator(timedTool.stream(context(timedParent, timedTool, { task: 'work' })))
    timedRun.release()

    expect(resultJson(cancelled.result).status).toBe('cancelled')
    expect(resultJson(timed.result).error).toBe('use_agent child exceeded its execution timeout')
  })
})
