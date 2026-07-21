import { describe, it, expect, vi } from 'vitest'

import { Agent } from '../../../agent/agent.js'
import type { ToolContext } from '../../../tools/tool.js'
import type { LocalAgent } from '../../../types/agent.js'
import { MockMessageModel } from '../../../__fixtures__/mock-message-model.js'
import { createMockTool } from '../../../__fixtures__/tool-helpers.js'

import {
  MAX_MULTIAGENT_DEPTH,
  MULTIAGENT_DEPTH_KEY,
  makeUseAgent,
  MultiagentDepthExceededError,
  useAgent,
} from '../use-agent.js'

function ctxFor(agent: LocalAgent, invocationState: Record<string, unknown> = {}): ToolContext {
  return {
    toolUse: {
      name: 'use_agent',
      toolUseId: 'test-id',
      input: {},
    },
    agent,
    invocationState,
    interrupt: () => {
      throw new Error('interrupt not available in mock context')
    },
  }
}

function mockChildModel(text: string): MockMessageModel {
  return new MockMessageModel().addTurn({ type: 'textBlock', text })
}

describe('use_agent input validation', () => {
  it('rejects empty systemPrompt', async () => {
    const parent = new Agent({ model: new MockMessageModel(), printer: false })
    await expect(useAgent.invoke({ systemPrompt: '   ', task: 'do it' }, ctxFor(parent))).rejects.toThrow(
      /systemPrompt must be non-empty/
    )
  })

  it('rejects oversized systemPrompt', async () => {
    const parent = new Agent({ model: new MockMessageModel(), printer: false })
    const huge = 'a'.repeat(8 * 1024 + 1)
    await expect(useAgent.invoke({ systemPrompt: huge, task: 'do it' }, ctxFor(parent))).rejects.toThrow(
      /exceeds size cap/
    )
  })

  it('rejects empty task', async () => {
    const parent = new Agent({ model: new MockMessageModel(), printer: false })
    await expect(useAgent.invoke({ systemPrompt: 'be helpful', task: '' }, ctxFor(parent))).rejects.toThrow(
      /task must be non-empty/
    )
  })

  it('rejects oversized task', async () => {
    const parent = new Agent({ model: new MockMessageModel(), printer: false })
    const huge = 'a'.repeat(32 * 1024 + 1)
    await expect(useAgent.invoke({ systemPrompt: 'ok', task: huge }, ctxFor(parent))).rejects.toThrow(
      /exceeds size cap/
    )
  })

  it('rejects boolean systemPrompt at the schema layer', async () => {
    // The schema rejects non-string inputs before the byte-cap check runs.
    // Guards against a future schema change that would coerce booleans to
    // strings.
    const parent = new Agent({ model: new MockMessageModel(), printer: false })
    await expect(
      useAgent.invoke({ systemPrompt: true as unknown as string, task: 'do it' }, ctxFor(parent))
    ).rejects.toThrow()
  })

  it('rejects boolean task at the schema layer', async () => {
    const parent = new Agent({ model: new MockMessageModel(), printer: false })
    await expect(
      useAgent.invoke({ systemPrompt: 'ok', task: false as unknown as string }, ctxFor(parent))
    ).rejects.toThrow()
  })
})

describe('use_agent tool allowlist', () => {
  it('rejects wildcard entry', async () => {
    const parent = new Agent({ model: new MockMessageModel(), printer: false })
    await expect(
      useAgent.invoke(
        {
          systemPrompt: 'ok',
          task: 'go',
          tools: ['*'],
        },
        ctxFor(parent)
      )
    ).rejects.toThrow(/wildcard/)
  })

  it('rejects unknown tool name', async () => {
    const other = createMockTool('search_docs', () => 'ok')
    const parent = new Agent({ model: new MockMessageModel(), printer: false, tools: [other] })
    await expect(
      useAgent.invoke(
        {
          systemPrompt: 'ok',
          task: 'go',
          tools: ['search_docs', 'not_a_real_tool'],
        },
        ctxFor(parent)
      )
    ).rejects.toThrow(/not present in the parent agent's tool registry/)
  })

  it('rejects non-string entry', async () => {
    const parent = new Agent({ model: new MockMessageModel(), printer: false })
    await expect(
      useAgent.invoke(
        {
          systemPrompt: 'ok',
          task: 'go',
          tools: [123 as unknown as string],
        },
        ctxFor(parent)
      )
    ).rejects.toThrow(/expected string/)
  })

  it('rejects multi-agent tool names', async () => {
    const parent = new Agent({ model: new MockMessageModel(), printer: false })
    for (const name of ['use_agent', 'swarm', 'graph', 'a2a_client']) {
      await expect(useAgent.invoke({ systemPrompt: 'ok', task: 'go', tools: [name] }, ctxFor(parent))).rejects.toThrow(
        /multi-agent tool/
      )
    }
  })

  it('rejects oversized allowlist', async () => {
    const parent = new Agent({ model: new MockMessageModel(), printer: false })
    const tools = Array.from({ length: 65 }, () => 'x')
    await expect(useAgent.invoke({ systemPrompt: 'ok', task: 'go', tools }, ctxFor(parent))).rejects.toThrow(
      /allowlist exceeds cap of 64/
    )
  })

  it('dedupes and preserves order', async () => {
    const one = createMockTool('one', () => 'one')
    const two = createMockTool('two', () => 'two')
    const parent = new Agent({ model: new MockMessageModel(), printer: false, tools: [one, two] })

    const invokeSpy = vi
      .spyOn(Agent.prototype, 'invoke')
      .mockResolvedValueOnce({ stopReason: 'endTurn', toString: () => 'done' } as never)

    await useAgent.invoke({ systemPrompt: 'ok', task: 'go', tools: ['one', 'two', 'one'] }, ctxFor(parent))

    const child = invokeSpy.mock.instances[0] as unknown as Agent
    const childToolNames = child.toolRegistry.list().map((t) => t.name)
    expect(childToolNames).toEqual(['one', 'two'])

    invokeSpy.mockRestore()
  })
})

describe('use_agent recursion cap', () => {
  it('refuses at the depth cap with a typed error', async () => {
    const parent = new Agent({ model: new MockMessageModel(), printer: false })
    await expect(
      useAgent.invoke(
        { systemPrompt: 'ok', task: 'go' },
        ctxFor(parent, { [MULTIAGENT_DEPTH_KEY]: MAX_MULTIAGENT_DEPTH })
      )
    ).rejects.toBeInstanceOf(MultiagentDepthExceededError)
  })

  it('forwards incremented depth to the child agent', async () => {
    const parent = new Agent({ model: new MockMessageModel(), printer: false })

    const invokeSpy = vi
      .spyOn(Agent.prototype, 'invoke')
      .mockResolvedValueOnce({ stopReason: 'endTurn', toString: () => 'done' } as never)

    await useAgent.invoke({ systemPrompt: 'ok', task: 'go' }, ctxFor(parent, { [MULTIAGENT_DEPTH_KEY]: 1 }))

    const options = invokeSpy.mock.calls[0]?.[1]
    expect(options).toBeDefined()
    expect((options?.invocationState as Record<string, unknown>)[MULTIAGENT_DEPTH_KEY]).toBe(2)

    invokeSpy.mockRestore()
  })

  it('preserves the parent invocationState and only overrides depth', async () => {
    const parent = new Agent({ model: new MockMessageModel(), printer: false })

    const invokeSpy = vi
      .spyOn(Agent.prototype, 'invoke')
      .mockResolvedValueOnce({ stopReason: 'endTurn', toString: () => 'done' } as never)

    await useAgent.invoke(
      { systemPrompt: 'ok', task: 'go' },
      ctxFor(parent, { traceId: 'abc-123', runId: 'xyz', [MULTIAGENT_DEPTH_KEY]: 0 })
    )

    const options = invokeSpy.mock.calls[0]?.[1]
    const state = options?.invocationState as Record<string, unknown>
    expect(state).toEqual({ traceId: 'abc-123', runId: 'xyz', [MULTIAGENT_DEPTH_KEY]: 1 })

    invokeSpy.mockRestore()
  })

  it('factory rejects non-finite maxDepth', () => {
    for (const bad of [Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NaN]) {
      expect(() => makeUseAgent({ maxDepth: bad })).toThrow(/maxDepth must be a positive integer/)
    }
  })

  it('factory rejects non-positive maxDepth', () => {
    expect(() => makeUseAgent({ maxDepth: 0 })).toThrow(/maxDepth/)
    expect(() => makeUseAgent({ maxDepth: -1 })).toThrow(/maxDepth/)
  })

  it('factory rejects non-integer maxDepth', () => {
    expect(() => makeUseAgent({ maxDepth: 2.5 })).toThrow(/maxDepth/)
  })
})

describe('use_agent model inheritance', () => {
  it('child agent uses the parent model instance', async () => {
    const parentModel = new MockMessageModel()
    const parent = new Agent({ model: parentModel, printer: false })

    const invokeSpy = vi
      .spyOn(Agent.prototype, 'invoke')
      .mockResolvedValueOnce({ stopReason: 'endTurn', toString: () => 'done' } as never)

    await useAgent.invoke({ systemPrompt: 'ok', task: 'go' }, ctxFor(parent))

    const child = invokeSpy.mock.instances[0] as unknown as Agent
    expect(child.model).toBe(parentModel)

    invokeSpy.mockRestore()
  })
})

describe('use_agent cancellation propagation', () => {
  it('forwards the parent cancelSignal to the child', async () => {
    const parent = new Agent({ model: new MockMessageModel(), printer: false })

    const invokeSpy = vi
      .spyOn(Agent.prototype, 'invoke')
      .mockResolvedValueOnce({ stopReason: 'endTurn', toString: () => 'done' } as never)

    await useAgent.invoke({ systemPrompt: 'ok', task: 'go' }, ctxFor(parent))

    const options = invokeSpy.mock.calls[0]?.[1]
    expect(options).toBeDefined()
    expect(options?.cancelSignal).toBe(parent.cancelSignal)
    invokeSpy.mockRestore()
  })

  it('re-raises the parent AbortError on cancelled child result', async () => {
    // Cross-SDK asymmetry: Python returns {status: cancelled}, TypeScript
    // re-raises AbortError so callers can distinguish cancellation from
    // other failures via `error.name === 'AbortError'`.
    const parent = new Agent({ model: new MockMessageModel(), printer: false })
    const parentAbort = new AbortController()
    Object.defineProperty(parent, 'cancelSignal', { get: () => parentAbort.signal })
    const abortReason = new DOMException('parent cancelled', 'AbortError')
    parentAbort.abort(abortReason)

    const invokeSpy = vi
      .spyOn(Agent.prototype, 'invoke')
      .mockResolvedValueOnce({ stopReason: 'cancelled', toString: () => 'partial' } as never)

    try {
      await useAgent.invoke({ systemPrompt: 'ok', task: 'go' }, ctxFor(parent))
      throw new Error('expected AbortError')
    } catch (error) {
      expect((error as Error).name).toBe('AbortError')
      // When the reason is an Error, it is rethrown as-is.
      expect(error).toBe(abortReason)
    }
    invokeSpy.mockRestore()
  })

  it('synthesizes an AbortError when cancelSignal.reason is not an Error', async () => {
    // cancelSignal.reason may be any value (a string, an object, undefined).
    // Guard against re-throwing a non-Error, which loses the .name === 'AbortError'
    // contract callers depend on.
    const parent = new Agent({ model: new MockMessageModel(), printer: false })
    const parentAbort = new AbortController()
    Object.defineProperty(parent, 'cancelSignal', { get: () => parentAbort.signal })
    parentAbort.abort('a plain string reason')

    const invokeSpy = vi
      .spyOn(Agent.prototype, 'invoke')
      .mockResolvedValueOnce({ stopReason: 'cancelled', toString: () => 'partial' } as never)

    try {
      await useAgent.invoke({ systemPrompt: 'ok', task: 'go' }, ctxFor(parent))
      throw new Error('expected AbortError')
    } catch (error) {
      expect((error as Error).name).toBe('AbortError')
    }
    invokeSpy.mockRestore()
  })
})

describe('use_agent happy path', () => {
  it('returns the child agent final text', async () => {
    const parent = new Agent({ model: mockChildModel('hello from the child'), printer: false })

    const result = (await useAgent.invoke({ systemPrompt: 'be a helper', task: 'say hi' }, ctxFor(parent))) as {
      status: string
      output: string
      executionTimeMs: number
    }

    expect(result.status).toBe('completed')
    expect(result.output).toContain('hello from the child')
    expect(typeof result.executionTimeMs).toBe('number')
  })
})

describe('use_agent stop reason mapping', () => {
  // Non-happy stop reasons must surface as failed. Otherwise a child that
  // hit limitTurns or a content filter looks like a delivered delegation
  // to the parent, hiding the failure.
  const failureReasons = [
    'limitTurns',
    'contentFiltered',
    'maxTokens',
    'guardrailIntervened',
    'limitOutputTokens',
    'limitTotalTokens',
  ] as const

  for (const stopReason of failureReasons) {
    it(`maps stopReason ${stopReason} to status=failed`, async () => {
      const parent = new Agent({ model: new MockMessageModel(), printer: false })
      const invokeSpy = vi
        .spyOn(Agent.prototype, 'invoke')
        .mockResolvedValueOnce({ stopReason, toString: () => 'partial' } as never)

      const result = (await useAgent.invoke({ systemPrompt: 'ok', task: 'go' }, ctxFor(parent))) as {
        status: string
        output: string
        executionTimeMs: number
      }

      expect(result.status).toBe('failed')
      expect(result.output).toBe('partial')
      expect(typeof result.executionTimeMs).toBe('number')

      invokeSpy.mockRestore()
    })
  }

  it('maps endTurn to status=completed', async () => {
    const parent = new Agent({ model: new MockMessageModel(), printer: false })
    const invokeSpy = vi
      .spyOn(Agent.prototype, 'invoke')
      .mockResolvedValueOnce({ stopReason: 'endTurn', toString: () => 'ok' } as never)

    const result = (await useAgent.invoke({ systemPrompt: 'ok', task: 'go' }, ctxFor(parent))) as {
      status: string
    }

    expect(result.status).toBe('completed')
    invokeSpy.mockRestore()
  })

  it('maps interrupt to status=interrupted', async () => {
    const parent = new Agent({ model: new MockMessageModel(), printer: false })
    const invokeSpy = vi
      .spyOn(Agent.prototype, 'invoke')
      .mockResolvedValueOnce({ stopReason: 'interrupt', toString: () => 'paused' } as never)

    const result = (await useAgent.invoke({ systemPrompt: 'ok', task: 'go' }, ctxFor(parent))) as {
      status: string
    }

    expect(result.status).toBe('interrupted')
    invokeSpy.mockRestore()
  })
})

describe('use_agent tool spec', () => {
  it('exposes the expected inputs and omits model config', () => {
    const spec = useAgent.toolSpec
    expect(spec.name).toBe('use_agent')
    const props = (spec.inputSchema as { properties: Record<string, unknown> }).properties
    for (const required of ['systemPrompt', 'task']) {
      expect(props).toHaveProperty(required)
    }
    expect(props).toHaveProperty('tools')
    // The credential-injection surface is deliberately absent.
    expect(props).not.toHaveProperty('modelProvider')
    expect(props).not.toHaveProperty('modelSettings')
  })
})
