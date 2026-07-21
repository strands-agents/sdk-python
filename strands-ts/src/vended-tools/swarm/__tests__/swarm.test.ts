import { describe, it, expect, vi, beforeEach } from 'vitest'

// Mock the SDK's Swarm before importing the shim so the shim's `new Swarm(...)`
// resolves to our controllable stub. Vitest hoists `vi.mock` above the imports,
// so this works even though Swarm is imported statically.
const invokeMock = vi.fn()
const swarmCtorMock = vi.fn()
vi.mock('../../../multiagent/swarm.js', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../../multiagent/swarm.js')>()
  class MockSwarm {
    constructor(options: unknown) {
      swarmCtorMock(options)
    }
    invoke(...args: unknown[]): unknown {
      return invokeMock(...args)
    }
  }
  return { ...actual, Swarm: MockSwarm }
})

import { swarm, makeSwarm, MAX_MULTIAGENT_DEPTH, MULTIAGENT_DEPTH_KEY, MultiagentDepthExceededError } from '../swarm.js'
import { MockMessageModel } from '../../../__fixtures__/mock-message-model.js'
import { createMockAgent } from '../../../__fixtures__/agent-helpers.js'
import { ToolRegistry } from '../../../registry/tool-registry.js'
import type { ToolContext } from '../../../tools/tool.js'
import type { Tool } from '../../../tools/tool.js'
import { Status, MultiAgentResult, NodeResult } from '../../../multiagent/state.js'
import { TextBlock } from '../../../types/messages.js'

/**
 * Type of a validated child spec. Kept loose (structural) so tests can splice
 * in intentionally-wrong fields (e.g. `model`) without fighting the compiler.
 */
type AgentSpec = { name: string; systemPrompt: string; tools: string[]; description?: string }

/**
 * Minimal valid child spec. Per the shared multi-agent dialect, `name`,
 * `systemPrompt`, and `tools` are all required. Individual tests override
 * fields to exercise specific validation branches.
 */
function makeSpec(name: string = 'a', overrides: Record<string, unknown> = {}): AgentSpec {
  return { name, systemPrompt: 'you are helpful', tools: [], ...overrides } as AgentSpec
}

/**
 * Bare `Tool` stub — just enough surface to survive `ToolRegistry.add`.
 */
function makeStubTool(name: string): Tool {
  return {
    name,
    description: `stub ${name}`,
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
    invoke: async () => ({}),
  } as unknown as Tool
}

/**
 * Build a ToolContext with a mock parent agent. Sets up a model and a tool
 * registry pre-populated with the given tool names.
 */
function buildContext(options?: {
  parentTools?: string[]
  cancelled?: boolean
  invocationState?: Record<string, unknown>
}): ToolContext {
  const toolRegistry = new ToolRegistry()
  for (const name of options?.parentTools ?? []) {
    toolRegistry.add(makeStubTool(name))
  }
  const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'ignored' })
  const abortController = new AbortController()
  if (options?.cancelled) abortController.abort()

  const agent = createMockAgent({
    toolRegistry,
    extra: { model, cancelSignal: abortController.signal },
  })
  return {
    toolUse: { name: 'swarm', toolUseId: 'tid', input: {} },
    agent,
    invocationState: options?.invocationState ?? {},
    interrupt: () => {
      throw new Error('interrupt not available in mock context')
    },
  }
}

/**
 * Build a MultiAgentResult that maps into a completed swarm run.
 */
function buildResult(options?: {
  nodeIds?: string[]
  finalText?: string
  status?: 'COMPLETED' | 'FAILED'
  duration?: number
}): MultiAgentResult {
  const nodeIds = options?.nodeIds ?? []
  const status = options?.status ?? 'COMPLETED'
  const results = nodeIds.map(
    (id) =>
      new NodeResult({
        nodeId: id,
        status: 'COMPLETED' as const,
        duration: 5,
        content: [new TextBlock(`from ${id}`)],
        usage: { inputTokens: 3, outputTokens: 4, totalTokens: 7 },
      })
  )
  return new MultiAgentResult({
    status: status as never,
    results,
    content: options?.finalText !== undefined ? [new TextBlock(options.finalText)] : [],
    duration: options?.duration ?? 42,
  })
}

describe('swarm tool', () => {
  describe('spec validation', () => {
    it('rejects an empty agents list', async () => {
      await expect(swarm.invoke({ agents: [], initialInput: 'go' }, buildContext())).rejects.toThrow()
    })

    it('rejects too many agents', async () => {
      const agents = Array.from({ length: 6 }, (_, i) => makeSpec(`a${i}`))
      await expect(swarm.invoke({ agents, initialInput: 'go' }, buildContext())).rejects.toThrow()
    })

    it('rejects a missing name', async () => {
      // Deliberate: exercise the tool boundary with malformed input the way a
      // model might send it. The tool accepts `unknown` at invoke time.
      await expect(
        swarm.invoke({ agents: [{ systemPrompt: 'hi', tools: [] }], initialInput: 'go' }, buildContext())
      ).rejects.toThrow()
    })

    it('rejects a name exceeding the 64-char cap', async () => {
      const overlong = 'a'.repeat(65)
      await expect(swarm.invoke({ agents: [makeSpec(overlong)], initialInput: 'go' }, buildContext())).rejects.toThrow()
    })

    it.each(['1leading-digit', 'has space', 'has-dash', 'has.dot', 'has$symbol'])(
      'rejects a name that fails the regex: %s',
      async (badName) => {
        await expect(
          swarm.invoke({ agents: [makeSpec(badName)], initialInput: 'go' }, buildContext())
        ).rejects.toThrow()
      }
    )

    it('rejects a missing systemPrompt', async () => {
      // `systemPrompt` is required per the shared dialect.
      await expect(
        swarm.invoke({ agents: [{ name: 'a', tools: [] }], initialInput: 'go' }, buildContext())
      ).rejects.toThrow()
    })

    it('rejects a missing tools list', async () => {
      // `tools` is required per the shared dialect (may be empty).
      await expect(
        swarm.invoke({ agents: [{ name: 'a', systemPrompt: 'sp' }], initialInput: 'go' }, buildContext())
      ).rejects.toThrow()
    })

    it('rejects tools exceeding the 64-entry cap', async () => {
      const tools = Array.from({ length: 65 }, (_, i) => `t${i}`)
      await expect(
        swarm.invoke({ agents: [makeSpec('a', { tools })], initialInput: 'go' }, buildContext())
      ).rejects.toThrow()
    })

    it('rejects unknown spec fields (guards against invented knobs like `model`)', async () => {
      await expect(
        swarm.invoke(
          {
            // `makeSpec` returns the loose `AgentSpec` type so the extra `model`
            // field survives compile — the whole point is that the tool boundary
            // rejects it at runtime via `.strict()`.
            agents: [makeSpec('a', { model: 'gpt-9000' })],
            initialInput: 'go',
          },
          buildContext()
        )
      ).rejects.toThrow()
    })

    it('rejects unknown top-level fields', async () => {
      // Defense-in-depth: `.strict()` rejects invented knobs at the top level.
      await expect(
        swarm.invoke(
          {
            agents: [makeSpec('a')],
            initialInput: 'go',
            hooks: [],
          },
          buildContext()
        )
      ).rejects.toThrow()
    })

    it('rejects an oversized initialInput', async () => {
      const oversized = 'x'.repeat(32 * 1024 + 1)
      await expect(swarm.invoke({ agents: [makeSpec('a')], initialInput: oversized }, buildContext())).rejects.toThrow(
        /initialInput exceeds size cap/
      )
    })

    it('rejects an oversized systemPrompt', async () => {
      const oversized = 'x'.repeat(8 * 1024 + 1)
      await expect(
        swarm.invoke({ agents: [makeSpec('a', { systemPrompt: oversized })], initialInput: 'go' }, buildContext())
      ).rejects.toThrow(/systemPrompt exceeds size cap/)
    })

    it('rejects empty initialInput', async () => {
      await expect(swarm.invoke({ agents: [makeSpec('a')], initialInput: '' }, buildContext())).rejects.toThrow()
    })

    it('rejects duplicate agent names', async () => {
      await expect(
        swarm.invoke({ agents: [makeSpec('a'), makeSpec('a')], initialInput: 'go' }, buildContext())
      ).rejects.toThrow(/duplicate/)
    })
  })

  describe('tool allowlist', () => {
    it('rejects an unknown tool name', async () => {
      await expect(
        swarm.invoke(
          {
            agents: [makeSpec('a', { tools: ['bash'] })],
            initialInput: 'go',
          },
          buildContext({ parentTools: [] })
        )
      ).rejects.toThrow(/unknown tool/)
    })

    it('rejects a tool not in the parent registry', async () => {
      await expect(
        swarm.invoke(
          {
            agents: [makeSpec('a', { tools: ['evil'] })],
            initialInput: 'go',
          },
          buildContext({ parentTools: ['safe'] })
        )
      ).rejects.toThrow(/unknown tool 'evil'/)
    })

    it('rejects literal wildcard names (they are not expanded)', async () => {
      await expect(
        swarm.invoke(
          {
            agents: [makeSpec('a', { tools: ['*'] })],
            initialInput: 'go',
          },
          buildContext({ parentTools: ['safe'] })
        )
      ).rejects.toThrow(/unknown tool/)
    })

    it.each(['use_agent', 'swarm', 'graph', 'a2a_client'])(
      'rejects multi-agent tool name in a child spec: %s',
      async (multiagentTool) => {
        // Defense-in-depth: even if the parent registered a multi-agent tool,
        // the model cannot grant it to a child — that would let it bypass the
        // shared depth counter by having a child re-invoke `swarm`/`graph`/etc.
        await expect(
          swarm.invoke(
            {
              agents: [makeSpec('a', { tools: [multiagentTool] })],
              initialInput: 'go',
            },
            buildContext({ parentTools: [multiagentTool] })
          )
        ).rejects.toThrow(/multi-agent tool/)
      }
    )
  })

  describe('entryAgent', () => {
    it('rejects an entryAgent not in the list', async () => {
      await expect(
        swarm.invoke(
          {
            agents: [makeSpec('a'), makeSpec('b')],
            initialInput: 'go',
            entryAgent: 'c',
          },
          buildContext()
        )
      ).rejects.toThrow(/entryAgent 'c' not in agents list/)
    })
  })

  describe('happy path', () => {
    beforeEach(() => {
      swarmCtorMock.mockReset()
      invokeMock.mockReset()
    })

    it('constructs a Swarm with fixed safety caps and maps the result', async () => {
      invokeMock.mockResolvedValue(
        buildResult({
          nodeIds: ['researcher', 'writer'],
          finalText: 'Here is your report.',
          duration: 250,
        })
      )

      const result = await swarm.invoke(
        {
          agents: [
            makeSpec('researcher', { systemPrompt: 'You research.' }),
            makeSpec('writer', { systemPrompt: 'You write.' }),
          ],
          initialInput: 'Write a report on octopi.',
        },
        buildContext()
      )

      // Wrapper wired the caps we care about, not user-configurable.
      expect(swarmCtorMock).toHaveBeenCalledOnce()
      const options = swarmCtorMock.mock.calls[0]![0]
      expect(options.maxSteps).toBe(10)
      expect(options.timeout).toBe(300_000)
      expect(options.nodeTimeout).toBe(120_000)
      expect(options.nodes).toHaveLength(2)

      // The child swarm actually ran with the initial input, and cancellation
      // + depth counter were composed with the parent's state.
      expect(invokeMock).toHaveBeenCalledOnce()
      const invokeCall = invokeMock.mock.calls[0]!
      expect(invokeCall[0]).toBe('Write a report on octopi.')
      expect(invokeCall[1]).toHaveProperty('cancelSignal')
      expect(invokeCall[1].invocationState).toEqual({ [MULTIAGENT_DEPTH_KEY]: 1 })

      // Status is translated from SDK's `COMPLETED` into the shared dialect's `success`.
      expect(result).toEqual({
        status: 'success',
        output: 'Here is your report.',
        nodeHistory: ['researcher', 'writer'],
        executionCount: 2,
        executionTimeMs: 250,
        usage: { inputTokens: 6, outputTokens: 8, totalTokens: 14 },
      })
    })

    it('maps a failed result to error with an empty output', async () => {
      invokeMock.mockResolvedValue(buildResult({ nodeIds: [], status: 'FAILED' }))

      const result = await swarm.invoke({ agents: [makeSpec('loner')], initialInput: 'go' }, buildContext())
      // SDK's `FAILED` maps to shared dialect's `error`.
      expect(result.status).toBe('error')
      expect(result.output).toBe('')
      expect(result.nodeHistory).toEqual([])
    })

    it('maps an interrupted result to cancelled', async () => {
      // SDK's `INTERRUPTED` maps to the shared dialect's `cancelled`.
      invokeMock.mockResolvedValue(
        new MultiAgentResult({
          status: Status.INTERRUPTED as never,
          results: [],
          content: [],
          duration: 10,
        })
      )
      const result = await swarm.invoke({ agents: [makeSpec('a')], initialInput: 'go' }, buildContext())
      expect(result.status).toBe('cancelled')
    })

    it('maps a cancelled result to cancelled', async () => {
      invokeMock.mockResolvedValue(
        new MultiAgentResult({
          status: Status.CANCELLED as never,
          results: [],
          content: [],
          duration: 10,
        })
      )
      const result = await swarm.invoke({ agents: [makeSpec('a')], initialInput: 'go' }, buildContext())
      expect(result.status).toBe('cancelled')
    })

    it('forwards entryAgent as the swarm start id', async () => {
      invokeMock.mockResolvedValue(buildResult({ nodeIds: ['b'], finalText: 'ok' }))

      await swarm.invoke(
        {
          agents: [makeSpec('a'), makeSpec('b')],
          initialInput: 'go',
          entryAgent: 'b',
        },
        buildContext()
      )
      const options = swarmCtorMock.mock.calls[0]![0]
      expect(options.start).toBe('b')
    })
  })

  describe('cancellation', () => {
    beforeEach(() => {
      swarmCtorMock.mockReset()
      invokeMock.mockReset()
    })

    it('propagates the parent cancel signal to the child swarm', async () => {
      invokeMock.mockResolvedValue(buildResult({ nodeIds: ['a'], finalText: 'ok' }))

      await swarm.invoke({ agents: [makeSpec('a')], initialInput: 'go' }, buildContext({ cancelled: true }))

      const invokeCall = invokeMock.mock.calls[0]!
      const passedSignal = invokeCall[1].cancelSignal as AbortSignal
      expect(passedSignal.aborted).toBe(true)
    })

    it('rethrows an AbortError when the child rejects while the parent signal is aborted', async () => {
      // Drives the actual catch/rethrow path: the child Swarm rejects (as it
      // does when cancelSignal fires mid-invoke) and the wrapper translates
      // that into an AbortError so callers can distinguish cancellation from a
      // real failure via `error.name === 'AbortError'`.
      invokeMock.mockRejectedValue(new Error('child swarm aborted'))

      let caught: unknown
      try {
        await swarm.invoke({ agents: [makeSpec('a')], initialInput: 'go' }, buildContext({ cancelled: true }))
      } catch (err) {
        caught = err
      }
      expect(caught).toBeInstanceOf(Error)
      expect((caught as Error).name).toBe('AbortError')
    })

    it('rethrows the original error when the child rejects and the parent is not aborted', async () => {
      // Non-cancellation failures must pass through untouched — otherwise a
      // real bug in the swarm gets misreported as a cancellation.
      const original = new Error('real swarm failure')
      invokeMock.mockRejectedValue(original)

      await expect(
        swarm.invoke({ agents: [makeSpec('a')], initialInput: 'go' }, buildContext({ cancelled: false }))
      ).rejects.toBe(original)
    })
  })

  describe('recursion depth', () => {
    beforeEach(() => {
      swarmCtorMock.mockReset()
      invokeMock.mockReset()
    })

    it('refuses at the depth cap with a MultiagentDepthExceededError', async () => {
      let caught: unknown
      try {
        await swarm.invoke(
          { agents: [makeSpec('a')], initialInput: 'go' },
          buildContext({ invocationState: { [MULTIAGENT_DEPTH_KEY]: MAX_MULTIAGENT_DEPTH } })
        )
      } catch (err) {
        caught = err
      }
      expect(caught).toBeInstanceOf(MultiagentDepthExceededError)
      expect((caught as Error).message).toMatch(/recursion depth cap/)
    })

    it('forwards incremented depth to the child swarm', async () => {
      invokeMock.mockResolvedValue(buildResult({ nodeIds: ['a'], finalText: 'ok' }))

      await swarm.invoke(
        { agents: [makeSpec('a')], initialInput: 'go' },
        buildContext({ invocationState: { [MULTIAGENT_DEPTH_KEY]: 1 } })
      )

      const invokeCall = invokeMock.mock.calls[0]!
      expect(invokeCall[1].invocationState).toEqual({ [MULTIAGENT_DEPTH_KEY]: 2 })
    })

    it('treats garbage depth as zero', async () => {
      invokeMock.mockResolvedValue(buildResult({ nodeIds: ['a'], finalText: 'ok' }))

      await swarm.invoke(
        { agents: [makeSpec('a')], initialInput: 'go' },
        buildContext({ invocationState: { [MULTIAGENT_DEPTH_KEY]: 'not-an-int' } })
      )

      const invokeCall = invokeMock.mock.calls[0]!
      expect(invokeCall[1].invocationState).toEqual({ [MULTIAGENT_DEPTH_KEY]: 1 })
    })

    it('preserves the rest of the parent invocationState when incrementing depth', async () => {
      // Tracing / telemetry / per-run keys flow through — the tool overrides
      // only the shared depth counter, not the whole state bag.
      invokeMock.mockResolvedValue(buildResult({ nodeIds: ['a'], finalText: 'ok' }))

      await swarm.invoke(
        { agents: [makeSpec('a')], initialInput: 'go' },
        buildContext({ invocationState: { traceId: 'abc', userId: 'u1', [MULTIAGENT_DEPTH_KEY]: 1 } })
      )

      const invokeCall = invokeMock.mock.calls[0]!
      expect(invokeCall[1].invocationState).toEqual({
        traceId: 'abc',
        userId: 'u1',
        [MULTIAGENT_DEPTH_KEY]: 2,
      })
    })
  })

  describe('makeSwarm factory', () => {
    beforeEach(() => {
      swarmCtorMock.mockReset()
      invokeMock.mockReset()
    })

    it('accepts a custom name and tighter caps', async () => {
      const custom = makeSwarm({
        name: 'mini-swarm',
        maxAgents: 2,
        executionTimeoutMs: 60_000,
        nodeTimeoutMs: 30_000,
        maxSteps: 4,
      })
      expect(custom.name).toBe('mini-swarm')

      invokeMock.mockResolvedValue(buildResult({ nodeIds: ['a'], finalText: 'ok' }))
      await custom.invoke({ agents: [makeSpec('a')], initialInput: 'go' }, buildContext())

      const options = swarmCtorMock.mock.calls[0]![0]
      expect(options.maxSteps).toBe(4)
      expect(options.timeout).toBe(60_000)
      expect(options.nodeTimeout).toBe(30_000)
    })

    it('enforces the custom maxAgents cap', async () => {
      const custom = makeSwarm({ maxAgents: 2 })
      const agents = [makeSpec('a'), makeSpec('b'), makeSpec('c')]
      await expect(custom.invoke({ agents, initialInput: 'go' }, buildContext())).rejects.toThrow()
    })

    it('honours a custom maxMultiagentDepth', async () => {
      const custom = makeSwarm({ maxMultiagentDepth: 1 })
      await expect(
        custom.invoke(
          { agents: [makeSpec('a')], initialInput: 'go' },
          buildContext({ invocationState: { [MULTIAGENT_DEPTH_KEY]: 1 } })
        )
      ).rejects.toBeInstanceOf(MultiagentDepthExceededError)
    })
  })
})
