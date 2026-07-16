/**
 * Tests for the graph vended tool.
 *
 * The tool is a thin shim over `Graph`. Security-focused tests hit the
 * validation surface (bad topology, oversized inputs, unknown tools,
 * wildcards, cycles); happy-path tests exercise a small end-to-end DAG
 * against `MockMessageModel` sub-agents.
 */

import { describe, expect, it } from 'vitest'
import { Agent } from '../../../agent/agent.js'
import { MockMessageModel } from '../../../__fixtures__/mock-message-model.js'
import { Graph } from '../../../multiagent/graph.js'
import { TextBlock } from '../../../types/messages.js'
import type { ToolContext } from '../../../tools/tool.js'
import { graph, makeGraph, MAX_MULTIAGENT_DEPTH, MULTIAGENT_DEPTH_KEY } from '../graph.js'
import { MAX_NODES, MAX_INITIAL_INPUT_LENGTH, MAX_SYSTEM_PROMPT_LENGTH, MAX_TOOLS_PER_NODE } from '../types.js'

/**
 * Build a parent agent whose model produces `count` deterministic replies.
 * Each sub-agent inside the graph will consume one reply from the shared
 * `MockMessageModel`, so callers pass `count = numberOfNodes` in happy paths.
 */
function makeParent(count: number): Agent {
  const model = new MockMessageModel()
  for (let i = 0; i < count; i++) {
    model.addTurn(new TextBlock(`reply ${i}`))
  }
  return new Agent({ model, printer: false })
}

/**
 * Real tool context so `context.agent.toolRegistry` / `cancelSignal` behave the
 * same way they do in production.
 */
function toolContext(agent: Agent): ToolContext {
  return {
    toolUse: { name: 'graph', toolUseId: 'test-id', input: {} },
    agent,
    invocationState: {},
    interrupt: () => {
      throw new Error('interrupt not available in tool tests')
    },
  }
}

describe('graph tool — validation (security surface)', () => {
  it('rejects a cycle', async () => {
    const agent = makeParent(0)
    await expect(
      graph.invoke(
        {
          nodes: [{ id: 'a' }, { id: 'b' }],
          edges: [
            { fromId: 'a', toId: 'b' },
            { fromId: 'b', toId: 'a' },
          ],
          initialInput: 'go',
        },
        toolContext(agent)
      )
    ).rejects.toThrow(/cycle/i)
  })

  it('rejects a self-loop', async () => {
    const agent = makeParent(0)
    await expect(
      graph.invoke(
        {
          nodes: [{ id: 'a' }],
          edges: [{ fromId: 'a', toId: 'a' }],
          initialInput: 'go',
        },
        toolContext(agent)
      )
    ).rejects.toThrow(/Self-loop/)
  })

  it('rejects a tool name not registered on the parent', async () => {
    const agent = makeParent(0)
    await expect(
      graph.invoke(
        {
          nodes: [{ id: 'a', tools: ['definitely_not_a_real_tool'] }],
          edges: [],
          initialInput: 'go',
        },
        toolContext(agent)
      )
    ).rejects.toThrow(/not registered on the parent/)
  })

  it('rejects a wildcard in a node tool allow-list', async () => {
    const agent = makeParent(0)
    await expect(
      graph.invoke(
        {
          nodes: [{ id: 'a', tools: ['*'] }],
          edges: [],
          initialInput: 'go',
        },
        toolContext(agent)
      )
    ).rejects.toThrow(/wildcard/i)
  })

  it('rejects an oversized per-node tool allow-list', async () => {
    const agent = makeParent(0)
    const tools = Array.from({ length: MAX_TOOLS_PER_NODE + 1 }, () => 'not_a_tool')
    await expect(
      graph.invoke(
        {
          nodes: [{ id: 'a', tools }],
          edges: [],
          initialInput: 'go',
        },
        toolContext(agent)
      )
    ).rejects.toThrow(/tools/)
  })

  it('rejects too many nodes', async () => {
    const agent = makeParent(0)
    const nodes = Array.from({ length: MAX_NODES + 1 }, (_, i) => ({ id: `n${i}` }))
    await expect(graph.invoke({ nodes, edges: [], initialInput: 'go' }, toolContext(agent))).rejects.toThrow(
      /Too many nodes/
    )
  })

  it('rejects duplicate node ids', async () => {
    const agent = makeParent(0)
    await expect(
      graph.invoke(
        {
          nodes: [{ id: 'a' }, { id: 'a' }],
          edges: [],
          initialInput: 'go',
        },
        toolContext(agent)
      )
    ).rejects.toThrow(/Duplicate node id/)
  })

  it('rejects edges to unknown nodes', async () => {
    const agent = makeParent(0)
    await expect(
      graph.invoke(
        {
          nodes: [{ id: 'a' }],
          edges: [{ fromId: 'a', toId: 'ghost' }],
          initialInput: 'go',
        },
        toolContext(agent)
      )
    ).rejects.toThrow(/unknown/)
  })

  it('rejects oversized initialInput', async () => {
    const agent = makeParent(0)
    await expect(
      graph.invoke(
        {
          nodes: [{ id: 'a' }],
          edges: [],
          initialInput: 'x'.repeat(MAX_INITIAL_INPUT_LENGTH + 1),
        },
        toolContext(agent)
      )
    ).rejects.toThrow(/initialInput/)
  })

  it('rejects oversized systemPrompt', async () => {
    const agent = makeParent(0)
    await expect(
      graph.invoke(
        {
          nodes: [{ id: 'a', systemPrompt: 'x'.repeat(MAX_SYSTEM_PROMPT_LENGTH + 1) }],
          edges: [],
          initialInput: 'go',
        },
        toolContext(agent)
      )
    ).rejects.toThrow(/systemPrompt/)
  })

  it('rejects a node id containing forbidden characters', async () => {
    const agent = makeParent(0)
    await expect(
      graph.invoke(
        {
          nodes: [{ id: 'bad id!' }],
          edges: [],
          initialInput: 'go',
        },
        toolContext(agent)
      )
    ).rejects.toThrow()
  })
})

describe('graph tool — happy path', () => {
  it('runs a linear chain a -> b and returns per-node text', async () => {
    // Two sub-agents, one model turn each.
    const agent = makeParent(2)
    const result = await graph.invoke(
      {
        nodes: [{ id: 'a' }, { id: 'b' }],
        edges: [{ fromId: 'a', toId: 'b' }],
        initialInput: 'start',
      },
      toolContext(agent)
    )

    expect(result.status).toBe('completed')
    expect(Object.keys(result.results).sort()).toEqual(['a', 'b'])
    const nodeA = result.results['a']!
    const nodeB = result.results['b']!
    expect(nodeA.status).toBe('completed')
    expect(nodeB.status).toBe('completed')
    expect(result.executionOrder).toEqual(['a', 'b'])
    // `output` is the sole terminal (leaf) node's output — here that's `b`.
    expect(result.output).toBe(nodeB.output)
  })

  it('runs a diamond a -> b/c -> d', async () => {
    // Four sub-agents, one model turn each.
    const agent = makeParent(4)
    const result = await graph.invoke(
      {
        nodes: [{ id: 'a' }, { id: 'b' }, { id: 'c' }, { id: 'd' }],
        edges: [
          { fromId: 'a', toId: 'b' },
          { fromId: 'a', toId: 'c' },
          { fromId: 'b', toId: 'd' },
          { fromId: 'c', toId: 'd' },
        ],
        initialInput: 'start',
      },
      toolContext(agent)
    )

    expect(result.status).toBe('completed')
    expect(Object.keys(result.results).sort()).toEqual(['a', 'b', 'c', 'd'])
    // a is first, d is last; b and c can be in either order.
    expect(result.executionOrder[0]).toBe('a')
    expect(result.executionOrder.at(-1)).toBe('d')
  })
})

describe('graph tool — multi-agent depth cap', () => {
  it('refuses to start when the shared depth cap is reached', async () => {
    const agent = makeParent(0)
    const ctx: ToolContext = {
      toolUse: { name: 'graph', toolUseId: 'test-id', input: {} },
      agent,
      invocationState: { [MULTIAGENT_DEPTH_KEY]: MAX_MULTIAGENT_DEPTH },
      interrupt: () => {
        throw new Error('interrupt not available in tool tests')
      },
    }
    await expect(graph.invoke({ nodes: [{ id: 'a' }], edges: [], initialInput: 'go' }, ctx)).rejects.toThrow(
      /recursion depth cap/
    )
  })

  it('honours a lower cap set via makeGraph', async () => {
    // A factory-configured cap is the only way to change the recursion limit
    // (nothing surfaces to the model). Set it to 1 and start a call already at
    // depth 1 — the tool must refuse.
    const shallow = makeGraph({ maxDepth: 1 })
    const agent = makeParent(0)
    const ctx: ToolContext = {
      toolUse: { name: 'graph', toolUseId: 'test-id', input: {} },
      agent,
      invocationState: { [MULTIAGENT_DEPTH_KEY]: 1 },
      interrupt: () => {
        throw new Error('interrupt not available in tool tests')
      },
    }
    await expect(shallow.invoke({ nodes: [{ id: 'a' }], edges: [], initialInput: 'go' }, ctx)).rejects.toThrow(
      /recursion depth cap of 1/
    )
  })

  it('threads depth + 1 into the child Graph.invoke invocationState', async () => {
    const agent = makeParent(1)
    const ctx: ToolContext = {
      toolUse: { name: 'graph', toolUseId: 'test-id', input: {} },
      agent,
      invocationState: { [MULTIAGENT_DEPTH_KEY]: 1 },
      interrupt: () => {
        throw new Error('interrupt not available in tool tests')
      },
    }

    // Monkeypatch `Graph.prototype.invoke` to capture the options threaded in
    // by the tool without disturbing execution. Direct patching (rather than
    // `vi.spyOn`) is necessary because the tool constructs `new Graph(...)` at
    // call time and the prototype patch must be visible on that instance.
    const original = Graph.prototype.invoke
    let capturedOptions: unknown = null
    Graph.prototype.invoke = async function patched(this: Graph, input, options) {
      capturedOptions = options
      return original.call(this, input, options)
    }
    try {
      await graph.invoke({ nodes: [{ id: 'a' }], edges: [], initialInput: 'go' }, ctx)
    } finally {
      Graph.prototype.invoke = original
    }

    expect(capturedOptions).toBeDefined()
    expect((capturedOptions as { invocationState?: Record<string, unknown> })?.invocationState).toEqual({
      [MULTIAGENT_DEPTH_KEY]: 2,
    })
  })
})

describe('graph tool — cancellation', () => {
  it("returns cancelled when the parent agent's cancelSignal is already aborted", async () => {
    const model = new MockMessageModel().addTurn(new TextBlock('reply'))
    const agent = new Agent({ model, printer: false })
    // Reach into the agent internals to pre-abort. The tool must not proceed
    // — and per spec, it must return `{status: 'cancelled'}` rather than raise.

    ;(agent as any)._abortController.abort()

    ;(agent as any)._abortSignal = (agent as any)._abortController.signal

    const result = await graph.invoke(
      {
        nodes: [{ id: 'a' }],
        edges: [],
        initialInput: 'go',
      },
      toolContext(agent)
    )
    expect(result.status).toBe('cancelled')
    expect(result.output).toBe('')
    expect(result.executionOrder).toEqual([])
    expect(result.results).toEqual({})
  })

  it('returns cancelled when the SDK Graph throws mid-flight and the parent signal is aborted', async () => {
    // The SDK's Graph throws `graph cancelled by external signal` when its
    // external cancelSignal aborts mid-flight (see multiagent/graph.ts:343).
    // The tool must catch that throw and return `{status: 'cancelled'}` per
    // spec — it must not propagate the exception past the tool boundary.
    // Monkey-patch `Graph.prototype.invoke` to arm the parent's cancel signal
    // and throw the SDK's mid-flight cancel error deterministically. Without
    // the round-3 catch, the tool would re-throw and this test would fail.
    const model = new MockMessageModel().addTurn(new TextBlock('reply'))
    const agent = new Agent({ model, printer: false })

    const original = Graph.prototype.invoke
    Graph.prototype.invoke = async function patchedInvoke(this: Graph) {
      // Abort the parent so the tool's catch handler observes it and knows the
      // throw was cancel-caused rather than a real failure.
      ;(agent as any)._abortController.abort()
      throw new Error(`graph_id=<${this.id}> | graph cancelled by external signal`)
    }
    try {
      const result = await graph.invoke(
        {
          nodes: [{ id: 'a' }, { id: 'b' }],
          edges: [{ fromId: 'a', toId: 'b' }],
          initialInput: 'go',
        },
        toolContext(agent)
      )
      expect(result.status).toBe('cancelled')
      expect(result.output).toBe('')
    } finally {
      Graph.prototype.invoke = original
    }
  })

  it('propagates non-cancellation errors from Graph.invoke past the tool boundary', async () => {
    // Verify the catch is scoped correctly: real failures still propagate.
    // Without this, the previous test could pass trivially by swallowing all
    // errors — this pins down that only cancellation-shaped errors turn into
    // a cancelled result.
    const model = new MockMessageModel().addTurn(new TextBlock('reply'))
    const agent = new Agent({ model, printer: false })

    const original = Graph.prototype.invoke
    Graph.prototype.invoke = async function throwsRealError() {
      throw new Error('some unrelated failure')
    }
    try {
      await expect(
        graph.invoke({ nodes: [{ id: 'a' }], edges: [], initialInput: 'go' }, toolContext(agent))
      ).rejects.toThrow(/some unrelated failure/)
    } finally {
      Graph.prototype.invoke = original
    }
  })
})

describe('graph tool — multi-agent tool rejection', () => {
  it('rejects a graph node that references a multi-agent tool by name', async () => {
    // Register the graph tool itself on the parent so `graph` resolves in the
    // registry — otherwise the "unknown tool" check would fire first and mask
    // this rejection.
    const model = new MockMessageModel().addTurn(new TextBlock('reply'))
    const agent = new Agent({ model, printer: false, tools: [graph] })

    await expect(
      graph.invoke(
        {
          nodes: [{ id: 'a', tools: ['graph'] }],
          edges: [],
          initialInput: 'go',
        },
        toolContext(agent)
      )
    ).rejects.toThrow(/multi-agent tool/)
  })
})
