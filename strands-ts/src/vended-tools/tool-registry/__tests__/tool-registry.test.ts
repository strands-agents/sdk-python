import { describe, it, expect, beforeEach } from 'vitest'
import type { ToolContext } from '../../../tools/tool.js'
import { FunctionTool } from '../../../tools/function-tool.js'
import { McpTool } from '../../../tools/mcp-tool.js'
import type { McpClient } from '../../../mcp/index.js'
import { ToolRegistry } from '../../../registry/tool-registry.js'
import { createMockAgent } from '../../../__fixtures__/agent-helpers.js'
import { makeToolRegistry } from '../tool-registry.js'
import { MAX_DYNAMIC_TOOLS, type ListResult, type MutationResult } from '../types.js'

/**
 * Minimal fake of McpClient. The tool_registry tool only calls `listTools()`,
 * so this stub returns a fixed array of McpTool instances. We construct real
 * McpTool objects (they don't dial out until the agent invokes them) but hand
 * them this fake in place of a real client. `unknown as McpClient` is safe
 * here because the wire call goes through a separate McpClient instance the
 * real agent owns.
 */
function makeFakeClient(tools: { name: string; description?: string }[]): McpClient {
  const client = {} as McpClient
  const mcpTools = tools.map(
    (t) =>
      new McpTool({
        name: t.name,
        description: t.description ?? `Fake ${t.name}`,
        inputSchema: { type: 'object', properties: {} },
        client,
      })
  )
  ;(client as unknown as { listTools: () => Promise<McpTool[]> }).listTools = async () => mcpTools
  return client
}

function makeContext(registry: ToolRegistry): ToolContext {
  const agent = createMockAgent({ toolRegistry: registry })
  return {
    toolUse: { name: 'tool_registry', toolUseId: 'test-id', input: {} },
    agent,
    invocationState: {},
    interrupt: () => {
      throw new Error('interrupt not available in mock context')
    },
  }
}

function makeDevTool(name: string): FunctionTool {
  return new FunctionTool({
    name,
    description: `developer-registered ${name}`,
    callback: () => 'ok',
  })
}

describe('tool_registry tool', () => {
  let registry: ToolRegistry
  let context: ToolContext
  let client: McpClient

  beforeEach(() => {
    registry = new ToolRegistry([makeDevTool('dev_echo'), makeDevTool('dev_ping')])
    context = makeContext(registry)
    client = makeFakeClient([{ name: 'remote_alpha' }, { name: 'remote_beta' }, { name: 'remote_gamma' }])
  })

  describe('list operation', () => {
    it('lists developer-registered tools as not owned', async () => {
      const registryTool = makeToolRegistry({ mcpClients: { weather: client } })
      const result = (await registryTool.invoke({ operation: 'list' }, context)) as unknown as ListResult
      const names = Object.fromEntries(result.tools.map((t) => [t.name, t]))
      expect(names.dev_echo).toBeDefined()
      expect(names.dev_echo!.registeredByToolRegistry).toBe(false)
      expect(names.dev_ping!.registeredByToolRegistry).toBe(false)
      expect(result.dynamicCount).toBe(0)
      expect(result.dynamicLimit).toBe(MAX_DYNAMIC_TOOLS)
    })
  })

  describe('create operation', () => {
    it('registers an MCP tool and marks it owned', async () => {
      const registryTool = makeToolRegistry({ mcpClients: { weather: client } })
      const result = (await registryTool.invoke(
        {
          operation: 'create',
          toolName: 'alpha',
          source: 'weather',
          remoteName: 'remote_alpha',
        },
        context
      )) as unknown as MutationResult
      expect(result.operation).toBe('create')
      expect(result.name).toBe('alpha')
      expect(result.dynamicCount).toBe(1)
      expect(registry.get('alpha')).toBeDefined()
    })

    it('uses toolName as default remoteName', async () => {
      const registryTool = makeToolRegistry({ mcpClients: { weather: client } })
      const result = (await registryTool.invoke(
        { operation: 'create', toolName: 'remote_beta', source: 'weather' },
        context
      )) as unknown as MutationResult
      expect(result.name).toBe('remote_beta')
      expect(registry.get('remote_beta')).toBeDefined()
    })

    it('binds a local alias to the remote name for the wire call', async () => {
      // Regression: an earlier implementation set McpTool.name = localName but
      // callTool sends tool.name on the wire, so aliased tools sent the
      // wrong name to the server. The bound tool must carry remoteName.
      const registryTool = makeToolRegistry({ mcpClients: { weather: client } })
      await registryTool.invoke(
        { operation: 'create', toolName: 'alpha', source: 'weather', remoteName: 'remote_alpha' },
        context
      )
      const bound = registry.get('alpha') as McpTool
      expect(bound.name).toBe('alpha')
      expect(bound.remoteName).toBe('remote_alpha')
    })

    it('applies descriptionOverride to the resulting tool spec', async () => {
      const registryTool = makeToolRegistry({ mcpClients: { weather: client } })
      await registryTool.invoke(
        {
          operation: 'create',
          toolName: 'alpha',
          source: 'weather',
          remoteName: 'remote_alpha',
          descriptionOverride: 'custom text',
        },
        context
      )
      expect(registry.get('alpha')!.toolSpec.description).toBe('custom text')
    })

    it('rejects unknown source', async () => {
      const registryTool = makeToolRegistry({ mcpClients: { weather: client } })
      await expect(
        registryTool.invoke({ operation: 'create', toolName: 'alpha', source: 'not_a_real_client' }, context)
      ).rejects.toThrow(/unknown source/)
    })

    it('rejects unknown remote tool', async () => {
      const registryTool = makeToolRegistry({ mcpClients: { weather: client } })
      await expect(
        registryTool.invoke(
          {
            operation: 'create',
            toolName: 'alpha',
            source: 'weather',
            remoteName: 'does_not_exist',
          },
          context
        )
      ).rejects.toThrow(/not found on MCP server/)
    })

    it('rejects registering over a developer tool', async () => {
      const registryTool = makeToolRegistry({ mcpClients: { weather: client } })
      await expect(
        registryTool.invoke(
          {
            operation: 'create',
            toolName: 'dev_echo',
            source: 'weather',
            remoteName: 'remote_alpha',
          },
          context
        )
      ).rejects.toThrow(/already registered/)
    })

    it('cannot register a tool with the tool_registry name itself', async () => {
      const registryTool = makeToolRegistry({ mcpClients: { weather: client } })
      await expect(
        registryTool.invoke(
          {
            operation: 'create',
            toolName: 'tool_registry',
            source: 'weather',
            remoteName: 'remote_alpha',
          },
          context
        )
      ).rejects.toThrow(/itself/)
    })

    describe.each(['1_starts_with_digit', 'has-dash', 'has space', 'toolname!', 'x'.repeat(65)])(
      'rejects invalid tool name %j',
      (bad) => {
        it('throws ToolRegistryError', async () => {
          const registryTool = makeToolRegistry({ mcpClients: { weather: client } })
          await expect(
            registryTool.invoke(
              { operation: 'create', toolName: bad, source: 'weather', remoteName: 'remote_alpha' },
              context
            )
          ).rejects.toThrow(/invalid tool name/)
        })
      }
    )

    it('enforces the dynamic tool cap', async () => {
      const registryTool = makeToolRegistry({
        mcpClients: { weather: client },
        maxDynamicTools: 2,
      })
      await registryTool.invoke(
        { operation: 'create', toolName: 'alpha', source: 'weather', remoteName: 'remote_alpha' },
        context
      )
      await registryTool.invoke(
        { operation: 'create', toolName: 'beta', source: 'weather', remoteName: 'remote_beta' },
        context
      )
      await expect(
        registryTool.invoke(
          { operation: 'create', toolName: 'gamma', source: 'weather', remoteName: 'remote_gamma' },
          context
        )
      ).rejects.toThrow(/dynamic tool cap reached/)
    })

    it('enforces the dynamic tool cap under concurrent create calls', async () => {
      const cap = 2
      const registryTool = makeToolRegistry({
        mcpClients: { weather: client },
        maxDynamicTools: cap,
      })
      // Fire 2× cap concurrent creates against a shared context. Under a
      // check-then-await-then-write ordering all N would race past the cap.
      const results = await Promise.allSettled(
        Array.from({ length: cap * 2 }, (_, i) =>
          registryTool.invoke(
            { operation: 'create', toolName: `t${i}`, source: 'weather', remoteName: 'remote_alpha' },
            context
          )
        )
      )
      const fulfilled = results.filter((r) => r.status === 'fulfilled').length
      expect(fulfilled).toBe(cap)
      const rejected = results.filter((r) => r.status === 'rejected') as PromiseRejectedResult[]
      expect(rejected.length).toBe(cap)
      for (const r of rejected) {
        expect(String(r.reason)).toMatch(/dynamic tool cap reached|already registered/)
      }
    })

    it('does not orphan a reservation if a concurrent delete lands during create', async () => {
      // A controllable client: `listTools` blocks until the test resolves the
      // gate, so we can interleave a `delete` between `create`'s reservation
      // (synchronous) and its call to `registry.add` (post-await).
      let releaseListTools!: () => void
      const gate = new Promise<void>((resolve) => {
        releaseListTools = resolve
      })
      const slowClient = {} as McpClient
      const remoteTools = [
        new McpTool({
          name: 'remote_alpha',
          description: 'fake',
          inputSchema: { type: 'object', properties: {} },
          client: slowClient,
        }),
      ]
      ;(slowClient as unknown as { listTools: () => Promise<McpTool[]> }).listTools = async () => {
        await gate
        return remoteTools
      }

      const registryTool = makeToolRegistry({ mcpClients: { weather: slowClient } })

      // Task B: starts create; reserves 'alpha' synchronously, then awaits.
      const createPromise = registryTool.invoke(
        { operation: 'create', toolName: 'alpha', source: 'weather', remoteName: 'remote_alpha' },
        context
      )

      // Yield the microtask queue so the create task has actually entered
      // `resolveMcpTool` and is waiting on `gate`.
      await Promise.resolve()
      await Promise.resolve()

      // Task A: delete lands on the reservation and clears it.
      const deleteResult = (await registryTool.invoke(
        { operation: 'delete', toolName: 'alpha' },
        context
      )) as unknown as MutationResult
      expect(deleteResult.operation).toBe('delete')

      // Task B resumes. Under the fix it must abort rather than write a tool
      // it can no longer track.
      releaseListTools()
      await expect(createPromise).rejects.toThrow(/cancelled by a concurrent delete/)

      // Final state: no orphan in the SDK registry, and this instance's
      // ownership set is empty (dynamicCount == 0), proving the reservation
      // was released.
      expect(registry.get('alpha')).toBeUndefined()
      const list = (await registryTool.invoke({ operation: 'list' }, context)) as unknown as ListResult
      expect(list.dynamicCount).toBe(0)
    })
  })

  describe('update operation', () => {
    it('updates a tool this instance registered', async () => {
      const registryTool = makeToolRegistry({ mcpClients: { weather: client } })
      await registryTool.invoke(
        { operation: 'create', toolName: 'alpha', source: 'weather', remoteName: 'remote_alpha' },
        context
      )
      const result = (await registryTool.invoke(
        {
          operation: 'update',
          toolName: 'alpha',
          source: 'weather',
          remoteName: 'remote_beta',
          descriptionOverride: 'now bound to beta',
        },
        context
      )) as unknown as MutationResult
      expect(result.operation).toBe('update')
      // The description confirms the re-binding took effect.
      expect(registry.get('alpha')!.toolSpec.description).toBe('now bound to beta')
    })

    it('cannot update a developer-registered tool', async () => {
      const registryTool = makeToolRegistry({ mcpClients: { weather: client } })
      await expect(
        registryTool.invoke(
          {
            operation: 'update',
            toolName: 'dev_echo',
            source: 'weather',
            remoteName: 'remote_alpha',
          },
          context
        )
      ).rejects.toThrow(/developer-registered/)
    })

    it('does not resurrect a deleted tool if a concurrent delete lands during update', async () => {
      // Mirror the create-during-delete race for `update`. Without the guard,
      // a delete landing between `update`'s pre-await ownership check and its
      // post-await write would resurrect an entry the model believed deleted.
      let releaseListTools!: () => void
      const gate = new Promise<void>((resolve) => {
        releaseListTools = resolve
      })
      const slowClient = {} as McpClient
      const remoteTools = [
        new McpTool({
          name: 'remote_alpha',
          description: 'fake',
          inputSchema: { type: 'object', properties: {} },
          client: slowClient,
        }),
        new McpTool({
          name: 'remote_beta',
          description: 'fake',
          inputSchema: { type: 'object', properties: {} },
          client: slowClient,
        }),
      ]
      let gated = false
      ;(slowClient as unknown as { listTools: () => Promise<McpTool[]> }).listTools = async () => {
        if (gated) {
          await gate
        }
        return remoteTools
      }

      const registryTool = makeToolRegistry({ mcpClients: { weather: slowClient } })

      // First create; MCP lookup returns immediately.
      await registryTool.invoke(
        { operation: 'create', toolName: 'alpha', source: 'weather', remoteName: 'remote_alpha' },
        context
      )
      expect(registry.get('alpha')).toBeDefined()

      // From now on, listTools blocks.
      gated = true

      // Task B: start `update`; it passes the pre-await ownership check, then parks.
      const updatePromise = registryTool.invoke(
        { operation: 'update', toolName: 'alpha', source: 'weather', remoteName: 'remote_beta' },
        context
      )

      // Yield so the update task is actually awaiting on the gate.
      await Promise.resolve()
      await Promise.resolve()

      // Task A: delete lands while the update is still awaiting.
      const deleteResult = (await registryTool.invoke(
        { operation: 'delete', toolName: 'alpha' },
        context
      )) as unknown as MutationResult
      expect(deleteResult.operation).toBe('delete')

      // Release the update; it must abort rather than resurrect the tool.
      releaseListTools()
      await expect(updatePromise).rejects.toThrow(/cancelled by a concurrent delete/)

      // No orphan in the SDK registry; ownership set is empty.
      expect(registry.get('alpha')).toBeUndefined()
      const list = (await registryTool.invoke({ operation: 'list' }, context)) as unknown as ListResult
      expect(list.dynamicCount).toBe(0)
    })
  })

  describe('delete operation', () => {
    it('deletes a tool this instance registered', async () => {
      const registryTool = makeToolRegistry({ mcpClients: { weather: client } })
      await registryTool.invoke(
        { operation: 'create', toolName: 'alpha', source: 'weather', remoteName: 'remote_alpha' },
        context
      )
      const result = (await registryTool.invoke(
        { operation: 'delete', toolName: 'alpha' },
        context
      )) as unknown as MutationResult
      expect(result.operation).toBe('delete')
      expect(result.dynamicCount).toBe(0)
      expect(registry.get('alpha')).toBeUndefined()
    })

    it('cannot delete a developer-registered tool', async () => {
      const registryTool = makeToolRegistry({ mcpClients: { weather: client } })
      await expect(registryTool.invoke({ operation: 'delete', toolName: 'dev_echo' }, context)).rejects.toThrow(
        /developer-registered/
      )
      expect(registry.get('dev_echo')).toBeDefined()
    })

    it('cannot delete a tool that was never registered', async () => {
      const registryTool = makeToolRegistry({ mcpClients: { weather: client } })
      await expect(registryTool.invoke({ operation: 'delete', toolName: 'never_registered' }, context)).rejects.toThrow(
        /developer-registered/
      )
    })

    it('does not leak ownership between agents sharing the tool instance', async () => {
      const registryTool = makeToolRegistry({ mcpClients: { weather: client } })

      const regA = new ToolRegistry()
      const regB = new ToolRegistry()
      const ctxA = makeContext(regA)
      const ctxB = makeContext(regB)

      await registryTool.invoke(
        { operation: 'create', toolName: 'alpha', source: 'weather', remoteName: 'remote_alpha' },
        ctxA
      )

      // Even though agent A has 'alpha', from agent B's perspective it was
      // never registered via tool_registry — deletion must be rejected.
      await expect(registryTool.invoke({ operation: 'delete', toolName: 'alpha' }, ctxB)).rejects.toThrow(
        /developer-registered/
      )
    })
  })

  describe('operation dispatch', () => {
    it('degrades to read-only when no MCP clients are configured', async () => {
      const registryTool = makeToolRegistry()
      const list = (await registryTool.invoke({ operation: 'list' }, context)) as unknown as ListResult
      expect(list.dynamicLimit).toBe(MAX_DYNAMIC_TOOLS)

      await expect(
        registryTool.invoke({ operation: 'create', toolName: 'alpha', source: 'anything' }, context)
      ).rejects.toThrow(/unknown source/)
    })
  })

  describe('factory validation', () => {
    it('rejects zero maxDynamicTools', () => {
      expect(() => makeToolRegistry({ maxDynamicTools: 0 })).toThrow(/at least 1/)
    })

    it('honors custom name and description', () => {
      const t = makeToolRegistry({ name: 'registry_ctl', description: 'custom desc' })
      expect(t.name).toBe('registry_ctl')
      expect(t.toolSpec.description).toBe('custom desc')
    })
  })
})
