import { describe, expect, it, vi } from 'vitest'
import { CedarAuthorization } from '../cedar.js'
import { Agent } from '../../../agent/agent.js'
import { MockMessageModel } from '../../../__fixtures__/mock-message-model.js'
import { createMockTool } from '../../../__fixtures__/tool-helpers.js'
import { resolve } from 'node:path'

const FIXTURES = resolve(import.meta.dirname!, 'fixtures')

describe('CedarAuthorization', () => {
  describe('real Cedar evaluation', () => {

    const entities = [
      { uid: { type: 'Resource', id: 'agent' }, attrs: {}, parents: [] },
      { uid: { type: 'User', id: 'alice' }, attrs: { role: 'admin' }, parents: [] },
      { uid: { type: 'User', id: 'bob' }, attrs: { role: 'analyst' }, parents: [] },
      { uid: { type: 'User', id: 'eve' }, attrs: { role: 'viewer' }, parents: [] },
    ]

    it('allows permitted tool calls', async () => {
      const model = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'search', toolUseId: 'tool-1', input: { query: 'test' } })
        .addTurn({ type: 'textBlock', text: 'Done' })

      let toolExecuted = false
      const tool = createMockTool('search', () => {
        toolExecuted = true
        return 'results'
      })

      const cedar = new CedarAuthorization({
        policies: `${FIXTURES}/test.cedar`,
        entities,
        principalResolver: (state) => {
          if (!state.user_id) return undefined
          return { type: 'User', id: String(state.user_id) }
        },
      })

      const agent = new Agent({ model, tools: [tool], interventions: [cedar], printer: false })
      const result = await agent.invoke('Search', { invocationState: { user_id: 'alice' } })

      expect(result.stopReason).toBe('endTurn')
      expect(toolExecuted).toBe(true)
    })

    it('denies tools not in any permit policy (default-deny)', async () => {
      const model = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'delete_record', toolUseId: 'tool-1', input: { id: '1' } })
        .addTurn({ type: 'textBlock', text: 'Ok' })

      let toolExecuted = false
      const tool = createMockTool('delete_record', () => {
        toolExecuted = true
        return 'deleted'
      })

      const cedar = new CedarAuthorization({
        policies: `${FIXTURES}/test.cedar`,
        entities,
        principalResolver: () => ({ type: 'User', id: 'alice' }),
      })

      const agent = new Agent({ model, tools: [tool], interventions: [cedar], printer: false })
      await agent.invoke('Delete it', { invocationState: {} })

      expect(toolExecuted).toBe(false)
    })

    it('enforces role-based access (admin can delete, analyst cannot)', async () => {
      const model = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'delete_record', toolUseId: 'tool-1', input: {} })
        .addTurn({ type: 'textBlock', text: 'Done' })

      let toolExecuted = false
      const tool = createMockTool('delete_record', () => {
        toolExecuted = true
        return 'deleted'
      })

      // Admin can delete
      const cedarAdmin = new CedarAuthorization({
        policies: `${FIXTURES}/role-based.cedar`,
        entities,
        principalResolver: () => ({ type: 'User', id: 'alice' }),
      })

      const agentAdmin = new Agent({ model, tools: [tool], interventions: [cedarAdmin], printer: false })
      await agentAdmin.invoke('Delete', { invocationState: {} })
      expect(toolExecuted).toBe(true)

      // Analyst cannot delete
      toolExecuted = false
      const model2 = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'delete_record', toolUseId: 'tool-2', input: {} })
        .addTurn({ type: 'textBlock', text: 'Denied' })

      const cedarAnalyst = new CedarAuthorization({
        policies: `${FIXTURES}/role-based.cedar`,
        entities,
        principalResolver: () => ({ type: 'User', id: 'bob' }),
      })

      const agentAnalyst = new Agent({ model: model2, tools: [tool], interventions: [cedarAnalyst], printer: false })
      await agentAnalyst.invoke('Delete', { invocationState: {} })
      expect(toolExecuted).toBe(false)
    })

    it('enforces role-based access (analyst can search, viewer cannot)', async () => {
      const model = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'search', toolUseId: 'tool-1', input: {} })
        .addTurn({ type: 'textBlock', text: 'Done' })

      let toolExecuted = false
      const tool = createMockTool('search', () => {
        toolExecuted = true
        return 'found'
      })

      // Analyst can search
      const cedarAnalyst = new CedarAuthorization({
        policies: `${FIXTURES}/role-based.cedar`,
        entities,
        principalResolver: () => ({ type: 'User', id: 'bob' }),
      })

      const agent1 = new Agent({ model, tools: [tool], interventions: [cedarAnalyst], printer: false })
      await agent1.invoke('Search', { invocationState: {} })
      expect(toolExecuted).toBe(true)

      // Viewer cannot search
      toolExecuted = false
      const model2 = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'search', toolUseId: 'tool-2', input: {} })
        .addTurn({ type: 'textBlock', text: 'Denied' })

      const cedarViewer = new CedarAuthorization({
        policies: `${FIXTURES}/role-based.cedar`,
        entities,
        principalResolver: () => ({ type: 'User', id: 'eve' }),
      })

      const agent2 = new Agent({ model: model2, tools: [tool], interventions: [cedarViewer], printer: false })
      await agent2.invoke('Search', { invocationState: {} })
      expect(toolExecuted).toBe(false)
    })

    it('enforces rate limits via call_count in session context', async () => {
      const model = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'send_email', toolUseId: 'tool-1', input: {} })
        .addTurn({ type: 'toolUseBlock', name: 'send_email', toolUseId: 'tool-2', input: {} })
        .addTurn({ type: 'toolUseBlock', name: 'send_email', toolUseId: 'tool-3', input: {} })
        .addTurn({ type: 'toolUseBlock', name: 'send_email', toolUseId: 'tool-4', input: {} })
        .addTurn({ type: 'textBlock', text: 'Done' })

      let callCount = 0
      const tool = createMockTool('send_email', () => {
        callCount++
        return 'sent'
      })

      const cedar = new CedarAuthorization({
        policies: `${FIXTURES}/rate-limited.cedar`,
        entities,
        principalResolver: () => ({ type: 'User', id: 'alice' }),
      })

      const agent = new Agent({ model, tools: [tool], interventions: [cedar], printer: false })
      await agent.invoke('Send 4 emails', { invocationState: {} })

      // Policy allows call_count < 3, so calls 1 and 2 succeed, 3+ denied
      expect(callCount).toBe(2)
    })

    it('enforces environment restrictions', async () => {
      const model = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'search', toolUseId: 'tool-1', input: {} })
        .addTurn({ type: 'textBlock', text: 'Done' })

      let toolExecuted = false
      const tool = createMockTool('search', () => {
        toolExecuted = true
        return 'results'
      })

      // Non-production: allowed
      const cedar = new CedarAuthorization({
        policies: `${FIXTURES}/env-restricted.cedar`,
        entities,
        principalResolver: () => ({ type: 'User', id: 'alice' }),
      })

      const agent = new Agent({ model, tools: [tool], interventions: [cedar], printer: false })
      await agent.invoke('Search', { invocationState: { environment: 'development' } })
      expect(toolExecuted).toBe(true)

      // Production: denied
      toolExecuted = false
      const model2 = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'search', toolUseId: 'tool-2', input: {} })
        .addTurn({ type: 'textBlock', text: 'Denied' })

      const agent2 = new Agent({ model: model2, tools: [tool], interventions: [cedar], printer: false })
      await agent2.invoke('Search', { invocationState: { environment: 'production' } })
      expect(toolExecuted).toBe(false)
    })

    it('denies when principal is missing (fail-closed)', async () => {
      const model = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'search', toolUseId: 'tool-1', input: {} })
        .addTurn({ type: 'textBlock', text: 'Ok' })

      let toolExecuted = false
      const tool = createMockTool('search', () => {
        toolExecuted = true
        return 'results'
      })

      const cedar = new CedarAuthorization({
        policies: `${FIXTURES}/test.cedar`,
        entities,
        principalResolver: (state) => {
          if (!state.user_id) return undefined
          return { type: 'User', id: String(state.user_id) }
        },
      })

      const agent = new Agent({ model, tools: [tool], interventions: [cedar], printer: false })
      await agent.invoke('Search', { invocationState: {} })
      expect(toolExecuted).toBe(false)
    })

    it('denies on malformed policy (evaluation failure)', async () => {
      const model = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'search', toolUseId: 'tool-1', input: {} })
        .addTurn({ type: 'textBlock', text: 'Ok' })

      let toolExecuted = false
      const tool = createMockTool('search', () => {
        toolExecuted = true
        return 'results'
      })

      const cedar = new CedarAuthorization({
        policies: 'this is not valid cedar syntax at all!!!',
        entities,
        principalResolver: () => ({ type: 'User', id: 'alice' }),
      })

      const agent = new Agent({ model, tools: [tool], interventions: [cedar], printer: false })
      await agent.invoke('Search', { invocationState: {} })
      expect(toolExecuted).toBe(false)
    })
  })

  describe('resource resolution', () => {
    it('defaults resource to Resource::"agent"', async () => {
      const model = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'search', toolUseId: 'tool-1', input: {} })
        .addTurn({ type: 'textBlock', text: 'Done' })

      let toolExecuted = false
      const tool = createMockTool('search', () => {
        toolExecuted = true
        return 'ok'
      })

      // Policy permits any resource — works with the default
      const cedar = new CedarAuthorization({
        policies: 'permit(principal, action == Action::"search", resource);',
        entities: [{ uid: { type: 'Resource', id: 'agent' }, attrs: {}, parents: [] }],
        principalResolver: () => ({ type: 'User', id: 'alice' }),
      })

      const agent = new Agent({ model, tools: [tool], interventions: [cedar], printer: false })
      await agent.invoke('Go', { invocationState: {} })
      expect(toolExecuted).toBe(true)
    })

    it('uses record-based resource resolver', async () => {
      const model = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'delete', toolUseId: 'tool-1', input: { record_id: '42' } })
        .addTurn({ type: 'textBlock', text: 'Done' })

      let toolExecuted = false
      const tool = createMockTool('delete', () => {
        toolExecuted = true
        return 'deleted'
      })

      // Policy permits deleting Record::"42" specifically
      const cedar = new CedarAuthorization({
        policies: 'permit(principal, action == Action::"delete", resource == Record::"42");',
        entities: [{ uid: { type: 'Record', id: '42' }, attrs: {}, parents: [] }],
        principalResolver: () => ({ type: 'User', id: 'alice' }),
        resourceResolver: { delete: { key: 'record_id', type: 'Record' } },
      })

      const agent = new Agent({ model, tools: [tool], interventions: [cedar], printer: false })
      await agent.invoke('Delete', { invocationState: {} })
      expect(toolExecuted).toBe(true)
    })

    it('denies when resource resolver maps to unauthorized resource', async () => {
      const model = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'delete', toolUseId: 'tool-1', input: { record_id: '99' } })
        .addTurn({ type: 'textBlock', text: 'Denied' })

      let toolExecuted = false
      const tool = createMockTool('delete', () => {
        toolExecuted = true
        return 'deleted'
      })

      // Policy only permits Record::"42"
      const cedar = new CedarAuthorization({
        policies: 'permit(principal, action == Action::"delete", resource == Record::"42");',
        entities: [
          { uid: { type: 'Record', id: '42' }, attrs: {}, parents: [] },
          { uid: { type: 'Record', id: '99' }, attrs: {}, parents: [] },
        ],
        principalResolver: () => ({ type: 'User', id: 'alice' }),
        resourceResolver: { delete: { key: 'record_id', type: 'Record' } },
      })

      const agent = new Agent({ model, tools: [tool], interventions: [cedar], printer: false })
      await agent.invoke('Delete 99', { invocationState: {} })
      expect(toolExecuted).toBe(false)
    })
  })

  describe('context enricher', () => {
    it('adds custom fields usable in policies', async () => {
      const model = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'search', toolUseId: 'tool-1', input: {} })
        .addTurn({ type: 'textBlock', text: 'Done' })

      let toolExecuted = false
      const tool = createMockTool('search', () => {
        toolExecuted = true
        return 'ok'
      })

      // Policy checks custom context field
      const cedar = new CedarAuthorization({
        policies: 'permit(principal, action, resource) when { context.session.department == "engineering" };',
        entities: [{ uid: { type: 'Resource', id: 'agent' }, attrs: {}, parents: [] }],
        principalResolver: () => ({ type: 'User', id: 'alice' }),
        contextEnricher: () => ({ department: 'engineering' }),
      })

      const agent = new Agent({ model, tools: [tool], interventions: [cedar], printer: false })
      await agent.invoke('Go', { invocationState: {} })
      expect(toolExecuted).toBe(true)
    })

    it('denies when enricher value does not match policy', async () => {
      const model = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'search', toolUseId: 'tool-1', input: {} })
        .addTurn({ type: 'textBlock', text: 'Denied' })

      let toolExecuted = false
      const tool = createMockTool('search', () => {
        toolExecuted = true
        return 'ok'
      })

      const cedar = new CedarAuthorization({
        policies: 'permit(principal, action, resource) when { context.session.department == "engineering" };',
        entities: [{ uid: { type: 'Resource', id: 'agent' }, attrs: {}, parents: [] }],
        principalResolver: () => ({ type: 'User', id: 'alice' }),
        contextEnricher: () => ({ department: 'marketing' }),
      })

      const agent = new Agent({ model, tools: [tool], interventions: [cedar], printer: false })
      await agent.invoke('Go', { invocationState: {} })
      expect(toolExecuted).toBe(false)
    })
  })

  describe('onError behavior', () => {
    it('throws by default when handler errors', async () => {
      vi.mock('@cedar-policy/cedar-wasm/nodejs', async (importOriginal) => {
        const orig = await importOriginal<typeof import('@cedar-policy/cedar-wasm/nodejs')>()
        return { ...orig }
      })

      const model = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'tool', toolUseId: 'tool-1', input: {} })
        .addTurn({ type: 'textBlock', text: 'Done' })

      const tool = createMockTool('tool', () => 'ok')

      // principalResolver throws
      const cedar = new CedarAuthorization({
        policies: 'permit(principal, action, resource);',
        principalResolver: () => {
          throw new Error('resolver crash')
        },
        onError: 'throw',
      })

      const agent = new Agent({ model, tools: [tool], interventions: [cedar], printer: false })
      await expect(agent.invoke('Go', { invocationState: {} })).rejects.toThrow('resolver crash')
    })

    it('denies when onError is "deny" and handler throws', async () => {
      const model = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'tool', toolUseId: 'tool-1', input: {} })
        .addTurn({ type: 'textBlock', text: 'Done' })

      let toolExecuted = false
      const tool = createMockTool('tool', () => {
        toolExecuted = true
        return 'ok'
      })

      const cedar = new CedarAuthorization({
        policies: 'permit(principal, action, resource);',
        principalResolver: () => {
          throw new Error('resolver crash')
        },
        onError: 'deny',
      })

      const agent = new Agent({ model, tools: [tool], interventions: [cedar], printer: false })
      const result = await agent.invoke('Go', { invocationState: {} })
      expect(result.stopReason).toBe('endTurn')
      expect(toolExecuted).toBe(false)
    })

    it('proceeds when onError is "proceed" and handler throws', async () => {
      const model = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'tool', toolUseId: 'tool-1', input: {} })
        .addTurn({ type: 'textBlock', text: 'Done' })

      let toolExecuted = false
      const tool = createMockTool('tool', () => {
        toolExecuted = true
        return 'ok'
      })

      const cedar = new CedarAuthorization({
        policies: 'permit(principal, action, resource);',
        principalResolver: () => {
          throw new Error('resolver crash')
        },
        onError: 'proceed',
      })

      const agent = new Agent({ model, tools: [tool], interventions: [cedar], printer: false })
      const result = await agent.invoke('Go', { invocationState: {} })
      expect(result.stopReason).toBe('endTurn')
      expect(toolExecuted).toBe(true)
    })
  })

  describe('file-based config', () => {
    it('reads .cedar file from disk', async () => {
      const fixturesDir = import.meta.url.replace('file://', '').replace('/cedar.test.ts', '/fixtures')

      const model = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'search', toolUseId: 'tool-1', input: {} })
        .addTurn({ type: 'textBlock', text: 'Done' })

      let toolExecuted = false
      const tool = createMockTool('search', () => {
        toolExecuted = true
        return 'ok'
      })

      const cedar = new CedarAuthorization({
        policies: `${fixturesDir}/test.cedar`,
        entities: [
          { uid: { type: 'Resource', id: 'agent' }, attrs: {}, parents: [] },
          { uid: { type: 'User', id: 'alice' }, attrs: { role: 'analyst' }, parents: [] },
        ],
        principalResolver: () => ({ type: 'User', id: 'alice' }),
      })

      const agent = new Agent({ model, tools: [tool], interventions: [cedar], printer: false })
      await agent.invoke('Search', { invocationState: {} })
      expect(toolExecuted).toBe(true)
    })

    it('reads .json entity file from disk', async () => {
      const fixturesDir = import.meta.url.replace('file://', '').replace('/cedar.test.ts', '/fixtures')

      const model = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'search', toolUseId: 'tool-1', input: {} })
        .addTurn({ type: 'textBlock', text: 'Done' })

      let toolExecuted = false
      const tool = createMockTool('search', () => {
        toolExecuted = true
        return 'ok'
      })

      const cedar = new CedarAuthorization({
        policies: 'permit(principal, action == Action::"search", resource);',
        entities: `${fixturesDir}/entities.json`,
        principalResolver: () => ({ type: 'User', id: 'alice' }),
      })

      const agent = new Agent({ model, tools: [tool], interventions: [cedar], printer: false })
      await agent.invoke('Search', { invocationState: {} })
      expect(toolExecuted).toBe(true)
    })

    it('treats non-existent .cedar path as inline policy text', () => {
      const cedar = new CedarAuthorization({
        policies: '/nonexistent/path.cedar',
        principalResolver: () => ({ type: 'User', id: 'alice' }),
      })
      expect(cedar.name).toBe('cedar-authorization')
    })
  })

  describe('session management', () => {
    it('resetSession clears call counts', async () => {
      const model = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'send_email', toolUseId: 'tool-1', input: {} })
        .addTurn({ type: 'textBlock', text: 'Done' })

      const tool = createMockTool('send_email', () => 'sent')

      // Rate limit: < 2 calls allowed
      const cedar = new CedarAuthorization({
        policies: 'permit(principal, action, resource) when { context.session.call_count < 2 };',
        entities: [{ uid: { type: 'Resource', id: 'agent' }, attrs: {}, parents: [] }],
        principalResolver: () => ({ type: 'User', id: 'alice' }),
      })

      // First call succeeds
      const agent1 = new Agent({ model, tools: [tool], interventions: [cedar], printer: false })
      await agent1.invoke('Send', { invocationState: { session_id: 'sess1' } })

      // Reset the session
      cedar.resetSession('sess1')

      // Next call succeeds again (counter reset)
      let toolExecuted = false
      const model2 = new MockMessageModel()
        .addTurn({ type: 'toolUseBlock', name: 'send_email', toolUseId: 'tool-2', input: {} })
        .addTurn({ type: 'textBlock', text: 'Done' })

      const tool2 = createMockTool('send_email', () => {
        toolExecuted = true
        return 'sent'
      })

      const agent2 = new Agent({ model: model2, tools: [tool2], interventions: [cedar], printer: false })
      await agent2.invoke('Send again', { invocationState: { session_id: 'sess1' } })
      expect(toolExecuted).toBe(true)
    })
  })
})
