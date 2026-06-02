import { describe, it, expect, beforeAll, beforeEach } from 'vitest'
import { createGuest, drainStream, LogEntry } from './harness'

describe('Level 2a: boundary smoke tests', () => {
  const anthropicModel = { tag: 'anthropic' as const, val: { apiKey: 'sk-fake-key-for-testing' } }
  let root: any
  const logEntries: LogEntry[] = []

  function createAgent(): any {
    return new root.api.Agent({ model: anthropicModel })
  }

  beforeAll(async () => {
    root = await createGuest({
      log: (entry) => logEntries.push(entry),
      callTool: () => JSON.stringify({ status: 'success', content: [{ text: 'mock result' }] }),
    })
  }, 120_000)

  beforeEach(() => {
    logEntries.length = 0
  })

  it('component loads and instantiate succeeds', () => {
    expect(root).toBeDefined()
    expect(root.api).toBeDefined()
    expect(root.api.Agent).toBeDefined()
  })

  it('Agent construction succeeds', () => {
    expect(createAgent()).toBeDefined()
  })

  it('getMessages returns empty array on fresh agent', () => {
    expect(createAgent().getMessages()).toBe('[]')
  })

  it('setMessages → getMessages round-trips correctly', () => {
    const agent = createAgent()
    const messages = JSON.stringify([{ role: 'user', content: [{ type: 'text', text: 'hello' }] }])
    agent.setMessages({ json: messages })
    expect(agent.getMessages()).toBe(messages)
  })

  it('host-log mock receives log entries during construction', () => {
    createAgent()
    expect(logEntries.length).toBeGreaterThan(0)
    expect(logEntries[0]).toMatchObject({
      level: expect.stringMatching(/^(trace|debug|info|warn|error)$/),
      message: expect.any(String),
    })
  })

  it('generate with fake API key returns error event', async () => {
    const agent = new root.api.Agent({
      model: {
        ...anthropicModel,
        val: { ...anthropicModel.val, additionalConfig: JSON.stringify({ timeout: 10_000 }) },
      },
    })
    const stream = agent.generate({ input: 'hello', tools: undefined, toolChoice: undefined })
    const events = await drainStream(stream)
    const errorEvent = events.find((e: any) => e.tag === 'error')
    expect(errorEvent).toBeDefined()
    expect(typeof errorEvent.val).toBe('string')
  })

  it('deleteSession throws not-yet-implemented error', () => {
    expect(() => createAgent().deleteSession()).toThrow()
  })
})
