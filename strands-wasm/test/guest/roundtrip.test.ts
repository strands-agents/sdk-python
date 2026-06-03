import { describe, it, expect, vi, beforeAll, beforeEach } from 'vitest'
import { createGuest, drainStream, LogEntry, CallToolArgs } from './harness'

interface ToolSpec {
  name: string
  description: string
  inputSchema: string
}

const bedrockConfig = {
  model: { tag: 'bedrock' as const, val: { modelId: 'anthropic.claude-3-haiku-20240307-v1:0' } },
  modelParams: { maxTokens: 256 },
}

function generate(agent: any, input: string): any {
  return agent.generate({ input, tools: undefined, toolChoice: undefined })
}

describe.runIf(process.env.STRANDS_INTEG === 'true')('Level 2b: full round-trip tests', () => {
  let root: any
  const logEntries: LogEntry[] = []
  const callToolMock = vi.fn((args: CallToolArgs) => {
    return JSON.stringify({ status: 'success', content: [{ text: `mock result for ${args.name}` }] })
  })

  beforeAll(async () => {
    root = await createGuest({
      log: (entry) => logEntries.push(entry),
      callTool: callToolMock,
    })
  }, 120_000)

  beforeEach(() => {
    logEntries.length = 0
    callToolMock.mockClear()
  })

  it('full generate produces text-delta and stop events', async () => {
    const agent = new root.api.Agent({
      ...bedrockConfig,
      systemPrompt: 'Respond with exactly one word: hello',
    })
    const stream = generate(agent, 'Say hello')
    const events = await drainStream(stream)
    const textDeltas = events.filter((e: any) => e.tag === 'text-delta')
    expect(textDeltas.length).toBeGreaterThan(0)
    for (const td of textDeltas) {
      expect(typeof td.val).toBe('string')
    }
    const stopEvent = events.find((e: any) => e.tag === 'stop')
    expect(stopEvent).toBeDefined()
    expect(stopEvent.val).toMatchObject({
      reason: 'end-turn',
    })
  })

  it('tool call flow — model calls tool, host mock receives it', async () => {
    const weatherTool: ToolSpec = {
      name: 'get_weather',
      description: 'Get the current weather for a location',
      inputSchema: JSON.stringify({
        type: 'object',
        properties: { location: { type: 'string', description: 'City name' } },
        required: ['location'],
      }),
    }
    const agent = new root.api.Agent({
      ...bedrockConfig,
      systemPrompt: 'You have a get_weather tool. Use it to answer weather questions. Do not ask for clarification.',
      tools: [weatherTool],
    })
    const stream = generate(agent, 'What is the weather in Seattle?')
    const events = await drainStream(stream)
    expect(callToolMock).toHaveBeenCalled()
    expect(callToolMock.mock.calls[0][0].name).toBe('get_weather')
    const toolUseEvent = events.find((e: any) => e.tag === 'tool-use')
    expect(toolUseEvent).toBeDefined()
    expect(toolUseEvent.val.name).toBe('get_weather')
    expect(typeof toolUseEvent.val.toolUseId).toBe('string')
    expect(toolUseEvent.val.toolUseId.length).toBeGreaterThan(0)
    expect(() => JSON.parse(toolUseEvent.val.input)).not.toThrow()
    const toolResultEvent = events.find((e: any) => e.tag === 'tool-result')
    expect(toolResultEvent).toBeDefined()
    expect(toolResultEvent.val.status).toBe('success')
    expect(typeof toolResultEvent.val.content).toBe('string')
  })

  it('lifecycle events appear in readNext batches', async () => {
    const agent = new root.api.Agent({ ...bedrockConfig, systemPrompt: 'Say hi' })
    const stream = generate(agent, 'hello')
    const events = await drainStream(stream)
    const lifecycleEvents = events.filter((e: any) => e.tag === 'lifecycle')
    expect(lifecycleEvents.length).toBeGreaterThan(0)
    const beforeModelCall = lifecycleEvents.find((e: any) => e.val.eventType === 'before-model-call')
    expect(beforeModelCall).toBeDefined()
    expect(beforeModelCall.val).toMatchObject({
      eventType: 'before-model-call',
      toolUse: undefined,
      toolResult: undefined,
    })
  })

  it('metadata event with usage tokens appears', async () => {
    const agent = new root.api.Agent({ ...bedrockConfig, systemPrompt: 'Say one word' })
    const stream = generate(agent, 'go')
    const events = await drainStream(stream)
    const metadataEvent = events.find((e: any) => e.tag === 'metadata')
    expect(metadataEvent).toBeDefined()
    expect(metadataEvent.val.usage).toBeDefined()
    expect(metadataEvent.val.usage.inputTokens).toBeGreaterThan(0)
    expect(metadataEvent.val.usage.outputTokens).toBeGreaterThanOrEqual(0)
    expect(metadataEvent.val.usage.totalTokens).toBeGreaterThan(0)
  })

  it('cancel terminates the stream', async () => {
    const agent = new root.api.Agent({
      ...bedrockConfig,
      systemPrompt: 'Write a very long story about a dragon',
    })
    const stream = generate(agent, 'begin')
    const firstBatch = await stream.readNext()
    expect(firstBatch).toBeDefined()
    stream.cancel()
    const afterCancel = await stream.readNext()
    expect(afterCancel).toBeUndefined()
  })

  it('multi-turn: setMessages then generate continues context', async () => {
    const agent = new root.api.Agent({
      ...bedrockConfig,
      systemPrompt: 'Remember what the user tells you',
    })
    const priorMessages = [
      { role: 'user', content: [{ type: 'text', text: 'My name is Alice' }] },
      { role: 'assistant', content: [{ type: 'text', text: 'Nice to meet you, Alice!' }] },
    ]
    agent.setMessages({ json: JSON.stringify(priorMessages) })
    const stream = generate(agent, 'What is my name?')
    const events = await drainStream(stream)
    const textDeltas = events.filter((e: any) => e.tag === 'text-delta')
    const fullText = textDeltas.map((e: any) => e.val).join('')
    expect(fullText.toLowerCase()).toContain('alice')
  })
})
