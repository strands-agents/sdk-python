import {
  BedrockAgentCoreClient,
  InvokeHarnessCommand,
  type InvokeHarnessStreamOutput,
} from '@aws-sdk/client-bedrock-agentcore'
import type { AgentExecutionEvent, ExecutionEventBus, RequestContext } from '@a2a-js/sdk/server'
import type { TaskArtifactUpdateEvent } from '@a2a-js/sdk'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createCancellableAgent } from '../../__fixtures__/agent-helpers.js'
import { collectGenerator } from '../../__fixtures__/model-test-helpers.js'
import { A2AExecutor } from '../../a2a/executor.js'
import {
  ConcurrentInvocationError,
  ContextWindowOverflowError,
  MaxTokensError,
  ModelError,
  ModelThrottledError,
} from '../../errors.js'
import { logger } from '../../logging/logger.js'
import { Graph } from '../../multiagent/graph.js'
import { Status } from '../../multiagent/state.js'
import { AgentResult } from '../../types/agent.js'
import { Message, ReasoningBlock, TextBlock, ToolResultBlock, ToolUseBlock } from '../../types/messages.js'
import { AgentCoreHarnessAgent } from '../agentcore-harness-agent.js'
import {
  AgentCoreHarnessResultEvent,
  AgentCoreHarnessStreamUpdateEvent,
  type AgentCoreHarnessEventData,
} from '../events.js'

const HARNESS_ARN = 'arn:aws:bedrock-agentcore:us-east-1:123456789012:harness/TestHarness-abcdefghij'
const SESSION_ID = 'session-id-padded-to-thirty-three'

afterEach(() => {
  vi.restoreAllMocks()
})

describe('AgentCoreHarnessAgent', () => {
  describe('constructor', () => {
    it('derives identity from the Harness session and accepts an override', () => {
      const agent = new AgentCoreHarnessAgent({ harnessArn: HARNESS_ARN, runtimeSessionId: SESSION_ID })
      const identifiedAgent = new AgentCoreHarnessAgent({
        harnessArn: HARNESS_ARN,
        runtimeSessionId: SESSION_ID,
        id: 'research-harness',
      })

      expect(agent).toMatchObject({
        harnessArn: HARNESS_ARN,
        runtimeSessionId: SESSION_ID,
        id: `${HARNESS_ARN}:${SESSION_ID}`,
      })
      expect(identifiedAgent).toMatchObject({
        harnessArn: HARNESS_ARN,
        runtimeSessionId: SESSION_ID,
        id: 'research-harness',
      })
    })

    it('uses standard AWS region resolution and defaults to one attempt', async () => {
      const agent = new AgentCoreHarnessAgent({
        harnessArn: HARNESS_ARN,
        runtimeSessionId: SESSION_ID,
        clientConfig: { region: 'us-east-2' },
      })
      const client = clientFrom(agent)

      await expect(client.config.region()).resolves.toBe('us-east-2')
      await expect(client.config.maxAttempts()).resolves.toBe(1)
    })

    it('forwards client configuration including explicit retry settings', async () => {
      const credentials = vi.fn().mockResolvedValue({
        accessKeyId: 'test-access-key',
        secretAccessKey: 'test-secret-key',
      })
      const agent = new AgentCoreHarnessAgent({
        harnessArn: HARNESS_ARN,
        runtimeSessionId: SESSION_ID,
        clientConfig: { credentials, maxAttempts: 4, region: 'eu-west-1' },
      })
      const client = clientFrom(agent)

      await expect(client.config.region()).resolves.toBe('eu-west-1')
      await expect(client.config.maxAttempts()).resolves.toBe(4)
      await expect(client.config.credentials()).resolves.toMatchObject({ accessKeyId: 'test-access-key' })
      expect(credentials).toHaveBeenCalledOnce()
    })

    it('uses an injected client without replacing its configuration', async () => {
      const client = new BedrockAgentCoreClient({ region: 'eu-west-1', maxAttempts: 6 })
      const send = vi.spyOn(client, 'send').mockResolvedValueOnce({
        stream: harnessStream(chunk.messageStart(), chunk.messageStop('end_turn')),
      } as never)
      const agent = new AgentCoreHarnessAgent({
        harnessArn: HARNESS_ARN,
        runtimeSessionId: SESSION_ID,
        client,
        clientConfig: { maxAttempts: 2, region: 'ap-southeast-2' },
      })

      await agent.invoke('Run it.')

      expect(send).toHaveBeenCalledOnce()
      await expect(client.config.region()).resolves.toBe('eu-west-1')
      await expect(client.config.maxAttempts()).resolves.toBe(6)
    })

    it('defers Harness identifier validation to AgentCore', () => {
      const harnessArn = 'validated-by-service'
      const runtimeSessionId = 'validated-by-service'

      expect(
        new AgentCoreHarnessAgent({ harnessArn, runtimeSessionId, clientConfig: { region: 'us-east-1' } })
      ).toMatchObject({
        harnessArn,
        runtimeSessionId,
      })
    })
  })

  describe('stream', () => {
    it('sends one text request and translates the complete Harness stream', async () => {
      const metadata = chunk.metadata(
        { inputTokens: 10, outputTokens: 4, totalTokens: 14 },
        25
      ) as AgentCoreHarnessEventData
      const send = mockHarness(
        harnessStream(
          chunk.messageStart(),
          chunk.textDelta('working'),
          chunk.messageStop('tool_use'),
          chunk.messageStart('user'),
          chunk.messageStop('tool_result'),
          chunk.messageStart(),
          chunk.reasoningDelta({ text: 'considering' }),
          chunk.reasoningDelta({ signature: 'signed' }),
          chunk.contentBlockStop(),
          chunk.textDelta('Complete.'),
          chunk.contentBlockStop(),
          chunk.messageStop('end_turn'),
          metadata
        )
      )
      const invocationState = { requestId: 'request-1' }
      const agent = new AgentCoreHarnessAgent({ harnessArn: HARNESS_ARN, runtimeSessionId: SESSION_ID })

      const { items, result } = await collectGenerator(agent.stream('Run it.', { invocationState }))

      expect(send).toHaveBeenCalledOnce()
      expect(send.mock.calls[0]![0]).toBeInstanceOf(InvokeHarnessCommand)
      expect(send.mock.calls[0]![0].input).toStrictEqual({
        harnessArn: HARNESS_ARN,
        runtimeSessionId: SESSION_ID,
        messages: [{ role: 'user', content: [{ text: 'Run it.' }] }],
      })
      expect(send.mock.calls[0]![1]).toStrictEqual({})
      expect(result).toEqual(
        new AgentResult({
          stopReason: 'endTurn',
          lastMessage: new Message({
            role: 'assistant',
            content: [new ReasoningBlock({ text: 'considering', signature: 'signed' }), new TextBlock('Complete.')],
            trackingId: result.lastMessage.trackingId,
          }),
          invocationState,
        })
      )
      expect(items.at(-1)).toBeInstanceOf(AgentCoreHarnessResultEvent)
      expect(items.slice(0, -1)).toHaveLength(13)
      expect(items[0]).toBeInstanceOf(AgentCoreHarnessStreamUpdateEvent)
      expect(items.at(-2)).toEqual(new AgentCoreHarnessStreamUpdateEvent(metadata))
    })

    it('preserves a pending tool block before implicit text content', async () => {
      mockHarness(
        harnessStream(
          chunk.messageStart(),
          chunk.toolUseStart('tool-1', 'lookup'),
          chunk.toolUseDelta('{"city":"Boston"}'),
          chunk.textDelta('Done.'),
          chunk.messageStop('end_turn')
        )
      )
      const agent = new AgentCoreHarnessAgent({ harnessArn: HARNESS_ARN, runtimeSessionId: SESSION_ID })

      const result = await agent.invoke('Run it.')

      expect(result).toEqual(
        new AgentResult({
          stopReason: 'endTurn',
          lastMessage: new Message({
            role: 'assistant',
            content: [
              new ToolUseBlock({ toolUseId: 'tool-1', name: 'lookup', input: { city: 'Boston' } }),
              new TextBlock('Done.'),
            ],
            trackingId: result.lastMessage.trackingId,
          }),
          invocationState: {},
        })
      )
    })

    it('warns when converting unknown tool-result content to text', async () => {
      const warn = vi.spyOn(logger, 'warn').mockImplementation(() => {})
      mockHarness(
        harnessStream(
          chunk.messageStart(),
          chunk.toolResultStart('tool-1'),
          chunk.toolResultDelta([{ futureContent: { value: 1 } }]),
          chunk.contentBlockStop(),
          chunk.messageStop('tool_result')
        )
      )
      const agent = new AgentCoreHarnessAgent({ harnessArn: HARNESS_ARN, runtimeSessionId: SESSION_ID })

      const result = await agent.invoke('Run it.')

      expect(result).toEqual(
        new AgentResult({
          stopReason: 'toolResult',
          lastMessage: new Message({
            role: 'assistant',
            content: [
              new ToolResultBlock({
                toolUseId: 'tool-1',
                status: 'success',
                content: [new TextBlock('{"futureContent":{"value":1}}')],
              }),
            ],
            trackingId: result.lastMessage.trackingId,
          }),
          invocationState: {},
        })
      )
      expect(warn).toHaveBeenCalledOnce()
      expect(warn).toHaveBeenCalledWith('fields=<futureContent> | unknown tool-result content, converting to JSON text')
    })

    it.each([
      { termination: 'throws', throwsAfterAbort: true },
      { termination: 'closes normally', throwsAfterAbort: false },
    ])('returns only pre-abort content when an aborted stream $termination', async ({ throwsAfterAbort }) => {
      const controller = new AbortController()
      const send = mockHarness(cancelledStream(controller, throwsAfterAbort))
      const agent = new AgentCoreHarnessAgent({ harnessArn: HARNESS_ARN, runtimeSessionId: SESSION_ID })

      const { items, result } = await collectGenerator(agent.stream('Run it.', { cancelSignal: controller.signal }))

      expect({ ...result.toJSON(), lastMessage: result.lastMessage.toJSON() }).toEqual({
        type: 'agentResult',
        stopReason: 'cancelled',
        lastMessage: {
          role: 'assistant',
          content: [{ text: 'partial' }],
          trackingId: expect.any(String),
        },
      })
      expect(result.invocationState).toEqual({})
      expect(items).toEqual([
        new AgentCoreHarnessStreamUpdateEvent(chunk.messageStart() as AgentCoreHarnessEventData),
        new AgentCoreHarnessStreamUpdateEvent(chunk.textDelta('partial') as AgentCoreHarnessEventData),
        new AgentCoreHarnessResultEvent({ result }),
      ])
      expect(send.mock.calls[0]![1]).toStrictEqual({ abortSignal: controller.signal })
    })

    it('does not make a request when already cancelled', async () => {
      const send = mockHarness(harnessStream())
      const agent = new AgentCoreHarnessAgent({ harnessArn: HARNESS_ARN, runtimeSessionId: SESSION_ID })

      const result = await agent.invoke('Run it.', { cancelSignal: AbortSignal.abort() })

      expect(result.stopReason).toBe('cancelled')
      expect(send).not.toHaveBeenCalled()
    })

    it.each([
      {
        chunk: {
          internalServerException: { message: 'internal failure' },
        } as InvokeHarnessStreamOutput,
        error: ModelError,
        message: 'internal failure',
      },
      {
        chunk: {
          validationException: { message: 'Prompt is too long' },
        } as InvokeHarnessStreamOutput,
        error: ContextWindowOverflowError,
        message: 'Prompt is too long',
      },
      {
        chunk: {
          runtimeClientError: { message: 'runtime failure' },
        } as InvokeHarnessStreamOutput,
        error: ModelError,
        message: 'runtime failure',
      },
    ])('translates a streamed service error', async ({ chunk: errorChunk, error, message }) => {
      mockHarness(harnessStream(errorChunk))
      const agent = new AgentCoreHarnessAgent({ harnessArn: HARNESS_ARN, runtimeSessionId: SESSION_ID })

      await expect(agent.invoke('Run it.')).rejects.toMatchObject({ constructor: error, message })
    })

    it('rejects malformed tool input before returning a tool-use result', async () => {
      mockHarness(
        harnessStream(
          chunk.messageStart(),
          chunk.toolUseStart('tool-1', 'deployed_inline'),
          chunk.toolUseDelta('{'),
          chunk.contentBlockStop(),
          chunk.messageStop('tool_use')
        )
      )
      const agent = new AgentCoreHarnessAgent({ harnessArn: HARNESS_ARN, runtimeSessionId: SESSION_ID })

      await expect(agent.invoke('Run it.')).rejects.toMatchObject({
        constructor: ModelError,
        message: 'Unable to parse Harness tool input JSON.',
      })
    })

    it('surfaces max-token termination as MaxTokensError', async () => {
      mockHarness(harnessStream(chunk.messageStart(), chunk.textDelta('partial'), chunk.messageStop('max_tokens')))
      const agent = new AgentCoreHarnessAgent({ harnessArn: HARNESS_ARN, runtimeSessionId: SESSION_ID })

      await expect(agent.invoke('Run it.')).rejects.toBeInstanceOf(MaxTokensError)
    })

    it('rejects an interrupted Harness turn as non-resumable', async () => {
      mockHarness(harnessStream(chunk.messageStart(), chunk.textDelta('partial'), chunk.messageStop('interrupted')))
      const agent = new AgentCoreHarnessAgent({ harnessArn: HARNESS_ARN, runtimeSessionId: SESSION_ID })

      await expect(agent.invoke('Run it.')).rejects.toMatchObject({
        constructor: ModelError,
        message: 'Harness ended the turn with an unrecoverable stop reason: interrupted',
      })
    })

    it('rejects a concurrent invocation and releases the instance afterward', async () => {
      let release: (response: { stream: AsyncGenerator<InvokeHarnessStreamOutput> }) => void
      const pending = new Promise<{ stream: AsyncGenerator<InvokeHarnessStreamOutput> }>((resolve) => {
        release = resolve
      })
      const send = vi.spyOn(BedrockAgentCoreClient.prototype, 'send').mockReturnValueOnce(pending as never)
      const agent = new AgentCoreHarnessAgent({ harnessArn: HARNESS_ARN, runtimeSessionId: SESSION_ID })

      const first = agent.invoke('First')
      await expect(agent.invoke('Second')).rejects.toBeInstanceOf(ConcurrentInvocationError)
      release!({ stream: harnessStream(chunk.messageStart(), chunk.messageStop('end_turn')) })
      await first

      send.mockResolvedValueOnce({
        stream: harnessStream(chunk.messageStart(), chunk.messageStop('end_turn')),
      } as never)
      await expect(agent.invoke('Third')).resolves.toMatchObject({ stopReason: 'endTurn' })
    })
  })

  describe('validation', () => {
    it.each([
      {
        args: [new TextBlock('Line one'), new TextBlock('Line two')],
        expected: 'Line one\nLine two',
      },
      {
        args: [{ text: 'Data one' }, { text: 'Data two' }],
        expected: 'Data one\nData two',
      },
      {
        args: [
          new Message({ role: 'user', content: [new TextBlock('Earlier')] }),
          new Message({ role: 'assistant', content: [new TextBlock('Reply')] }),
          new Message({ role: 'user', content: [new TextBlock('Latest'), new TextBlock('request')] }),
        ],
        expected: 'Latest\nrequest',
      },
      {
        args: [
          { role: 'user' as const, content: [{ text: 'Serialized earlier' }] },
          { role: 'user' as const, content: [{ text: 'Serialized latest' }] },
        ],
        expected: 'Serialized latest',
      },
    ])('normalizes supported text input before calling AgentCore', async ({ args, expected }) => {
      const send = mockHarness(harnessStream(chunk.messageStart(), chunk.messageStop('end_turn')))
      const agent = new AgentCoreHarnessAgent({ harnessArn: HARNESS_ARN, runtimeSessionId: SESSION_ID })

      await agent.invoke(args)

      expect(send.mock.calls[0]![0].input.messages).toStrictEqual([{ role: 'user', content: [{ text: expected }] }])
    })

    it.each([
      { args: '', options: undefined, message: 'input must contain non-empty text' },
      { args: [], options: undefined, message: 'input must contain non-empty text' },
      {
        args: [new ReasoningBlock({ text: 'private reasoning' })],
        options: undefined,
        message: 'accepts only text content blocks; received reasoningBlock',
      },
      {
        args: [new Message({ role: 'assistant', content: [new TextBlock('No user message')] })],
        options: undefined,
        message: 'must include at least one user message',
      },
      {
        args: [{ interruptResponse: { interruptId: 'interrupt-1', response: 'continue' } }],
        options: undefined,
        message: 'does not support interrupt response input',
      },
      {
        args: { checkpointResume: { checkpoint: { position: 'afterModel' } } },
        options: undefined,
        message: 'does not support checkpoint resume input',
      },
      {
        args: 'Hi',
        options: { structuredOutputSchema: { _output: undefined } },
        message: 'structuredOutputSchema is not supported',
      },
      { args: 'Hi', options: { limits: { turns: 1 } }, message: 'limits is not supported' },
    ])('rejects unsupported input and options before calling AgentCore', async ({ args, options, message }) => {
      const send = mockHarness(harnessStream())
      const agent = new AgentCoreHarnessAgent({ harnessArn: HARNESS_ARN, runtimeSessionId: SESSION_ID })

      await expect(agent.invoke(args as never, options as never)).rejects.toThrow(message)
      expect(send).not.toHaveBeenCalled()
    })
  })

  describe('composition', () => {
    it('accepts dependency output from a Graph predecessor', async () => {
      const send = mockHarness(
        harnessStream(
          chunk.messageStart(),
          chunk.textDelta('Harness graph result'),
          chunk.contentBlockStop(),
          chunk.messageStop('end_turn')
        )
      )
      const source = createCancellableAgent('source', 0, { message: 'Source result' })
      const harness = new AgentCoreHarnessAgent({ harnessArn: HARNESS_ARN, runtimeSessionId: SESSION_ID })
      const graph = new Graph({
        nodes: [source, harness],
        edges: [[source.id, harness.id]],
        maxSteps: 2,
      })

      const result = await graph.invoke('Original task')

      expect(result.status).toBe(Status.COMPLETED)
      expect(result.content).toStrictEqual([new TextBlock('Harness graph result')])
      expect(send.mock.calls[0]![0].input.messages).toStrictEqual([
        { role: 'user', content: [{ text: 'Original task\n[node: source]\nSource result' }] },
      ])
    })

    it('returns final Harness text through A2A execution', async () => {
      const send = mockHarness(
        harnessStream(
          chunk.messageStart(),
          chunk.textDelta('Harness A2A result'),
          chunk.contentBlockStop(),
          chunk.messageStop('end_turn')
        )
      )
      const harness = new AgentCoreHarnessAgent({ harnessArn: HARNESS_ARN, runtimeSessionId: SESSION_ID })
      const executor = new A2AExecutor({ agentFactory: () => harness })
      const eventBus = createMockEventBus()
      const context = createRequestContext('A2A request')

      await executor.execute(context, eventBus)

      const artifactEvents = eventBus.events.filter(
        (event): event is TaskArtifactUpdateEvent => event.kind === 'artifact-update'
      )
      expect(artifactEvents).toStrictEqual([
        {
          kind: 'artifact-update',
          taskId: 'task-1',
          contextId: 'context-1',
          artifact: {
            artifactId: expect.any(String),
            parts: [{ kind: 'text', text: 'Harness A2A result' }],
          },
          append: false,
          lastChunk: true,
        },
      ])
      expect(send.mock.calls[0]![0].input.messages).toStrictEqual([
        { role: 'user', content: [{ text: 'A2A request' }] },
      ])
    })
  })

  describe('errors', () => {
    it.each([
      { name: 'ThrottlingException', error: ModelThrottledError },
      { name: 'OtherException', error: ModelError },
    ])('normalizes a rejected AgentCore request', async ({ name, error }) => {
      const original = Object.assign(new Error('request failed'), { name })
      vi.spyOn(BedrockAgentCoreClient.prototype, 'send').mockRejectedValueOnce(original)
      const agent = new AgentCoreHarnessAgent({ harnessArn: HARNESS_ARN, runtimeSessionId: SESSION_ID })

      await expect(agent.invoke('Run it.')).rejects.toMatchObject({
        constructor: error,
        message: 'request failed',
        cause: original,
      })
    })
  })
})

function clientFrom(agent: AgentCoreHarnessAgent): BedrockAgentCoreClient {
  return (agent as unknown as { _client: BedrockAgentCoreClient })._client
}

function mockHarness(stream: AsyncGenerator<InvokeHarnessStreamOutput>): ReturnType<typeof vi.spyOn> {
  return vi.spyOn(BedrockAgentCoreClient.prototype, 'send').mockResolvedValueOnce({ stream } as never)
}

async function* harnessStream(...chunks: InvokeHarnessStreamOutput[]): AsyncGenerator<InvokeHarnessStreamOutput> {
  yield* chunks
}

async function* cancelledStream(
  controller: AbortController,
  throwsAfterAbort: boolean
): AsyncGenerator<InvokeHarnessStreamOutput> {
  yield chunk.messageStart()
  yield chunk.textDelta('partial')
  controller.abort()
  if (throwsAfterAbort) throw new Error('aborted')
  yield chunk.textDelta(' buffered after abort')
  yield chunk.contentBlockStop()
  yield chunk.messageStop('end_turn')
}

function createMockEventBus(): ExecutionEventBus & { events: AgentExecutionEvent[] } {
  const events: AgentExecutionEvent[] = []
  return {
    events,
    publish: vi.fn((event) => {
      events.push(event)
    }),
    on: vi.fn().mockReturnThis(),
    off: vi.fn().mockReturnThis(),
    once: vi.fn().mockReturnThis(),
    removeAllListeners: vi.fn().mockReturnThis(),
    finished: vi.fn(),
  }
}

function createRequestContext(text: string): RequestContext {
  return {
    taskId: 'task-1',
    contextId: 'context-1',
    userMessage: {
      kind: 'message',
      messageId: 'message-1',
      role: 'user',
      parts: [{ kind: 'text', text }],
    },
  }
}

const chunk = {
  messageStart: (role: 'assistant' | 'user' = 'assistant'): InvokeHarnessStreamOutput =>
    ({ messageStart: { role } }) as InvokeHarnessStreamOutput,
  textDelta: (text: string): InvokeHarnessStreamOutput =>
    ({ contentBlockDelta: { delta: { text } } }) as InvokeHarnessStreamOutput,
  reasoningDelta: (reasoningContent: { text?: string; signature?: string }): InvokeHarnessStreamOutput =>
    ({ contentBlockDelta: { delta: { reasoningContent } } }) as InvokeHarnessStreamOutput,
  toolUseStart: (toolUseId: string, name: string): InvokeHarnessStreamOutput =>
    ({ contentBlockStart: { start: { toolUse: { toolUseId, name } } } }) as InvokeHarnessStreamOutput,
  toolUseDelta: (input: string): InvokeHarnessStreamOutput =>
    ({ contentBlockDelta: { delta: { toolUse: { input } } } }) as InvokeHarnessStreamOutput,
  toolResultStart: (toolUseId: string): InvokeHarnessStreamOutput =>
    ({
      contentBlockStart: { start: { toolResult: { toolUseId, status: 'success' } } },
    }) as InvokeHarnessStreamOutput,
  toolResultDelta: (toolResult: Record<string, unknown>[]): InvokeHarnessStreamOutput =>
    ({ contentBlockDelta: { delta: { toolResult } } }) as InvokeHarnessStreamOutput,
  contentBlockStop: (): InvokeHarnessStreamOutput => ({ contentBlockStop: {} }) as InvokeHarnessStreamOutput,
  messageStop: (stopReason: string): InvokeHarnessStreamOutput =>
    ({ messageStop: { stopReason } }) as InvokeHarnessStreamOutput,
  metadata: (usage: Record<string, number>, latencyMs: number): InvokeHarnessStreamOutput =>
    ({ metadata: { usage, metrics: { latencyMs } } }) as InvokeHarnessStreamOutput,
}
