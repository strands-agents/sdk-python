import { describe, expect, it, vi } from 'vitest'
import type { BedrockAgentCoreClient, InvokeHarnessStreamOutput } from '@aws-sdk/client-bedrock-agentcore'
import { z } from 'zod'
import { AgentCoreHarnessAgent, type AgentCoreHarnessAgentConfig } from '../agentcore-harness-agent.js'
import { AgentCoreHarnessStreamUpdateEvent, AgentCoreHarnessResultEvent } from '../events.js'
import { tool } from '../../tools/tool-factory.js'
import { Message, TextBlock } from '../../types/messages.js'
import { ImageBlock } from '../../types/media.js'
import { AgentResult } from '../../types/agent.js'
import { collectGenerator } from '../../__fixtures__/model-test-helpers.js'
import {
  ConcurrentInvocationError,
  ContextWindowOverflowError,
  MaxTokensError,
  ModelError,
  ModelThrottledError,
} from '../../errors.js'
import {
  HARNESS_ARN,
  SESSION_ID,
  chunk,
  harnessStream,
  mockClient,
  mockControlClient,
} from './harness-test-fixtures.js'

describe('AgentCoreHarnessAgent', () => {
  describe('identity and validation', () => {
    it('defaults id to the harness ARN', () => {
      const agent = new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID })
      expect(agent.id).toBe('arn:harness')
    })

    it('uses provided id, name, and description', () => {
      const agent = new AgentCoreHarnessAgent({
        harnessArn: 'arn:harness',
        runtimeSessionId: SESSION_ID,
        id: 'custom',
        name: 'My Agent',
        description: 'Does things',
      })
      expect(agent.id).toBe('custom')
      expect(agent.name).toBe('My Agent')
      expect(agent.description).toBe('Does things')
    })

    it('has undefined name and description when not provided', () => {
      const agent = new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID })
      expect(agent.name).toBeUndefined()
      expect(agent.description).toBeUndefined()
    })

    it.each([
      ['an empty string', ''],
      ['an empty array', []],
      ['only empty text blocks', [new TextBlock('')]],
    ] as [string, string | TextBlock[]][])('rejects a systemPrompt containing %s', (_description, systemPrompt) => {
      expect(
        () => new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID, systemPrompt })
      ).toThrow(TypeError)
    })

    it.each([
      { desc: 'too short', id: 'short' },
      { desc: 'too long', id: 'x'.repeat(101) },
    ])('rejects a runtimeSessionId that is $desc', ({ id }) => {
      expect(() => new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: id })).toThrow(
        /33.*100 characters/
      )
    })

    it.each([
      { desc: 'starts with a hyphen', id: `-${'a'.repeat(32)}` },
      { desc: 'starts with an underscore', id: `_${'a'.repeat(32)}` },
      { desc: 'contains a period', id: `${'a'.repeat(32)}.` },
      { desc: 'contains a space', id: `${'a'.repeat(32)} ` },
    ])('rejects a runtimeSessionId that $desc', ({ id }) => {
      expect(() => new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: id })).toThrow(
        /must start with an alphanumeric character/
      )
    })

    it('accepts hyphens and underscores after the first character', () => {
      const runtimeSessionId = `a${'-_'.repeat(16)}`

      expect(new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId }).runtimeSessionId).toBe(
        runtimeSessionId
      )
    })

    it.each(['concurrent', 'sequential'] as const)("accepts toolExecutor '%s'", (toolExecutor) => {
      expect(
        () => new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID, toolExecutor })
      ).not.toThrow()
    })

    it.each(['parallel', '', null])('rejects invalid runtime toolExecutor %#', (toolExecutor) => {
      expect(
        () =>
          new AgentCoreHarnessAgent({
            harnessArn: 'arn:harness',
            runtimeSessionId: SESSION_ID,
            toolExecutor: toolExecutor as never,
          })
      ).toThrow(`toolExecutor must be 'concurrent' or 'sequential', got '${String(toolExecutor)}'`)
    })

    it('rejects duplicate host tool names', () => {
      const first = tool({
        name: 'duplicate',
        description: 'first',
        inputSchema: z.object({}),
        callback: () => 'first',
      })
      const second = tool({
        name: 'duplicate',
        description: 'second',
        inputSchema: z.object({}),
        callback: () => 'second',
      })

      expect(
        () =>
          new AgentCoreHarnessAgent({
            harnessArn: 'arn:harness',
            runtimeSessionId: SESSION_ID,
            tools: [first, second],
          })
      ).toThrow("Tool with name 'duplicate' already registered")
    })

    it.each(['builtin', 'shell', 'file_operations'])("rejects reserved host tool name '%s'", (name) => {
      const hostTool = tool({
        name,
        description: 'reserved name',
        inputSchema: z.object({}),
        callback: () => 'result',
      })

      expect(
        () =>
          new AgentCoreHarnessAgent({
            harnessArn: 'arn:harness',
            runtimeSessionId: SESSION_ID,
            tools: [hostTool],
          })
      ).toThrow(`Host tool name '${name}' is reserved by AgentCore Harness.`)
    })

    it.each([
      ['all tools', ['*']],
      ['the inline-function namespace', ['@get_weather']],
      ['the fully qualified inline function', ['@get_weather/get_weather']],
      ['a matching namespace and tool glob', ['@get_*/get_w?ather']],
    ])('accepts a host tool allowed by %s', (_description, allowedTools) => {
      const hostTool = tool({
        name: 'get_weather',
        description: 'Get weather',
        inputSchema: z.object({}),
        callback: () => 'sunny',
      })

      expect(
        () =>
          new AgentCoreHarnessAgent({
            harnessArn: HARNESS_ARN,
            runtimeSessionId: SESSION_ID,
            tools: [hostTool],
            allowedTools,
          })
      ).not.toThrow()
    })

    it.each([
      ['an empty list', []],
      ['an unqualified name', ['get_weather']],
      ['another namespace', ['@other']],
      ['another tool in the namespace', ['@get_weather/get_forecast']],
    ])('rejects a host tool excluded by %s', (_description, allowedTools) => {
      const hostTool = tool({
        name: 'get_weather',
        description: 'Get weather',
        inputSchema: z.object({}),
        callback: () => 'sunny',
      })

      expect(
        () =>
          new AgentCoreHarnessAgent({
            harnessArn: HARNESS_ARN,
            runtimeSessionId: SESSION_ID,
            tools: [hostTool],
            allowedTools,
          })
      ).toThrow(
        "Host tool 'get_weather' is excluded by effective allowedTools. Add '@get_weather', '@get_weather/get_weather', a matching namespace glob, or '*' to allowedTools."
      )
    })
  })

  describe('invoke', () => {
    it('returns an AgentResult with the assembled text and endTurn stop reason', async () => {
      const { client } = mockClient(
        harnessStream(
          chunk.messageStart(),
          chunk.textDelta('Hello '),
          chunk.textDelta('world'),
          chunk.contentBlockStop(),
          chunk.messageStop('end_turn')
        )
      )
      const agent = new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID, client })

      const result = await agent.invoke('Hi')

      expect(result).toStrictEqual(
        new AgentResult({
          stopReason: 'endTurn',
          lastMessage: new Message({
            role: 'assistant',
            content: [new TextBlock('Hello world')],
            trackingId: result.lastMessage.trackingId,
          }),
          invocationState: {},
        })
      )
    })

    it.each([
      { name: 'turns', limits: { turns: 1 } },
      { name: 'output tokens', limits: { outputTokens: 100 } },
      { name: 'total tokens', limits: { totalTokens: 100 } },
      { name: 'an empty object', limits: {} },
    ])('rejects unsupported $name limits before invoking the harness', async ({ limits }) => {
      const { client, send } = mockClient()
      const agent = new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID, client })

      await expect(agent.invoke('Hi', { limits })).rejects.toThrow(
        'InvokeOptions.limits is not supported by AgentCoreHarnessAgent. Configure Harness-side maxIterations, maxTokens, or timeoutSeconds when constructing the agent.'
      )
      expect(send).not.toHaveBeenCalled()
    })

    it.each([
      { name: 'an empty content array', args: [] },
      { name: 'an empty string', args: '' },
      {
        name: 'mixed text and unsupported media',
        args: [
          new TextBlock('Do not send only this text'),
          new ImageBlock({ format: 'png', source: { bytes: new Uint8Array([1]) } }),
        ],
      },
    ])('rejects $name before invoking the harness', async ({ args }) => {
      const { client, send } = mockClient()
      const agent = new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID, client })

      await expect(agent.invoke(args as never)).rejects.toThrow()
      expect(send).not.toHaveBeenCalled()
    })

    it('forwards modelConfig and systemPrompt overrides', async () => {
      const { client, send } = mockClient(harnessStream(chunk.messageStop('end_turn')))
      const agent = new AgentCoreHarnessAgent({
        harnessArn: 'arn:harness',
        runtimeSessionId: SESSION_ID,
        client,
        modelConfig: { bedrockModelConfig: { modelId: 'anthropic.claude' } },
        systemPrompt: 'Be concise.',
      })

      await agent.invoke('Hi')

      // modelConfig is sent on the wire under the InvokeHarness `model` field.
      expect(send.mock.calls[0]![0].input).toStrictEqual({
        harnessArn: 'arn:harness',
        runtimeSessionId: SESSION_ID,
        messages: [{ role: 'user', content: [{ text: 'Hi' }] }],
        model: { bedrockModelConfig: { modelId: 'anthropic.claude' } },
        systemPrompt: [{ text: 'Be concise.' }],
      })
    })

    it('forwards optional request fields verbatim to the wire request', async () => {
      const { client, send } = mockClient(harnessStream(chunk.messageStop('end_turn')))
      const skills = [{ path: '/skills/researcher' }]
      const agent = new AgentCoreHarnessAgent({
        harnessArn: 'arn:harness',
        runtimeSessionId: SESSION_ID,
        client,
        skills,
        allowedTools: ['shell', 'browser'],
        maxIterations: 10,
        maxTokens: 4096,
        timeoutSeconds: 300,
        qualifier: 'prod',
        runtimeUserId: 'user-42',
        actorId: 'actor-7',
      })

      await agent.invoke('Hi')

      expect(send.mock.calls[0]![0].input).toStrictEqual({
        harnessArn: 'arn:harness',
        runtimeSessionId: SESSION_ID,
        messages: [{ role: 'user', content: [{ text: 'Hi' }] }],
        skills,
        allowedTools: ['shell', 'browser'],
        maxIterations: 10,
        maxTokens: 4096,
        timeoutSeconds: 300,
        qualifier: 'prod',
        runtimeUserId: 'user-42',
        actorId: 'actor-7',
      })
    })

    it('snapshots request overrides and isolates later invocations from command mutation', async () => {
      const modelConfig: NonNullable<AgentCoreHarnessAgentConfig['modelConfig']> = {
        bedrockModelConfig: {
          modelId: 'original-model',
          additionalParams: { nested: { value: 'original' } },
        },
      }
      const systemPrompt = [new TextBlock('Original system prompt')]
      const skills: NonNullable<AgentCoreHarnessAgentConfig['skills']> = [
        {
          git: {
            url: 'https://example.com/original.git',
            path: 'skills/original',
            auth: { credentialArn: 'arn:original', username: 'original-user' },
          },
        },
      ]
      const allowedTools = ['shell']
      const { client, send } = mockClient(
        harnessStream(chunk.messageStop('end_turn')),
        harnessStream(chunk.messageStop('end_turn'))
      )
      const config: AgentCoreHarnessAgentConfig = {
        harnessArn: HARNESS_ARN,
        runtimeSessionId: SESSION_ID,
        client,
        modelConfig,
        systemPrompt,
        skills,
        allowedTools,
        maxIterations: 10,
        maxTokens: 1000,
        timeoutSeconds: 60,
        qualifier: 'original-endpoint',
        runtimeUserId: 'original-user',
        actorId: 'original-actor',
      }
      const agent = new AgentCoreHarnessAgent(config)

      modelConfig.bedrockModelConfig.modelId = 'mutated-model'
      ;(modelConfig.bedrockModelConfig.additionalParams as { nested: { value: string } }).nested.value = 'mutated'
      systemPrompt.push(new TextBlock('Mutated prompt'))
      skills[0]!.git!.url = 'https://example.com/mutated.git'
      skills[0]!.git!.auth!.credentialArn = 'arn:mutated'
      allowedTools[0] = 'browser'
      config.modelConfig = { bedrockModelConfig: { modelId: 'replacement-model' } }
      config.systemPrompt = 'Replacement prompt'
      config.skills = [{ path: '/replacement' }]
      config.allowedTools = ['file_operations']
      config.maxIterations = 99
      config.maxTokens = 99
      config.timeoutSeconds = 99
      config.qualifier = 'replacement-endpoint'
      config.runtimeUserId = 'replacement-user'
      config.actorId = 'replacement-actor'

      await agent.invoke('First')

      expect(send.mock.calls[0]![0].input).toStrictEqual({
        harnessArn: HARNESS_ARN,
        runtimeSessionId: SESSION_ID,
        messages: [{ role: 'user', content: [{ text: 'First' }] }],
        model: {
          bedrockModelConfig: {
            modelId: 'original-model',
            additionalParams: { nested: { value: 'original' } },
          },
        },
        systemPrompt: [{ text: 'Original system prompt' }],
        skills: [
          {
            git: {
              url: 'https://example.com/original.git',
              path: 'skills/original',
              auth: { credentialArn: 'arn:original', username: 'original-user' },
            },
          },
        ],
        allowedTools: ['shell'],
        maxIterations: 10,
        maxTokens: 1000,
        timeoutSeconds: 60,
        qualifier: 'original-endpoint',
        runtimeUserId: 'original-user',
        actorId: 'original-actor',
      })

      const firstInput = send.mock.calls[0]![0].input
      firstInput.model!.bedrockModelConfig!.modelId = 'client-mutated-model'
      firstInput.systemPrompt![0]!.text = 'Client-mutated prompt'
      firstInput.skills![0]!.git!.url = 'https://example.com/client-mutated.git'
      firstInput.allowedTools![0] = 'client-mutated-tool'

      await agent.invoke('Second')

      expect(send.mock.calls[1]![0].input).toStrictEqual({
        harnessArn: HARNESS_ARN,
        runtimeSessionId: SESSION_ID,
        messages: [{ role: 'user', content: [{ text: 'Second' }] }],
        model: {
          bedrockModelConfig: {
            modelId: 'original-model',
            additionalParams: { nested: { value: 'original' } },
          },
        },
        systemPrompt: [{ text: 'Original system prompt' }],
        skills: [
          {
            git: {
              url: 'https://example.com/original.git',
              path: 'skills/original',
              auth: { credentialArn: 'arn:original', username: 'original-user' },
            },
          },
        ],
        allowedTools: ['shell'],
        maxIterations: 10,
        maxTokens: 1000,
        timeoutSeconds: 60,
        qualifier: 'original-endpoint',
        runtimeUserId: 'original-user',
        actorId: 'original-actor',
      })
    })

    it('forwards an explicit zero rather than dropping it as falsy', async () => {
      const { client, send } = mockClient(harnessStream(chunk.messageStop('end_turn')))
      const agent = new AgentCoreHarnessAgent({
        harnessArn: 'arn:harness',
        runtimeSessionId: SESSION_ID,
        client,
        maxTokens: 0,
      })

      await agent.invoke('Hi')

      expect(send.mock.calls[0]![0].input).toStrictEqual({
        harnessArn: 'arn:harness',
        runtimeSessionId: SESSION_ID,
        messages: [{ role: 'user', content: [{ text: 'Hi' }] }],
        maxTokens: 0,
      })
    })

    it('forwards an empty allowedTools override rather than treating it as omitted', async () => {
      const { client, send } = mockClient(harnessStream(chunk.messageStop('end_turn')))
      const agent = new AgentCoreHarnessAgent({
        harnessArn: 'arn:harness',
        runtimeSessionId: SESSION_ID,
        client,
        allowedTools: [],
      })

      await agent.invoke('Hi')

      expect(send.mock.calls[0]![0].input).toStrictEqual({
        harnessArn: 'arn:harness',
        runtimeSessionId: SESSION_ID,
        messages: [{ role: 'user', content: [{ text: 'Hi' }] }],
        allowedTools: [],
      })
    })

    it('omits optional request fields that are not configured', async () => {
      const { client, send } = mockClient(harnessStream(chunk.messageStop('end_turn')))
      const agent = new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID, client })

      await agent.invoke('Hi')

      expect(send.mock.calls[0]![0].input).toStrictEqual({
        harnessArn: 'arn:harness',
        runtimeSessionId: SESSION_ID,
        messages: [{ role: 'user', content: [{ text: 'Hi' }] }],
      })
    })
  })

  describe('concurrency', () => {
    it('rejects a parallel invocation and allows another after the active invocation completes', async () => {
      let releaseFirstRequest: () => void
      const firstRequestGate = new Promise<void>((resolve) => {
        releaseFirstRequest = resolve
      })
      const send = vi
        .fn()
        .mockImplementationOnce(async () => {
          await firstRequestGate
          return { stream: harnessStream(chunk.messageStop('end_turn')) }
        })
        .mockResolvedValueOnce({ stream: harnessStream(chunk.messageStop('end_turn')) })
      const client = { send } as unknown as BedrockAgentCoreClient
      const agent = new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID, client })

      const firstInvocation = agent.invoke('First')
      await vi.waitFor(() => expect(send).toHaveBeenCalledTimes(1))

      await expect(agent.invoke('Second')).rejects.toBeInstanceOf(ConcurrentInvocationError)

      releaseFirstRequest!()
      await expect(firstInvocation).resolves.toMatchObject({ stopReason: 'endTurn' })
      await expect(agent.invoke('Third')).resolves.toMatchObject({ stopReason: 'endTurn' })
      expect(send).toHaveBeenCalledTimes(2)
    })

    it('releases the invocation lock after an error', async () => {
      const originalError = new Error('request failed')
      const send = vi
        .fn()
        .mockRejectedValueOnce(originalError)
        .mockResolvedValueOnce({ stream: harnessStream(chunk.messageStop('end_turn')) })
      const client = { send } as unknown as BedrockAgentCoreClient
      const agent = new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID, client })

      await expect(agent.invoke('First')).rejects.toMatchObject({ constructor: ModelError, cause: originalError })
      await expect(agent.invoke('Second')).resolves.toMatchObject({ stopReason: 'endTurn' })
      expect(send).toHaveBeenCalledTimes(2)
    })

    it('releases the invocation lock when a stream is closed early', async () => {
      const { client, send } = mockClient(
        harnessStream(chunk.messageStart(), chunk.messageStop('end_turn')),
        harnessStream(chunk.messageStop('end_turn'))
      )
      const agent = new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID, client })
      const firstStream = agent.stream('First')

      await expect(firstStream.next()).resolves.toMatchObject({ done: false })
      await expect(agent.invoke('Second')).rejects.toBeInstanceOf(ConcurrentInvocationError)

      await firstStream.return(undefined as never)
      await expect(agent.invoke('Third')).resolves.toMatchObject({ stopReason: 'endTurn' })
      expect(send).toHaveBeenCalledTimes(2)
    })
  })

  describe('stream', () => {
    it('yields an update event per chunk and a final result event', async () => {
      const messageStart = chunk.messageStart()
      const textDelta = chunk.textDelta('Hi')
      const messageStop = chunk.messageStop('end_turn')
      const { client } = mockClient(harnessStream(messageStart, textDelta, messageStop))
      const agent = new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID, client })

      const { items: events, result } = await collectGenerator(agent.stream('Hi'))

      expect(result).toStrictEqual(
        new AgentResult({
          stopReason: 'endTurn',
          lastMessage: new Message({
            role: 'assistant',
            content: [new TextBlock('Hi')],
            trackingId: result.lastMessage.trackingId,
          }),
          invocationState: {},
        })
      )
      expect(events).toStrictEqual([
        new AgentCoreHarnessStreamUpdateEvent(messageStart),
        new AgentCoreHarnessStreamUpdateEvent(textDelta),
        new AgentCoreHarnessStreamUpdateEvent(messageStop),
        new AgentCoreHarnessResultEvent({ result }),
      ])
    })

    it('throws MaxTokensError when the turn stops for max tokens', async () => {
      const { client } = mockClient(
        harnessStream(chunk.messageStart(), chunk.textDelta('partial'), chunk.messageStop('max_tokens'))
      )
      const agent = new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID, client })

      await expect(agent.invoke('Hi')).rejects.toBeInstanceOf(MaxTokensError)
    })

    it.each(['malformed_model_output', 'malformed_tool_use'])(
      'throws ModelError when the turn stops for %s',
      async (stopReason) => {
        const { client } = mockClient(
          harnessStream(chunk.messageStart(), chunk.textDelta('partial'), chunk.messageStop(stopReason))
        )
        const agent = new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID, client })

        await expect(agent.invoke('Hi')).rejects.toBeInstanceOf(ModelError)
      }
    )

    it('throws ModelError when a tool-use input is not valid JSON', async () => {
      const { client } = mockClient(
        harnessStream(
          chunk.messageStart(),
          chunk.toolUseStart('tu-1', 'get_weather'),
          chunk.toolUseDelta('{"city":'),
          chunk.contentBlockStop(),
          chunk.messageStop('tool_use')
        )
      )
      const agent = new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID, client })

      await expect(agent.invoke('Hi')).rejects.toMatchObject({
        constructor: ModelError,
        cause: expect.any(SyntaxError),
      })
    })
  })

  describe('metrics', () => {
    it('surfaces token usage and latency from the terminal metadata event', async () => {
      const { client } = mockClient(
        harnessStream(
          chunk.messageStart(),
          chunk.textDelta('done'),
          chunk.contentBlockStop(),
          chunk.messageStop('end_turn'),
          chunk.metadata({ inputTokens: 100, outputTokens: 20, totalTokens: 120 }, 1500)
        )
      )
      const agent = new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID, client })

      const result = await agent.invoke('Hi')

      expect(result.metrics?.accumulatedUsage).toStrictEqual({ inputTokens: 100, outputTokens: 20, totalTokens: 120 })
      expect(result.metrics?.accumulatedMetrics).toStrictEqual({ latencyMs: 1500 })
      expect(result.metrics?.latestContextSize).toBe(100)
      expect(result.metrics?.projectedContextSize).toBe(120)
      expect(result.metrics?.cycleCount).toBe(1)
      expect(result.metrics?.totalDuration).toBeGreaterThanOrEqual(0)
    })

    it('passes cache-token counts through when the harness reports them', async () => {
      const { client } = mockClient(
        harnessStream(
          chunk.messageStop('end_turn'),
          chunk.metadata({
            inputTokens: 100,
            outputTokens: 20,
            totalTokens: 120,
            cacheReadInputTokens: 40,
            cacheWriteInputTokens: 10,
          })
        )
      )
      const agent = new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID, client })

      const result = await agent.invoke('Hi')

      expect(result.metrics?.accumulatedUsage).toStrictEqual({
        inputTokens: 100,
        outputTokens: 20,
        totalTokens: 120,
        cacheReadInputTokens: 40,
        cacheWriteInputTokens: 10,
      })
    })

    it('accumulates multiple metadata events from one Harness call', async () => {
      const { client } = mockClient(
        harnessStream(
          chunk.messageStart(),
          chunk.messageStop('tool_use'),
          chunk.metadata({ inputTokens: 50, outputTokens: 5, totalTokens: 55 }, 100),
          chunk.messageStart('user'),
          chunk.messageStop('tool_result'),
          chunk.messageStart(),
          chunk.textDelta('done'),
          chunk.messageStop('end_turn'),
          chunk.metadata({ inputTokens: 200, outputTokens: 30, totalTokens: 230 }, 900)
        )
      )
      const agent = new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID, client })

      const result = await agent.invoke('Hi')

      expect(result.metrics?.accumulatedUsage).toStrictEqual({ inputTokens: 250, outputTokens: 35, totalTokens: 285 })
      expect(result.metrics?.accumulatedMetrics.latencyMs).toBe(1000)
      expect(result.metrics?.latestContextSize).toBe(200)
      expect(result.metrics?.projectedContextSize).toBe(230)
      expect(result.metrics?.cycleCount).toBe(2)
    })

    it('omits metrics when the harness reports no usage', async () => {
      const { client } = mockClient(harnessStream(chunk.messageStart(), chunk.messageStop('end_turn')))
      const agent = new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID, client })

      const result = await agent.invoke('Hi')

      expect(result.metrics).toBeUndefined()
    })

    it('counts only the terminal turn usage across a host-tool bounce', async () => {
      // The tool_use turn carries no metadata event (matching the live harness); only the terminal
      // turn reports usage, so the accumulated total is that single terminal turn's counts.
      const bounced = tool({ name: 'ping', description: 'ping', inputSchema: z.object({}), callback: () => 'pong' })
      const { client } = mockClient(
        harnessStream(chunk.toolUseStart('tu-1', 'ping'), chunk.contentBlockStop(), chunk.messageStop('tool_use')),
        harnessStream(
          chunk.messageStart(),
          chunk.textDelta('ok'),
          chunk.messageStop('end_turn'),
          chunk.metadata({ inputTokens: 200, outputTokens: 30, totalTokens: 230 }, 900)
        )
      )
      const { controlClient } = mockControlClient()
      const agent = new AgentCoreHarnessAgent({
        harnessArn: 'arn:harness',
        runtimeSessionId: SESSION_ID,
        client,
        controlClient,
        tools: [bounced],
      })

      const result = await agent.invoke('go')

      expect(result.metrics?.accumulatedUsage).toStrictEqual({ inputTokens: 200, outputTokens: 30, totalTokens: 230 })
      expect(result.metrics?.accumulatedMetrics.latencyMs).toBe(900)
      expect(result.metrics?.cycleCount).toBe(2)
    })

    it('sums usage across turns when multiple terminal turns report it', async () => {
      // Defends the accumulation logic: if the harness ever reports usage on more than one turn of a
      // single invocation, the counts add up rather than overwrite.
      const bounced = tool({ name: 'ping', description: 'ping', inputSchema: z.object({}), callback: () => 'pong' })
      const { client } = mockClient(
        harnessStream(
          chunk.toolUseStart('tu-1', 'ping'),
          chunk.contentBlockStop(),
          chunk.messageStop('tool_use'),
          chunk.metadata({ inputTokens: 50, outputTokens: 5, totalTokens: 55 }, 100)
        ),
        harnessStream(
          chunk.messageStop('end_turn'),
          chunk.metadata({ inputTokens: 200, outputTokens: 30, totalTokens: 230 }, 900)
        )
      )
      const { controlClient } = mockControlClient()
      const agent = new AgentCoreHarnessAgent({
        harnessArn: 'arn:harness',
        runtimeSessionId: SESSION_ID,
        client,
        controlClient,
        tools: [bounced],
      })

      const result = await agent.invoke('go')

      expect(result.metrics?.accumulatedUsage).toStrictEqual({ inputTokens: 250, outputTokens: 35, totalTokens: 285 })
      expect(result.metrics?.accumulatedMetrics.latencyMs).toBe(1000)
      // latest/projected describe the last reporting turn (200/30), not the accumulated total (250/35).
      expect(result.metrics?.latestContextSize).toBe(200)
      expect(result.metrics?.projectedContextSize).toBe(230)
      expect(result.metrics?.cycleCount).toBe(2)
    })
  })

  describe('cancellation', () => {
    it('does not call the harness when the signal is already aborted', async () => {
      const { client, send } = mockClient(harnessStream(chunk.messageStop('end_turn')))
      const agent = new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID, client })

      const result = await agent.invoke('Hi', { cancelSignal: AbortSignal.abort() })

      expect(send).not.toHaveBeenCalled()
      expect(result.stopReason).toBe('cancelled')
    })

    it('returns cancelled without throwing when aborted mid-stream', async () => {
      const controller = new AbortController()
      const send = vi.fn().mockImplementation(() => {
        controller.abort()
        return Promise.reject(new Error('aborted'))
      })
      const client = { send } as unknown as BedrockAgentCoreClient
      const agent = new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID, client })

      const result = await agent.invoke('Hi', { cancelSignal: controller.signal })
      expect(result.stopReason).toBe('cancelled')
    })
  })

  describe('errors', () => {
    it.each(['ThrottlingException', 'ThrottledException'])(
      'maps a %s from send() to ModelThrottledError',
      async (name) => {
        const send = vi.fn().mockRejectedValue(Object.assign(new Error('slow down'), { name }))
        const client = { send } as unknown as BedrockAgentCoreClient
        const agent = new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID, client })

        await expect(agent.invoke('Hi')).rejects.toBeInstanceOf(ModelThrottledError)
      }
    )

    it('maps a context-window overflow thrown from send() to ContextWindowOverflowError', async () => {
      const original = new Error('Input is too long for requested model.')
      const send = vi.fn().mockRejectedValue(original)
      const client = { send } as unknown as BedrockAgentCoreClient
      const agent = new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID, client })

      await expect(agent.invoke('Hi')).rejects.toMatchObject({
        constructor: ContextWindowOverflowError,
        cause: original,
      })
    })

    it('wraps a non-typed send() error in ModelError, preserving the cause', async () => {
      const original = new Error('unexpected')
      const send = vi.fn().mockRejectedValue(original)
      const client = { send } as unknown as BedrockAgentCoreClient
      const agent = new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID, client })

      await expect(agent.invoke('Hi')).rejects.toMatchObject({ constructor: ModelError, cause: original })
    })

    it('throws a ModelError when the stream carries a validationException', async () => {
      const original = { message: 'bad input' }
      const { client } = mockClient(harnessStream({ validationException: original } as InvokeHarnessStreamOutput))
      const agent = new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID, client })

      await expect(agent.invoke('Hi')).rejects.toMatchObject({
        constructor: ModelError,
        message: 'bad input',
        cause: original,
      })
    })

    it('maps a context-window overflow in a validationException to ContextWindowOverflowError', async () => {
      const original = { message: 'prompt is too long' }
      const { client } = mockClient(harnessStream({ validationException: original } as InvokeHarnessStreamOutput))
      const agent = new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID, client })

      try {
        await agent.invoke('Hi')
        expect.fail('Expected error to be thrown')
      } catch (error) {
        expect(error).toBeInstanceOf(ContextWindowOverflowError)
        expect((error as ContextWindowOverflowError).cause).toBe(original)
      }
    })

    it('throws a ModelError when the stream carries an internalServerException', async () => {
      const original = { message: 'boom' }
      const { client } = mockClient(harnessStream({ internalServerException: original } as InvokeHarnessStreamOutput))
      const agent = new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID, client })

      await expect(agent.invoke('Hi')).rejects.toMatchObject({
        constructor: ModelError,
        message: 'boom',
        cause: original,
      })
    })

    it('throws a ModelError when the stream carries a runtimeClientError', async () => {
      const original = { message: 'runtime failed' }
      const { client } = mockClient(harnessStream({ runtimeClientError: original } as InvokeHarnessStreamOutput))
      const agent = new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID, client })

      await expect(agent.invoke('Hi')).rejects.toMatchObject({
        constructor: ModelError,
        message: 'runtime failed',
        cause: original,
      })
    })
  })
})
