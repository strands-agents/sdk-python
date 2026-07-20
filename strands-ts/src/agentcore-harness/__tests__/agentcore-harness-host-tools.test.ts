import { describe, expect, it, vi } from 'vitest'
import type { HarnessTool } from '@aws-sdk/client-bedrock-agentcore'
import {
  AccessDeniedException,
  type BedrockAgentCoreControlClient,
  GetHarnessCommand,
  GetHarnessEndpointCommand,
  ResourceNotFoundException,
  ValidationException,
} from '@aws-sdk/client-bedrock-agentcore-control'
import { z } from 'zod'
import { AgentCoreHarnessAgent } from '../agentcore-harness-agent.js'
import { ModelError, ModelThrottledError, ToolValidationError } from '../../errors.js'
import { tool } from '../../tools/tool-factory.js'
import { Tool, type ToolStreamGenerator } from '../../tools/tool.js'
import { TextBlock, ToolResultBlock } from '../../types/messages.js'
import { DocumentBlock, ImageBlock, VideoBlock } from '../../types/media.js'
import {
  HARNESS_ARN,
  HARNESS_ID,
  SESSION_ID,
  chunk,
  harnessStream,
  mockClient,
  mockControlClient,
} from './harness-test-fixtures.js'

describe('AgentCoreHarnessAgent host tools', () => {
  describe('tool configuration', () => {
    it('does not call GetHarness or override tools when no host tools are configured', async () => {
      const deployedTool: HarnessTool = { type: 'agentcore_browser', name: 'browser' }
      const { controlClient, send: controlSend } = mockControlClient([deployedTool])
      const { client, send } = mockClient(harnessStream(chunk.messageStop('end_turn')))
      const agent = new AgentCoreHarnessAgent({
        harnessArn: HARNESS_ARN,
        runtimeSessionId: SESSION_ID,
        client,
        controlClient,
      })

      await agent.invoke('Hi')

      expect(controlSend).not.toHaveBeenCalled()
      expect(send.mock.calls[0]![0].input).not.toHaveProperty('tools')
    })

    it('rejects invalid invocation content before loading deployed tool configuration', async () => {
      const getWeather = tool({
        name: 'get_weather',
        description: 'Get weather',
        inputSchema: z.object({}),
        callback: () => 'sunny',
      })
      const { controlClient, send: controlSend } = mockControlClient()
      const { client, send } = mockClient()
      const agent = new AgentCoreHarnessAgent({
        harnessArn: HARNESS_ARN,
        runtimeSessionId: SESSION_ID,
        client,
        controlClient,
        tools: [getWeather],
      })

      await expect(
        agent.invoke([
          new TextBlock('Do not send only this text'),
          new ImageBlock({ format: 'png', source: { bytes: new Uint8Array([1]) } }),
        ])
      ).rejects.toThrow("unsupported type 'imageBlock'")
      expect(controlSend).not.toHaveBeenCalled()
      expect(send).not.toHaveBeenCalled()
    })

    it('refreshes the default Harness configuration before every invocation', async () => {
      const deployedTool: HarnessTool = { type: 'agentcore_browser', name: 'browser' }
      const getWeather = tool({
        name: 'get_weather',
        description: 'Get weather',
        inputSchema: z.object({ city: z.string() }),
        callback: () => 'sunny',
      })
      const { controlClient, send: controlSend } = mockControlClient([deployedTool], ['@get_weather'])
      const { client, send } = mockClient(
        harnessStream(chunk.messageStop('end_turn')),
        harnessStream(chunk.messageStop('end_turn'))
      )
      const agent = new AgentCoreHarnessAgent({
        harnessArn: HARNESS_ARN,
        runtimeSessionId: SESSION_ID,
        client,
        controlClient,
        tools: [getWeather],
      })

      await agent.invoke('First')
      await agent.invoke('Second')

      expect(controlSend).toHaveBeenCalledTimes(2)
      expect(controlSend.mock.calls.map((call) => call[0].input)).toStrictEqual([
        { harnessId: HARNESS_ID },
        { harnessId: HARNESS_ID },
      ])
      expect(
        send.mock.calls.map((call) => ({
          tools: call[0].input.tools?.map((harnessTool: HarnessTool) => harnessTool.name),
          allowedTools: call[0].input.allowedTools,
        }))
      ).toStrictEqual([
        { tools: ['browser', 'get_weather'], allowedTools: ['@get_weather'] },
        { tools: ['browser', 'get_weather'], allowedTools: ['@get_weather'] },
      ])
      expect(send.mock.calls[0]![0].input.tools[0]).toStrictEqual(deployedTool)
      expect(send.mock.calls[0]![0].input.tools[0]).not.toBe(deployedTool)
    })

    it('refreshes an explicit DEFAULT qualifier without reading a named endpoint', async () => {
      const hostTool = tool({
        name: 'host',
        description: 'Host tool',
        inputSchema: z.object({}),
        callback: () => 'host result',
      })
      const { controlClient, send: controlSend } = mockControlClient()
      const { client } = mockClient(harnessStream(chunk.messageStop('end_turn')))
      const agent = new AgentCoreHarnessAgent({
        harnessArn: HARNESS_ARN,
        runtimeSessionId: SESSION_ID,
        client,
        controlClient,
        tools: [hostTool],
        qualifier: 'DEFAULT',
      })

      await agent.invoke('Hi')

      expect(controlSend).toHaveBeenCalledOnce()
      expect(controlSend.mock.calls[0]![0]).toBeInstanceOf(GetHarnessCommand)
      expect(controlSend.mock.calls[0]![0].input).toStrictEqual({ harnessId: HARNESS_ID })
    })

    it('tracks a named endpoint live version and caches only that version configuration', async () => {
      const v1Tool: HarnessTool = { type: 'agentcore_browser', name: 'browser-v1' }
      const v2Tool: HarnessTool = { type: 'agentcore_browser', name: 'browser-v2' }
      const hostTool = tool({
        name: 'host',
        description: 'Host tool',
        inputSchema: z.object({}),
        callback: () => 'host result',
      })
      const liveVersions = ['1', '1', '2', '1']
      let endpointRead = 0
      const controlSend = vi.fn((command: GetHarnessCommand | GetHarnessEndpointCommand) => {
        if (command instanceof GetHarnessEndpointCommand) {
          return Promise.resolve({ endpoint: { liveVersion: liveVersions[endpointRead++] } })
        }
        if (command instanceof GetHarnessCommand) {
          const version = command.input.harnessVersion
          return Promise.resolve({
            harness: {
              harnessVersion: version,
              tools: version === '1' ? [v1Tool] : [v2Tool],
            },
          })
        }
        throw new Error('unexpected control-plane command')
      })
      const controlClient = { send: controlSend } as unknown as BedrockAgentCoreControlClient
      const { client, send } = mockClient(
        harnessStream(chunk.messageStop('end_turn')),
        harnessStream(chunk.messageStop('end_turn')),
        harnessStream(chunk.messageStop('end_turn')),
        harnessStream(chunk.messageStop('end_turn'))
      )
      const agent = new AgentCoreHarnessAgent({
        harnessArn: HARNESS_ARN,
        runtimeSessionId: SESSION_ID,
        client,
        controlClient,
        tools: [hostTool],
        qualifier: 'prod',
      })

      await agent.invoke('First')
      await agent.invoke('Second')
      await agent.invoke('Third')
      await agent.invoke('Fourth')

      expect(
        controlSend.mock.calls.map((call) => ({
          command: call[0].constructor.name,
          input: call[0].input,
        }))
      ).toStrictEqual([
        {
          command: 'GetHarnessEndpointCommand',
          input: { harnessId: HARNESS_ID, endpointName: 'prod' },
        },
        {
          command: 'GetHarnessCommand',
          input: { harnessId: HARNESS_ID, harnessVersion: '1' },
        },
        {
          command: 'GetHarnessEndpointCommand',
          input: { harnessId: HARNESS_ID, endpointName: 'prod' },
        },
        {
          command: 'GetHarnessEndpointCommand',
          input: { harnessId: HARNESS_ID, endpointName: 'prod' },
        },
        {
          command: 'GetHarnessCommand',
          input: { harnessId: HARNESS_ID, harnessVersion: '2' },
        },
        {
          command: 'GetHarnessEndpointCommand',
          input: { harnessId: HARNESS_ID, endpointName: 'prod' },
        },
        {
          command: 'GetHarnessCommand',
          input: { harnessId: HARNESS_ID, harnessVersion: '1' },
        },
      ])
      expect(
        send.mock.calls.map((call) => call[0].input.tools.map((harnessTool: HarnessTool) => harnessTool.name))
      ).toStrictEqual([
        ['browser-v1', 'host'],
        ['browser-v1', 'host'],
        ['browser-v2', 'host'],
        ['browser-v1', 'host'],
      ])
    })

    it('rejects a named endpoint without a live version before InvokeHarness', async () => {
      const hostTool = tool({
        name: 'host',
        description: 'Host tool',
        inputSchema: z.object({}),
        callback: () => 'host result',
      })
      const controlSend = vi.fn().mockResolvedValue({ endpoint: {} })
      const controlClient = { send: controlSend } as unknown as BedrockAgentCoreControlClient
      const { client, send } = mockClient()
      const agent = new AgentCoreHarnessAgent({
        harnessArn: HARNESS_ARN,
        runtimeSessionId: SESSION_ID,
        client,
        controlClient,
        tools: [hostTool],
        qualifier: 'prod',
      })

      await expect(agent.invoke('Hi')).rejects.toMatchObject({
        constructor: Error,
        message: "Harness endpoint 'prod' has no live version",
      })
      expect(controlSend).toHaveBeenCalledOnce()
      expect(controlSend.mock.calls[0]![0]).toBeInstanceOf(GetHarnessEndpointCommand)
      expect(send).not.toHaveBeenCalled()
    })

    it('rejects an empty GetHarness response instead of replacing unknown deployed tools', async () => {
      const hostTool = tool({
        name: 'host',
        description: 'Host tool',
        inputSchema: z.object({}),
        callback: () => 'host result',
      })
      const controlSend = vi.fn().mockResolvedValue({})
      const controlClient = { send: controlSend } as unknown as BedrockAgentCoreControlClient
      const { client, send } = mockClient()
      const agent = new AgentCoreHarnessAgent({
        harnessArn: HARNESS_ARN,
        runtimeSessionId: SESSION_ID,
        client,
        controlClient,
        tools: [hostTool],
      })

      await expect(agent.invoke('Hi')).rejects.toMatchObject({
        constructor: Error,
        message: 'GetHarness returned no Harness configuration',
      })
      expect(controlSend).toHaveBeenCalledOnce()
      expect(send).not.toHaveBeenCalled()
    })

    it('rejects pending DEFAULT configuration while the data plane still serves the previous version', async () => {
      const hostTool = tool({
        name: 'host',
        description: 'Host tool',
        inputSchema: z.object({}),
        callback: () => 'host result',
      })
      const controlSend = vi.fn().mockResolvedValue({
        harness: {
          status: 'UPDATING',
          tools: [{ type: 'agentcore_browser', name: 'pending-browser' }],
        },
      })
      const controlClient = { send: controlSend } as unknown as BedrockAgentCoreControlClient
      const { client, send } = mockClient()
      const agent = new AgentCoreHarnessAgent({
        harnessArn: HARNESS_ARN,
        runtimeSessionId: SESSION_ID,
        client,
        controlClient,
        tools: [hostTool],
      })

      await expect(agent.invoke('Hi')).rejects.toMatchObject({
        constructor: Error,
        message:
          'Default Harness configuration cannot be resolved safely while the Harness is UPDATING. Retry after the update completes or use a named endpoint.',
      })
      expect(controlSend).toHaveBeenCalledOnce()
      expect(send).not.toHaveBeenCalled()
    })

    it('uses an explicit allowedTools override instead of the deployed list', async () => {
      const getWeather = tool({
        name: 'get_weather',
        description: 'Get weather',
        inputSchema: z.object({ city: z.string() }),
        callback: () => 'sunny',
      })
      const { controlClient } = mockControlClient([], [])
      const { client, send } = mockClient(harnessStream(chunk.messageStop('end_turn')))
      const agent = new AgentCoreHarnessAgent({
        harnessArn: HARNESS_ARN,
        runtimeSessionId: SESSION_ID,
        client,
        controlClient,
        tools: [getWeather],
        allowedTools: ['@get_weather'],
      })

      await agent.invoke('Hi')

      expect(send.mock.calls[0]![0].input.allowedTools).toStrictEqual(['@get_weather'])
    })

    it('rejects a host tool excluded by the deployed allowlist before InvokeHarness', async () => {
      const getWeather = tool({
        name: 'get_weather',
        description: 'Get weather',
        inputSchema: z.object({ city: z.string() }),
        callback: () => 'sunny',
      })
      const { controlClient, send: controlSend } = mockControlClient([], [])
      const { client, send } = mockClient()
      const agent = new AgentCoreHarnessAgent({
        harnessArn: HARNESS_ARN,
        runtimeSessionId: SESSION_ID,
        client,
        controlClient,
        tools: [getWeather],
      })

      await expect(agent.invoke('Hi')).rejects.toMatchObject({
        constructor: ToolValidationError,
        message:
          "Host tool 'get_weather' is excluded by effective allowedTools. Add '@get_weather', '@get_weather/get_weather', a matching namespace glob, or '*' to allowedTools.",
      })
      expect(controlSend).toHaveBeenCalledOnce()
      expect(send).not.toHaveBeenCalled()
    })

    it('preserves non-throttling control-plane errors and retries the next invocation', async () => {
      const originals = [
        new AccessDeniedException({ $metadata: {}, message: 'access denied' }),
        new ResourceNotFoundException({ $metadata: {}, message: 'not found' }),
        new ValidationException({ $metadata: {}, message: 'invalid request', reason: undefined }),
        new Error('control plane unavailable'),
      ]
      const deployedTool: HarnessTool = { type: 'agentcore_browser', name: 'browser' }
      const getWeather = tool({
        name: 'get_weather',
        description: 'Get weather',
        inputSchema: z.object({ city: z.string() }),
        callback: () => 'sunny',
      })
      const controlSend = vi
        .fn()
        .mockRejectedValueOnce(originals[0])
        .mockRejectedValueOnce(originals[1])
        .mockRejectedValueOnce(originals[2])
        .mockRejectedValueOnce(originals[3])
        .mockResolvedValueOnce({ harness: { tools: [deployedTool] } })
      const controlClient = { send: controlSend } as unknown as BedrockAgentCoreControlClient
      const { client, send } = mockClient(harnessStream(chunk.messageStop('end_turn')))
      const agent = new AgentCoreHarnessAgent({
        harnessArn: HARNESS_ARN,
        runtimeSessionId: SESSION_ID,
        client,
        controlClient,
        tools: [getWeather],
      })

      for (const original of originals) {
        await expect(agent.invoke('First')).rejects.toBe(original)
      }
      expect(send).not.toHaveBeenCalled()

      await agent.invoke('Second')

      expect(controlSend).toHaveBeenCalledTimes(5)
      expect(send).toHaveBeenCalledOnce()
      expect(send.mock.calls[0]![0].input.tools.map((tool: HarnessTool) => tool.name)).toStrictEqual([
        'browser',
        'get_weather',
      ])
    })

    it.each(['ThrottlingException', 'ThrottledException'])(
      'maps a GetHarness %s to ModelThrottledError and preserves the cause',
      async (name) => {
        const original = Object.assign(new Error('slow down'), { name })
        const controlSend = vi.fn().mockRejectedValue(original)
        const controlClient = { send: controlSend } as unknown as BedrockAgentCoreControlClient
        const { client, send } = mockClient()
        const hostTool = tool({
          name: 'get_weather',
          description: 'Get weather',
          inputSchema: z.object({ city: z.string() }),
          callback: () => 'sunny',
        })
        const agent = new AgentCoreHarnessAgent({
          harnessArn: HARNESS_ARN,
          runtimeSessionId: SESSION_ID,
          client,
          controlClient,
          tools: [hostTool],
        })

        await expect(agent.invoke('Hi')).rejects.toMatchObject({
          constructor: ModelThrottledError,
          message: 'slow down',
          cause: original,
        })
        expect(controlSend).toHaveBeenCalledOnce()
        expect(send).not.toHaveBeenCalled()
      }
    )

    it('rejects a host tool name that conflicts with a deployed harness tool', async () => {
      const deployedTool: HarnessTool = { type: 'agentcore_browser', name: 'duplicate' }
      const hostTool = tool({
        name: 'duplicate',
        description: 'duplicate',
        inputSchema: z.object({}),
        callback: () => 'host result',
      })
      const { controlClient, send: controlSend } = mockControlClient([deployedTool])
      const { client, send } = mockClient(harnessStream(chunk.messageStop('end_turn')))
      const agent = new AgentCoreHarnessAgent({
        harnessArn: HARNESS_ARN,
        runtimeSessionId: SESSION_ID,
        client,
        controlClient,
        tools: [hostTool],
      })

      await expect(agent.invoke('Hi')).rejects.toMatchObject({
        constructor: ToolValidationError,
        message:
          "Host tool 'duplicate' conflicts with deployed tool type 'agentcore_browser'. A local handler can bind only to a deployed inline_function.",
      })
      expect(controlSend).toHaveBeenCalledTimes(1)
      expect(send).not.toHaveBeenCalled()
    })

    it('rejects a stream-only host tool during construction', () => {
      const streamOnlyTool = new (class extends Tool {
        name = 'streamer'
        description = 'stream only'
        toolSpec = {
          name: 'streamer',
          description: 'stream only',
          inputSchema: { type: 'object' as const, properties: {} },
        }
        // eslint-disable-next-line require-yield
        async *stream(): ToolStreamGenerator {
          return new ToolResultBlock({ toolUseId: 'x', status: 'success', content: [] })
        }
      })()

      expect(
        () =>
          new AgentCoreHarnessAgent({
            harnessArn: HARNESS_ARN,
            runtimeSessionId: SESSION_ID,
            tools: [streamOnlyTool as never],
          })
      ).toThrowError(
        expect.objectContaining({
          name: 'ToolValidationError',
          message:
            "Host tool 'streamer' must implement invoke(). AgentCoreHarnessAgent does not support stream-only tools.",
        })
      )
    })
  })

  describe('host tool bounce', () => {
    it('runs a host tool and resumes with a second InvokeHarness call carrying toolUse and toolResult', async () => {
      const callback = vi.fn().mockResolvedValue('72F, sunny')
      const getWeather = tool({
        name: 'get_weather',
        description: 'Get weather',
        inputSchema: z.object({ city: z.string() }),
        callback,
      })

      // First call streams a tool_use; second call streams the final answer.
      const { client, send } = mockClient(
        harnessStream(
          chunk.messageStart(),
          chunk.toolUseStart('tu-1', 'get_weather'),
          chunk.toolUseDelta('{"city":"Seattle"}'),
          chunk.contentBlockStop(),
          chunk.messageStop('tool_use')
        ),
        harnessStream(chunk.messageStart(), chunk.textDelta('It is 72F.'), chunk.messageStop('end_turn'))
      )
      const { controlClient, send: controlSend } = mockControlClient()
      const agent = new AgentCoreHarnessAgent({
        harnessArn: 'arn:harness',
        runtimeSessionId: SESSION_ID,
        client,
        controlClient,
        tools: [getWeather],
      })

      const result = await agent.invoke('weather in Seattle?')

      // The callback ran on the host with the parsed input.
      expect(callback.mock.calls[0]![0]).toStrictEqual({ city: 'Seattle' })

      // Two InvokeHarness calls were made.
      expect(send).toHaveBeenCalledTimes(2)
      expect(controlSend).toHaveBeenCalledTimes(2)
      expect(
        send.mock.calls.map((call) => call[0].input.tools?.map((harnessTool: HarnessTool) => harnessTool.name))
      ).toStrictEqual([['get_weather'], ['get_weather']])
      expect(send.mock.calls.map((call) => call[0].input.allowedTools)).toStrictEqual([undefined, undefined])

      // The resume call carries the assistant toolUse followed by the user toolResult.
      const resumeMessages = send.mock.calls[1]![0].input.messages
      expect(resumeMessages).toStrictEqual([
        {
          role: 'assistant',
          content: [
            {
              toolUse: {
                toolUseId: 'tu-1',
                name: 'get_weather',
                input: { city: 'Seattle' },
                type: 'tool_use',
              },
            },
          ],
        },
        {
          role: 'user',
          content: [
            {
              toolResult: {
                toolUseId: 'tu-1',
                content: [{ text: '72F, sunny' }],
                status: 'success',
                type: 'tool_use',
              },
            },
          ],
        },
      ])

      expect(result.stopReason).toBe('endTurn')
      expect((result.lastMessage.content[0] as TextBlock).text).toBe('It is 72F.')
    })

    it('uses the previous snapshot when a continuation refresh fails after a host side effect', async () => {
      const warning = vi.spyOn(console, 'warn').mockImplementation(() => {})
      const callback = vi.fn(() => 'completed side effect')
      const hostTool = tool({
        name: 'host',
        description: 'Host tool',
        inputSchema: z.object({}),
        callback,
      })
      const controlSend = vi
        .fn()
        .mockResolvedValueOnce({ harness: { tools: [] } })
        .mockRejectedValueOnce(new Error('control plane unavailable'))
      const controlClient = { send: controlSend } as unknown as BedrockAgentCoreControlClient
      const { client, send } = mockClient(
        harnessStream(chunk.toolUseStart('tu-1', 'host'), chunk.contentBlockStop(), chunk.messageStop('tool_use')),
        harnessStream(chunk.messageStop('end_turn'))
      )
      const agent = new AgentCoreHarnessAgent({
        harnessArn: HARNESS_ARN,
        runtimeSessionId: SESSION_ID,
        client,
        controlClient,
        tools: [hostTool],
      })

      const result = await agent.invoke('Run host')

      expect(result.stopReason).toBe('endTurn')
      expect(callback).toHaveBeenCalledOnce()
      expect(controlSend).toHaveBeenCalledTimes(2)
      expect(send).toHaveBeenCalledTimes(2)
      expect(send.mock.calls[1]![0].input.tools.map((harnessTool: HarnessTool) => harnessTool.name)).toStrictEqual([
        'host',
      ])
      expect(warning).toHaveBeenCalledWith(
        `harness_arn=<${HARNESS_ARN}>, qualifier=<DEFAULT>, error=<control plane unavailable> | unable to refresh harness tool configuration, using previous snapshot for mandatory host result continuation`
      )
      warning.mockRestore()
    })

    it('runs multiple host tools in one bounce concurrently by default', async () => {
      // beta resolves the gate alpha awaits, so alpha can only finish if beta starts first — which
      // requires concurrent execution. Sequential execution would deadlock (alpha never returns).
      let releaseAlpha: () => void = () => {}
      const betaStarted = new Promise<void>((resolve) => {
        releaseAlpha = resolve
      })
      const alpha = tool({
        name: 'alpha',
        description: 'Waits for beta to start',
        inputSchema: z.object({}),
        callback: async () => {
          await betaStarted
          return 'alpha'
        },
      })
      const beta = tool({
        name: 'beta',
        description: 'Releases alpha',
        inputSchema: z.object({}),
        callback: () => {
          releaseAlpha()
          return 'beta'
        },
      })
      const { client, send } = mockClient(
        harnessStream(
          chunk.messageStart(),
          chunk.toolUseStart('tu-1', 'alpha'),
          chunk.contentBlockStop(),
          chunk.toolUseStart('tu-2', 'beta'),
          chunk.contentBlockStop(),
          chunk.messageStop('tool_use')
        ),
        harnessStream(chunk.messageStop('end_turn'))
      )
      const { controlClient } = mockControlClient()
      const agent = new AgentCoreHarnessAgent({
        harnessArn: 'arn:harness',
        runtimeSessionId: SESSION_ID,
        client,
        controlClient,
        tools: [alpha, beta],
      })

      const result = await agent.invoke('run both')

      expect(result.stopReason).toBe('endTurn')
      // Results resume in input order regardless of completion order.
      expect(send.mock.calls[1]![0].input.messages[1]).toStrictEqual({
        role: 'user',
        content: [
          {
            toolResult: {
              toolUseId: 'tu-1',
              content: [{ text: 'alpha' }],
              status: 'success',
              type: 'tool_use',
            },
          },
          {
            toolResult: {
              toolUseId: 'tu-2',
              content: [{ text: 'beta' }],
              status: 'success',
              type: 'tool_use',
            },
          },
        ],
      })
    })

    it('runs multiple host tools one at a time when toolExecutor is sequential', async () => {
      const order: string[] = []
      const alpha = tool({
        name: 'alpha',
        description: 'First',
        inputSchema: z.object({}),
        callback: async () => {
          order.push('alpha:start')
          await new Promise((resolve) => setTimeout(resolve, 0))
          order.push('alpha:end')
          return 'alpha'
        },
      })
      const beta = tool({
        name: 'beta',
        description: 'Second',
        inputSchema: z.object({}),
        callback: () => {
          order.push('beta:start')
          return 'beta'
        },
      })
      const { client } = mockClient(
        harnessStream(
          chunk.messageStart(),
          chunk.toolUseStart('tu-1', 'alpha'),
          chunk.contentBlockStop(),
          chunk.toolUseStart('tu-2', 'beta'),
          chunk.contentBlockStop(),
          chunk.messageStop('tool_use')
        ),
        harnessStream(chunk.messageStop('end_turn'))
      )
      const { controlClient } = mockControlClient()
      const agent = new AgentCoreHarnessAgent({
        harnessArn: 'arn:harness',
        runtimeSessionId: SESSION_ID,
        client,
        controlClient,
        tools: [alpha, beta],
        toolExecutor: 'sequential',
      })

      await agent.invoke('run both')

      // alpha fully completes before beta starts; concurrent execution would interleave them.
      expect(order).toStrictEqual(['alpha:start', 'alpha:end', 'beta:start'])
    })

    it('preserves an empty string returned by a host tool', async () => {
      const empty = tool({
        name: 'empty',
        description: 'Returns an empty string',
        inputSchema: z.object({}),
        callback: () => '',
      })
      const { client, send } = mockClient(
        harnessStream(chunk.toolUseStart('tu-1', 'empty'), chunk.contentBlockStop(), chunk.messageStop('tool_use')),
        harnessStream(chunk.messageStop('end_turn'))
      )
      const { controlClient } = mockControlClient()
      const agent = new AgentCoreHarnessAgent({
        harnessArn: 'arn:harness',
        runtimeSessionId: SESSION_ID,
        client,
        controlClient,
        tools: [empty],
      })

      await agent.invoke('run empty')

      expect(send.mock.calls[1]![0].input.messages[1]).toStrictEqual({
        role: 'user',
        content: [
          {
            toolResult: {
              toolUseId: 'tu-1',
              content: [{ text: '""' }],
              status: 'success',
              type: 'tool_use',
            },
          },
        ],
      })
    })

    it('encodes a JSON-compatible host tool value as JSON text', async () => {
      const json = tool({
        name: 'json',
        description: 'Returns structured data',
        inputSchema: z.object({}),
        callback: () => ({ weather: 'sunny', temperature: 72 }),
      })
      const { client, send } = mockClient(
        harnessStream(chunk.toolUseStart('tu-1', 'json'), chunk.contentBlockStop(), chunk.messageStop('tool_use')),
        harnessStream(chunk.messageStop('end_turn'))
      )
      const { controlClient } = mockControlClient()
      const agent = new AgentCoreHarnessAgent({
        harnessArn: HARNESS_ARN,
        runtimeSessionId: SESSION_ID,
        client,
        controlClient,
        tools: [json],
      })

      await agent.invoke('run json')

      expect(send.mock.calls[1]![0].input.messages[1]).toStrictEqual({
        role: 'user',
        content: [
          {
            toolResult: {
              toolUseId: 'tu-1',
              content: [{ text: '{"weather":"sunny","temperature":72}' }],
              status: 'success',
              type: 'tool_use',
            },
          },
        ],
      })
    })

    it.each([
      {
        description: 'image',
        value: new ImageBlock({ format: 'png', source: { bytes: new Uint8Array([1]) } }),
      },
      {
        description: 'video',
        value: new VideoBlock({ format: 'mp4', source: { bytes: new Uint8Array([2]) } }),
      },
      {
        description: 'document',
        value: new DocumentBlock({ name: 'notes', format: 'txt', source: { text: 'hello' } }),
      },
    ])('commits an error result when a host tool returns unsupported $description content', async ({ value }) => {
      const media = tool({
        name: 'media',
        description: 'Returns unsupported media',
        inputSchema: z.object({}),
        callback: () => value,
      })
      const { client, send } = mockClient(
        harnessStream(chunk.toolUseStart('tu-1', 'media'), chunk.contentBlockStop(), chunk.messageStop('tool_use')),
        harnessStream(chunk.messageStop('end_turn'))
      )
      const { controlClient } = mockControlClient()
      const agent = new AgentCoreHarnessAgent({
        harnessArn: HARNESS_ARN,
        runtimeSessionId: SESSION_ID,
        client,
        controlClient,
        tools: [media],
      })

      await agent.invoke('run media')

      expect(send).toHaveBeenCalledTimes(2)
      expect(send.mock.calls[1]![0].input.messages[1]).toStrictEqual({
        role: 'user',
        content: [
          {
            toolResult: {
              toolUseId: 'tu-1',
              content: [
                {
                  text: `Error: Tool-result content at block index 0, item index 0 has unsupported type '${value.type}'. AgentCore Harness host tools can return only text or JSON-compatible values.`,
                },
              ],
              status: 'error',
              type: 'tool_use',
            },
          },
        ],
      })
    })

    it('returns an error tool result when a host tool throws, then resumes', async () => {
      const failing = tool({
        name: 'boom',
        description: 'Fails',
        inputSchema: z.object({}),
        callback: () => {
          throw new Error('kaboom')
        },
      })
      const { client, send } = mockClient(
        harnessStream(chunk.toolUseStart('tu-1', 'boom'), chunk.contentBlockStop(), chunk.messageStop('tool_use')),
        harnessStream(chunk.messageStop('end_turn'))
      )
      const { controlClient } = mockControlClient()
      const agent = new AgentCoreHarnessAgent({
        harnessArn: 'arn:harness',
        runtimeSessionId: SESSION_ID,
        client,
        controlClient,
        tools: [failing],
      })

      await agent.invoke('run boom')

      expect(send).toHaveBeenCalledTimes(2)
      expect(send.mock.calls[1]![0].input.messages[1]).toStrictEqual({
        role: 'user',
        content: [
          {
            toolResult: {
              toolUseId: 'tu-1',
              content: [{ text: 'Error: kaboom' }],
              status: 'error',
              type: 'tool_use',
            },
          },
        ],
      })
    })

    it('does not bounce for a tool the agent does not own', async () => {
      const { client, send } = mockClient(
        harnessStream(
          chunk.toolUseStart('tu-1', 'vended_tool'),
          chunk.contentBlockStop(),
          chunk.messageStop('tool_use')
        )
      )
      const agent = new AgentCoreHarnessAgent({ harnessArn: 'arn:harness', runtimeSessionId: SESSION_ID, client })

      const result = await agent.invoke('Hi')

      // Unknown (vended) tool use is terminal; no resume call.
      expect(send).toHaveBeenCalledTimes(1)
      expect(result.stopReason).toBe('toolUse')
    })

    it('rejects an unexpected stream request for a registered host tool excluded by effective allowedTools', async () => {
      const callback = vi.fn(() => 'sunny')
      const getWeather = tool({
        name: 'get_weather',
        description: 'Get weather',
        inputSchema: z.object({}),
        callback,
      })
      const { client, send } = mockClient(
        harnessStream(
          chunk.toolUseStart('tu-1', 'get_weather'),
          chunk.contentBlockStop(),
          chunk.messageStop('tool_use')
        )
      )
      const agent = new AgentCoreHarnessAgent({
        harnessArn: HARNESS_ARN,
        runtimeSessionId: SESSION_ID,
        client,
        tools: [getWeather],
      })
      ;(
        agent as unknown as {
          _resolveToolConfiguration(): Promise<{ tools: HarnessTool[]; allowedTools?: string[] }>
        }
      )._resolveToolConfiguration = vi.fn().mockResolvedValue({ tools: [], allowedTools: [] })

      await expect(agent.invoke('Hi')).rejects.toMatchObject({
        constructor: ModelError,
        message:
          "Harness requested host tool 'get_weather' even though it is excluded by effective allowedTools. The callback was not executed.",
      })
      expect(callback).not.toHaveBeenCalled()
      expect(send).toHaveBeenCalledOnce()
    })
  })
})
