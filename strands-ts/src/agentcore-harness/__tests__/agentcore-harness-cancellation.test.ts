import type { InvokeHarnessStreamOutput } from '@aws-sdk/client-bedrock-agentcore'
import type { BedrockAgentCoreControlClient } from '@aws-sdk/client-bedrock-agentcore-control'
import { describe, expect, it, vi } from 'vitest'
import { z } from 'zod'
import { ModelError } from '../../errors.js'
import { tool } from '../../tools/tool-factory.js'
import { TextBlock } from '../../types/messages.js'
import { AgentCoreHarnessAgent } from '../agentcore-harness-agent.js'
import { SESSION_ID, chunk, harnessStream, mockClient, mockControlClient } from './harness-test-fixtures.js'

describe('AgentCoreHarnessAgent host-tool cancellation', () => {
  it('commits sequential results, drains later tool requests, and allows reuse after cancellation', async () => {
    const controller = new AbortController()
    const firstCallback = vi.fn(() => {
      controller.abort()
      return 'first result'
    })
    const secondCallback = vi.fn(() => 'second result')
    const first = tool({
      name: 'first',
      description: 'First tool',
      inputSchema: z.object({}),
      callback: firstCallback,
    })
    const second = tool({
      name: 'second',
      description: 'Second tool',
      inputSchema: z.object({}),
      callback: secondCallback,
    })
    const { client, send } = mockClient(
      harnessStream(
        chunk.messageStart(),
        chunk.toolUseStart('tu-1', 'first'),
        chunk.contentBlockStop(),
        chunk.toolUseStart('tu-2', 'second'),
        chunk.contentBlockStop(),
        chunk.messageStop('tool_use')
      ),
      harnessStream(
        chunk.messageStart(),
        chunk.toolUseStart('tu-3', 'second'),
        chunk.contentBlockStop(),
        chunk.messageStop('tool_use')
      ),
      harnessStream(chunk.messageStart(), chunk.textDelta('Cancellation committed.'), chunk.messageStop('max_tokens')),
      harnessStream(chunk.messageStart(), chunk.textDelta('Ready again.'), chunk.messageStop('end_turn'))
    )
    const { controlClient } = mockControlClient()
    const agent = new AgentCoreHarnessAgent({
      harnessArn: 'arn:harness',
      runtimeSessionId: SESSION_ID,
      client,
      controlClient,
      tools: [first, second],
      toolExecutor: 'sequential',
    })

    const cancelledResult = await agent.invoke('Hi', { cancelSignal: controller.signal })
    const nextResult = await agent.invoke('Continue')

    expect(cancelledResult.stopReason).toBe('cancelled')
    expect(nextResult.stopReason).toBe('endTurn')
    expect(nextResult.lastMessage.content).toStrictEqual([new TextBlock('Ready again.')])
    expect(firstCallback).toHaveBeenCalledOnce()
    expect(secondCallback).not.toHaveBeenCalled()
    expect(send).toHaveBeenCalledTimes(4)
    expect(send.mock.calls.map((call) => call[1])).toStrictEqual([{ abortSignal: controller.signal }, {}, {}, {}])
    expect(send.mock.calls[1]![0].input.messages[1]).toStrictEqual({
      role: 'user',
      content: [
        {
          toolResult: {
            toolUseId: 'tu-1',
            content: [{ text: 'first result' }],
            status: 'success',
            type: 'tool_use',
          },
        },
        {
          toolResult: {
            toolUseId: 'tu-2',
            content: [{ text: 'Tool execution cancelled' }],
            status: 'error',
            type: 'tool_use',
          },
        },
      ],
    })
    expect(send.mock.calls[2]![0].input.messages[1]).toStrictEqual({
      role: 'user',
      content: [
        {
          toolResult: {
            toolUseId: 'tu-3',
            content: [{ text: 'Tool execution cancelled' }],
            status: 'error',
            type: 'tool_use',
          },
        },
      ],
    })
  })

  it('does not launch concurrent host tools when cancellation precedes execution', async () => {
    const controller = new AbortController()
    const firstCallback = vi.fn(() => 'first result')
    const secondCallback = vi.fn(() => 'second result')
    const first = tool({
      name: 'first',
      description: 'First tool',
      inputSchema: z.object({}),
      callback: firstCallback,
    })
    const second = tool({
      name: 'second',
      description: 'Second tool',
      inputSchema: z.object({}),
      callback: secondCallback,
    })
    async function* toolRequestThenCancel(): AsyncGenerator<InvokeHarnessStreamOutput> {
      yield chunk.messageStart()
      yield chunk.toolUseStart('tu-1', 'first')
      yield chunk.contentBlockStop()
      yield chunk.toolUseStart('tu-2', 'second')
      yield chunk.contentBlockStop()
      yield chunk.messageStop('tool_use')
      controller.abort()
    }
    const { client, send } = mockClient(
      toolRequestThenCancel(),
      harnessStream(chunk.messageStart(), chunk.messageStop('end_turn'))
    )
    const { controlClient } = mockControlClient()
    const agent = new AgentCoreHarnessAgent({
      harnessArn: 'arn:harness',
      runtimeSessionId: SESSION_ID,
      client,
      controlClient,
      tools: [first, second],
    })

    const result = await agent.invoke('Hi', { cancelSignal: controller.signal })

    expect(result.stopReason).toBe('cancelled')
    expect(firstCallback).not.toHaveBeenCalled()
    expect(secondCallback).not.toHaveBeenCalled()
    expect(send).toHaveBeenCalledTimes(2)
    expect(send.mock.calls[1]![0].input.messages[1].content).toStrictEqual([
      {
        toolResult: {
          toolUseId: 'tu-1',
          content: [{ text: 'Tool execution cancelled' }],
          status: 'error',
          type: 'tool_use',
        },
      },
      {
        toolResult: {
          toolUseId: 'tu-2',
          content: [{ text: 'Tool execution cancelled' }],
          status: 'error',
          type: 'tool_use',
        },
      },
    ])
  })

  it('finishes every concurrent host tool that started before cancellation', async () => {
    const controller = new AbortController()
    const firstCallback = vi.fn(() => {
      controller.abort()
      return 'first result'
    })
    const secondCallback = vi.fn(() => 'second result')
    const first = tool({
      name: 'first',
      description: 'First tool',
      inputSchema: z.object({}),
      callback: firstCallback,
    })
    const second = tool({
      name: 'second',
      description: 'Second tool',
      inputSchema: z.object({}),
      callback: secondCallback,
    })
    const { client, send } = mockClient(
      harnessStream(
        chunk.messageStart(),
        chunk.toolUseStart('tu-1', 'first'),
        chunk.contentBlockStop(),
        chunk.toolUseStart('tu-2', 'second'),
        chunk.contentBlockStop(),
        chunk.messageStop('tool_use')
      ),
      harnessStream(chunk.messageStart(), chunk.messageStop('end_turn'))
    )
    const { controlClient } = mockControlClient()
    const agent = new AgentCoreHarnessAgent({
      harnessArn: 'arn:harness',
      runtimeSessionId: SESSION_ID,
      client,
      controlClient,
      tools: [first, second],
    })

    const result = await agent.invoke('Hi', { cancelSignal: controller.signal })

    expect(result.stopReason).toBe('cancelled')
    expect(firstCallback).toHaveBeenCalledOnce()
    expect(secondCallback).toHaveBeenCalledOnce()
    expect(send.mock.calls[1]![0].input.messages[1].content).toStrictEqual([
      {
        toolResult: {
          toolUseId: 'tu-1',
          content: [{ text: 'first result' }],
          status: 'success',
          type: 'tool_use',
        },
      },
      {
        toolResult: {
          toolUseId: 'tu-2',
          content: [{ text: 'second result' }],
          status: 'success',
          type: 'tool_use',
        },
      },
    ])
  })

  it('throws when a mandatory host-result continuation fails after cancellation', async () => {
    const controller = new AbortController()
    const original = new Error('commit status unknown')
    const hostTool = tool({
      name: 'host',
      description: 'Host tool',
      inputSchema: z.object({}),
      callback: () => {
        controller.abort()
        return 'completed result'
      },
    })
    const { client, send } = mockClient(
      harnessStream(
        chunk.messageStart(),
        chunk.toolUseStart('tu-1', 'host'),
        chunk.contentBlockStop(),
        chunk.messageStop('tool_use')
      )
    )
    send.mockRejectedValueOnce(original)
    const { controlClient } = mockControlClient()
    const agent = new AgentCoreHarnessAgent({
      harnessArn: 'arn:harness',
      runtimeSessionId: SESSION_ID,
      client,
      controlClient,
      tools: [hostTool],
    })

    await expect(agent.invoke('Hi', { cancelSignal: controller.signal })).rejects.toMatchObject({
      constructor: ModelError,
      message: 'commit status unknown',
      cause: original,
    })
    expect(send).toHaveBeenCalledTimes(2)
    expect(send.mock.calls[1]![1]).toStrictEqual({})
  })

  it('returns cancelled without invoking the harness when aborted during tool resolution', async () => {
    const warning = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const controller = new AbortController()
    const { client, send } = mockClient(harnessStream(chunk.messageStop('end_turn')))
    const controlSend = vi.fn().mockImplementation(() => {
      controller.abort()
      return Promise.reject(new Error('aborted'))
    })
    const controlClient = { send: controlSend } as unknown as BedrockAgentCoreControlClient
    const hostTool = tool({ name: 'host', description: 'Host tool', inputSchema: z.object({}), callback: () => 'ok' })
    const agent = new AgentCoreHarnessAgent({
      harnessArn: 'arn:harness',
      runtimeSessionId: SESSION_ID,
      client,
      controlClient,
      tools: [hostTool],
    })

    const result = await agent.invoke('Hi', { cancelSignal: controller.signal })

    expect(result.stopReason).toBe('cancelled')
    expect(send).not.toHaveBeenCalled()
    expect(warning).not.toHaveBeenCalled()
    warning.mockRestore()
  })
})
