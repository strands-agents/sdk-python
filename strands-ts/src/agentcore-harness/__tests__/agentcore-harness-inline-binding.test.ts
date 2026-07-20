import type { HarnessInlineFunctionConfig, HarnessTool } from '@aws-sdk/client-bedrock-agentcore'
import { describe, expect, it, vi } from 'vitest'
import { z } from 'zod'
import { AgentCoreHarnessAgent } from '../agentcore-harness-agent.js'
import { ToolValidationError } from '../../errors.js'
import { tool } from '../../tools/tool-factory.js'
import {
  HARNESS_ARN,
  SESSION_ID,
  chunk,
  harnessStream,
  mockClient,
  mockControlClient,
} from './harness-test-fixtures.js'

describe('AgentCoreHarnessAgent deployed inline-function binding', () => {
  it('binds a compatible local handler without duplicating the deployed declaration', async () => {
    const deployedTool = inlineFunction({
      description: 'Request approval',
      inputSchema: {
        required: ['requestId'],
        properties: { requestId: { type: 'string' } },
        additionalProperties: false,
        type: 'object',
      },
    })
    const callback = vi.fn().mockResolvedValue({ approved: true })
    const localTool = tool({
      name: 'approval',
      description: 'Request approval',
      inputSchema: z.object({ requestId: z.string() }),
      callback,
    })
    const { controlClient } = mockControlClient([deployedTool], ['@approval'])
    const { client, send } = mockClient(
      harnessStream(
        chunk.messageStart(),
        chunk.toolUseStart('approval-1', 'approval'),
        chunk.toolUseDelta('{"requestId":"req-1"}'),
        chunk.contentBlockStop(),
        chunk.messageStop('tool_use')
      ),
      harnessStream(chunk.messageStart(), chunk.textDelta('Approved.'), chunk.messageStop('end_turn'))
    )
    const agent = new AgentCoreHarnessAgent({
      harnessArn: HARNESS_ARN,
      runtimeSessionId: SESSION_ID,
      client,
      controlClient,
      tools: [localTool],
    })

    const result = await agent.invoke('Approve request req-1')

    expect(callback.mock.calls).toStrictEqual([[{ requestId: 'req-1' }, undefined]])
    expect(
      send.mock.calls.map((call) => ({
        tools: call[0].input.tools,
        allowedTools: call[0].input.allowedTools,
      }))
    ).toStrictEqual([
      { tools: [deployedTool], allowedTools: ['@approval'] },
      { tools: [deployedTool], allowedTools: ['@approval'] },
    ])
    expect(result.stopReason).toBe('endTurn')
  })

  it.each([
    {
      mismatch: 'description',
      deployedTool: inlineFunction({
        description: 'Approve a payment',
        inputSchema: {
          type: 'object',
          properties: { requestId: { type: 'string' } },
          required: ['requestId'],
          additionalProperties: false,
        },
      }),
      message:
        "Host tool 'approval' cannot bind to the deployed inline function because its description differs. Make the local tool definition match the deployed inline function.",
    },
    {
      mismatch: 'input schema',
      deployedTool: inlineFunction({
        description: 'Request approval',
        inputSchema: {
          type: 'object',
          properties: { paymentId: { type: 'string' } },
          required: ['paymentId'],
          additionalProperties: false,
        },
      }),
      message:
        "Host tool 'approval' cannot bind to the deployed inline function because its input schema differs. Make the local tool definition match the deployed inline function.",
    },
  ])('rejects a same-name inline function with a different $mismatch', async ({ deployedTool, message }) => {
    const localTool = tool({
      name: 'approval',
      description: 'Request approval',
      inputSchema: z.object({ requestId: z.string() }),
      callback: () => ({ approved: true }),
    })
    const { controlClient, send: controlSend } = mockControlClient([deployedTool], ['@approval'])
    const { client, send } = mockClient()
    const agent = new AgentCoreHarnessAgent({
      harnessArn: HARNESS_ARN,
      runtimeSessionId: SESSION_ID,
      client,
      controlClient,
      tools: [localTool],
    })

    await expect(agent.invoke('Approve request req-1')).rejects.toMatchObject({
      constructor: ToolValidationError,
      message,
    })
    expect(controlSend).toHaveBeenCalledOnce()
    expect(send).not.toHaveBeenCalled()
  })

  it('returns a deployed inline-function request when no local handler is configured', async () => {
    const { controlClient, send: controlSend } = mockControlClient()
    const { client, send } = mockClient(
      harnessStream(
        chunk.messageStart(),
        chunk.toolUseStart('approval-1', 'approval'),
        chunk.toolUseDelta('{"requestId":"req-1"}'),
        chunk.contentBlockStop(),
        chunk.messageStop('tool_use')
      )
    )
    const agent = new AgentCoreHarnessAgent({
      harnessArn: HARNESS_ARN,
      runtimeSessionId: SESSION_ID,
      client,
      controlClient,
    })

    const result = await agent.invoke('Approve request req-1')

    expect(controlSend).not.toHaveBeenCalled()
    expect(send.mock.calls[0]![0].input).not.toHaveProperty('tools')
    expect({
      stopReason: result.stopReason,
      content: result.lastMessage.content.map((block) => block.toJSON()),
    }).toStrictEqual({
      stopReason: 'toolUse',
      content: [
        {
          toolUse: {
            toolUseId: 'approval-1',
            name: 'approval',
            input: { requestId: 'req-1' },
          },
        },
      ],
    })
  })

  it.each([
    ['remote_mcp', 'mcp_1'],
    ['agentcore_gateway', 'gateway_1'],
    ['agentcore_browser', 'browser_1'],
    ['agentcore_code_interpreter', 'code_interpreter_1'],
    ['inline_function', 'inline_function_1'],
  ] as const)(
    'rejects a host tool that conflicts with an unnamed %s tool runtime name',
    async (deployedToolType, generatedName) => {
      const deployedTool: HarnessTool =
        deployedToolType === 'inline_function'
          ? {
              type: deployedToolType,
              config: {
                inlineFunction: {
                  description: 'Generated-name collision',
                  inputSchema: {
                    type: 'object',
                    properties: {},
                    additionalProperties: false,
                  },
                },
              },
            }
          : { type: deployedToolType }
      const localTool = tool({
        name: generatedName,
        description: 'Generated-name collision',
        inputSchema: z.object({}),
        callback: () => 'host result',
      })
      const { controlClient, send: controlSend } = mockControlClient([
        { type: 'agentcore_gateway', name: 'existing' },
        deployedTool,
      ])
      const { client, send } = mockClient()
      const agent = new AgentCoreHarnessAgent({
        harnessArn: HARNESS_ARN,
        runtimeSessionId: SESSION_ID,
        client,
        controlClient,
        tools: [localTool],
      })

      await expect(agent.invoke('Run the tool')).rejects.toMatchObject({
        constructor: ToolValidationError,
        message: `Host tool '${generatedName}' conflicts with the runtime-generated name of the unnamed deployed tool at index 1 (type '${deployedToolType}'). Assign the deployed tool an explicit unique name or rename the host tool.`,
      })
      expect(controlSend).toHaveBeenCalledOnce()
      expect(send).not.toHaveBeenCalled()
    }
  )

  it('rejects collisions between explicit and inferred deployed tool names', async () => {
    const deployedTools: HarnessTool[] = [{ type: 'remote_mcp' }, { type: 'agentcore_gateway', name: 'mcp_0' }]
    const unrelatedHostTool = tool({
      name: 'approval',
      description: 'Request approval',
      inputSchema: z.object({}),
      callback: () => 'host result',
    })
    const { controlClient, send: controlSend } = mockControlClient(deployedTools)
    const { client, send } = mockClient()
    const agent = new AgentCoreHarnessAgent({
      harnessArn: HARNESS_ARN,
      runtimeSessionId: SESSION_ID,
      client,
      controlClient,
      tools: [unrelatedHostTool],
    })

    await expect(agent.invoke('Run the tool')).rejects.toMatchObject({
      constructor: ToolValidationError,
      message:
        "Deployed harness tools at indexes 0 and 1 resolve to runtime name 'mcp_0'. Assign explicit unique names to the unnamed deployed tools.",
    })
    expect(controlSend).toHaveBeenCalledOnce()
    expect(send).not.toHaveBeenCalled()
  })
})

function inlineFunction({
  description,
  inputSchema,
}: {
  description: string
  inputSchema: Record<string, unknown>
}): HarnessTool {
  return {
    type: 'inline_function',
    name: 'approval',
    config: {
      inlineFunction: {
        description,
        inputSchema: inputSchema as HarnessInlineFunctionConfig['inputSchema'],
      },
    },
  }
}
