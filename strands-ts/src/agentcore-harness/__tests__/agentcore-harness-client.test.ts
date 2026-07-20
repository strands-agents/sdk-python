import { BedrockAgentCoreClient, type BedrockAgentCoreClientConfig } from '@aws-sdk/client-bedrock-agentcore'
import { describe, expect, it, vi } from 'vitest'
import { z } from 'zod'
import { ModelError } from '../../errors.js'
import { tool } from '../../tools/tool-factory.js'
import { AgentCoreHarnessAgent } from '../agentcore-harness-agent.js'
import { createHarnessClient, createHarnessControlClient } from '../clients.js'
import {
  HARNESS_ARN,
  SESSION_ID,
  chunk,
  harnessStream,
  mockClient,
  mockControlClient,
} from './harness-test-fixtures.js'

describe('AgentCoreHarnessAgent clients', () => {
  it('reuses an injected client across host-tool turns', async () => {
    const { client, send } = mockClient(
      harnessStream(chunk.toolUseStart('tu-1', 'noop'), chunk.contentBlockStop(), chunk.messageStop('tool_use')),
      harnessStream(chunk.messageStop('end_turn'))
    )
    const noop = tool({ name: 'noop', description: 'noop', inputSchema: z.object({}), callback: () => 'ok' })
    const { controlClient } = mockControlClient()
    const agent = new AgentCoreHarnessAgent({
      harnessArn: HARNESS_ARN,
      runtimeSessionId: SESSION_ID,
      client,
      controlClient,
      tools: [noop],
    })

    await agent.invoke('Hi')

    expect(send).toHaveBeenCalledTimes(2)
  })

  it('constructs clients with the intended region and retry ownership', async () => {
    const defaultClient = createHarnessClient({}, 'us-east-1')
    const configuredClient = createHarnessClient({ region: 'us-west-2', maxAttempts: 2 }, 'us-east-1')
    const controlClient = createHarnessControlClient({ maxAttempts: 4 }, 'us-east-1')
    const injectedClient = new BedrockAgentCoreClient({ region: 'us-east-1', maxAttempts: 3 })

    expect(await defaultClient.config.maxAttempts()).toBe(1)
    expect(await configuredClient.config.region()).toBe('us-east-1')
    expect(await configuredClient.config.maxAttempts()).toBe(2)
    expect(await controlClient.config.maxAttempts()).toBe(4)
    expect(await injectedClient.config.maxAttempts()).toBe(3)
  })

  it('retries retryable pre-stream failures only when clientConfig opts in', async () => {
    const credentials = { accessKeyId: 'test', secretAccessKey: 'test' }
    const cases = [
      {
        name: 'retryable HTTP 500',
        fail: () => ({
          response: {
            statusCode: 500,
            headers: {
              'content-type': 'application/json',
              'x-amzn-errortype': 'InternalServerException',
            },
            body: new TextEncoder().encode(JSON.stringify({ message: 'server failed' })),
          },
        }),
      },
      {
        name: 'ambiguous timeout',
        fail: () => {
          throw Object.assign(new Error('socket timed out'), { name: 'TimeoutError' })
        },
      },
    ]

    for (const testCase of cases) {
      for (const expectedAttempts of [1, 2]) {
        const handle = vi.fn().mockImplementation(testCase.fail)
        const clientConfig: BedrockAgentCoreClientConfig = {
          credentials,
          requestHandler: { handle },
          ...(expectedAttempts === 2 && { maxAttempts: 2 }),
        }
        const agent = new AgentCoreHarnessAgent({
          harnessArn: HARNESS_ARN,
          runtimeSessionId: SESSION_ID,
          region: 'us-east-1',
          clientConfig,
        })

        await expect(agent.invoke('Hi')).rejects.toBeDefined()
        expect(handle, `${testCase.name} with ${expectedAttempts} attempt(s)`).toHaveBeenCalledTimes(expectedAttempts)
      }
    }
  })

  it('does not replay a request after response streaming begins', async () => {
    const send = vi.fn().mockResolvedValue({
      stream: (async function* () {
        yield chunk.textDelta('partial')
        throw new Error('stream failed')
      })(),
    })
    const agent = new AgentCoreHarnessAgent({
      harnessArn: HARNESS_ARN,
      runtimeSessionId: SESSION_ID,
      client: { send } as unknown as BedrockAgentCoreClient,
    })

    await expect(agent.invoke('Hi')).rejects.toMatchObject({
      constructor: ModelError,
      message: 'stream failed',
    })
    expect(send).toHaveBeenCalledOnce()
  })
})
