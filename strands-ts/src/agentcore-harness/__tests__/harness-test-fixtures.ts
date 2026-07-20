import type { BedrockAgentCoreClient, HarnessTool, InvokeHarnessStreamOutput } from '@aws-sdk/client-bedrock-agentcore'
import type { BedrockAgentCoreControlClient } from '@aws-sdk/client-bedrock-agentcore-control'
import { vi } from 'vitest'
import type { AgentCoreHarnessEventData } from '../events.js'

// A session id at the minimum valid length (33 chars).
export const SESSION_ID = 'session-id-padded-to-thirty-three'
export const HARNESS_ID = 'TestHarness-abcdefghij'
export const HARNESS_ARN = `arn:aws:bedrock-agentcore:us-east-1:123456789012:harness/${HARNESS_ID}`

/** Builds a fake InvokeHarness stream from chunk objects. */
export async function* harnessStream(
  ...chunks: InvokeHarnessStreamOutput[]
): AsyncGenerator<InvokeHarnessStreamOutput> {
  for (const chunk of chunks) {
    yield chunk
  }
}

/** Chunk builders mirroring the InvokeHarness wire shapes. */
export const chunk = {
  messageStart: (role: 'assistant' | 'user' = 'assistant'): AgentCoreHarnessEventData =>
    ({ messageStart: { role } }) as AgentCoreHarnessEventData,
  textDelta: (text: string): AgentCoreHarnessEventData =>
    ({ contentBlockDelta: { delta: { text } } }) as AgentCoreHarnessEventData,
  toolUseStart: (toolUseId: string, name: string): AgentCoreHarnessEventData =>
    ({ contentBlockStart: { start: { toolUse: { toolUseId, name } } } }) as AgentCoreHarnessEventData,
  toolUseDelta: (input: string): AgentCoreHarnessEventData =>
    ({ contentBlockDelta: { delta: { toolUse: { input } } } }) as AgentCoreHarnessEventData,
  contentBlockStop: (): AgentCoreHarnessEventData => ({ contentBlockStop: {} }) as AgentCoreHarnessEventData,
  messageStop: (stopReason: string): AgentCoreHarnessEventData =>
    ({ messageStop: { stopReason } }) as AgentCoreHarnessEventData,
  metadata: (usage: Record<string, number>, latencyMs?: number): AgentCoreHarnessEventData =>
    ({ metadata: { usage, ...(latencyMs !== undefined && { metrics: { latencyMs } }) } }) as AgentCoreHarnessEventData,
}

/** A mock client whose `send` returns the queued streams in order. */
export function mockClient(...streams: AsyncGenerator<InvokeHarnessStreamOutput>[]): {
  client: BedrockAgentCoreClient
  send: ReturnType<typeof vi.fn>
} {
  const send = vi.fn()
  streams.forEach((stream) => send.mockResolvedValueOnce({ stream }))
  return { client: { send } as unknown as BedrockAgentCoreClient, send }
}

/** A mock control client that returns the supplied deployed harness tool configuration. */
export function mockControlClient(
  deployedTools: HarnessTool[] = [],
  allowedTools?: string[]
): {
  controlClient: BedrockAgentCoreControlClient
  send: ReturnType<typeof vi.fn>
} {
  const send = vi.fn().mockResolvedValue({
    harness: { tools: deployedTools, ...(allowedTools !== undefined && { allowedTools }) },
  })
  return { controlClient: { send } as unknown as BedrockAgentCoreControlClient, send }
}
