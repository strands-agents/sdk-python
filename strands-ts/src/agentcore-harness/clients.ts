import { BedrockAgentCoreClient, type BedrockAgentCoreClientConfig } from '@aws-sdk/client-bedrock-agentcore'
import {
  BedrockAgentCoreControlClient,
  type BedrockAgentCoreControlClientConfig,
} from '@aws-sdk/client-bedrock-agentcore-control'

/**
 * Creates the Harness data-plane client owned by an AgentCoreHarnessAgent.
 *
 * @internal
 */
export function createHarnessClient(
  config: BedrockAgentCoreClientConfig | undefined,
  region: string | undefined
): BedrockAgentCoreClient {
  return new BedrockAgentCoreClient({
    ...config,
    maxAttempts: config?.maxAttempts ?? 1,
    ...(region !== undefined && { region }),
  })
}

/**
 * Creates the Harness control-plane client owned by an AgentCoreHarnessAgent.
 *
 * @internal
 */
export function createHarnessControlClient(
  config: BedrockAgentCoreControlClientConfig | undefined,
  region: string | undefined
): BedrockAgentCoreControlClient {
  return new BedrockAgentCoreControlClient({
    ...config,
    ...(region !== undefined && { region }),
  })
}
