/**
 * AgentCore Harness support for the Strands Agents SDK.
 *
 * Provides {@link AgentCoreHarnessAgent}, an {@link InvokableAgent} that runs its agent loop in a
 * managed AgentCore Harness microVM via the `InvokeHarness` API, while executing host-side (custom)
 * tools on the client.
 */

export { AgentCoreHarnessAgent, type AgentCoreHarnessAgentConfig } from './agentcore-harness-agent.js'
export {
  AgentCoreHarnessStreamUpdateEvent,
  AgentCoreHarnessResultEvent,
  type AgentCoreHarnessEventData,
  type AgentCoreHarnessStreamEvent,
} from './events.js'
