/**
 * use_agent vended tool for delegating tasks to a nested agent.
 */

export {
  MAX_MULTIAGENT_DEPTH,
  MULTIAGENT_DEPTH_KEY,
  makeUseAgent,
  MultiagentDepthExceededError,
  useAgent,
} from './use-agent.js'
export type { MakeUseAgentOptions } from './use-agent.js'
export type { UseAgentInput, UseAgentResult, UseAgentStatus } from './types.js'
