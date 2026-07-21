/**
 * Swarm vended tool — spin up a handoff-based sub-agent team at runtime.
 */

export { swarm, makeSwarm, MAX_MULTIAGENT_DEPTH, MULTIAGENT_DEPTH_KEY, MultiagentDepthExceededError } from './swarm.js'
export type { MakeSwarmOptions } from './swarm.js'
export { SWARM_TOOL_DESCRIPTION } from './types.js'
export type { SwarmToolResult } from './types.js'
