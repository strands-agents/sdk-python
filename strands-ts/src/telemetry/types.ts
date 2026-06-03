/**
 * Type definitions for OpenTelemetry telemetry support.
 */

import type { AttributeValue } from '@opentelemetry/api'
import type { Message, SystemPrompt, ToolResultBlock } from '../types/messages.js'
import type { InvokeArgs } from '../types/agent.js'
import type { ToolSpec, ToolUse } from '../tools/types.js'
import type { Usage, Metrics } from '../models/streaming.js'

// Re-export for convenience
export type { Usage, Metrics }

/**
 * Options for starting an agent span.
 */
export interface StartAgentSpanOptions {
  /** Conversation messages to record as span events. */
  messages: Message[]
  /** Name of the agent being invoked. */
  agentName: string
  /** Unique identifier for the agent instance. */
  agentId?: string
  /** Model identifier used by the agent. */
  modelId?: string
  /** List of tools available to the agent. */
  tools?: { name: string }[]
  /** Custom attributes to merge onto the span. */
  traceAttributes?: Record<string, AttributeValue>
  /** Tool configuration map, included when gen_ai_tool_definitions opt-in is enabled. */
  toolsConfig?: Record<string, ToolSpec>
  /** System prompt provided to the agent. */
  systemPrompt?: SystemPrompt
}

/**
 * Options for ending an agent span.
 */
export interface EndAgentSpanOptions {
  /** Final response from the agent. */
  response?: Message
  /** Error that caused the agent invocation to fail. */
  error?: Error
  /** Accumulated token usage across all model calls in this invocation. */
  accumulatedUsage?: Usage
  /** Reason the agent stopped (e.g., 'end_turn', 'tool_use'). */
  stopReason?: string
}

/**
 * Options for starting a model invocation span.
 */
export interface StartModelInvokeSpanOptions {
  /** Conversation messages sent to the model. */
  messages: Message[]
  /** Model identifier being invoked. */
  modelId?: string
  /** System prompt provided to the model for this invocation. */
  systemPrompt?: SystemPrompt
}

/**
 * Options for ending a model invocation span.
 */
export interface EndModelSpanOptions {
  /** Token usage from this model call. */
  usage?: Usage
  /** Performance metrics from this model call. */
  metrics?: Metrics
  /** Error that caused the model invocation to fail. */
  error?: Error
  /** Message-like object with 'content' and 'role' properties. */
  output?: Message
  /** Reason the model stopped generating (e.g., 'end_turn', 'tool_use'). */
  stopReason?: string
}

/**
 * Options for starting a tool call span.
 */
export interface StartToolCallSpanOptions {
  /** Tool use request containing name, id, and input arguments. */
  tool: ToolUse
}

/**
 * Options for ending a tool call span.
 */
export interface EndToolCallSpanOptions {
  /** Result returned by the tool execution. */
  toolResult?: ToolResultBlock
  /** Error that caused the tool call to fail. */
  error?: Error
}

/**
 * Options for starting an agent loop cycle span.
 */
export interface StartAgentLoopSpanOptions {
  /** Unique identifier for this loop cycle. */
  cycleId: string
  /** Conversation messages at the start of this cycle. */
  messages: Message[]
}

/**
 * Options for ending an agent loop cycle span.
 */
export interface EndAgentLoopSpanOptions {
  /** Error that caused the loop cycle to fail. */
  error?: Error
}

/**
 * Options for starting a multi-agent orchestration span.
 */
export interface StartMultiAgentSpanOptions {
  /** Unique identifier for the orchestrator instance. */
  orchestratorId: string
  /** Orchestration pattern type. */
  orchestratorType: 'graph' | 'swarm'
  /** Input task or prompt passed to the orchestrator. */
  input?: InvokeArgs | undefined
  /** Custom attributes to merge onto the span. */
  traceAttributes?: Record<string, AttributeValue> | undefined
}

/**
 * Options for ending a multi-agent orchestration span.
 */
export interface EndMultiAgentSpanOptions {
  /** Error that caused the orchestration to fail. */
  error?: Error | undefined
  /** Total duration of the orchestration in milliseconds. */
  duration?: number | undefined
  /** Aggregated token usage across all node executions. */
  usage?: Usage | undefined
}

/**
 * Options for starting a node execution span.
 */
export interface StartNodeSpanOptions {
  /** Unique identifier for the node. */
  nodeId: string
  /** Node type identifier (e.g., 'agentNode', 'multiAgentNode'). */
  nodeType: string
  /** Custom attributes to merge onto the span. */
  traceAttributes?: Record<string, AttributeValue> | undefined
}

/**
 * Options for ending a node execution span.
 */
export interface EndNodeSpanOptions {
  /** Final status of the node execution. */
  status?: string | undefined
  /** Duration of the node execution in milliseconds. */
  duration?: number | undefined
  /** Token usage from the node execution. */
  usage?: Usage | undefined
  /** Error that caused the node execution to fail. */
  error?: Error | undefined
}
