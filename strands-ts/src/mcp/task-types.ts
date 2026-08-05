import type {
  CallToolResult,
  CreateMessageRequest,
  CreateMessageResult,
  ElicitRequest,
  ElicitResult,
  ListRootsRequest,
  ListRootsResult,
} from '@modelcontextprotocol/client'

/** Status reported by a task in the MCP tasks extension. */
export type McpTaskStatus = 'working' | 'input_required' | 'completed' | 'failed' | 'cancelled'

/** Operational metadata shared by every MCP task state. */
export interface McpTask {
  /** Stable server-generated task identifier. */
  taskId: string
  /** Current task status. */
  status: McpTaskStatus
  /** Human-readable context about the current task state. */
  statusMessage?: string
  /** ISO 8601 timestamp at which the task was created. */
  createdAt: string
  /** ISO 8601 timestamp at which the task was last updated. */
  lastUpdatedAt: string
  /** Retention duration from creation in milliseconds, or `null` for unlimited retention. */
  ttlMs: number | null
  /** Server-suggested polling interval in milliseconds. */
  pollIntervalMs?: number
}

/** Server-to-client request that can be surfaced by an input-required task. */
export type McpInputRequest = CreateMessageRequest | ListRootsRequest | ElicitRequest

/** Client response to a request surfaced by an input-required task. */
export type McpInputResponse = CreateMessageResult | ListRootsResult | ElicitResult

/** Outstanding task input requests keyed by identifiers unique within the task. */
export interface McpInputRequests {
  /** Server-to-client request keyed by its task-scoped identifier. */
  [key: string]: McpInputRequest
}

/** Task input responses keyed by their corresponding request identifiers. */
export interface McpInputResponses {
  /** Client response keyed by the corresponding task-scoped request identifier. */
  [key: string]: McpInputResponse
}

/** JSON-RPC error stored by a task whose underlying request failed. */
export interface McpTaskError {
  /** JSON-RPC error code. */
  code: number
  /** Concise description of the error. */
  message: string
  /** Sender-defined error details. */
  data?: unknown
}

/** Task that is actively processing its underlying request. */
export interface McpWorkingTask extends McpTask {
  /** Working-state discriminator. */
  status: 'working'
}

/** Task that is waiting for one or more client responses. */
export interface McpInputRequiredTask extends McpTask {
  /** Input-required-state discriminator. */
  status: 'input_required'
  /** Outstanding server-to-client requests. */
  inputRequests: McpInputRequests
}

/** Task whose tool call completed at the protocol level. */
export interface McpCompletedTask extends McpTask {
  /** Completed-state discriminator. */
  status: 'completed'
  /** Final result of the tool call represented by this task. */
  result: CallToolResult
}

/** Task whose underlying request failed with a JSON-RPC error. */
export interface McpFailedTask extends McpTask {
  /** Failed-state discriminator. */
  status: 'failed'
  /** JSON-RPC error that caused the task to fail. */
  error: McpTaskError
}

/** Task that was cancelled before completion. */
export interface McpCancelledTask extends McpTask {
  /** Cancelled-state discriminator. */
  status: 'cancelled'
}

/** Status-specific task state returned by `tasks/get` and task notifications. */
export type McpDetailedTask =
  McpWorkingTask | McpInputRequiredTask | McpCompletedTask | McpFailedTask | McpCancelledTask

/** Common result envelope used by the MCP tasks extension. */
export interface McpTaskResult {
  /** Wire discriminator for task handles and completed task operations. */
  resultType: 'task' | 'complete'
  /** Optional MCP result metadata. */
  _meta?: Record<string, unknown>
  /** Additional fields defined by MCP result extensions. */
  [key: string]: unknown
}

/** Direct tool result whose wire-only task discriminator has been removed. */
export type McpDirectCallToolResult = CallToolResult & {
  resultType?: never
}

/** Task handle returned instead of an immediate tool result. */
export type McpCreateTaskResult = McpTaskResult & McpTask & { resultType: 'task' }

/** Direct result or task handle returned by `callToolWithTask`. */
export type McpCallToolWithTaskResult = McpDirectCallToolResult | McpCreateTaskResult

/** Complete status-specific result returned by `tasks/get`. */
export type McpGetTaskResult = McpTaskResult & McpDetailedTask & { resultType: 'complete' }

/** Empty acknowledgement returned by `tasks/update`. */
export interface McpUpdateTaskResult extends McpTaskResult {
  /** Completed-result discriminator. */
  resultType: 'complete'
}

/** Empty acknowledgement returned by `tasks/cancel`. */
export interface McpCancelTaskResult extends McpTaskResult {
  /** Completed-result discriminator. */
  resultType: 'complete'
}

/** Complete task state carried by a `notifications/tasks` notification. */
export type McpTaskStatusNotificationParams = McpDetailedTask & {
  /** Optional MCP notification metadata. */
  _meta?: Record<string, unknown>
  /** Additional fields defined by MCP notification extensions. */
  [key: string]: unknown
}

/** Status update emitted by a server for a subscribed MCP task. */
export interface McpTaskStatusNotification {
  /** JSON-RPC protocol version. */
  jsonrpc: '2.0'
  /** Task status notification method. */
  method: 'notifications/tasks'
  /** Complete task state at notification time. */
  params: McpTaskStatusNotificationParams
}
