import type {
  AuthInfo,
  ClientRequest,
  ClientNotification,
  ElicitRequestParams,
  ElicitResult,
  RequestId,
  RequestMeta,
  RequestOptions,
  StandardSchemaV1,
} from '@modelcontextprotocol/client'

/**
 * Context provided to an elicitation callback, including the abort signal for the in-flight request.
 */
export interface ElicitationContext {
  /** Abort signal for the request that triggered elicitation. */
  signal: AbortSignal
  /** Authentication information associated with the request, when available. */
  authInfo?: AuthInfo
  /** MCP transport session identifier, when available. */
  sessionId?: string
  /** Non-protocol metadata from the original request. */
  _meta?: RequestMeta
  /** JSON-RPC request identifier for correlation. */
  requestId: RequestId
  /** Task identifier when elicitation is part of SEP-2663 task execution. */
  taskId?: string
  /** Incoming HTTP headers, when the server request arrived over HTTP. */
  requestInfo?: { headers: Record<string, string | string[] | undefined> }
  /** Sends a notification associated with the elicitation request. */
  sendNotification: (notification: ClientNotification) => Promise<void>
  /** Sends a request associated with the elicitation request. */
  sendRequest: <Schema extends StandardSchemaV1>(
    request: ClientRequest,
    resultSchema: Schema,
    options?: RequestOptions
  ) => Promise<StandardSchemaV1.InferOutput<Schema>>
}

/**
 * Callback invoked when an MCP server sends an elicitation request to gather user input during tool execution.
 *
 * @param context - Request context including abort signal.
 * @param params - The elicitation parameters from the server (message, requested schema or URL).
 * @returns The user's response: accept (with content), decline, or cancel.
 */
export type ElicitationCallback = (context: ElicitationContext, params: ElicitRequestParams) => Promise<ElicitResult>
