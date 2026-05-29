import type {
  ElicitResult,
  ElicitRequestParams,
  ClientRequest,
  ClientNotification,
} from '@modelcontextprotocol/sdk/types.js'
import type { RequestHandlerExtra } from '@modelcontextprotocol/sdk/shared/protocol.js'

/**
 * Context provided to an elicitation callback, including the abort signal for the in-flight request.
 */
export type ElicitationContext = RequestHandlerExtra<ClientRequest, ClientNotification>

/**
 * Callback invoked when an MCP server sends an elicitation request to gather user input during tool execution.
 *
 * @param context - Request context including abort signal.
 * @param params - The elicitation parameters from the server (message, requested schema or URL).
 * @returns The user's response: accept (with content), decline, or cancel.
 */
export type ElicitationCallback = (context: ElicitationContext, params: ElicitRequestParams) => Promise<ElicitResult>
