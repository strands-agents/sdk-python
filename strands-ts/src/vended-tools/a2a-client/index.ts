/**
 * A2A client tool for invoking remote A2A agents.
 */

export { DEFAULT_A2A_CLIENT_DESCRIPTION, a2aClient, makeA2AClient } from './a2a-client.js'
export type { A2AClientOutput, A2AClientRemoteCard, MakeA2AClientOptions } from './types.js'
export { UrlNotAllowedError } from './url-guard.js'
