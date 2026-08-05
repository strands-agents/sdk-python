import {
  CLIENT_CAPABILITIES_META_KEY,
  CLIENT_INFO_META_KEY,
  PROTOCOL_VERSION_META_KEY,
  ProtocolError,
  SdkError,
  SdkErrorCode,
} from '@modelcontextprotocol/client'

import type {
  CallToolRequest,
  ClientCapabilities,
  FetchLike,
  Implementation,
  JSONRPCErrorResponse,
  JSONRPCMessage,
  JSONRPCRequest,
  JSONRPCResponse,
  MessageExtraInfo,
  RequestId,
  Transport,
  TransportSendOptions,
} from '@modelcontextprotocol/client'

const TOOL_CALL_TOKEN_META_KEY = 'com.strandsagents/task-call-token'
const TOOL_CALL_TOKEN_PREFIX = 'strands-tool-call:'
const TASK_REQUEST_ID_PREFIX = 'strands-task:'
const TASKS_EXTENSION = 'io.modelcontextprotocol/tasks'
const TASK_METHODS = new Set(['tasks/get', 'tasks/update', 'tasks/cancel'])

interface PendingRequest {
  cleanup: () => void
  reject: (error: unknown) => void
  resolve: (result: unknown) => void
}

interface TaskRequestOptions {
  signal?: AbortSignal
  timeoutMs: number
}

interface TaskTransportOptions {
  capabilities: ClientCapabilities
  clientInfo: Implementation
  protocolVersion: string
}

interface PreparedToolCall {
  params: CallToolRequest['params']
  token: string
}

/**
 * Bridges the SEP-2663 extension around stable MCP v2's protocol-era registry.
 *
 * @internal
 */
export class TaskTransport implements Transport {
  private readonly _inner: Transport
  private readonly _capabilities: ClientCapabilities
  private readonly _clientInfo: Implementation
  private readonly _protocolVersion: string
  private readonly _activeToolCalls = new Set<string>()
  private readonly _pendingRequests = new Map<string, PendingRequest>()
  private readonly _taskResults = new Map<string, unknown>()
  private readonly _toolCallRequestIds = new Map<string, RequestId>()
  private readonly _toolCallTokens = new Map<RequestId, string>()
  private _nextId = 0
  private _nextToolCallToken = 0
  private _taskNotificationHandler: ((params: unknown) => void) | undefined

  public onclose: (() => void) | undefined
  public onerror: ((error: Error) => void) | undefined
  public onmessage: (<Message extends JSONRPCMessage>(message: Message, extra?: MessageExtraInfo) => void) | undefined

  public constructor(inner: Transport, options: TaskTransportOptions) {
    this._inner = inner
    this._capabilities = options.capabilities
    this._clientInfo = options.clientInfo
    this._protocolVersion = options.protocolVersion
    exposeDisposableStdioProbeShape(this, inner)
  }

  public get sessionId(): string | undefined {
    return this._inner.sessionId
  }

  public get hasPerRequestStream(): boolean {
    return this._inner.hasPerRequestStream ?? false
  }

  public async start(): Promise<void> {
    this._inner.onclose = (): void => {
      const error = new SdkError(SdkErrorCode.ConnectionClosed, 'MCP transport closed')
      for (const pending of this._pendingRequests.values()) {
        pending.cleanup()
        pending.reject(error)
      }
      this._pendingRequests.clear()
      this._clearToolCalls()
      this.onclose?.()
    }
    this._inner.onerror = (error): void => {
      this.onerror?.(error)
    }
    this._inner.onmessage = (message, extra): void => {
      this._routeMessage(message, extra)
    }
    await this._inner.start()
  }

  public async send(message: JSONRPCMessage, options?: TransportSendOptions): Promise<void> {
    const prepared = extractToolCallToken(message)
    const outbound = withoutLegacyTaskCapability(prepared?.message ?? message)
    if (prepared && this._activeToolCalls.has(prepared.token)) {
      this._toolCallRequestIds.set(prepared.token, prepared.message.id)
      this._toolCallTokens.set(prepared.message.id, prepared.token)
    }

    try {
      await this._inner.send(outbound, options)
    } catch (error) {
      if (prepared) this._releaseToolCallRequest(prepared.message.id)
      throw error
    }
  }

  public async close(): Promise<void> {
    this._clearToolCalls()
    await this._inner.close()
  }

  /**
   * Reaps a disposable stdio probe when MCP v2 reflects this method during era negotiation.
   *
   * @internal
   */
  public async _dispose(): Promise<void> {
    const dispose = Reflect.get(this._inner, '_dispose')
    if (typeof dispose === 'function') {
      await dispose.call(this._inner)
      return
    }
    await this._inner.close()
  }

  public setProtocolVersion(version: string): void {
    this._inner.setProtocolVersion?.(version)
  }

  public setSupportedProtocolVersions(versions: string[]): void {
    this._inner.setSupportedProtocolVersions?.(versions)
  }

  public setTaskNotificationHandler(handler: ((params: unknown) => void) | undefined): void {
    this._taskNotificationHandler = handler
  }

  public prepareToolCall(params: CallToolRequest['params']): PreparedToolCall {
    const token = `${TOOL_CALL_TOKEN_PREFIX}${this._nextToolCallToken++}`
    this._activeToolCalls.add(token)
    return {
      token,
      params: {
        ...params,
        _meta: {
          ...(isRecord(params._meta) ? params._meta : undefined),
          [TOOL_CALL_TOKEN_META_KEY]: token,
        },
      },
    }
  }

  public takeTaskResult(token: string): unknown | undefined {
    const taskResult = this._taskResults.get(token)
    this._taskResults.delete(token)
    return taskResult
  }

  public finishToolCall(token: string): void {
    this._activeToolCalls.delete(token)
    this._taskResults.delete(token)
    const requestId = this._toolCallRequestIds.get(token)
    if (requestId !== undefined) {
      this._toolCallRequestIds.delete(token)
      this._toolCallTokens.delete(requestId)
    }
  }

  public request(
    method: 'tasks/get' | 'tasks/update' | 'tasks/cancel',
    params: Record<string, unknown>,
    options: TaskRequestOptions
  ): Promise<unknown> {
    const id = `${TASK_REQUEST_ID_PREFIX}${this._nextId++}`
    const request: JSONRPCRequest = {
      jsonrpc: '2.0',
      id,
      method,
      params: {
        ...params,
        _meta: {
          ...(isRecord(params._meta) ? params._meta : undefined),
          [PROTOCOL_VERSION_META_KEY]: this._protocolVersion,
          [CLIENT_INFO_META_KEY]: this._clientInfo,
          [CLIENT_CAPABILITIES_META_KEY]: this._capabilities,
        },
      },
    }

    return new Promise((resolve, reject) => {
      const requestController = new AbortController()
      let timeout: ReturnType<typeof setTimeout> | undefined
      const abort = (): void => {
        const pending = this._pendingRequests.get(id)
        if (!pending) return
        this._pendingRequests.delete(id)
        pending.cleanup()
        const reason = abortReason(options.signal)
        requestController.abort(reason)
        reject(reason)
      }
      const cleanup = (): void => {
        if (timeout !== undefined) clearTimeout(timeout)
        options.signal?.removeEventListener('abort', abort)
      }

      if (options.signal?.aborted) {
        reject(abortReason(options.signal))
        return
      }

      timeout = setTimeout(() => {
        const pending = this._pendingRequests.get(id)
        if (!pending) return
        this._pendingRequests.delete(id)
        pending.cleanup()
        const error = new SdkError(SdkErrorCode.RequestTimeout, `MCP ${method} request timed out`, {
          method,
          timeoutMs: options.timeoutMs,
        })
        requestController.abort(error)
        reject(error)
      }, options.timeoutMs)
      options.signal?.addEventListener('abort', abort, { once: true })

      this._pendingRequests.set(id, { cleanup, reject, resolve })

      const taskId = params.taskId
      const headers =
        typeof taskId === 'string'
          ? {
              'Mcp-Method': method,
              'Mcp-Name': encodeMcpHeaderValue(taskId),
            }
          : undefined

      void this._inner
        .send(request, {
          ...(headers && { headers }),
          requestSignal: requestController.signal,
        })
        .catch((error: unknown) => {
          const pending = this._pendingRequests.get(id)
          if (!pending) return
          this._pendingRequests.delete(id)
          pending.cleanup()
          pending.reject(error)
        })
    })
  }

  private _routeMessage(message: JSONRPCMessage, extra?: MessageExtraInfo): void {
    if (isTaskNotification(message)) {
      this._taskNotificationHandler?.(message.params)
      return
    }

    if (isJsonRpcResponse(message) && typeof message.id === 'string' && message.id.startsWith(TASK_REQUEST_ID_PREFIX)) {
      const pending = this._pendingRequests.get(message.id)
      if (!pending) return

      this._pendingRequests.delete(message.id)
      pending.cleanup()
      if (isJsonRpcErrorResponse(message)) {
        pending.reject(ProtocolError.fromError(message.error.code, message.error.message, message.error.data))
      } else {
        pending.resolve(message.result)
      }
      return
    }

    if (isJsonRpcResponse(message) && message.id !== undefined) {
      const token = this._releaseToolCallRequest(message.id)
      if (token === undefined) {
        this.onmessage?.(message, extra)
        return
      }

      if (!isJsonRpcErrorResponse(message) && isTaskResult(message.result)) {
        this._taskResults.set(token, message.result)
        this.onmessage?.(
          {
            ...message,
            result: {
              resultType: 'complete',
              content: [],
            },
          } as JSONRPCMessage,
          extra
        )
        return
      }
    }

    this.onmessage?.(message, extra)
  }

  private _releaseToolCallRequest(requestId: RequestId): string | undefined {
    const token = this._toolCallTokens.get(requestId)
    if (token === undefined) return undefined
    this._toolCallTokens.delete(requestId)
    if (this._toolCallRequestIds.get(token) === requestId) {
      this._toolCallRequestIds.delete(token)
    }
    return token
  }

  private _clearToolCalls(): void {
    this._activeToolCalls.clear()
    this._toolCallRequestIds.clear()
    this._toolCallTokens.clear()
    this._taskResults.clear()
  }
}

/**
 * Adds the SEP-2663 task routing name after the official transport has applied auth and standard headers.
 *
 * @internal
 */
export function createTaskRoutingFetch(fetchImplementation?: FetchLike): FetchLike {
  return async (input, init): Promise<Response> => {
    const routing = readTaskRouting(init?.body)
    if (!routing) {
      return await (fetchImplementation ?? globalThis.fetch)(input, init)
    }

    const headers = new Headers(init?.headers)
    headers.set('Mcp-Method', routing.method)
    headers.set('Mcp-Name', encodeMcpHeaderValue(routing.taskId))
    return await (fetchImplementation ?? globalThis.fetch)(input, { ...init, headers })
  }
}

function readTaskRouting(body: BodyInit | null | undefined): { method: string; taskId: string } | undefined {
  if (typeof body !== 'string') return undefined

  try {
    const message: unknown = JSON.parse(body)
    if (!isRecord(message) || typeof message.method !== 'string' || !TASK_METHODS.has(message.method)) {
      return undefined
    }
    const params = message.params
    if (!isRecord(params) || typeof params.taskId !== 'string') return undefined
    return { method: message.method, taskId: params.taskId }
  } catch {
    return undefined
  }
}

function abortReason(signal: AbortSignal | undefined): unknown {
  if (signal?.reason instanceof Error) return signal.reason
  return new DOMException('The operation was aborted', 'AbortError')
}

function withoutLegacyTaskCapability(message: JSONRPCMessage): JSONRPCMessage {
  if (!isJsonRpcRequest(message) || message.method !== 'initialize' || !isRecord(message.params)) return message

  const capabilities = message.params.capabilities
  if (!isRecord(capabilities) || !isRecord(capabilities.extensions)) return message
  if (!(TASKS_EXTENSION in capabilities.extensions)) return message

  const extensions = { ...capabilities.extensions }
  delete extensions[TASKS_EXTENSION]
  const nextCapabilities = { ...capabilities }
  if (Object.keys(extensions).length === 0) {
    delete nextCapabilities.extensions
  } else {
    nextCapabilities.extensions = extensions
  }

  return {
    ...message,
    params: {
      ...message.params,
      capabilities: nextCapabilities,
    },
  } as JSONRPCMessage
}

function extractToolCallToken(message: JSONRPCMessage): { message: JSONRPCRequest; token: string } | undefined {
  if (!isJsonRpcRequest(message) || message.method !== 'tools/call' || !isRecord(message.params)) return undefined

  const meta = message.params._meta
  if (!isRecord(meta) || typeof meta[TOOL_CALL_TOKEN_META_KEY] !== 'string') return undefined

  const token = meta[TOOL_CALL_TOKEN_META_KEY]
  const nextMeta = { ...meta }
  delete nextMeta[TOOL_CALL_TOKEN_META_KEY]
  const params = { ...message.params }
  if (Object.keys(nextMeta).length === 0) {
    delete params._meta
  } else {
    params._meta = nextMeta
  }

  return {
    token,
    message: {
      ...message,
      params,
    },
  }
}

function exposeDisposableStdioProbeShape(wrapper: TaskTransport, inner: Transport): void {
  if (!isDisposableSdkStdioTransport(inner)) return

  Object.defineProperties(wrapper, {
    constructor: {
      configurable: true,
      value: inner.constructor,
    },
    stderr: {
      configurable: true,
      get: () => Reflect.get(inner, 'stderr'),
    },
    pid: {
      configurable: true,
      get: () => Reflect.get(inner, 'pid'),
    },
    _serverParams: {
      configurable: true,
      value: Reflect.get(inner, '_serverParams'),
    },
  })
}

function isDisposableSdkStdioTransport(transport: Transport): boolean {
  if (!('stderr' in transport) || !('pid' in transport)) return false

  const prototype = Object.getPrototypeOf(transport)
  const serverParams = Reflect.get(transport, '_serverParams')
  return (
    prototype !== null &&
    Object.hasOwn(prototype, '_dispose') &&
    isRecord(serverParams) &&
    typeof serverParams.command === 'string'
  )
}

function encodeMcpHeaderValue(value: string): string {
  if (isSafeMcpHeaderValue(value)) return value

  const bytes = new TextEncoder().encode(value)
  let binary = ''
  for (const byte of bytes) binary += String.fromCodePoint(byte)
  return `=?base64?${globalThis.btoa(binary)}?=`
}

function isSafeMcpHeaderValue(value: string): boolean {
  if (value.length === 0 || value !== value.trim()) return false
  if (value.startsWith('=?base64?') && value.endsWith('?=')) return false

  for (let index = 0; index < value.length; index++) {
    const codePoint = value.codePointAt(index)!
    if (codePoint === 9 || (codePoint >= 32 && codePoint <= 126)) continue
    return false
  }
  return true
}

function isJsonRpcRequest(message: JSONRPCMessage): message is JSONRPCRequest {
  return 'method' in message && 'id' in message
}

function isJsonRpcResponse(message: JSONRPCMessage): message is JSONRPCResponse | JSONRPCErrorResponse {
  return 'id' in message && ('result' in message || 'error' in message)
}

function isJsonRpcErrorResponse(message: JSONRPCResponse | JSONRPCErrorResponse): message is JSONRPCErrorResponse {
  return 'error' in message
}

function isTaskResult(result: unknown): boolean {
  return isRecord(result) && result.resultType === 'task'
}

function isTaskNotification(message: JSONRPCMessage): message is JSONRPCMessage & { params: unknown } {
  return 'method' in message && !('id' in message) && message.method === 'notifications/tasks' && 'params' in message
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
