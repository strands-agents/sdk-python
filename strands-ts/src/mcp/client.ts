import {
  Client,
  ClientCredentialsProvider,
  ProtocolError,
  SdkError,
  SdkErrorCode,
  StreamableHTTPClientTransport,
} from '@modelcontextprotocol/client'
import { context, propagation, trace } from '@opentelemetry/api'

import type { JSONSchema, JSONValue } from '../types/json.js'
import type { ElicitationCallback, ElicitationContext } from '../types/elicitation.js'
import { McpTool } from '../tools/mcp-tool.js'
import { logger } from '../logging/index.js'
import { type McpServerConfig, mcpServerLoader } from './config.js'
import { createTaskRoutingFetch, TaskTransport } from './task-transport.js'

import type {
  CallToolResult,
  ClientCapabilities,
  ClientContext,
  ClientNotification,
  ClientRequest,
  ElicitRequestParams,
  Implementation,
  LoggingMessageNotificationParams,
  McpSubscription,
  OAuthClientProvider,
  RequestMeta,
  RequestOptions,
  ServerCapabilities,
  StandardSchemaV1,
  SubscriptionFilter,
  Transport,
} from '@modelcontextprotocol/client'
import type {
  McpCallToolWithTaskResult,
  McpCancelTaskResult,
  McpCreateTaskResult,
  McpGetTaskResult,
  McpInputRequests,
  McpInputResponses,
  McpTaskStatus,
  McpUpdateTaskResult,
} from './task-types.js'
import {
  parseMcpCancelTaskResult,
  parseMcpCreateTaskResult,
  parseMcpGetTaskResult,
  parseMcpTaskStatusNotificationParams,
  parseMcpUpdateTaskResult,
} from './task-schemas.js'

const TASKS_EXTENSION = 'io.modelcontextprotocol/tasks'
const TASKS_PROTOCOL_VERSION = '2026-07-28'
const MINIMUM_POLL_INTERVAL_MS = 10
const MAX_TIMER_DELAY_MS = 2_147_483_647

/**
 * Widened transport type that accepts MCP transport implementations without requiring explicit casts.
 *
 * Under `exactOptionalPropertyTypes`, `StreamableHTTPClientTransport` is not directly assignable
 * to `Transport` because its `sessionId` getter returns `string | undefined`, while `Transport`
 * declares `sessionId?: string` (absent or string, but not explicitly undefined).
 * This type relaxes that constraint so users can pass any MCP transport without `as Transport`.
 */
export type McpTransport = Omit<Transport, 'sessionId'> & { sessionId?: string | undefined }

/** Temporary placeholder for RuntimeConfig */
export interface RuntimeConfig {
  applicationName?: string
  applicationVersion?: string
}

/**
 * Configuration for SEP-2663 MCP task execution.
 */
export interface TasksConfig {
  /** Overall timeout for creation, polling, input handling, and cancellation. Defaults to 300000. */
  timeoutMs?: number

  /** Timeout for each task lifecycle request. Defaults to 60000. */
  requestTimeoutMs?: number

  /** Polling delay used when the server omits `pollIntervalMs`. Defaults to 1000. */
  pollIntervalMs?: number

  /** Whether to request task notifications in addition to polling. Defaults to true. */
  useNotifications?: boolean

  /** @deprecated Use `requestTimeoutMs`. */
  ttl?: number

  /** @deprecated Use `timeoutMs`. */
  pollTimeout?: number
}

interface ResolvedTasksConfig {
  timeoutMs: number
  requestTimeoutMs: number
  pollIntervalMs: number
  useNotifications: boolean
}

interface TaskOperation {
  deadline: number
  dispose: () => void
  signal: AbortSignal
}

interface TaskNotificationChannel {
  latest?: McpGetTaskResult
  wake?: () => void
}

/** Connection state of an MCP client. */
export type McpConnectionState = 'disconnected' | 'connected' | 'failed'

/** Error thrown when a server reports a task as cancelled. */
export class McpTaskCancelledError extends Error {
  /** Optional server-provided context for the cancellation. */
  public readonly statusMessage: string | undefined

  public constructor(statusMessage?: string) {
    super(statusMessage ? `MCP task was cancelled: ${statusMessage}` : 'MCP task was cancelled')
    this.name = 'McpTaskCancelledError'
    this.statusMessage = statusMessage
  }
}

/** Options for MCP tool invocation. */
export interface McpCallToolOptions {
  /** AbortSignal to cancel the in-flight request. */
  signal?: AbortSignal
  /** Overrides the configured overall timeout in milliseconds for this call. */
  timeoutMs?: number
}

/** Options for an explicit SEP-2663 lifecycle request. */
export interface McpTaskRequestOptions {
  /** AbortSignal to cancel the lifecycle request. */
  signal?: AbortSignal
  /** Overrides the configured per-request timeout in milliseconds. */
  timeoutMs?: number
}

/** OAuth client credentials for machine-to-machine authentication. */
export interface McpClientCredentials {
  clientId: string
  clientSecret: string
  /** OAuth scopes to request. Joined with spaces before sending to the token endpoint. */
  scopes?: string[]
}

/** Decides whether a tool matches a filter. Receives the tool under its agent-facing name. */
export type McpToolFilterCallback = (tool: McpTool) => boolean

/**
 * Matches a tool for filtering. A string matches the server-side tool name exactly; a `RegExp`
 * matches it from the start (as Python's `Pattern.match` does); a callback receives the tool.
 */
export type McpToolMatcher = string | RegExp | McpToolFilterCallback

/** Filters controlling which MCP tools a client exposes. */
export interface McpToolFilters {
  /** When present, only tools matching at least one matcher are exposed. */
  allowed?: McpToolMatcher[]
  /** Tools matching at least one matcher are excluded, even when also allowed. */
  rejected?: McpToolMatcher[]
}

/** Per-call overrides for {@link McpClient.listTools}. */
export interface McpListToolsOptions {
  /** Prefix for agent-facing tool names. An empty string disables a prefix set on the client. */
  prefix?: string
  /** Tool filters. An empty object disables filters set on the client. */
  toolFilters?: McpToolFilters
}

/** Behavioral options shared by all MCP client configurations. */
export interface McpClientOptions extends RuntimeConfig {
  /** Disable OpenTelemetry MCP instrumentation. */
  disableMcpInstrumentation?: boolean

  /** Prefix for agent-facing tool names, applied as `<prefix>_<toolName>`. */
  prefix?: string

  /** Filters controlling which tools this client exposes. */
  toolFilters?: McpToolFilters

  /** Enables SEP-2663 capability advertisement and task lifecycle handling. */
  tasksConfig?: TasksConfig

  /**
   * Callback to handle server-initiated elicitation requests.
   * When provided, the client advertises elicitation support (form + url modes)
   * and routes incoming elicitation requests to this callback.
   */
  elicitationCallback?: ElicitationCallback

  /** When true, connection failures are logged as warnings instead of throwing. */
  continueOnError?: boolean

  /** Called when the server emits a log message. Defaults to routing through the Strands logger. */
  logHandler?: (params: LoggingMessageNotificationParams) => void
}

/** Arguments for configuring an MCP Client. */
export type McpClientConfig = McpClientOptions & {
  /** Pre-constructed transport. Mutually exclusive with `url`. */
  transport?: McpTransport

  /** Server URL. When provided, a StreamableHTTP transport is constructed automatically. */
  url?: string | URL

  /** Client credentials for OAuth machine-to-machine auth. Requires `url`. */
  auth?: McpClientCredentials

  /** Custom OAuth provider for advanced auth flows. Requires `url`. Mutually exclusive with `auth`. */
  authProvider?: OAuthClientProvider

  /** Custom headers to include on every request to the server. Requires `url`. */
  headers?: Record<string, string>
}

/** MCP Client for interacting with Model Context Protocol servers. */
export class McpClient {
  /** Default timeout for each task lifecycle request in milliseconds. */
  public static readonly DEFAULT_TTL = 60000

  /** Default overall timeout for automatic task completion in milliseconds. */
  public static readonly DEFAULT_POLL_TIMEOUT = 300000

  /** Default polling interval when a task response omits `pollIntervalMs`. */
  public static readonly DEFAULT_TASK_POLL_INTERVAL = 1000

  /**
   * Parses an MCP servers config (file path or object) and returns McpClient instances.
   *
   * @param config - A file path to a JSON config, or a flat server map object.
   * @param defaults - Options applied to all clients unless overridden per-server.
   * @returns An array of McpClient instances ready to be passed to an Agent.
   */
  public static async loadServers(
    config: string | Record<string, McpServerConfig>,
    defaults?: McpClientOptions
  ): Promise<McpClient[]> {
    return (await mcpServerLoader.get()(config, defaults)).map((c) => new McpClient(c))
  }

  private _clientName: string
  private _clientVersion: string
  private _transport: Transport
  private _taskTransport: TaskTransport | undefined
  private _state: McpConnectionState
  private _client: Client
  private _continueOnError: boolean
  private _logHandler: (params: LoggingMessageNotificationParams) => void
  private _disableMcpInstrumentation: boolean
  private _tasksConfig: ResolvedTasksConfig | undefined
  private _clientCapabilities: ClientCapabilities
  private _elicitationCallback: ElicitationCallback | undefined
  private _prefix: string | undefined
  private _toolFilters: McpToolFilters | undefined
  /** Server-side name of each listed tool, which differs from `tool.name` when a prefix is set. */
  private _serverToolNames = new WeakMap<McpTool, string>()
  private _registeredToolNames = new Set<string>()
  private _onToolsChanged: ((oldTools: string[], newTools: McpTool[]) => void) | undefined
  private _refreshingTools = false
  private _pendingRefresh = false
  private _connectionPromise: Promise<void> | undefined
  private _taskNotificationChannels = new Map<string, TaskNotificationChannel>()

  constructor(args: McpClientConfig) {
    this._clientName = args.applicationName || 'strands-agents-ts-sdk'
    this._clientVersion = args.applicationVersion || '0.0.1'
    this._state = 'disconnected'
    this._continueOnError = args.continueOnError ?? false
    this._logHandler = args.logHandler ?? defaultLogHandler
    this._tasksConfig = resolveTasksConfig(args.tasksConfig)
    this._elicitationCallback = args.elicitationCallback
    this._prefix = args.prefix
    this._toolFilters = args.toolFilters
    this._clientCapabilities = {
      ...(this._elicitationCallback ? { elicitation: { form: {}, url: {} } } : undefined),
      ...(this._tasksConfig ? { extensions: { [TASKS_EXTENSION]: {} } } : undefined),
    }

    const transport = McpClient._resolveTransport(args)
    if (this._tasksConfig) {
      this._taskTransport = new TaskTransport(transport, {
        capabilities: this._clientCapabilities,
        clientInfo: { name: this._clientName, version: this._clientVersion },
        protocolVersion: TASKS_PROTOCOL_VERSION,
      })
      this._taskTransport.setTaskNotificationHandler((params) => {
        this._handleTaskNotification(params)
      })
      this._transport = this._taskTransport
    } else {
      this._transport = transport
    }

    this._client = new Client(
      {
        name: this._clientName,
        version: this._clientVersion,
      },
      {
        capabilities: this._clientCapabilities,
        versionNegotiation: { mode: 'auto' },
        listChanged: {
          tools: {
            autoRefresh: false,
            debounceMs: 300,
            onChanged: (): void => {
              this._handleToolsChanged()
            },
          },
        },
      }
    )

    this._client.setNotificationHandler('notifications/message', (notification) => {
      this._logHandler(notification.params)
    })

    this._disableMcpInstrumentation = args.disableMcpInstrumentation ?? false
  }

  private static _resolveTransport(args: McpClientConfig): Transport {
    if (args.transport && args.url) {
      throw new Error('McpClientConfig: provide either "transport" or "url", not both')
    }
    if (!args.transport && !args.url) {
      throw new Error('McpClientConfig: either "transport" or "url" must be provided')
    }
    if (args.transport) {
      if (args.auth || args.authProvider || args.headers) {
        throw new Error(
          'McpClientConfig: "auth", "authProvider", and "headers" require "url" (not compatible with "transport")'
        )
      }
      if (args.tasksConfig !== undefined && args.transport instanceof StreamableHTTPClientTransport) {
        throw new Error(
          'McpClientConfig: SEP-2663 tasks require the "url" configuration for Streamable HTTP so Mcp-Name task routing headers can be applied'
        )
      }
      return args.transport as Transport
    }
    if (args.auth && args.authProvider) {
      throw new Error('McpClientConfig: provide either "auth" or "authProvider", not both')
    }

    const authProvider = args.auth
      ? new ClientCredentialsProvider({
          clientId: args.auth.clientId,
          clientSecret: args.auth.clientSecret,
          ...(args.auth.scopes && { scope: args.auth.scopes.join(' ') }),
        })
      : args.authProvider

    const url = args.url instanceof URL ? args.url : new URL(args.url!)
    return new StreamableHTTPClientTransport(url, {
      ...(authProvider && { authProvider }),
      ...(args.headers && { requestInit: { headers: args.headers } }),
      ...(args.tasksConfig !== undefined && { fetch: createTaskRoutingFetch() }),
    }) as Transport
  }

  get client(): Client {
    return this._client
  }

  get serverCapabilities(): ServerCapabilities | undefined {
    return this._client.getServerCapabilities()
  }

  get serverVersion(): Implementation | undefined {
    return this._client.getServerVersion()
  }

  get serverInstructions(): string | undefined {
    return this._client.getInstructions()
  }

  get connectionState(): McpConnectionState {
    return this._state
  }

  get clientName(): string {
    return this._clientName
  }

  get continueOnError(): boolean {
    return this._continueOnError
  }

  /**
   * Connects the MCP client to the server.
   *
   * Called lazily before any operation that requires a connection. When `continueOnError` is true,
   * connection failures are swallowed and the client enters a `'failed'` state — subsequent
   * calls are no-ops until `connect(true)` is called explicitly to retry.
   *
   * @param reconnect - When true, forces a reconnect even if already connected or failed.
   * @returns A promise that resolves when the connection is established.
   */
  public async connect(reconnect: boolean = false): Promise<void> {
    if (this._connectionPromise) {
      try {
        await this._connectionPromise
      } catch (error) {
        if (!reconnect) throw error
      }
      if (!reconnect) return
    }

    if (this._state !== 'disconnected' && !reconnect) return

    const connectionPromise = this._connect(reconnect)
    this._connectionPromise = connectionPromise
    try {
      await connectionPromise
    } finally {
      if (this._connectionPromise === connectionPromise) {
        this._connectionPromise = undefined
      }
    }
  }

  private async _connect(reconnect: boolean): Promise<void> {
    if (this._state === 'connected' && reconnect) {
      await this._client.close()
      this._state = 'disconnected'
    }

    if (this._elicitationCallback) {
      const callback = this._elicitationCallback
      this._client.setRequestHandler('elicitation/create', async (request, requestContext) => {
        return await callback(toElicitationContext(requestContext), request.params)
      })
    }

    try {
      await this._client.connect(this._transport)
      this._state = 'connected'
    } catch (error) {
      if (!this._continueOnError) throw error
      this._state = 'failed'
      logger.warn(
        `client=<${this._clientName}>, error=<${error}> | MCP server failed to connect, continuing (continueOnError)`
      )
    }
  }

  /**
   * Disconnects the MCP client from the server and cleans up resources.
   *
   * @returns A promise that resolves when the disconnection is complete.
   */
  public async disconnect(): Promise<void> {
    // Must be done sequentially
    await this._client.close()
    await this._transport.close()
    this._state = 'disconnected'
  }

  /**
   * Enables the `await using` pattern for automatic resource cleanup.
   * Delegates to {@link McpClient.disconnect}.
   */
  async [Symbol.asyncDispose](): Promise<void> {
    await this.disconnect()
  }

  /**
   * Lists the tools available on the server and returns them as executable McpTool instances.
   *
   * A prefix renames tools for the agent only; tools are always invoked, and matched by string and
   * `RegExp` filters, under their server-side name.
   *
   * @param options - Overrides for the prefix and filters set on the client. An omitted field uses
   *                  the client's value; an explicit empty string or empty object disables it.
   * @returns A promise that resolves with an array of McpTool instances.
   */
  public async listTools(options?: McpListToolsOptions): Promise<McpTool[]> {
    await this.connect()
    if (this._state === 'failed') return []

    const prefix = options?.prefix === undefined ? this._prefix : options.prefix
    const toolFilters = options?.toolFilters === undefined ? this._toolFilters : options.toolFilters
    const tools: McpTool[] = []
    let cursor: string | undefined

    do {
      const result = await this._client.listTools(cursor ? { cursor } : undefined)

      for (const toolSpec of result.tools) {
        const toolName = prefix ? `${prefix}_${toolSpec.name}` : toolSpec.name
        if (prefix) {
          logger.debug(`tool_rename=<${toolSpec.name}->${toolName}> | renamed tool`)
        }

        const tool = new McpTool({
          name: toolName,
          description: toolSpec.description || `Tool which performs ${toolSpec.name}`,
          inputSchema: toolSpec.inputSchema as JSONSchema,
          ...(toolSpec.outputSchema !== undefined && { outputSchema: toolSpec.outputSchema as JSONSchema }),
          client: this,
        })
        this._serverToolNames.set(tool, toolSpec.name)

        if (shouldIncludeTool(tool, toolSpec.name, toolFilters)) tools.push(tool)
      }

      cursor = result.nextCursor
    } while (cursor)

    // Per-call overrides are transient, so they must not become the baseline that a later
    // tools-changed refresh reports as the previously registered names.
    if (options?.prefix === undefined && options?.toolFilters === undefined) {
      this._registeredToolNames = new Set(tools.map((tool) => tool.name))
    }

    return tools
  }

  /**
   * Sets a callback invoked when the MCP server's tool list changes at runtime.
   *
   * @param callback - Handler receiving the previous tool names and the refreshed tool instances,
   *                   or undefined to remove the callback.
   */
  set onToolsChanged(callback: ((oldTools: string[], newTools: McpTool[]) => void) | undefined) {
    this._onToolsChanged = callback
  }

  private async _handleToolsChanged(): Promise<void> {
    if (this._refreshingTools) {
      this._pendingRefresh = true
      return
    }
    this._refreshingTools = true
    try {
      do {
        this._pendingRefresh = false
        const oldTools = [...this._registeredToolNames]
        const newTools = await this.listTools()
        this._onToolsChanged?.(oldTools, newTools)
      } while (this._pendingRefresh)
    } catch (err) {
      logger.warn(
        `client=<${this._clientName}>, error=<${err}> | failed to refresh tools after toolsChanged notification`
      )
    } finally {
      this._refreshingTools = false
    }
  }

  /**
   * Invoke a tool on the connected MCP server using an McpTool instance.
   *
   * When the server returns a SEP-2663 task, this method handles input requests and
   * polls until the task reaches a terminal state. Direct tool results are returned
   * unchanged.
   *
   * @param tool - The McpTool instance to invoke.
   * @param args - The arguments to pass to the tool.
   * @param options - Optional settings for the request.
   * @returns The final tool result.
   * @throws {@link McpTaskCancelledError} When the server reports a cancelled task.
   * @throws {@link ProtocolError} When the task fails with a JSON-RPC error.
   */
  public async callTool(tool: McpTool, args: JSONValue, options?: McpCallToolOptions): Promise<JSONValue> {
    if (!this._tasksConfig) {
      return (await this.callToolWithTask(tool, args, options)) as JSONValue
    }

    const operation = createTaskOperation(options?.signal, options?.timeoutMs ?? this._tasksConfig.timeoutMs)
    let task: McpCreateTaskResult | undefined
    try {
      const initialResult = await this._callToolWithTask(tool, args, {
        signal: operation.signal,
        timeoutMs: remainingTime(operation.deadline, this._tasksConfig.requestTimeoutMs),
      })
      if (!isCreateTaskResult(initialResult)) return initialResult as JSONValue

      task = initialResult
      return (await this._completeTask(initialResult, operation)) as JSONValue
    } catch (error) {
      const primaryError = operation.signal.aborted ? abortReason(operation.signal) : error
      if (task && (operation.signal.aborted || isRequestTimeoutError(error))) {
        void this._cancelAfterFailure(task.taskId, operation.deadline)
      }
      throw primaryError
    } finally {
      operation.dispose()
    }
  }

  /**
   * Invokes a tool once and returns either its direct result or a server-created task handle.
   *
   * This operation never polls or consumes a returned task handle. Task creation is controlled
   * by the server; the client only advertises support when `tasksConfig` is configured.
   *
   * @param tool - The McpTool instance to invoke.
   * @param args - The arguments to pass to the tool.
   * @param options - Optional settings for the request.
   * @returns The direct tool result or task handle returned by the server.
   */
  public async callToolWithTask(
    tool: McpTool,
    args: JSONValue,
    options?: McpCallToolOptions
  ): Promise<McpCallToolWithTaskResult> {
    const timeoutMs = options?.timeoutMs ?? this._tasksConfig?.requestTimeoutMs
    return await this._callToolWithTask(tool, args, {
      ...(options?.signal && { signal: options.signal }),
      ...(timeoutMs !== undefined && { timeoutMs }),
    })
  }

  /**
   * Retrieves the current state of a SEP-2663 task.
   *
   * @param taskId - Server-issued task identifier.
   * @param options - Optional lifecycle request settings.
   * @returns The validated task state, including terminal result or error data when present.
   */
  public async getTask(taskId: string, options?: McpTaskRequestOptions): Promise<McpGetTaskResult> {
    assertTaskId(taskId)
    const result = await this._requestTask('tasks/get', { taskId }, options)
    const task = parseTaskResponse(result, parseMcpGetTaskResult, 'tasks/get')
    if (task.taskId !== taskId) {
      throw new SdkError(SdkErrorCode.InvalidResult, 'MCP tasks/get response returned a different taskId')
    }
    return task
  }

  /**
   * Supplies one or more responses to outstanding task input requests.
   *
   * Partial response sets are supported; the server may keep the task in `input_required`
   * until every outstanding request has been answered.
   *
   * @param taskId - Server-issued task identifier.
   * @param inputResponses - Responses keyed by the corresponding input request key.
   * @param options - Optional lifecycle request settings.
   * @returns The server's validated acknowledgement.
   */
  public async updateTask(
    taskId: string,
    inputResponses: McpInputResponses,
    options?: McpTaskRequestOptions
  ): Promise<McpUpdateTaskResult> {
    assertTaskId(taskId)
    if (!isRecord(inputResponses)) {
      throw new TypeError('MCP tasks/update inputResponses must be an object')
    }
    return parseTaskResponse(
      await this._requestTask('tasks/update', { taskId, inputResponses }, options),
      parseMcpUpdateTaskResult,
      'tasks/update'
    )
  }

  /**
   * Requests cooperative cancellation of a SEP-2663 task.
   *
   * A successful acknowledgement does not guarantee that the server stopped the work or that
   * the task will eventually report `cancelled`.
   *
   * @param taskId - Server-issued task identifier.
   * @param options - Optional lifecycle request settings.
   * @returns The server's validated acknowledgement.
   */
  public async cancelTask(taskId: string, options?: McpTaskRequestOptions): Promise<McpCancelTaskResult> {
    assertTaskId(taskId)
    return parseTaskResponse(
      await this._requestTask('tasks/cancel', { taskId }, options),
      parseMcpCancelTaskResult,
      'tasks/cancel'
    )
  }

  private async _callToolWithTask(
    tool: McpTool,
    args: JSONValue,
    options: McpCallToolOptions
  ): Promise<McpCallToolWithTaskResult> {
    await this.connect()
    if (this._state === 'failed') throw new Error('MCP server failed to connect. Call connect(true) to retry.')

    if (args === null || args === undefined) {
      args = {}
    }

    if (typeof args !== 'object' || Array.isArray(args)) {
      throw new Error(
        `MCP Protocol Error: Tool arguments must be a JSON Object (named parameters). Received: ${Array.isArray(args) ? 'Array' : typeof args}`
      )
    }

    // Inject OpenTelemetry trace context into tool arguments for distributed tracing
    const enhancedArgs = this._disableMcpInstrumentation ? args : injectTraceContext(args)
    const toolArgs = enhancedArgs as Record<string, unknown>

    const toolName = this._serverToolNames.get(tool) ?? tool.name
    const params = {
      name: toolName,
      arguments: toolArgs,
    }
    const prepared = this._taskTransport?.prepareToolCall(params)
    try {
      const result = await this._client.callTool(prepared?.params ?? params, {
        ...(options.signal && { signal: options.signal }),
        ...(options.timeoutMs !== undefined && {
          timeout: options.timeoutMs,
          maxTotalTimeout: options.timeoutMs,
        }),
      })

      const rawTask = prepared ? this._taskTransport?.takeTaskResult(prepared.token) : undefined
      if (rawTask === undefined) return result as McpCallToolWithTaskResult

      this._assertTaskLifecycleAvailable()
      return parseTaskResponse(rawTask, parseMcpCreateTaskResult, 'tools/call')
    } finally {
      if (prepared) this._taskTransport?.finishToolCall(prepared.token)
    }
  }

  private async _requestTask(
    method: 'tasks/get' | 'tasks/update' | 'tasks/cancel',
    params: Record<string, unknown>,
    options?: McpTaskRequestOptions
  ): Promise<unknown> {
    await this.connect()
    if (this._state === 'failed') throw new Error('MCP server failed to connect. Call connect(true) to retry.')
    this._assertTaskLifecycleAvailable()

    const timeoutMs = options?.timeoutMs ?? this._tasksConfig!.requestTimeoutMs
    assertPositiveDuration(timeoutMs, 'MCP task request timeout')
    return await this._taskTransport!.request(method, params, {
      ...(options?.signal && { signal: options.signal }),
      timeoutMs,
    })
  }

  private _assertTaskLifecycleAvailable(): void {
    if (!this._tasksConfig || !this._taskTransport) {
      throw new Error('SEP-2663 task operations require McpClient tasksConfig')
    }
    if (
      this._client.getProtocolEra() !== 'modern' ||
      this._client.getNegotiatedProtocolVersion() !== TASKS_PROTOCOL_VERSION
    ) {
      throw new Error(`SEP-2663 task operations require negotiated MCP protocol ${TASKS_PROTOCOL_VERSION}`)
    }

    const extensions = this._client.getServerCapabilities()?.extensions
    if (!isRecord(extensions) || !isRecord(extensions[TASKS_EXTENSION])) {
      throw new Error(`MCP server did not advertise the ${TASKS_EXTENSION} extension`)
    }
  }

  private async _completeTask(task: McpCreateTaskResult, operation: TaskOperation): Promise<CallToolResult> {
    const channel: TaskNotificationChannel = {}
    this._taskNotificationChannels.set(task.taskId, channel)
    const subscriptionController = new AbortController()
    const forwardAbort = (): void => subscriptionController.abort(operation.signal.reason)
    operation.signal.addEventListener('abort', forwardAbort, { once: true })
    const subscriptionPromise = this._openTaskSubscription(
      task.taskId,
      subscriptionController.signal,
      operation.deadline
    )

    let current: McpCreateTaskResult | McpGetTaskResult = task
    let hasDetailedState = false
    const answeredInputKeys = new Set<string>()

    try {
      while (true) {
        throwIfAborted(operation.signal)

        if (hasDetailedState) {
          const detailed = current as McpGetTaskResult
          if (isTerminalTaskStatus(detailed.status)) return taskTerminalResult(detailed)
          if (detailed.status === 'input_required') {
            await this._handleTaskInput(detailed, answeredInputKeys, operation)
          }
        }

        const shouldPollImmediately = !hasDetailedState && current.status !== 'working'
        if (!shouldPollImmediately) {
          const interval = Math.max(
            MINIMUM_POLL_INTERVAL_MS,
            current.pollIntervalMs ?? this._tasksConfig!.pollIntervalMs
          )
          const notification = await this._waitForTaskNotification(task.taskId, interval, operation.signal)
          if (notification) {
            current = reconcileTaskState(current, notification, hasDetailedState)
            hasDetailedState = true
            continue
          }
        }

        const polled = await this.getTask(task.taskId, {
          signal: operation.signal,
          timeoutMs: remainingTime(operation.deadline, this._tasksConfig!.requestTimeoutMs),
        })
        current = reconcileTaskState(current, polled, hasDetailedState)
        hasDetailedState = true

        const queued = this._takeTaskNotification(task.taskId)
        if (queued) current = reconcileTaskState(current, queued, true)
      }
    } finally {
      operation.signal.removeEventListener('abort', forwardAbort)
      subscriptionController.abort()
      channel.wake?.()
      this._taskNotificationChannels.delete(task.taskId)
      const subscription = await subscriptionPromise
      await subscription?.close().catch(() => undefined)
    }
  }

  private async _handleTaskInput(
    task: McpGetTaskResult & { status: 'input_required'; inputRequests: McpInputRequests },
    answeredInputKeys: Set<string>,
    operation: TaskOperation
  ): Promise<void> {
    for (const [key, request] of Object.entries(task.inputRequests)) {
      if (answeredInputKeys.has(key)) continue

      const response = await this._fulfillTaskInput(task.taskId, key, request, operation.signal)
      await this.updateTask(
        task.taskId,
        { [key]: response },
        {
          signal: operation.signal,
          timeoutMs: remainingTime(operation.deadline, this._tasksConfig!.requestTimeoutMs),
        }
      )
      answeredInputKeys.add(key)
    }
  }

  private async _fulfillTaskInput(
    taskId: string,
    key: string,
    request: McpInputRequests[string],
    signal: AbortSignal
  ): Promise<McpInputResponses[string]> {
    if (request.method !== 'elicitation/create') {
      throw new Error(`Unsupported MCP task input request method "${request.method}" for key "${key}"`)
    }
    if (!this._elicitationCallback) {
      throw new Error('MCP task requires elicitation, but no elicitationCallback is configured')
    }

    const params = request.params as ElicitRequestParams
    return await raceWithAbort(
      this._elicitationCallback(
        createTaskElicitationContext(this._client, this._transport, taskId, key, signal, params._meta),
        params
      ),
      signal
    )
  }

  private async _openTaskSubscription(
    taskId: string,
    signal: AbortSignal,
    deadline: number
  ): Promise<McpSubscription | undefined> {
    if (!this._tasksConfig?.useNotifications) return undefined

    try {
      return await this._client.listen({ taskIds: [taskId] } as SubscriptionFilter, {
        signal,
        timeout: remainingTime(deadline, this._tasksConfig.requestTimeoutMs),
      })
    } catch {
      return undefined
    }
  }

  private _handleTaskNotification(params: unknown): void {
    if (!isRecord(params) || typeof params.taskId !== 'string') return
    const channel = this._taskNotificationChannels.get(params.taskId)
    if (!channel) return

    try {
      channel.latest = parseMcpTaskStatusNotificationParams(params)
      channel.wake?.()
    } catch {
      logger.warn('notification=<notifications/tasks> | ignored malformed MCP task notification')
    }
  }

  private _takeTaskNotification(taskId: string): McpGetTaskResult | undefined {
    const channel = this._taskNotificationChannels.get(taskId)
    const latest = channel?.latest
    if (channel) delete channel.latest
    return latest
  }

  private async _waitForTaskNotification(
    taskId: string,
    delayMs: number,
    signal: AbortSignal
  ): Promise<McpGetTaskResult | undefined> {
    const queued = this._takeTaskNotification(taskId)
    if (queued) return queued

    const channel = this._taskNotificationChannels.get(taskId)
    if (!channel) return await abortableDelay(delayMs, signal)
    const activeChannel = channel

    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(finish, Math.min(delayMs, MAX_TIMER_DELAY_MS))
      const abort = (): void => {
        cleanup()
        reject(abortReason(signal))
      }
      function cleanup(): void {
        clearTimeout(timeout)
        signal.removeEventListener('abort', abort)
        delete activeChannel.wake
      }
      function finish(): void {
        cleanup()
        resolve()
      }

      activeChannel.wake = finish
      if (signal.aborted) {
        abort()
      } else {
        signal.addEventListener('abort', abort, { once: true })
      }
    })
    return this._takeTaskNotification(taskId)
  }

  private async _cancelAfterFailure(taskId: string, deadline: number): Promise<void> {
    try {
      await this.cancelTask(taskId, {
        timeoutMs: Math.max(1, remainingTime(deadline, this._tasksConfig!.requestTimeoutMs, false)),
      })
    } catch {
      // The triggering abort or timeout remains the primary error.
    }
  }
}

function resolveTasksConfig(config: TasksConfig | undefined): ResolvedTasksConfig | undefined {
  if (config === undefined) return undefined

  const resolved = {
    timeoutMs: config.timeoutMs ?? config.pollTimeout ?? McpClient.DEFAULT_POLL_TIMEOUT,
    requestTimeoutMs: config.requestTimeoutMs ?? config.ttl ?? McpClient.DEFAULT_TTL,
    pollIntervalMs: config.pollIntervalMs ?? McpClient.DEFAULT_TASK_POLL_INTERVAL,
    useNotifications: config.useNotifications ?? true,
  }
  assertPositiveDuration(resolved.timeoutMs, 'MCP task overall timeout')
  assertPositiveDuration(resolved.requestTimeoutMs, 'MCP task request timeout')
  assertPositiveDuration(resolved.pollIntervalMs, 'MCP task poll interval')
  return resolved
}

function createTaskOperation(externalSignal: AbortSignal | undefined, timeoutMs: number): TaskOperation {
  assertPositiveDuration(timeoutMs, 'MCP task overall timeout')
  const controller = new AbortController()
  const deadline = Date.now() + timeoutMs
  const abortFromExternal = (): void => controller.abort(abortReason(externalSignal))
  const timeout = setTimeout(() => {
    controller.abort(
      new SdkError(SdkErrorCode.RequestTimeout, `MCP task did not complete within ${timeoutMs}ms`, { timeoutMs })
    )
  }, timeoutMs)

  if (externalSignal?.aborted) {
    abortFromExternal()
  } else {
    externalSignal?.addEventListener('abort', abortFromExternal, { once: true })
  }

  return {
    deadline,
    signal: controller.signal,
    dispose: (): void => {
      clearTimeout(timeout)
      externalSignal?.removeEventListener('abort', abortFromExternal)
    },
  }
}

function remainingTime(deadline: number, limit: number, throwOnExpired: boolean = true): number {
  const remaining = deadline - Date.now()
  if (remaining <= 0 && throwOnExpired) {
    throw new SdkError(SdkErrorCode.RequestTimeout, 'MCP task operation timed out')
  }
  return Math.max(1, Math.min(limit, remaining))
}

function throwIfAborted(signal: AbortSignal): void {
  if (signal.aborted) throw abortReason(signal)
}

function abortReason(signal: AbortSignal | undefined): Error {
  if (signal?.reason instanceof Error) return signal.reason
  return new DOMException('The operation was aborted', 'AbortError')
}

function isRequestTimeoutError(error: unknown): boolean {
  return error instanceof SdkError && error.code === SdkErrorCode.RequestTimeout
}

async function abortableDelay(delayMs: number, signal: AbortSignal): Promise<undefined> {
  await new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(finish, Math.min(delayMs, MAX_TIMER_DELAY_MS))
    const abort = (): void => {
      cleanup()
      reject(abortReason(signal))
    }
    function cleanup(): void {
      clearTimeout(timeout)
      signal.removeEventListener('abort', abort)
    }
    function finish(): void {
      cleanup()
      resolve()
    }

    if (signal.aborted) {
      abort()
    } else {
      signal.addEventListener('abort', abort, { once: true })
    }
  })
  return undefined
}

async function raceWithAbort<Result>(promise: Promise<Result>, signal: AbortSignal): Promise<Result> {
  if (signal.aborted) throw abortReason(signal)

  return await new Promise<Result>((resolve, reject) => {
    const abort = (): void => {
      cleanup()
      reject(abortReason(signal))
    }
    const cleanup = (): void => {
      signal.removeEventListener('abort', abort)
    }

    signal.addEventListener('abort', abort, { once: true })
    promise.then(
      (result) => {
        cleanup()
        resolve(result)
      },
      (error: unknown) => {
        cleanup()
        reject(error)
      }
    )
  })
}

function reconcileTaskState(
  previous: McpCreateTaskResult | McpGetTaskResult,
  next: McpGetTaskResult,
  previousIsDetailed: boolean
): McpGetTaskResult {
  if (next.taskId !== previous.taskId) {
    throw new SdkError(SdkErrorCode.InvalidResult, 'MCP task response changed taskId')
  }
  if (next.createdAt !== previous.createdAt) {
    throw new SdkError(SdkErrorCode.InvalidResult, 'MCP task response changed createdAt')
  }

  const previousUpdatedAt = Date.parse(previous.lastUpdatedAt)
  const nextUpdatedAt = Date.parse(next.lastUpdatedAt)
  if (nextUpdatedAt < previousUpdatedAt) {
    if (previousIsDetailed) return previous as McpGetTaskResult
    throw new SdkError(SdkErrorCode.InvalidResult, 'MCP task response predates the task handle')
  }

  if (isTerminalTaskStatus(previous.status)) {
    if (previous.status !== next.status || (previousIsDetailed && !sameTaskState(previous as McpGetTaskResult, next))) {
      throw new SdkError(SdkErrorCode.InvalidResult, 'MCP task changed after reaching a terminal state')
    }
    if (previousIsDetailed) return previous as McpGetTaskResult
  }

  // Task timestamps can have coarser precision than state transitions, so equality alone cannot
  // distinguish a valid transition from a contradiction.
  return next
}

function taskTerminalResult(task: McpGetTaskResult): CallToolResult {
  if (task.status === 'completed') return task.result
  if (task.status === 'cancelled') throw new McpTaskCancelledError(task.statusMessage)
  if (task.status === 'failed') {
    const message = task.statusMessage ? `${task.error.message}: ${task.statusMessage}` : task.error.message
    throw ProtocolError.fromError(task.error.code, message, task.error.data)
  }
  throw new SdkError(SdkErrorCode.InvalidResult, `MCP task status "${task.status}" is not terminal`)
}

function isCreateTaskResult(result: McpCallToolWithTaskResult): result is McpCreateTaskResult {
  return result.resultType === 'task'
}

function isTerminalTaskStatus(status: McpTaskStatus): status is 'completed' | 'failed' | 'cancelled' {
  return status === 'completed' || status === 'failed' || status === 'cancelled'
}

function sameTaskState(left: McpGetTaskResult, right: McpGetTaskResult): boolean {
  const leftState = { ...left }
  const rightState = { ...right }
  delete leftState._meta
  delete rightState._meta
  return sameJson(leftState, rightState)
}

function sameJson(left: unknown, right: unknown): boolean {
  if (Object.is(left, right)) return true
  if (Array.isArray(left) || Array.isArray(right)) {
    if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) return false
    return left.every((value, index) => sameJson(value, right[index]))
  }
  if (!isRecord(left) || !isRecord(right)) return false

  const leftKeys = Object.keys(left).sort()
  const rightKeys = Object.keys(right).sort()
  return (
    leftKeys.length === rightKeys.length &&
    leftKeys.every((key, index) => key === rightKeys[index] && sameJson(left[key], right[key]))
  )
}

function assertTaskId(taskId: string): void {
  if (taskId.length === 0) throw new TypeError('MCP taskId must not be empty')
}

function assertPositiveDuration(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value <= 0 || value > MAX_TIMER_DELAY_MS) {
    throw new TypeError(`${name} must be a positive safe integer no greater than ${MAX_TIMER_DELAY_MS}`)
  }
}

function parseTaskResponse<Result>(value: unknown, parse: (value: unknown) => Result, operation: string): Result {
  try {
    return parse(value)
  } catch {
    throw new SdkError(SdkErrorCode.InvalidResult, `MCP ${operation} returned a malformed SEP-2663 response`)
  }
}

function toElicitationContext(context: ClientContext): ElicitationContext {
  return {
    signal: context.mcpReq.signal,
    ...(context.http?.authInfo && { authInfo: context.http.authInfo }),
    ...(context.sessionId && { sessionId: context.sessionId }),
    ...(context.mcpReq._meta && { _meta: context.mcpReq._meta }),
    requestId: context.mcpReq.id,
    sendNotification: async (notification): Promise<void> => {
      await context.mcpReq.notify(notification)
    },
    sendRequest: async <Schema extends StandardSchemaV1>(
      request: ClientRequest,
      resultSchema: Schema,
      options?: RequestOptions
    ): Promise<StandardSchemaV1.InferOutput<Schema>> => {
      return await context.mcpReq.send(request, resultSchema, options)
    },
  }
}

function createTaskElicitationContext(
  client: Client,
  transport: Transport,
  taskId: string,
  inputKey: string,
  signal: AbortSignal,
  requestMeta: RequestMeta | undefined
): ElicitationContext {
  return {
    signal,
    ...(transport.sessionId && { sessionId: transport.sessionId }),
    ...(requestMeta && { _meta: requestMeta }),
    requestId: `task:${taskId}:${inputKey}`,
    taskId,
    sendNotification: async (notification: ClientNotification): Promise<void> => {
      await client.notification(notification)
    },
    sendRequest: async <Schema extends StandardSchemaV1>(
      request: ClientRequest,
      resultSchema: Schema,
      options?: RequestOptions
    ): Promise<StandardSchemaV1.InferOutput<Schema>> => {
      return await client.request(request, resultSchema, options)
    },
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/**
 * Decides whether a listed tool is exposed: allowed is applied first, then rejected, so a rejected
 * tool is excluded even when also allowed.
 */
function shouldIncludeTool(tool: McpTool, serverToolName: string, filters: McpToolFilters | undefined): boolean {
  if (!filters) return true
  if (filters.allowed !== undefined && !matchesAnyMatcher(tool, serverToolName, filters.allowed)) return false
  if (filters.rejected !== undefined && matchesAnyMatcher(tool, serverToolName, filters.rejected)) return false
  return true
}

function matchesAnyMatcher(tool: McpTool, serverToolName: string, matchers: McpToolMatcher[]): boolean {
  return matchers.some((matcher) => {
    if (typeof matcher === 'function') return matcher(tool)
    if (typeof matcher === 'string') return matcher === serverToolName

    // The sticky flag anchors the match at the start of the name, matching Python's Pattern.match.
    // A fresh RegExp keeps the caller's lastIndex untouched.
    const anchored = new RegExp(matcher.source, matcher.flags.includes('y') ? matcher.flags : `${matcher.flags}y`)
    return anchored.test(serverToolName)
  })
}

function defaultLogHandler(params: LoggingMessageNotificationParams): void {
  const { level, logger: serverLogger, data } = params
  const message = `logger=<${serverLogger ?? 'mcp'}>, data=<${JSON.stringify(data)}> | MCP server log`
  if (level === 'debug') {
    logger.debug(message)
  } else if (level === 'info' || level === 'notice') {
    logger.info(message)
  } else if (level === 'warning') {
    logger.warn(message)
  } else {
    logger.error(message)
  }
}

/**
 * Carrier object for OpenTelemetry context propagation.
 */
interface ContextCarrier {
  [key: string]: string | string[] | undefined
}

/**
 * Injects OpenTelemetry trace context into MCP tool call arguments.
 * Returns the args with a `_meta` field containing W3C traceparent headers.
 * If no active span exists or injection fails, returns the original args unchanged.
 *
 * @param args - The tool call arguments (must be a non-null object)
 * @returns The args with trace context injected, or the original args on failure
 */
function injectTraceContext(args: JSONValue): JSONValue {
  try {
    const currentContext = context.active()
    const currentSpan = trace.getSpan(currentContext)

    if (!currentSpan || !currentSpan.spanContext().traceId) {
      return args
    }

    const carrier: ContextCarrier = {}
    propagation.inject(currentContext, carrier)

    const existingMeta = (args as Record<string, unknown>)._meta
    const mergedMeta =
      existingMeta && typeof existingMeta === 'object' && !Array.isArray(existingMeta)
        ? { ...existingMeta, ...carrier }
        : carrier

    return {
      ...(args as Record<string, unknown>),
      _meta: mergedMeta as unknown as JSONValue,
    }
  } catch (error) {
    logger.warn(`error=<${error}> | failed to inject trace context into mcp tool call args`)
    return args
  }
}
