import {
  CLIENT_CAPABILITIES_META_KEY,
  SERVER_INFO_META_KEY,
  SUBSCRIPTION_ID_META_KEY,
  createMcpHandler,
  inputResponse,
  McpServer,
} from '@modelcontextprotocol/server'
import { toNodeHandler } from '@modelcontextprotocol/node'
import { Buffer } from 'node:buffer'
import { randomUUID } from 'node:crypto'
import { createServer } from 'node:http'
import * as z from 'zod'

import type {
  CallToolResult,
  InputRequests,
  InputResponses,
  JSONRPCErrorResponse,
  RequestId,
  ServerContext,
} from '@modelcontextprotocol/server'
import type { IncomingMessage, Server as HttpServer, ServerResponse } from 'node:http'
import type { AddressInfo } from 'node:net'

const TASK_EXTENSION_ID = 'io.modelcontextprotocol/tasks'
const TASK_TTL_MS = 60_000
const TASK_POLL_INTERVAL_MS = 10
const ELICITATION_REQUEST_KEY = 'elicitation'

const SERVER_INFO = {
  name: 'test-mcp-task-server',
  version: '2.0.0',
}

const directResultInputSchema = z.object({
  value: z.string().optional(),
})

const instantTaskInputSchema = z.object({
  value: z.string().optional(),
})

const longRunningTaskInputSchema = z.object({
  duration: z.number().int().nonnegative().max(5_000).optional(),
  message: z.string().optional(),
})

const failingTaskInputSchema = z.object({
  error_message: z.string().optional(),
})

const inputRequiredTaskInputSchema = z.object({
  prompt: z.string().optional(),
})

const cancellableTaskInputSchema = z.object({
  message: z.string().optional(),
})

const cancelledTaskInputSchema = z.object({
  reason: z.string().optional(),
})

const jsonRpcMessageSchema = z.object({
  jsonrpc: z.literal('2.0'),
  id: z.union([z.string(), z.number().int()]).optional(),
  method: z.string(),
  params: z.looseObject({}).optional(),
})

const taskOperationParamsSchema = z.looseObject({
  taskId: z.string(),
})

const updateTaskParamsSchema = taskOperationParamsSchema.extend({
  inputResponses: z.record(z.string(), z.unknown()),
})

const taskSubscriptionParamsSchema = z.looseObject({
  notifications: z.looseObject({
    taskIds: z.array(z.string()),
  }),
})

type TaskStatus = 'working' | 'input_required' | 'completed' | 'failed' | 'cancelled'
type TaskToolName =
  'instant_task' | 'long_running_task' | 'failing_task' | 'input_required_task' | 'cancellable_task' | 'cancelled_task'

interface BaseTask {
  taskId: string
  status: TaskStatus
  statusMessage?: string
  createdAt: string
  lastUpdatedAt: string
  ttlMs: number | null
  pollIntervalMs?: number
}

interface WorkingTask extends BaseTask {
  status: 'working'
}

interface InputRequiredTask extends BaseTask {
  status: 'input_required'
  inputRequests: InputRequests
}

interface CompletedTask extends BaseTask {
  status: 'completed'
  result: CallToolResult
}

interface FailedTask extends BaseTask {
  status: 'failed'
  error: JSONRPCErrorResponse['error']
}

interface CancelledTask extends BaseTask {
  status: 'cancelled'
}

type DetailedTask = WorkingTask | InputRequiredTask | CompletedTask | FailedTask | CancelledTask

interface CreateTaskResult extends BaseTask {
  resultType: 'task'
}

interface GetTaskResult extends BaseTask {
  resultType: 'complete'
  inputRequests?: InputRequests
  result?: CallToolResult
  error?: JSONRPCErrorResponse['error']
}

interface JsonRpcMessage {
  jsonrpc: '2.0'
  id?: RequestId
  method: string
  params?: Record<string, unknown>
}

interface JsonRpcRequest extends JsonRpcMessage {
  id: RequestId
}

interface TaskSubscription {
  close: () => void
  response: ServerResponse
}

interface StoreSubscriber {
  listener: (task: DetailedTask) => void
  taskIds: Set<string>
}

/**
 * Captured Streamable HTTP routing headers for one JSON-RPC request.
 */
export interface TaskHttpRequestObservation {
  /** JSON-RPC method from the request body. */
  method: string
  /** Tool or task name carried by the request body, when present. */
  name?: string
  /** Task identifier carried by the request body, when present. */
  taskId?: string
  /** Raw Mcp-Method request header. */
  mcpMethod?: string
  /** Raw Mcp-Name request header. */
  mcpName?: string
  /** Raw MCP-Protocol-Version request header. */
  protocolVersion?: string
}

/**
 * Information for a running task-enabled HTTP fixture.
 */
export interface TaskHttpServerInfo {
  /** Node HTTP server instance. */
  server: HttpServer
  /** Random loopback port selected by the operating system. */
  port: number
  /** Streamable HTTP endpoint URL. */
  url: string
  /** Requests observed by the HTTP edge, in arrival order. */
  requests: readonly TaskHttpRequestObservation[]
  /** Stops timers, subscriptions, MCP handlers, and the HTTP server. */
  close: () => Promise<void>
}

class InProcessTaskStore {
  private readonly _tasks = new Map<string, DetailedTask>()
  private readonly _timers = new Set<ReturnType<typeof setTimeout>>()
  private readonly _subscribers = new Set<StoreSubscriber>()

  createInstant(value: string): CreateTaskResult {
    const task = this._newTask('completed', 'Task completed', {
      result: textResult(value),
    })
    return toCreateTaskResult(task)
  }

  createLongRunning(duration: number, message: string): CreateTaskResult {
    const task = this._newTask('working', 'Step 1: Initializing')
    const progressDelay = Math.max(1, Math.floor(duration / 2))

    this._schedule(task.taskId, progressDelay, (current) => {
      if (current.status !== 'working') return current
      return transitionTask(current, 'working', 'Step 2: Processing')
    })
    this._schedule(task.taskId, duration, (current) => {
      if (current.status !== 'working') return current
      return transitionTask(current, 'completed', 'Task completed', {
        result: textResult(message),
      })
    })
    return toCreateTaskResult(task)
  }

  createFailing(errorMessage: string): CreateTaskResult {
    const task = this._newTask('working', 'Task is about to fail')
    this._schedule(task.taskId, 30, (current) => {
      if (current.status !== 'working') return current
      return transitionTask(current, 'failed', errorMessage, {
        error: {
          code: -32603,
          message: errorMessage,
        },
      })
    })
    return toCreateTaskResult(task)
  }

  createInputRequired(prompt: string): CreateTaskResult {
    const task = this._newTask('input_required', 'User input is required', {
      inputRequests: {
        [ELICITATION_REQUEST_KEY]: {
          method: 'elicitation/create',
          params: {
            mode: 'form',
            message: prompt,
            requestedSchema: {
              type: 'object',
              properties: {
                value: {
                  type: 'string',
                  description: 'Value returned to the task',
                },
              },
              required: ['value'],
            },
          },
        },
      },
    })
    return toCreateTaskResult(task)
  }

  createCancellable(message: string): CreateTaskResult {
    const task = this._newTask('working', message)
    return toCreateTaskResult(task)
  }

  createCancelled(reason: string): CreateTaskResult {
    const task = this._newTask('cancelled', reason)
    return toCreateTaskResult(task)
  }

  get(taskId: string): DetailedTask | undefined {
    return this._tasks.get(taskId)
  }

  update(taskId: string, responses: InputResponses | Record<string, unknown>): boolean {
    const task = this._tasks.get(taskId)
    if (!task) return false
    if (task.status !== 'input_required') return true

    const response = inputResponse(responses, ELICITATION_REQUEST_KEY)
    if (response.kind !== 'elicit') return true

    if (response.action !== 'accept') {
      const statusMessage = response.action === 'decline' ? 'Input was declined' : 'Input was cancelled'
      this._replace(transitionTask(task, 'cancelled', statusMessage))
      return true
    }

    const value = response.content?.value
    if (typeof value !== 'string') return true

    const working = transitionTask(task, 'working', 'Input accepted')
    this._replace(working)
    this._schedule(taskId, 10, (current) => {
      if (current.status !== 'working') return current
      return transitionTask(current, 'completed', 'Task completed', {
        result: textResult(`Input received: ${value}`),
      })
    })
    return true
  }

  cancel(taskId: string): boolean {
    const task = this._tasks.get(taskId)
    if (!task) return false
    if (task.status === 'working' || task.status === 'input_required') {
      this._replace(transitionTask(task, 'cancelled', 'Task cancelled by client'))
    }
    return true
  }

  subscribe(taskIds: string[], listener: (task: DetailedTask) => void): () => void {
    const subscriber: StoreSubscriber = {
      listener,
      taskIds: new Set(taskIds),
    }
    this._subscribers.add(subscriber)
    return (): void => {
      this._subscribers.delete(subscriber)
    }
  }

  cleanup(): void {
    for (const timer of this._timers) clearTimeout(timer)
    this._timers.clear()
    this._subscribers.clear()
    this._tasks.clear()
  }

  private _newTask(status: 'working', statusMessage: string, details?: Record<string, never>): WorkingTask
  private _newTask(
    status: 'input_required',
    statusMessage: string,
    details: Pick<InputRequiredTask, 'inputRequests'>
  ): InputRequiredTask
  private _newTask(status: 'completed', statusMessage: string, details: Pick<CompletedTask, 'result'>): CompletedTask
  private _newTask(status: 'failed', statusMessage: string, details: Pick<FailedTask, 'error'>): FailedTask
  private _newTask(status: 'cancelled', statusMessage: string, details?: Record<string, never>): CancelledTask
  private _newTask(
    status: TaskStatus,
    statusMessage: string,
    details: Partial<
      Pick<InputRequiredTask, 'inputRequests'> & Pick<CompletedTask, 'result'> & Pick<FailedTask, 'error'>
    > = {}
  ): DetailedTask {
    const now = new Date().toISOString()
    const base = {
      taskId: randomUUID(),
      statusMessage,
      createdAt: now,
      lastUpdatedAt: now,
      ttlMs: TASK_TTL_MS,
      pollIntervalMs: TASK_POLL_INTERVAL_MS,
    }

    let task: DetailedTask
    switch (status) {
      case 'working':
        task = { ...base, status }
        break
      case 'input_required':
        task = { ...base, status, inputRequests: details.inputRequests! }
        break
      case 'completed':
        task = { ...base, status, result: details.result! }
        break
      case 'failed':
        task = { ...base, status, error: details.error! }
        break
      case 'cancelled':
        task = { ...base, status }
        break
    }

    this._replace(task)
    return task
  }

  private _schedule(taskId: string, delay: number, transition: (current: DetailedTask) => DetailedTask): void {
    const timer = setTimeout(() => {
      this._timers.delete(timer)
      const current = this._tasks.get(taskId)
      if (!current) return
      const next = transition(current)
      if (next !== current) this._replace(next)
    }, delay)
    timer.unref()
    this._timers.add(timer)
  }

  private _replace(task: DetailedTask): void {
    this._tasks.set(task.taskId, task)
    for (const subscriber of this._subscribers) {
      if (subscriber.taskIds.has(task.taskId)) subscriber.listener(task)
    }
  }
}

/**
 * Starts a stable-v2 MCP server plus the fixture-local SEP-2663 extension routes.
 */
export async function startTaskHTTPServer(): Promise<TaskHttpServerInfo> {
  const taskStore = new InProcessTaskStore()
  const requests: TaskHttpRequestObservation[] = []
  const subscriptions = new Set<TaskSubscription>()

  const protocolHandler = createMcpHandler(() => createTaskTestServer(taskStore), {
    legacy: 'reject',
  })
  const protocolNodeHandler = toNodeHandler(protocolHandler)

  const httpServer = createServer((request, response) => {
    void handleHttpRequest(request, response, taskStore, requests, subscriptions, protocolNodeHandler).catch(
      (error: unknown) => {
        writeUnexpectedError(response, error)
      }
    )
  })

  await listen(httpServer)
  const address = httpServer.address() as AddressInfo

  return {
    server: httpServer,
    port: address.port,
    url: `http://127.0.0.1:${address.port}/mcp`,
    requests,
    close: async (): Promise<void> => {
      for (const subscription of subscriptions) subscription.close()
      subscriptions.clear()
      taskStore.cleanup()
      await protocolHandler.close()
      await closeServer(httpServer)
    },
  }
}

function createTaskTestServer(taskStore: InProcessTaskStore): McpServer {
  const server = new McpServer(SERVER_INFO, {
    capabilities: {
      extensions: {
        [TASK_EXTENSION_ID]: {},
      },
    },
  })

  server.registerTool(
    'direct_result',
    {
      title: 'Direct Result',
      description: 'Returns a direct CallToolResult without creating a task',
      inputSchema: directResultInputSchema,
    },
    async ({ value }): Promise<CallToolResult> => textResult(value ?? 'direct result')
  )

  server.registerTool(
    'instant_task',
    {
      title: 'Instant Task',
      description: 'Creates a task that is already completed',
      inputSchema: instantTaskInputSchema,
    },
    async ({ value }, context): Promise<CallToolResult> =>
      toSdkTaskResult(requireTaskCapability(context, taskStore.createInstant(value ?? 'instant result')))
  )

  server.registerTool(
    'long_running_task',
    {
      title: 'Long Running Task',
      description: 'Creates a working task that completes after a short delay',
      inputSchema: longRunningTaskInputSchema,
    },
    async ({ duration, message }, context): Promise<CallToolResult> =>
      toSdkTaskResult(
        requireTaskCapability(context, taskStore.createLongRunning(duration ?? 200, message ?? 'Task completed!'))
      )
  )

  server.registerTool(
    'failing_task',
    {
      title: 'Failing Task',
      description: 'Creates a task that fails with a JSON-RPC execution error',
      inputSchema: failingTaskInputSchema,
    },
    async ({ error_message }, context): Promise<CallToolResult> =>
      toSdkTaskResult(
        requireTaskCapability(context, taskStore.createFailing(error_message ?? 'Task intentionally failed'))
      )
  )

  server.registerTool(
    'input_required_task',
    {
      title: 'Input Required Task',
      description: 'Creates a task with an embedded elicitation request',
      inputSchema: inputRequiredTaskInputSchema,
    },
    async ({ prompt }, context): Promise<CallToolResult> =>
      toSdkTaskResult(
        requireTaskCapability(context, taskStore.createInputRequired(prompt ?? 'Provide a value for this task'))
      )
  )

  server.registerTool(
    'cancellable_task',
    {
      title: 'Cancellable Task',
      description: 'Creates a working task that remains active until cancelled',
      inputSchema: cancellableTaskInputSchema,
    },
    async ({ message }, context): Promise<CallToolResult> =>
      toSdkTaskResult(
        requireTaskCapability(context, taskStore.createCancellable(message ?? 'Waiting for cancellation'))
      )
  )

  server.registerTool(
    'cancelled_task',
    {
      title: 'Cancelled Task',
      description: 'Creates a task in the cancelled state',
      inputSchema: cancelledTaskInputSchema,
    },
    async ({ reason }, context): Promise<CallToolResult> =>
      toSdkTaskResult(requireTaskCapability(context, taskStore.createCancelled(reason ?? 'Task was cancelled')))
  )

  return server
}

async function handleHttpRequest(
  request: IncomingMessage,
  response: ServerResponse,
  taskStore: InProcessTaskStore,
  observations: TaskHttpRequestObservation[],
  subscriptions: Set<TaskSubscription>,
  protocolHandler: ReturnType<typeof toNodeHandler>
): Promise<void> {
  const protocolRequest = request as Parameters<typeof protocolHandler>[0]

  if (new URL(request.url ?? '/', 'http://127.0.0.1').pathname !== '/mcp') {
    response.writeHead(404)
    response.end()
    return
  }

  if (request.method !== 'POST') {
    await protocolHandler(protocolRequest, response)
    return
  }

  const body = await readJsonBody(request)
  const parsedMessage = jsonRpcMessageSchema.safeParse(body)
  if (!parsedMessage.success) {
    writeJsonRpcError(response, null, -32700, 'Parse error')
    return
  }

  const rpcMessage = parsedMessage.data as JsonRpcMessage
  observations.push(observeRequest(request, rpcMessage))

  if (!hasRequestId(rpcMessage)) {
    await protocolHandler(protocolRequest, response, body)
    return
  }

  if (rpcMessage.method === 'tools/call' && isTaskToolRequest(rpcMessage)) {
    handleTaskToolCall(response, rpcMessage, taskStore)
    return
  }

  if (rpcMessage.method === 'tasks/get') {
    handleGetTask(response, rpcMessage, taskStore)
    return
  }

  if (rpcMessage.method === 'tasks/update') {
    handleUpdateTask(response, rpcMessage, taskStore)
    return
  }

  if (rpcMessage.method === 'tasks/cancel') {
    handleCancelTask(response, rpcMessage, taskStore)
    return
  }

  if (rpcMessage.method === 'subscriptions/listen' && isTaskSubscriptionRequest(rpcMessage)) {
    handleTaskSubscription(response, rpcMessage, taskStore, subscriptions)
    return
  }

  await protocolHandler(protocolRequest, response, body)
}

function handleTaskToolCall(response: ServerResponse, request: JsonRpcRequest, taskStore: InProcessTaskStore): void {
  if (!hasTaskCapability(request.params?._meta)) {
    writeMissingTaskCapability(response, request.id)
    return
  }

  const name = request.params?.name
  const argumentsValue = request.params?.arguments ?? {}
  if (!isTaskToolName(name)) {
    writeJsonRpcError(response, request.id, -32602, 'Unknown task tool')
    return
  }

  try {
    const result = createTaskForTool(taskStore, name, argumentsValue)
    writeJsonRpcResult(response, request.id, result)
  } catch (error) {
    if (error instanceof z.ZodError) {
      writeJsonRpcError(response, request.id, -32602, 'Invalid tool arguments', error.issues)
      return
    }
    throw error
  }
}

function handleGetTask(response: ServerResponse, request: JsonRpcRequest, taskStore: InProcessTaskStore): void {
  if (!hasTaskCapability(request.params?._meta)) {
    writeMissingTaskCapability(response, request.id)
    return
  }

  const params = taskOperationParamsSchema.safeParse(request.params)
  if (!params.success) {
    writeJsonRpcError(response, request.id, -32602, 'Invalid tasks/get parameters')
    return
  }

  const task = taskStore.get(params.data.taskId)
  if (!task) {
    writeJsonRpcError(response, request.id, -32602, 'Failed to retrieve task: Task not found')
    return
  }

  writeJsonRpcResult(response, request.id, toGetTaskResult(task))
}

function handleUpdateTask(response: ServerResponse, request: JsonRpcRequest, taskStore: InProcessTaskStore): void {
  if (!hasTaskCapability(request.params?._meta)) {
    writeMissingTaskCapability(response, request.id)
    return
  }

  const params = updateTaskParamsSchema.safeParse(request.params)
  if (!params.success) {
    writeJsonRpcError(response, request.id, -32602, 'Invalid tasks/update parameters')
    return
  }

  if (!taskStore.update(params.data.taskId, params.data.inputResponses)) {
    writeJsonRpcError(response, request.id, -32602, 'Failed to update task: Task not found')
    return
  }

  writeJsonRpcResult(response, request.id, { resultType: 'complete' })
}

function handleCancelTask(response: ServerResponse, request: JsonRpcRequest, taskStore: InProcessTaskStore): void {
  if (!hasTaskCapability(request.params?._meta)) {
    writeMissingTaskCapability(response, request.id)
    return
  }

  const params = taskOperationParamsSchema.safeParse(request.params)
  if (!params.success) {
    writeJsonRpcError(response, request.id, -32602, 'Invalid tasks/cancel parameters')
    return
  }

  if (!taskStore.cancel(params.data.taskId)) {
    writeJsonRpcError(response, request.id, -32602, 'Failed to cancel task: Task not found')
    return
  }

  writeJsonRpcResult(response, request.id, { resultType: 'complete' })
}

function handleTaskSubscription(
  response: ServerResponse,
  request: JsonRpcRequest,
  taskStore: InProcessTaskStore,
  subscriptions: Set<TaskSubscription>
): void {
  if (!hasTaskCapability(request.params?._meta)) {
    writeMissingTaskCapability(response, request.id)
    return
  }

  const params = taskSubscriptionParamsSchema.safeParse(request.params)
  if (!params.success) {
    writeJsonRpcError(response, request.id, -32602, 'Invalid subscriptions/listen parameters')
    return
  }

  const taskIds = params.data.notifications.taskIds.filter((taskId) => taskStore.get(taskId) !== undefined)
  response.writeHead(200, {
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
    'Content-Type': 'text/event-stream',
  })
  response.flushHeaders()

  const subscriptionMeta = {
    [SUBSCRIPTION_ID_META_KEY]: request.id,
  }
  writeSseMessage(response, {
    jsonrpc: '2.0',
    method: 'notifications/subscriptions/acknowledged',
    params: {
      _meta: subscriptionMeta,
      notifications: {
        taskIds,
      },
    },
  })

  const unsubscribe = taskStore.subscribe(taskIds, (task) => {
    writeSseMessage(response, {
      jsonrpc: '2.0',
      method: 'notifications/tasks',
      params: {
        ...task,
        _meta: subscriptionMeta,
      },
    })
  })

  for (const taskId of taskIds) {
    const task = taskStore.get(taskId)
    if (task) {
      writeSseMessage(response, {
        jsonrpc: '2.0',
        method: 'notifications/tasks',
        params: {
          ...task,
          _meta: subscriptionMeta,
        },
      })
    }
  }

  let closed = false
  const subscription: TaskSubscription = {
    response,
    close: (): void => {
      if (closed) return
      closed = true
      unsubscribe()
      subscriptions.delete(subscription)
      writeSseMessage(response, {
        jsonrpc: '2.0',
        id: request.id,
        result: {
          resultType: 'complete',
          _meta: subscriptionMeta,
        },
      })
      response.end()
    },
  }
  subscriptions.add(subscription)
  response.on('close', subscription.close)
}

function createTaskForTool(
  taskStore: InProcessTaskStore,
  name: TaskToolName,
  argumentsValue: unknown
): CreateTaskResult {
  switch (name) {
    case 'instant_task': {
      const { value } = instantTaskInputSchema.parse(argumentsValue)
      return taskStore.createInstant(value ?? 'instant result')
    }
    case 'long_running_task': {
      const { duration, message } = longRunningTaskInputSchema.parse(argumentsValue)
      return taskStore.createLongRunning(duration ?? 200, message ?? 'Task completed!')
    }
    case 'failing_task': {
      const { error_message } = failingTaskInputSchema.parse(argumentsValue)
      return taskStore.createFailing(error_message ?? 'Task intentionally failed')
    }
    case 'input_required_task': {
      const { prompt } = inputRequiredTaskInputSchema.parse(argumentsValue)
      return taskStore.createInputRequired(prompt ?? 'Provide a value for this task')
    }
    case 'cancellable_task': {
      const { message } = cancellableTaskInputSchema.parse(argumentsValue)
      return taskStore.createCancellable(message ?? 'Waiting for cancellation')
    }
    case 'cancelled_task': {
      const { reason } = cancelledTaskInputSchema.parse(argumentsValue)
      return taskStore.createCancelled(reason ?? 'Task was cancelled')
    }
  }
}

function requireTaskCapability(context: ServerContext, result: CreateTaskResult): CreateTaskResult {
  if (!hasTaskCapability(context.mcpReq.envelope)) {
    throw new Error(`Missing required client capability: ${TASK_EXTENSION_ID}`)
  }
  return result
}

function transitionTask(
  task: DetailedTask,
  status: 'working',
  statusMessage: string,
  details?: Record<string, never>
): WorkingTask
function transitionTask(
  task: DetailedTask,
  status: 'completed',
  statusMessage: string,
  details: Pick<CompletedTask, 'result'>
): CompletedTask
function transitionTask(
  task: DetailedTask,
  status: 'failed',
  statusMessage: string,
  details: Pick<FailedTask, 'error'>
): FailedTask
function transitionTask(
  task: DetailedTask,
  status: 'cancelled',
  statusMessage: string,
  details?: Record<string, never>
): CancelledTask
function transitionTask(
  task: DetailedTask,
  status: Exclude<TaskStatus, 'input_required'>,
  statusMessage: string,
  details: Partial<Pick<CompletedTask, 'result'> & Pick<FailedTask, 'error'>> = {}
): Exclude<DetailedTask, InputRequiredTask> {
  const base = {
    taskId: task.taskId,
    statusMessage,
    createdAt: task.createdAt,
    lastUpdatedAt: new Date().toISOString(),
    ttlMs: task.ttlMs,
    ...(task.pollIntervalMs !== undefined && { pollIntervalMs: task.pollIntervalMs }),
  }

  switch (status) {
    case 'working':
      return { ...base, status }
    case 'completed':
      return { ...base, status, result: details.result! }
    case 'failed':
      return { ...base, status, error: details.error! }
    case 'cancelled':
      return { ...base, status }
  }
}

function toCreateTaskResult(task: DetailedTask): CreateTaskResult {
  return {
    resultType: 'task',
    taskId: task.taskId,
    status: task.status,
    ...(task.statusMessage !== undefined && { statusMessage: task.statusMessage }),
    createdAt: task.createdAt,
    lastUpdatedAt: task.lastUpdatedAt,
    ttlMs: task.ttlMs,
    ...(task.pollIntervalMs !== undefined && { pollIntervalMs: task.pollIntervalMs }),
  }
}

function toGetTaskResult(task: DetailedTask): GetTaskResult {
  return {
    resultType: 'complete',
    ...task,
  }
}

function toSdkTaskResult(task: CreateTaskResult): CallToolResult {
  return {
    ...task,
    content: [],
  }
}

function textResult(text: string): CallToolResult {
  return {
    content: [{ type: 'text', text }],
  }
}

function isTaskToolRequest(request: JsonRpcRequest): boolean {
  return isTaskToolName(request.params?.name)
}

function hasRequestId(message: JsonRpcMessage): message is JsonRpcRequest {
  return message.id !== undefined
}

function isTaskToolName(value: unknown): value is TaskToolName {
  return (
    value === 'instant_task' ||
    value === 'long_running_task' ||
    value === 'failing_task' ||
    value === 'input_required_task' ||
    value === 'cancellable_task' ||
    value === 'cancelled_task'
  )
}

function isTaskSubscriptionRequest(request: JsonRpcRequest): boolean {
  return taskSubscriptionParamsSchema.safeParse(request.params).success
}

function hasTaskCapability(metaValue: unknown): boolean {
  if (!isRecord(metaValue)) return false
  const capabilities = metaValue[CLIENT_CAPABILITIES_META_KEY]
  if (!isRecord(capabilities)) return false
  const extensions = capabilities.extensions
  if (!isRecord(extensions)) return false
  return isRecord(extensions[TASK_EXTENSION_ID])
}

function observeRequest(request: IncomingMessage, rpcMessage: JsonRpcMessage): TaskHttpRequestObservation {
  const params = rpcMessage.params
  const name = typeof params?.name === 'string' ? params.name : undefined
  const taskId = typeof params?.taskId === 'string' ? params.taskId : undefined
  const mcpMethod = readHeader(request, 'mcp-method')
  const mcpName = readHeader(request, 'mcp-name')
  const protocolVersion = readHeader(request, 'mcp-protocol-version')

  return {
    method: rpcMessage.method,
    ...(name !== undefined && { name }),
    ...(taskId !== undefined && { taskId }),
    ...(mcpMethod !== undefined && { mcpMethod }),
    ...(mcpName !== undefined && { mcpName }),
    ...(protocolVersion !== undefined && { protocolVersion }),
  }
}

function readHeader(request: IncomingMessage, name: string): string | undefined {
  const value = request.headers[name]
  return Array.isArray(value) ? value[0] : value
}

async function readJsonBody(request: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = []
  for await (const chunk of request) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk))
  }
  const body = Buffer.concat(chunks).toString('utf8')
  return body.length > 0 ? JSON.parse(body) : undefined
}

function writeJsonRpcResult(response: ServerResponse, id: RequestId, result: object): void {
  writeJson(response, 200, {
    jsonrpc: '2.0',
    id,
    result: {
      ...result,
      _meta: {
        [SERVER_INFO_META_KEY]: SERVER_INFO,
      },
    },
  })
}

function writeJsonRpcError(
  response: ServerResponse,
  id: RequestId | null,
  code: number,
  message: string,
  data?: unknown
): void {
  writeJson(response, 200, {
    jsonrpc: '2.0',
    id,
    error: {
      code,
      message,
      ...(data !== undefined && { data }),
    },
  })
}

function writeMissingTaskCapability(response: ServerResponse, id: RequestId): void {
  writeJsonRpcError(response, id, -32003, 'Missing required client capability', {
    requiredCapabilities: {
      extensions: {
        [TASK_EXTENSION_ID]: {},
      },
    },
  })
}

function writeJson(response: ServerResponse, status: number, body: unknown): void {
  response.writeHead(status, {
    'Content-Type': 'application/json',
  })
  response.end(JSON.stringify(body))
}

function writeSseMessage(response: ServerResponse, message: unknown): void {
  if (response.destroyed || response.writableEnded) return
  response.write(`event: message\ndata: ${JSON.stringify(message)}\n\n`)
}

function writeUnexpectedError(response: ServerResponse, error: unknown): void {
  if (response.headersSent) {
    response.end()
    return
  }
  const message = error instanceof Error ? error.message : String(error)
  writeJsonRpcError(response, null, -32603, 'Internal server error', { message })
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function listen(server: HttpServer): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      server.off('error', reject)
      resolve()
    })
  })
}

function closeServer(server: HttpServer): Promise<void> {
  return new Promise((resolve, reject) => {
    server.close((error) => {
      if (error) {
        reject(error)
        return
      }
      resolve()
    })
  })
}
