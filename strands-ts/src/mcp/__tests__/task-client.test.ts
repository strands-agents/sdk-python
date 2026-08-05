import {
  CLIENT_CAPABILITIES_META_KEY,
  CLIENT_INFO_META_KEY,
  InMemoryTransport,
  PROTOCOL_VERSION_META_KEY,
  ProtocolError,
  ProtocolErrorCode,
  SERVER_INFO_META_KEY,
  SUBSCRIPTION_ID_META_KEY,
  SdkError,
  SdkErrorCode,
} from '@modelcontextprotocol/client'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { McpClient, McpTaskCancelledError, type TasksConfig } from '../client.js'
import { McpTool } from '../../tools/mcp-tool.js'

import type {
  ClientCapabilities,
  JSONRPCErrorResponse,
  JSONRPCMessage,
  JSONRPCRequest,
  RequestId,
  ServerCapabilities,
  Transport,
} from '@modelcontextprotocol/client'
import type { McpCreateTaskResult, McpGetTaskResult, McpInputRequests } from '../task-types.js'

const TASKS_EXTENSION = 'io.modelcontextprotocol/tasks'
const MODERN_PROTOCOL_VERSION = '2026-07-28'
const LEGACY_PROTOCOL_VERSION = '2025-11-25'
const TASK_ID = 'task-1'
const CREATED_AT = '2026-08-04T12:00:00.000Z'
const NO_RESPONSE = Symbol('no-response')

type ScriptedResponse = unknown | typeof NO_RESPONSE
type RequestHandler = (request: JSONRPCRequest) => ScriptedResponse | Promise<ScriptedResponse>

interface ScriptedServerOptions {
  era?: 'modern' | 'legacy'
  capabilities?: ServerCapabilities
}

interface TaskHarness {
  client: McpClient
  server: ScriptedServer
  tool: McpTool
  cleanup: () => Promise<void>
}

class ScriptedServer {
  public readonly messages: JSONRPCMessage[] = []

  private readonly _transport: Transport
  private readonly _era: 'modern' | 'legacy'
  private readonly _capabilities: ServerCapabilities
  private readonly _handlers = new Map<string, RequestHandler>()

  private constructor(transport: Transport, options: ScriptedServerOptions) {
    this._transport = transport
    this._era = options.era ?? 'modern'
    this._capabilities = options.capabilities ?? {
      tools: {},
      extensions: { [TASKS_EXTENSION]: {} },
    }
    this._transport.onmessage = (message): void => {
      this._receive(message)
    }
  }

  public static async create(options: ScriptedServerOptions = {}): Promise<{
    clientTransport: Transport
    server: ScriptedServer
  }> {
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair()
    const server = new ScriptedServer(serverTransport, options)
    await serverTransport.start()
    return { clientTransport, server }
  }

  public handle(method: string, handler: RequestHandler): void {
    this._handlers.set(method, handler)
  }

  public requests(method: string): JSONRPCRequest[] {
    return this.messages.filter(
      (message): message is JSONRPCRequest => isJsonRpcRequest(message) && message.method === method
    )
  }

  public async notify(method: string, params: Record<string, unknown>): Promise<void> {
    await this._transport.send({
      jsonrpc: '2.0',
      method,
      params,
    })
  }

  public async close(): Promise<void> {
    await this._transport.close()
  }

  private _receive(message: JSONRPCMessage): void {
    this.messages.push(message)
    if (!isJsonRpcRequest(message)) return

    if (message.method === 'server/discover') {
      if (this._era === 'legacy') {
        void this._sendError(message.id, ProtocolErrorCode.MethodNotFound, 'Method not found')
      } else {
        void this._sendResult(message.id, {
          resultType: 'complete',
          supportedVersions: [MODERN_PROTOCOL_VERSION],
          capabilities: this._capabilities,
          _meta: {
            [SERVER_INFO_META_KEY]: { name: 'task-test-server', version: '1.0.0' },
          },
        })
      }
      return
    }

    if (message.method === 'initialize') {
      void this._sendResult(message.id, {
        protocolVersion: LEGACY_PROTOCOL_VERSION,
        capabilities: this._capabilities,
        serverInfo: { name: 'legacy-task-test-server', version: '1.0.0' },
      })
      return
    }

    const handler = this._handlers.get(message.method)
    if (!handler) {
      void this._sendError(message.id, ProtocolErrorCode.MethodNotFound, `No handler for ${message.method}`)
      return
    }

    void Promise.resolve()
      .then(() => handler(message))
      .then(async (result) => {
        if (result !== NO_RESPONSE) await this._sendResult(message.id, result)
      })
      .catch(async (error: unknown) => {
        if (error instanceof ProtocolError) {
          await this._sendError(message.id, error.code, error.message, error.data)
        } else {
          await this._sendError(message.id, ProtocolErrorCode.InternalError, 'Scripted server failure')
        }
      })
  }

  private async _sendResult(id: RequestId, result: unknown): Promise<void> {
    await this._transport.send({
      jsonrpc: '2.0',
      id,
      result,
    } as JSONRPCMessage)
  }

  private async _sendError(id: RequestId, code: number, message: string, data?: unknown): Promise<void> {
    const response: JSONRPCErrorResponse = {
      jsonrpc: '2.0',
      id,
      error: {
        code,
        message,
        ...(data !== undefined && { data }),
      },
    }
    await this._transport.send(response)
  }
}

const activeHarnesses: TaskHarness[] = []

afterEach(async () => {
  for (const harness of activeHarnesses.splice(0)) {
    await harness.cleanup()
  }
  vi.useRealTimers()
  vi.restoreAllMocks()
})

async function createHarness(
  options: ScriptedServerOptions & {
    tasksConfig?: TasksConfig | false
    elicitationCallback?: ConstructorParameters<typeof McpClient>[0]['elicitationCallback']
  } = {}
): Promise<TaskHarness> {
  const { clientTransport, server } = await ScriptedServer.create(options)
  const tasksConfig =
    options.tasksConfig === false
      ? undefined
      : {
          timeoutMs: 1_000,
          requestTimeoutMs: 500,
          pollIntervalMs: 10,
          useNotifications: false,
          ...options.tasksConfig,
        }
  const client = new McpClient({
    applicationName: 'task-test-client',
    applicationVersion: '1.2.3',
    transport: clientTransport,
    ...(tasksConfig !== undefined && { tasksConfig }),
    ...(options.elicitationCallback && { elicitationCallback: options.elicitationCallback }),
  })
  const tool = new McpTool({
    name: 'task_tool',
    description: 'Task tool',
    inputSchema: { type: 'object' },
    client,
  })
  let cleaned = false
  const harness: TaskHarness = {
    client,
    server,
    tool,
    cleanup: async (): Promise<void> => {
      if (cleaned) return
      cleaned = true
      await client.disconnect().catch(() => undefined)
      await server.close().catch(() => undefined)
    },
  }
  activeHarnesses.push(harness)
  return harness
}

function updatedAt(revision: number): string {
  return new Date(Date.parse(CREATED_AT) + revision * 1_000).toISOString()
}

function createTask(
  status: McpCreateTaskResult['status'] = 'working',
  overrides: Partial<McpCreateTaskResult> = {}
): McpCreateTaskResult {
  return {
    resultType: 'task',
    taskId: TASK_ID,
    status,
    createdAt: CREATED_AT,
    lastUpdatedAt: CREATED_AT,
    ttlMs: 60_000,
    pollIntervalMs: 10,
    ...overrides,
  }
}

function workingTask(revision: number, overrides: Partial<McpGetTaskResult> = {}): McpGetTaskResult {
  return {
    resultType: 'complete',
    taskId: TASK_ID,
    status: 'working',
    createdAt: CREATED_AT,
    lastUpdatedAt: updatedAt(revision),
    ttlMs: 60_000,
    pollIntervalMs: 10,
    ...overrides,
  } as McpGetTaskResult
}

function completedTask(
  revision: number,
  text: string = 'done',
  overrides: Partial<McpGetTaskResult> = {}
): McpGetTaskResult {
  return {
    resultType: 'complete',
    taskId: TASK_ID,
    status: 'completed',
    createdAt: CREATED_AT,
    lastUpdatedAt: updatedAt(revision),
    ttlMs: 60_000,
    result: {
      content: [{ type: 'text', text }],
    },
    ...overrides,
  } as McpGetTaskResult
}

function failedTask(revision: number, statusMessage?: string): McpGetTaskResult {
  return {
    resultType: 'complete',
    taskId: TASK_ID,
    status: 'failed',
    statusMessage,
    createdAt: CREATED_AT,
    lastUpdatedAt: updatedAt(revision),
    ttlMs: 60_000,
    error: {
      code: -32_603,
      message: 'Task execution failed',
      data: { retryable: false },
    },
  } as McpGetTaskResult
}

function cancelledTask(revision: number, statusMessage?: string): McpGetTaskResult {
  return {
    resultType: 'complete',
    taskId: TASK_ID,
    status: 'cancelled',
    statusMessage,
    createdAt: CREATED_AT,
    lastUpdatedAt: updatedAt(revision),
    ttlMs: 60_000,
  } as McpGetTaskResult
}

function inputRequiredTask(
  revision: number,
  inputRequests: McpInputRequests,
  overrides: Partial<McpGetTaskResult> = {}
): McpGetTaskResult {
  return {
    resultType: 'complete',
    taskId: TASK_ID,
    status: 'input_required',
    createdAt: CREATED_AT,
    lastUpdatedAt: updatedAt(revision),
    ttlMs: 60_000,
    pollIntervalMs: 10,
    inputRequests,
    ...overrides,
  } as McpGetTaskResult
}

function directResult(text: string = 'direct', isError: boolean = false): Record<string, unknown> {
  return {
    resultType: 'complete',
    content: [{ type: 'text', text }],
    ...(isError && { isError: true }),
  }
}

function elicitationRequest(message: string, requestMeta?: Record<string, unknown>): McpInputRequests[string] {
  return {
    method: 'elicitation/create',
    params: {
      mode: 'form',
      message,
      requestedSchema: {
        type: 'object',
        properties: {
          value: { type: 'string' },
        },
        required: ['value'],
      },
      ...(requestMeta && { _meta: requestMeta }),
    },
  }
}

function isJsonRpcRequest(message: JSONRPCMessage): message is JSONRPCRequest {
  return 'method' in message && 'id' in message
}

function requestParams(request: JSONRPCRequest): Record<string, unknown> {
  return request.params as Record<string, unknown>
}

function requestMeta(request: JSONRPCRequest): Record<string, unknown> {
  return requestParams(request)._meta as Record<string, unknown>
}

async function acknowledgeTaskSubscription(server: ScriptedServer, request: JSONRPCRequest): Promise<void> {
  await server.notify('notifications/subscriptions/acknowledged', {
    _meta: { [SUBSCRIPTION_ID_META_KEY]: request.id },
    notifications: { taskIds: [TASK_ID] },
  })
}

function taskNotificationParams(task: McpGetTaskResult, meta: Record<string, unknown> = {}): Record<string, unknown> {
  const params = { ...task } as Record<string, unknown>
  delete params.resultType
  params._meta = meta
  return params
}

describe('McpClient SEP-2663 tasks', () => {
  describe('tool invocation', () => {
    it('returns a direct CallToolResult unchanged from both invocation APIs', async () => {
      const { client, server, tool } = await createHarness()
      const results = [directResult('high-level'), directResult('low-level')]
      server.handle('tools/call', () => results.shift()!)

      await expect(client.callTool(tool, { value: 1 })).resolves.toEqual({
        content: [{ type: 'text', text: 'high-level' }],
      })
      await expect(client.callToolWithTask(tool, { value: 2 })).resolves.toEqual({
        content: [{ type: 'text', text: 'low-level' }],
      })
      expect(server.requests('tasks/get')).toEqual([])
    })

    it('returns a validated task handle from callToolWithTask without polling', async () => {
      const { client, server, tool } = await createHarness()
      const task = createTask()
      server.handle('tools/call', () => task)

      await expect(client.callToolWithTask(tool, { value: 'keep' })).resolves.toEqual(task)
      expect(server.requests('tasks/get')).toEqual([])

      const callRequest = server.requests('tools/call')[0]!
      expect(requestParams(callRequest)).toEqual({
        name: 'task_tool',
        arguments: { value: 'keep' },
        _meta: {
          [PROTOCOL_VERSION_META_KEY]: MODERN_PROTOCOL_VERSION,
          [CLIENT_INFO_META_KEY]: { name: 'task-test-client', version: '1.2.3' },
          [CLIENT_CAPABILITIES_META_KEY]: {
            extensions: { [TASKS_EXTENSION]: {} },
          },
        },
      })
    })

    it('polls working tasks and returns the nested CallToolResult', async () => {
      const { client, server, tool } = await createHarness()
      server.handle('tools/call', () => createTask())
      server.handle('tasks/get', () => completedTask(1, 'finished'))

      await expect(client.callTool(tool, {})).resolves.toEqual({
        content: [{ type: 'text', text: 'finished' }],
      })
      expect(server.requests('tasks/get')).toHaveLength(1)
    })

    it('preserves completed tool results with isError true', async () => {
      const { client, server, tool } = await createHarness()
      server.handle('tools/call', () => createTask('completed'))
      server.handle('tasks/get', () =>
        completedTask(1, 'tool-level failure', {
          result: {
            content: [{ type: 'text', text: 'tool-level failure' }],
            isError: true,
          },
        })
      )

      await expect(client.callTool(tool, {})).resolves.toEqual({
        content: [{ type: 'text', text: 'tool-level failure' }],
        isError: true,
      })
    })
  })

  describe('explicit lifecycle operations', () => {
    it('validates get, update, and cancel independently with exact request metadata', async () => {
      const { client, server } = await createHarness()
      const getResult = workingTask(1)
      server.handle('tasks/get', () => getResult)
      server.handle('tasks/update', () => ({ resultType: 'complete', _meta: { ack: 'update' } }))
      server.handle('tasks/cancel', () => ({ resultType: 'complete', _meta: { ack: 'cancel' } }))

      await expect(client.getTask(TASK_ID)).resolves.toEqual(getResult)
      await expect(
        client.updateTask(TASK_ID, {
          answer: { action: 'accept', content: { value: 'yes' } },
        })
      ).resolves.toEqual({ resultType: 'complete', _meta: { ack: 'update' } })
      await expect(client.cancelTask(TASK_ID)).resolves.toEqual({
        resultType: 'complete',
        _meta: { ack: 'cancel' },
      })

      expect(
        ['tasks/get', 'tasks/update', 'tasks/cancel'].map((method) => {
          const request = server.requests(method)[0]!
          return {
            method,
            params: requestParams(request),
          }
        })
      ).toEqual([
        {
          method: 'tasks/get',
          params: {
            taskId: TASK_ID,
            _meta: expect.objectContaining({
              [PROTOCOL_VERSION_META_KEY]: MODERN_PROTOCOL_VERSION,
              [CLIENT_CAPABILITIES_META_KEY]: {
                extensions: { [TASKS_EXTENSION]: {} },
              },
            }),
          },
        },
        {
          method: 'tasks/update',
          params: {
            taskId: TASK_ID,
            inputResponses: {
              answer: { action: 'accept', content: { value: 'yes' } },
            },
            _meta: expect.objectContaining({
              [PROTOCOL_VERSION_META_KEY]: MODERN_PROTOCOL_VERSION,
              [CLIENT_CAPABILITIES_META_KEY]: {
                extensions: { [TASKS_EXTENSION]: {} },
              },
            }),
          },
        },
        {
          method: 'tasks/cancel',
          params: {
            taskId: TASK_ID,
            _meta: expect.objectContaining({
              [PROTOCOL_VERSION_META_KEY]: MODERN_PROTOCOL_VERSION,
              [CLIENT_CAPABILITIES_META_KEY]: {
                extensions: { [TASKS_EXTENSION]: {} },
              },
            }),
          },
        },
      ])
    })

    it.each([
      {
        name: 'tasks/get',
        method: 'tasks/get',
        invoke: async (client: McpClient, signal: AbortSignal): Promise<unknown> =>
          await client.getTask(TASK_ID, { signal }),
      },
      {
        name: 'tasks/update',
        method: 'tasks/update',
        invoke: async (client: McpClient, signal: AbortSignal): Promise<unknown> =>
          await client.updateTask(
            TASK_ID,
            {
              answer: { action: 'accept', content: { value: 'yes' } },
            },
            { signal }
          ),
      },
      {
        name: 'tasks/cancel',
        method: 'tasks/cancel',
        invoke: async (client: McpClient, signal: AbortSignal): Promise<unknown> =>
          await client.cancelTask(TASK_ID, { signal }),
      },
    ])('honors AbortSignal for $name', async ({ invoke, method }) => {
      const { client, server } = await createHarness()
      const controller = new AbortController()
      const primaryError = new Error(`${method} cancelled`)
      server.handle(method, () => NO_RESPONSE)

      const resultPromise = invoke(client, controller.signal)
      const rejection = expect(resultPromise).rejects.toBe(primaryError)
      await vi.waitFor(() => {
        expect(server.requests(method)).toHaveLength(1)
      })
      controller.abort(primaryError)

      await rejection
    })
  })

  describe('poll scheduling', () => {
    it('honors changing poll intervals returned by the server', async () => {
      vi.useFakeTimers()
      const { client, server, tool } = await createHarness({ tasksConfig: { pollIntervalMs: 25 } })
      server.handle('tools/call', () => createTask('working', { pollIntervalMs: 50 }))
      const states = [workingTask(1, { pollIntervalMs: 80 }), completedTask(2, 'after changing intervals')]
      server.handle('tasks/get', () => states.shift()!)

      const resultPromise = client.callTool(tool, {})
      await vi.advanceTimersByTimeAsync(0)
      expect(server.requests('tasks/get')).toHaveLength(0)

      await vi.advanceTimersByTimeAsync(49)
      expect(server.requests('tasks/get')).toHaveLength(0)
      await vi.advanceTimersByTimeAsync(1)
      expect(server.requests('tasks/get')).toHaveLength(1)

      await vi.advanceTimersByTimeAsync(79)
      expect(server.requests('tasks/get')).toHaveLength(1)
      await vi.advanceTimersByTimeAsync(1)

      await expect(resultPromise).resolves.toEqual({
        content: [{ type: 'text', text: 'after changing intervals' }],
      })
      expect(server.requests('tasks/get')).toHaveLength(2)
    })

    it('uses the configured bounded default when pollIntervalMs is absent', async () => {
      vi.useFakeTimers()
      const { client, server, tool } = await createHarness({ tasksConfig: { pollIntervalMs: 35 } })
      const seed = createTask()
      delete seed.pollIntervalMs
      server.handle('tools/call', () => seed)
      server.handle('tasks/get', () => completedTask(1, 'default interval'))

      const resultPromise = client.callTool(tool, {})
      await vi.advanceTimersByTimeAsync(34)
      expect(server.requests('tasks/get')).toHaveLength(0)
      await vi.advanceTimersByTimeAsync(1)

      await expect(resultPromise).resolves.toEqual({
        content: [{ type: 'text', text: 'default interval' }],
      })
    })

    it('clamps a zero server interval to avoid a busy loop', async () => {
      vi.useFakeTimers()
      const { client, server, tool } = await createHarness()
      server.handle('tools/call', () => createTask('working', { pollIntervalMs: 0 }))
      server.handle('tasks/get', () => completedTask(1))

      const resultPromise = client.callTool(tool, {})
      await vi.advanceTimersByTimeAsync(9)
      expect(server.requests('tasks/get')).toHaveLength(0)
      await vi.advanceTimersByTimeAsync(1)

      await expect(resultPromise).resolves.toEqual({
        content: [{ type: 'text', text: 'done' }],
      })
    })
  })

  describe('timeouts and cancellation', () => {
    it('times out task creation without attempting cancellation before a task handle exists', async () => {
      vi.useFakeTimers()
      const { client, server, tool } = await createHarness({ tasksConfig: { timeoutMs: 50 } })
      server.handle('tools/call', () => NO_RESPONSE)
      server.handle('tasks/cancel', () => ({ resultType: 'complete' }))

      const resultPromise = client.callTool(tool, {})
      const rejection = expect(resultPromise).rejects.toEqual(
        expect.objectContaining({ code: SdkErrorCode.RequestTimeout })
      )
      await vi.advanceTimersByTimeAsync(50)

      await rejection
      expect(server.requests('tasks/cancel')).toEqual([])
    })

    it('cancels exactly once on overall timeout after receiving a task handle', async () => {
      vi.useFakeTimers()
      const { client, server, tool } = await createHarness({ tasksConfig: { timeoutMs: 50 } })
      server.handle('tools/call', () => createTask('working', { pollIntervalMs: 1_000 }))
      server.handle('tasks/cancel', () => ({ resultType: 'complete' }))

      const resultPromise = client.callTool(tool, {})
      const rejection = expect(resultPromise).rejects.toEqual(
        expect.objectContaining({ code: SdkErrorCode.RequestTimeout })
      )
      await vi.advanceTimersByTimeAsync(50)

      await rejection
      expect(server.requests('tasks/cancel')).toHaveLength(1)
      expect(server.requests('tasks/get')).toEqual([])
    })

    it('cancels exactly once when a lifecycle request times out', async () => {
      vi.useFakeTimers()
      const { client, server, tool } = await createHarness({
        tasksConfig: { timeoutMs: 500, requestTimeoutMs: 25 },
      })
      server.handle('tools/call', () => createTask('working', { pollIntervalMs: 10 }))
      server.handle('tasks/get', () => NO_RESPONSE)
      server.handle('tasks/cancel', () => ({ resultType: 'complete' }))

      const resultPromise = client.callTool(tool, {})
      const rejection = expect(resultPromise).rejects.toEqual(
        expect.objectContaining({
          code: SdkErrorCode.RequestTimeout,
          message: 'MCP tasks/get request timed out',
        })
      )
      await vi.advanceTimersByTimeAsync(35)
      await rejection
      await vi.advanceTimersByTimeAsync(0)

      expect(server.requests('tasks/get')).toHaveLength(1)
      expect(server.requests('tasks/cancel')).toHaveLength(1)
      await vi.advanceTimersByTimeAsync(500)
      expect(server.requests('tasks/cancel')).toHaveLength(1)
    })

    it('cancels exactly once on AbortSignal and stops polling promptly', async () => {
      vi.useFakeTimers()
      const { client, server, tool } = await createHarness()
      const controller = new AbortController()
      const primaryError = new Error('caller cancelled')
      server.handle('tools/call', () => createTask('working', { pollIntervalMs: 500 }))
      server.handle('tasks/cancel', () => ({ resultType: 'complete' }))

      const resultPromise = client.callTool(tool, {}, { signal: controller.signal })
      const rejection = expect(resultPromise).rejects.toBe(primaryError)
      await vi.advanceTimersByTimeAsync(0)
      controller.abort(primaryError)
      await vi.advanceTimersByTimeAsync(0)

      await rejection
      expect(server.requests('tasks/cancel')).toHaveLength(1)
      await vi.advanceTimersByTimeAsync(2_000)
      expect(server.requests('tasks/get')).toEqual([])
      expect(server.requests('tasks/cancel')).toHaveLength(1)
    })

    it('rejects promptly on AbortSignal when remote cancellation does not respond', async () => {
      vi.useFakeTimers()
      const { client, server, tool } = await createHarness({
        tasksConfig: { requestTimeoutMs: 500 },
      })
      const controller = new AbortController()
      const primaryError = new Error('caller cancelled')
      server.handle('tools/call', () => createTask('working', { pollIntervalMs: 500 }))
      server.handle('tasks/cancel', () => NO_RESPONSE)

      const rejected = vi.fn()
      void client.callTool(tool, {}, { signal: controller.signal }).catch(rejected)
      await vi.advanceTimersByTimeAsync(0)
      controller.abort(primaryError)
      await vi.advanceTimersByTimeAsync(0)

      expect(rejected).toHaveBeenCalledWith(primaryError)
      expect(server.requests('tasks/cancel')).toHaveLength(1)
      expect(server.requests('tasks/get')).toEqual([])
    })

    it('preserves the primary cancellation error when remote cancellation fails', async () => {
      vi.useFakeTimers()
      const { client, server, tool } = await createHarness()
      const controller = new AbortController()
      const primaryError = new Error('caller cancelled')
      server.handle('tools/call', () => createTask('working', { pollIntervalMs: 500 }))
      server.handle('tasks/cancel', () => {
        throw new ProtocolError(ProtocolErrorCode.InternalError, 'remote cancellation failed')
      })

      const resultPromise = client.callTool(tool, {}, { signal: controller.signal })
      const rejection = expect(resultPromise).rejects.toBe(primaryError)
      await vi.advanceTimersByTimeAsync(0)
      controller.abort(primaryError)
      await vi.advanceTimersByTimeAsync(0)

      await rejection
      expect(server.requests('tasks/cancel')).toHaveLength(1)
    })
  })

  describe('terminal states', () => {
    it('preserves failed-task error data and status context', async () => {
      const { client, server, tool } = await createHarness()
      server.handle('tools/call', () => createTask('failed'))
      server.handle('tasks/get', () => failedTask(1, 'The remote worker stopped'))

      const error = await client.callTool(tool, {}).catch((caught: unknown) => caught)

      expect(error).toBeInstanceOf(ProtocolError)
      expect(error).toEqual(
        expect.objectContaining({
          code: -32_603,
          data: { retryable: false },
          message: 'Task execution failed: The remote worker stopped',
        })
      )
    })

    it('translates cancelled tasks with useful status context', async () => {
      const { client, server, tool } = await createHarness()
      server.handle('tools/call', () => createTask('cancelled'))
      server.handle('tasks/get', () => cancelledTask(1, 'Cancelled by policy'))

      await expect(client.callTool(tool, {})).rejects.toEqual(
        expect.objectContaining({
          name: 'McpTaskCancelledError',
          statusMessage: 'Cancelled by policy',
          message: 'MCP task was cancelled: Cancelled by policy',
        })
      )
      await expect(Promise.reject(new McpTaskCancelledError())).rejects.toBeInstanceOf(McpTaskCancelledError)
    })
  })

  describe('input_required handling', () => {
    it('uses the elicitation handler, preserves request context, and deduplicates repeated keys', async () => {
      const callback = vi.fn().mockResolvedValue({
        action: 'accept',
        content: { value: 'approved' },
      })
      const { client, server, tool } = await createHarness({ elicitationCallback: callback })
      const request = elicitationRequest('Approve this task', { trustPolicy: 'strict' })
      server.handle('tools/call', () => createTask('input_required'))
      const states = [
        inputRequiredTask(1, { approval: request }),
        inputRequiredTask(2, { approval: request }),
        completedTask(3, 'approved'),
      ]
      server.handle('tasks/get', () => states.shift()!)
      server.handle('tasks/update', () => ({ resultType: 'complete' }))

      await expect(client.callTool(tool, {})).resolves.toEqual({
        content: [{ type: 'text', text: 'approved' }],
      })

      expect(callback).toHaveBeenCalledTimes(1)
      expect(callback.mock.calls[0]).toEqual([
        expect.objectContaining({
          taskId: TASK_ID,
          requestId: `task:${TASK_ID}:approval`,
          _meta: { trustPolicy: 'strict' },
          signal: expect.any(AbortSignal),
        }),
        request.params,
      ])
      expect(server.requests('tasks/update').map(requestParams)).toEqual([
        expect.objectContaining({
          taskId: TASK_ID,
          inputResponses: {
            approval: {
              action: 'accept',
              content: { value: 'approved' },
            },
          },
        }),
      ])
      expect(server.requests('tasks/get')).toHaveLength(3)
    })

    it('supports partial response sets through separate tasks/update acknowledgements', async () => {
      const callback = vi.fn().mockResolvedValue({
        action: 'accept',
        content: { value: 'answer' },
      })
      const { client, server, tool } = await createHarness({ elicitationCallback: callback })
      server.handle('tools/call', () => createTask('input_required'))
      const states = [
        inputRequiredTask(1, {
          first: elicitationRequest('First question'),
          second: elicitationRequest('Second question'),
        }),
        completedTask(2, 'both answered'),
      ]
      server.handle('tasks/get', () => states.shift()!)
      server.handle('tasks/update', () => ({ resultType: 'complete' }))

      await expect(client.callTool(tool, {})).resolves.toEqual({
        content: [{ type: 'text', text: 'both answered' }],
      })

      expect(callback).toHaveBeenCalledTimes(2)
      expect(server.requests('tasks/update').map((request) => requestParams(request).inputResponses)).toEqual([
        {
          first: {
            action: 'accept',
            content: { value: 'answer' },
          },
        },
        {
          second: {
            action: 'accept',
            content: { value: 'answer' },
          },
        },
      ])
    })

    it('propagates tasks/update failures without prompting the same key again', async () => {
      const callback = vi.fn().mockResolvedValue({
        action: 'accept',
        content: { value: 'answer' },
      })
      const { client, server, tool } = await createHarness({ elicitationCallback: callback })
      server.handle('tools/call', () => createTask('input_required'))
      server.handle('tasks/get', () =>
        inputRequiredTask(1, {
          answer: elicitationRequest('Question'),
        })
      )
      server.handle('tasks/update', () => {
        throw new ProtocolError(ProtocolErrorCode.InvalidParams, 'Input response rejected', {
          field: 'answer',
        })
      })

      const error = await client.callTool(tool, {}).catch((caught: unknown) => caught)

      expect(error).toEqual(
        expect.objectContaining({
          code: ProtocolErrorCode.InvalidParams,
          message: 'Input response rejected',
          data: { field: 'answer' },
        })
      )
      expect(callback).toHaveBeenCalledTimes(1)
      expect(server.requests('tasks/update')).toHaveLength(1)
    })

    it('fails clearly for unsupported input request methods', async () => {
      const { client, server, tool } = await createHarness()
      server.handle('tools/call', () => createTask('input_required'))
      server.handle('tasks/get', () =>
        inputRequiredTask(1, {
          sample: {
            method: 'sampling/createMessage',
            params: {
              messages: [{ role: 'user', content: { type: 'text', text: 'Summarize' } }],
              maxTokens: 64,
            },
          },
        })
      )

      await expect(client.callTool(tool, {})).rejects.toThrow(
        'Unsupported MCP task input request method "sampling/createMessage" for key "sample"'
      )
      expect(server.requests('tasks/update')).toEqual([])
    })

    it('enforces the overall timeout while an elicitation handler is still pending', async () => {
      vi.useFakeTimers()
      const callback = vi.fn(
        (): Promise<{ action: 'accept'; content: { value: string } }> =>
          new Promise(() => {
            // The timeout must not depend on callback cooperation.
          })
      )
      const { client, server, tool } = await createHarness({
        tasksConfig: { timeoutMs: 50 },
        elicitationCallback: callback,
      })
      server.handle('tools/call', () => createTask('input_required'))
      server.handle('tasks/get', () =>
        inputRequiredTask(1, {
          answer: elicitationRequest('Question'),
        })
      )
      server.handle('tasks/cancel', () => ({ resultType: 'complete' }))

      const resultPromise = client.callTool(tool, {})
      const rejection = expect(resultPromise).rejects.toEqual(
        expect.objectContaining({ code: SdkErrorCode.RequestTimeout })
      )
      await vi.advanceTimersByTimeAsync(50)

      await rejection
      expect(callback).toHaveBeenCalledTimes(1)
      expect(server.requests('tasks/cancel')).toHaveLength(1)
    })
  })

  describe('response validation and state reconciliation', () => {
    it('sanitizes malformed task-handle errors without echoing server data', async () => {
      const { client, server, tool } = await createHarness()
      server.handle('tools/call', () => ({
        ...createTask(),
        taskId: '',
        statusMessage: 'credential=server-secret',
      }))

      const error = await client.callToolWithTask(tool, {}).catch((caught: unknown) => caught)

      expect(error).toBeInstanceOf(SdkError)
      expect(error).toEqual(expect.objectContaining({ code: SdkErrorCode.InvalidResult }))
      expect(String(error)).toContain('MCP tools/call returned a malformed SEP-2663 response')
      expect(String(error)).not.toContain('server-secret')
    })

    it.each([
      {
        name: 'tasks/get',
        invoke: async (client: McpClient): Promise<unknown> => await client.getTask(TASK_ID),
        method: 'tasks/get',
        result: {
          ...workingTask(1),
          status: 'completed',
        },
      },
      {
        name: 'tasks/update',
        invoke: async (client: McpClient): Promise<unknown> =>
          await client.updateTask(TASK_ID, {
            answer: { action: 'accept', content: { value: 'yes' } },
          }),
        method: 'tasks/update',
        result: workingTask(1),
      },
      {
        name: 'tasks/cancel',
        invoke: async (client: McpClient): Promise<unknown> => await client.cancelTask(TASK_ID),
        method: 'tasks/cancel',
        result: workingTask(1),
      },
    ])('rejects malformed $name responses predictably', async ({ invoke, method, result }) => {
      const { client, server } = await createHarness()
      server.handle(method, () => result)

      await expect(invoke(client)).rejects.toEqual(expect.objectContaining({ code: SdkErrorCode.InvalidResult }))
    })

    it('rejects a terminal seed that changes status', async () => {
      const { client, server, tool } = await createHarness()
      server.handle('tools/call', () => createTask('completed'))
      server.handle('tasks/get', () => workingTask(1))

      await expect(client.callTool(tool, {})).rejects.toThrow('MCP task changed after reaching a terminal state')
    })

    it('accepts a valid state transition that shares the previous timestamp', async () => {
      const { client, server, tool } = await createHarness()
      server.handle('tools/call', () => createTask('working'))
      server.handle('tasks/get', () => completedTask(0, 'same timestamp'))

      await expect(client.callTool(tool, {})).resolves.toEqual({
        content: [{ type: 'text', text: 'same timestamp' }],
      })
    })

    it('ignores stale duplicate states and continues to a newer terminal state', async () => {
      const { client, server, tool } = await createHarness()
      server.handle('tools/call', () => createTask())
      const states = [
        workingTask(2, { statusMessage: 'newest working state' }),
        workingTask(1, { statusMessage: 'stale working state' }),
        completedTask(3, 'latest result'),
      ]
      server.handle('tasks/get', () => states.shift()!)

      await expect(client.callTool(tool, {})).resolves.toEqual({
        content: [{ type: 'text', text: 'latest result' }],
      })
      expect(server.requests('tasks/get')).toHaveLength(3)
    })
  })

  describe('concurrent tasks', () => {
    it('keeps polling state isolated across simultaneous calls', async () => {
      const { client, server, tool } = await createHarness()
      server.handle('tools/call', (request) => {
        const argumentsValue = requestParams(request).arguments as { id: string }
        return createTask('working', { taskId: `task-${argumentsValue.id}` })
      })
      server.handle('tasks/get', (request) => {
        const taskId = requestParams(request).taskId as string
        return completedTask(1, `result for ${taskId}`, { taskId })
      })

      const [first, second] = await Promise.all([
        client.callTool(tool, { id: 'a' }),
        client.callTool(tool, { id: 'b' }),
      ])

      expect([first, second]).toEqual([
        { content: [{ type: 'text', text: 'result for task-a' }] },
        { content: [{ type: 'text', text: 'result for task-b' }] },
      ])
      expect(
        server
          .requests('tasks/get')
          .map((request) => requestParams(request).taskId)
          .sort()
      ).toEqual(['task-a', 'task-b'])
    })
  })

  describe('task notifications', () => {
    it('completes from notifications/tasks through subscriptions/listen', async () => {
      const { client, server, tool } = await createHarness({
        tasksConfig: { useNotifications: true, pollIntervalMs: 100 },
      })
      server.handle('tools/call', () => createTask('working', { pollIntervalMs: 100 }))
      server.handle('subscriptions/listen', async (request) => {
        await acknowledgeTaskSubscription(server, request)
        await server.notify(
          'notifications/tasks',
          taskNotificationParams(completedTask(1, 'notification result'), {
            [SUBSCRIPTION_ID_META_KEY]: request.id,
          })
        )
        return NO_RESPONSE
      })

      await expect(client.callTool(tool, {})).resolves.toEqual({
        content: [{ type: 'text', text: 'notification result' }],
      })
      expect(server.requests('subscriptions/listen').map(requestParams)).toEqual([
        expect.objectContaining({
          notifications: { taskIds: [TASK_ID] },
        }),
      ])
      expect(server.requests('tasks/get')).toEqual([])
    })

    it('accepts poll and notification duplicates that differ only in envelope metadata', async () => {
      const { client, server, tool } = await createHarness({ tasksConfig: { useNotifications: true } })
      server.handle('tools/call', () => createTask())
      server.handle('subscriptions/listen', async (request) => {
        await acknowledgeTaskSubscription(server, request)
        return NO_RESPONSE
      })
      server.handle('tasks/get', async () => {
        const state = completedTask(1, 'same state', { _meta: { source: 'poll' } })
        await server.notify(
          'notifications/tasks',
          taskNotificationParams(state, {
            source: 'subscription',
            [SUBSCRIPTION_ID_META_KEY]: server.requests('subscriptions/listen')[0]!.id,
          })
        )
        return state
      })

      await expect(client.callTool(tool, {})).resolves.toEqual({
        content: [{ type: 'text', text: 'same state' }],
      })
      expect(server.requests('tasks/get')).toHaveLength(1)
    })

    it('rejects contradictory poll and notification states for the same update time', async () => {
      const { client, server, tool } = await createHarness({ tasksConfig: { useNotifications: true } })
      server.handle('tools/call', () => createTask())
      server.handle('subscriptions/listen', async (request) => {
        await acknowledgeTaskSubscription(server, request)
        return NO_RESPONSE
      })
      server.handle('tasks/get', async () => {
        await server.notify(
          'notifications/tasks',
          taskNotificationParams(completedTask(1, 'notification value'), {
            [SUBSCRIPTION_ID_META_KEY]: server.requests('subscriptions/listen')[0]!.id,
          })
        )
        return completedTask(1, 'poll value')
      })

      await expect(client.callTool(tool, {})).rejects.toThrow('MCP task changed after reaching a terminal state')
    })

    it('falls back to polling when subscription setup fails', async () => {
      const { client, server, tool } = await createHarness({ tasksConfig: { useNotifications: true } })
      server.handle('tools/call', () => createTask())
      server.handle('subscriptions/listen', () => {
        throw new ProtocolError(ProtocolErrorCode.InternalError, 'Subscriptions unavailable')
      })
      server.handle('tasks/get', () => completedTask(1, 'poll fallback'))

      await expect(client.callTool(tool, {})).resolves.toEqual({
        content: [{ type: 'text', text: 'poll fallback' }],
      })
      expect(server.requests('subscriptions/listen')).toHaveLength(1)
      expect(server.requests('tasks/get')).toHaveLength(1)
    })

    it('ignores malformed task notifications and falls back to polling', async () => {
      const { client, server, tool } = await createHarness({ tasksConfig: { useNotifications: true } })
      server.handle('tools/call', () => createTask())
      server.handle('subscriptions/listen', async (request) => {
        await acknowledgeTaskSubscription(server, request)
        const malformed = taskNotificationParams(completedTask(1, 'invalid notification'))
        delete malformed.result
        await server.notify('notifications/tasks', malformed)
        return NO_RESPONSE
      })
      server.handle('tasks/get', () => completedTask(2, 'poll fallback'))

      await expect(client.callTool(tool, {})).resolves.toEqual({
        content: [{ type: 'text', text: 'poll fallback' }],
      })
      expect(server.requests('tasks/get')).toHaveLength(1)
    })

    it('continues polling after an acknowledged subscription ends early', async () => {
      const { client, server, tool } = await createHarness({ tasksConfig: { useNotifications: true } })
      server.handle('tools/call', () => createTask())
      server.handle('subscriptions/listen', async (request) => {
        await acknowledgeTaskSubscription(server, request)
        return { resultType: 'complete' }
      })
      server.handle('tasks/get', () => completedTask(1, 'poll after close'))

      await expect(client.callTool(tool, {})).resolves.toEqual({
        content: [{ type: 'text', text: 'poll after close' }],
      })
      expect(server.requests('tasks/get')).toHaveLength(1)
    })
  })

  describe('protocol compatibility', () => {
    it('advertises task capability only when tasks are enabled', async () => {
      const enabled = await createHarness()
      enabled.server.handle('tools/call', () => directResult())
      await enabled.client.callTool(enabled.tool, {})

      const disabled = await createHarness({ tasksConfig: false })
      disabled.server.handle('tools/call', () => directResult())
      await disabled.client.callTool(disabled.tool, {})

      expect(requestMeta(enabled.server.requests('server/discover')[0]!)).toEqual(
        expect.objectContaining({
          [CLIENT_CAPABILITIES_META_KEY]: {
            extensions: { [TASKS_EXTENSION]: {} },
          },
        })
      )
      expect(requestMeta(disabled.server.requests('server/discover')[0]!)[CLIENT_CAPABILITIES_META_KEY]).toEqual({})
      expect(requestMeta(disabled.server.requests('tools/call')[0]!)[CLIENT_CAPABILITIES_META_KEY]).toEqual({})
    })

    it('keeps legacy direct calls working without task capability metadata', async () => {
      const { client, server, tool } = await createHarness({ era: 'legacy' })
      server.handle('tools/call', () => ({ content: [{ type: 'text', text: 'legacy direct' }] }))

      await expect(client.callTool(tool, {})).resolves.toEqual({
        content: [{ type: 'text', text: 'legacy direct' }],
      })

      const initialize = server.requests('initialize')[0]!
      const initializeCapabilities = requestParams(initialize).capabilities as ClientCapabilities
      expect(initializeCapabilities.extensions?.[TASKS_EXTENSION]).toBeUndefined()
      expect(requestParams(server.requests('tools/call')[0]!)).toEqual({
        name: 'task_tool',
        arguments: {},
      })
      await expect(client.getTask(TASK_ID)).rejects.toThrow(
        `SEP-2663 task operations require negotiated MCP protocol ${MODERN_PROTOCOL_VERSION}`
      )
      expect(server.requests('tasks/get')).toEqual([])
    })

    it('rejects task handles and lifecycle methods when the server omits the extension capability', async () => {
      const { client, server, tool } = await createHarness({ capabilities: { tools: {} } })
      server.handle('tools/call', () => createTask())

      await expect(client.callToolWithTask(tool, {})).rejects.toThrow(
        `MCP server did not advertise the ${TASKS_EXTENSION} extension`
      )
      await expect(client.cancelTask(TASK_ID)).rejects.toThrow(
        `MCP server did not advertise the ${TASKS_EXTENSION} extension`
      )
      expect(server.requests('tasks/cancel')).toEqual([])
    })
  })
})
