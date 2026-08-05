import { describe, it, expect, beforeAll, afterAll, vi } from 'vitest'
import { McpClient, Agent } from '@strands-agents/sdk'
import { startTaskHTTPServer, type TaskHttpServerInfo } from '../__fixtures__/test-mcp-task-server.js'
import { startHTTPServer, type HttpServerInfo } from '../__fixtures__/test-mcp-server.js'
import { bedrock } from '../__fixtures__/model-providers.js'
import { hasToolUse, countToolResults } from '../__fixtures__/test-helpers.js'

import type { ElicitationCallback, TasksConfig } from '@strands-agents/sdk'

const MODERN_PROTOCOL_VERSION = '2026-07-28'

/**
 * Creates a connected McpClient for the given server URL.
 * Returns the client - caller is responsible for disconnecting.
 * @param serverUrl - The URL of the MCP server
 * @param appName - The application name for the client
 * @param tasksConfig - Optional tasks configuration. When provided, enables task-based tool invocation.
 */
function createClient(serverUrl: string, appName: string, tasksConfig?: TasksConfig): McpClient {
  return new McpClient({
    applicationName: appName,
    url: serverUrl,
    ...(tasksConfig !== undefined && { tasksConfig }),
  })
}

describe('MCP Task Integration Tests', () => {
  let taskServerInfo: TaskHttpServerInfo | undefined
  let nonTaskServerInfo: HttpServerInfo | undefined

  beforeAll(async () => {
    // Start both servers in parallel
    ;[taskServerInfo, nonTaskServerInfo] = await Promise.all([startTaskHTTPServer(), startHTTPServer()])
  }, 30000)

  afterAll(async () => {
    // Clean up both servers
    await Promise.all([taskServerInfo?.close(), nonTaskServerInfo?.close()])
  }, 30000)

  describe('McpClient.callTool() with Task-Enabled Server', () => {
    it('preserves direct tool results when task support is enabled', async () => {
      if (!taskServerInfo) throw new Error('Task server not started')

      const client = createClient(taskServerInfo.url, 'test-direct-result-client', {})
      try {
        const tools = await client.listTools()
        const directTool = tools.find((tool) => tool.name === 'direct_result')
        if (!directTool) throw new Error('direct_result tool not found')

        await expect(client.callTool(directTool, { value: 'direct response' })).resolves.toMatchObject({
          content: [{ type: 'text', text: 'direct response' }],
        })
        expect(taskServerInfo.requests.filter((request) => request.method === 'tasks/get')).toEqual([])
      } finally {
        await client.disconnect()
      }
    }, 30000)

    it('returns a task handle without polling and supports explicit lifecycle operations', async () => {
      if (!taskServerInfo) throw new Error('Task server not started')

      const client = createClient(taskServerInfo.url, 'test-task-lifecycle-client', {
        useNotifications: false,
      })
      try {
        const tools = await client.listTools()
        const cancellableTool = tools.find((tool) => tool.name === 'cancellable_task')
        if (!cancellableTool) throw new Error('cancellable_task tool not found')
        const requestStart = taskServerInfo.requests.length

        const task = await client.callToolWithTask(cancellableTool, { message: 'waiting' })
        expect(task).toMatchObject({
          resultType: 'task',
          status: 'working',
          statusMessage: 'waiting',
        })
        if (task.resultType !== 'task' || typeof task.taskId !== 'string') {
          throw new Error('Expected a task handle')
        }
        const taskId = task.taskId

        expect(taskServerInfo.requests.slice(requestStart).filter((request) => request.method === 'tasks/get')).toEqual(
          []
        )
        await expect(client.getTask(taskId)).resolves.toMatchObject({
          taskId,
          status: 'working',
        })
        await expect(client.cancelTask(taskId)).resolves.toEqual({
          resultType: 'complete',
          _meta: expect.any(Object),
        })
        await expect(client.getTask(taskId)).resolves.toMatchObject({
          taskId,
          status: 'cancelled',
        })

        expect(
          taskServerInfo.requests
            .slice(requestStart)
            .filter((request) => request.method.startsWith('tasks/'))
            .map(({ method, taskId, mcpMethod, mcpName, protocolVersion }) => ({
              method,
              taskId,
              mcpMethod,
              mcpName,
              protocolVersion,
            }))
        ).toEqual([
          {
            method: 'tasks/get',
            taskId,
            mcpMethod: 'tasks/get',
            mcpName: taskId,
            protocolVersion: MODERN_PROTOCOL_VERSION,
          },
          {
            method: 'tasks/cancel',
            taskId,
            mcpMethod: 'tasks/cancel',
            mcpName: taskId,
            protocolVersion: MODERN_PROTOCOL_VERSION,
          },
          {
            method: 'tasks/get',
            taskId,
            mcpMethod: 'tasks/get',
            mcpName: taskId,
            protocolVersion: MODERN_PROTOCOL_VERSION,
          },
        ])
      } finally {
        await client.disconnect()
      }
    }, 30000)

    it('extracts result from task tool that completes immediately', async () => {
      if (!taskServerInfo) throw new Error('Task server not started')

      const client = createClient(taskServerInfo.url, 'test-task-client', {})
      try {
        await client.connect()
        const tools = await client.listTools()
        const instantTool = tools.find((t) => t.name === 'instant_task')
        expect(instantTool).toBeDefined()

        const result = await client.callTool(instantTool!, { value: 'hello from instant task' })

        expect(result).toMatchObject({
          content: expect.arrayContaining([expect.objectContaining({ type: 'text', text: 'hello from instant task' })]),
        })
      } finally {
        await client.disconnect()
      }
    }, 30000)

    it('extracts result from long-running task with progress updates', async () => {
      if (!taskServerInfo) throw new Error('Task server not started')

      const client = createClient(taskServerInfo.url, 'test-task-client', {})
      try {
        await client.connect()
        const tools = await client.listTools()
        const longRunningTool = tools.find((t) => t.name === 'long_running_task')
        expect(longRunningTool).toBeDefined()

        const result = await client.callTool(longRunningTool!, {
          duration: 300,
          message: 'Long task completed successfully!',
        })

        expect(result).toMatchObject({
          content: expect.arrayContaining([
            expect.objectContaining({ type: 'text', text: 'Long task completed successfully!' }),
          ]),
        })
      } finally {
        await client.disconnect()
      }
    }, 30000)

    it('throws error for failed tasks (MCP SDK behavior)', async () => {
      if (!taskServerInfo) throw new Error('Task server not started')

      const client = createClient(taskServerInfo.url, 'test-task-client', {})
      try {
        await client.connect()
        const tools = await client.listTools()
        const failingTool = tools.find((t) => t.name === 'failing_task')
        expect(failingTool).toBeDefined()

        await expect(client.callTool(failingTool!, { error_message: 'This task failed on purpose!' })).rejects.toThrow(
          /failed/i
        )
      } finally {
        await client.disconnect()
      }
    }, 30000)

    it('handles task elicitation through tasks/update and returns the final result', async () => {
      if (!taskServerInfo) throw new Error('Task server not started')

      const elicitationCallback: ElicitationCallback = vi.fn().mockResolvedValue({
        action: 'accept',
        content: { value: 'integration answer' },
      })
      const client = new McpClient({
        applicationName: 'test-task-elicitation-client',
        url: taskServerInfo.url,
        tasksConfig: {
          pollIntervalMs: 10,
          timeoutMs: 5_000,
        },
        elicitationCallback,
      })
      try {
        const tools = await client.listTools()
        const inputTool = tools.find((tool) => tool.name === 'input_required_task')
        if (!inputTool) throw new Error('input_required_task tool not found')
        const requestStart = taskServerInfo.requests.length

        await expect(client.callTool(inputTool, { prompt: 'Provide integration input' })).resolves.toMatchObject({
          content: [{ type: 'text', text: 'Input received: integration answer' }],
        })
        expect(elicitationCallback).toHaveBeenCalledOnce()
        const updateRequests = taskServerInfo.requests
          .slice(requestStart)
          .filter((request) => request.method === 'tasks/update')
        expect(updateRequests).toHaveLength(1)
        expect(updateRequests[0]).toEqual(
          expect.objectContaining({
            method: 'tasks/update',
            taskId: expect.any(String),
            mcpMethod: 'tasks/update',
            mcpName: updateRequests[0]!.taskId,
            protocolVersion: MODERN_PROTOCOL_VERSION,
          })
        )
      } finally {
        await client.disconnect()
      }
    }, 30000)

    it('translates a server-cancelled task consistently', async () => {
      if (!taskServerInfo) throw new Error('Task server not started')

      const client = createClient(taskServerInfo.url, 'test-cancelled-task-client', {
        useNotifications: false,
      })
      try {
        const tools = await client.listTools()
        const cancelledTool = tools.find((tool) => tool.name === 'cancelled_task')
        if (!cancelledTool) throw new Error('cancelled_task tool not found')

        await expect(client.callTool(cancelledTool, { reason: 'Cancelled by integration fixture' })).rejects.toEqual(
          expect.objectContaining({
            name: 'McpTaskCancelledError',
            statusMessage: 'Cancelled by integration fixture',
          })
        )
      } finally {
        await client.disconnect()
      }
    }, 30000)
  })

  describe('McpClient.callTool() with Non-Task Server (Backward Compatibility)', () => {
    it('extracts result from regular (non-task) tools', async () => {
      if (!nonTaskServerInfo) throw new Error('Non-task server not started')

      const client = createClient(nonTaskServerInfo.url, 'test-compat-client')
      try {
        await client.connect()
        const tools = await client.listTools()
        const echoTool = tools.find((t) => t.name === 'echo')
        expect(echoTool).toBeDefined()

        const result = await client.callTool(echoTool!, { message: 'backward compat test' })

        expect(result).toMatchObject({
          content: expect.arrayContaining([expect.objectContaining({ type: 'text', text: 'backward compat test' })]),
        })
      } finally {
        await client.disconnect()
      }
    }, 30000)

    it('handles calculator tool with complex arguments', async () => {
      if (!nonTaskServerInfo) throw new Error('Non-task server not started')

      const client = createClient(nonTaskServerInfo.url, 'test-compat-client')
      try {
        await client.connect()
        const tools = await client.listTools()
        const calculatorTool = tools.find((t) => t.name === 'calculator')
        expect(calculatorTool).toBeDefined()

        const result = await client.callTool(calculatorTool!, { operation: 'multiply', a: 6, b: 7 })

        expect(result).toMatchObject({
          content: expect.arrayContaining([expect.objectContaining({ type: 'text', text: 'Result: 42' })]),
        })
      } finally {
        await client.disconnect()
      }
    }, 30000)
  })

  describe('Agent Integration with Task Tools', () => {
    it('agent can use task tools in a conversation', async () => {
      if (!taskServerInfo) throw new Error('Task server not started')

      const client = createClient(taskServerInfo.url, 'test-agent-task-client', {})
      try {
        const model = bedrock.createModel({ maxTokens: 300 })
        const agent = new Agent({
          systemPrompt:
            'You are a helpful assistant. When asked to run a task, use the instant_task tool with the value provided by the user.',
          tools: [client],
          model,
        })

        const result = await agent.invoke('Please run an instant task with the value "agent test message"')

        expect(result).toBeDefined()
        expect(result.stopReason).toBeDefined()
        expect(hasToolUse(agent.messages, 'instant_task')).toBe(true)
        expect(countToolResults(agent.messages, 'success')).toBeGreaterThan(0)
      } finally {
        await client.disconnect()
      }
    }, 60000)

    it('agent handles task tool errors gracefully', async () => {
      if (!taskServerInfo) throw new Error('Task server not started')

      const client = createClient(taskServerInfo.url, 'test-agent-task-client', {})
      try {
        const model = bedrock.createModel({ maxTokens: 300 })
        const agent = new Agent({
          systemPrompt: 'You are a helpful assistant. When asked to test error handling, use the failing_task tool.',
          tools: [client],
          model,
        })

        const result = await agent.invoke('Please use the failing_task tool to test error handling.')

        expect(result).toBeDefined()
        expect(hasToolUse(agent.messages, 'failing_task')).toBe(true)
        expect(countToolResults(agent.messages, 'error')).toBeGreaterThan(0)
      } finally {
        await client.disconnect()
      }
    }, 60000)

    it('agent can use multiple task tools in a multi-turn conversation', async () => {
      if (!taskServerInfo) throw new Error('Task server not started')

      const client = createClient(taskServerInfo.url, 'test-agent-multi-task-client', {})
      try {
        const model = bedrock.createModel({ maxTokens: 300 })
        const agent = new Agent({
          systemPrompt:
            'You are a helpful assistant. Use task tools when requested. Available tools: instant_task (quick), long_running_task (takes time).',
          tools: [client],
          model,
        })

        // First turn: use instant_task
        await agent.invoke('Run an instant task with value "first turn"')
        expect(hasToolUse(agent.messages, 'instant_task')).toBe(true)

        // Second turn: use long_running_task
        await agent.invoke('Now run a long running task with message "second turn complete"')
        expect(hasToolUse(agent.messages, 'long_running_task')).toBe(true)

        // Both tool results should be successful
        expect(countToolResults(agent.messages, 'success')).toBeGreaterThanOrEqual(2)
      } finally {
        await client.disconnect()
      }
    }, 90000)
  })
})
