import { describe, expect, it } from 'vitest'

import {
  McpInputResponsesSchema,
  McpTaskStatusNotificationSchema,
  McpTaskStatusNotificationParamsSchema,
  parseMcpCancelTaskResult,
  parseMcpCreateTaskResult,
  parseMcpGetTaskResult,
  parseMcpTaskStatusNotificationParams,
  parseMcpUpdateTaskResult,
} from '../task-schemas.js'

const TASK_BASE = {
  taskId: 'task-123',
  createdAt: '2026-08-04T12:00:00Z',
  lastUpdatedAt: '2026-08-04T12:01:00Z',
  ttlMs: 60_000,
  pollIntervalMs: 1_000,
} as const

const CALL_TOOL_RESULT = {
  content: [{ type: 'text', text: 'finished' }],
  structuredContent: { answer: 42 },
  isError: false,
} as const

const INPUT_REQUESTS = {
  sampling: {
    method: 'sampling/createMessage',
    params: {
      messages: [{ role: 'user', content: { type: 'text', text: 'Summarize this task' } }],
      maxTokens: 128,
    },
  },
  roots: {
    method: 'roots/list',
  },
  elicitation: {
    method: 'elicitation/create',
    params: {
      message: 'Choose a format',
      requestedSchema: {
        type: 'object',
        properties: {
          format: { type: 'string' },
        },
        required: ['format'],
      },
    },
  },
} as const

const INPUT_RESPONSES = {
  sampling: {
    model: 'test-model',
    role: 'assistant',
    content: { type: 'text', text: 'Task summary' },
    stopReason: 'endTurn',
  },
  roots: {
    roots: [{ uri: 'file:///workspace', name: 'workspace' }],
  },
  elicitation: {
    action: 'accept',
    content: { format: 'json' },
  },
} as const

describe('MCP task schemas', () => {
  describe('parseMcpCreateTaskResult', () => {
    it('parses a task handle and preserves result extensions', () => {
      const result = {
        ...TASK_BASE,
        resultType: 'task',
        status: 'working',
        statusMessage: 'Processing',
        _meta: { traceId: 'trace-123' },
        'example.com/priority': 'high',
      }

      expect(parseMcpCreateTaskResult(result)).toEqual(result)
    })

    it('rejects a non-task result discriminator', () => {
      expect(() =>
        parseMcpCreateTaskResult({
          ...TASK_BASE,
          resultType: 'complete',
          status: 'working',
        })
      ).toThrow()
    })

    it.each([
      { name: 'empty task identifier', override: { taskId: '' } },
      { name: 'invalid creation timestamp', override: { createdAt: 'not-a-timestamp' } },
      { name: 'invalid update timestamp', override: { lastUpdatedAt: '2026-02-30T12:00:00Z' } },
      { name: 'negative ttl', override: { ttlMs: -1 } },
      { name: 'fractional ttl', override: { ttlMs: 1.5 } },
      { name: 'infinite ttl', override: { ttlMs: Number.POSITIVE_INFINITY } },
      { name: 'unsafe ttl', override: { ttlMs: Number.MAX_SAFE_INTEGER + 1 } },
      { name: 'negative poll interval', override: { pollIntervalMs: -1 } },
      { name: 'fractional poll interval', override: { pollIntervalMs: 1.5 } },
      {
        name: 'update timestamp before creation',
        override: {
          createdAt: '2026-08-04T12:01:00Z',
          lastUpdatedAt: '2026-08-04T12:00:00Z',
        },
      },
    ])('rejects task metadata with $name', ({ override }) => {
      expect(() =>
        parseMcpCreateTaskResult({
          ...TASK_BASE,
          ...override,
          resultType: 'task',
          status: 'working',
        })
      ).toThrow()
    })

    it('accepts unlimited retention and a zero poll interval', () => {
      const result = {
        ...TASK_BASE,
        resultType: 'task',
        status: 'working',
        ttlMs: null,
        pollIntervalMs: 0,
      }

      expect(parseMcpCreateTaskResult(result)).toEqual(result)
    })
  })

  describe('parseMcpGetTaskResult', () => {
    it.each([
      {
        name: 'working',
        result: {
          ...TASK_BASE,
          resultType: 'complete',
          status: 'working',
          statusMessage: 'Processing',
        },
      },
      {
        name: 'input_required',
        result: {
          ...TASK_BASE,
          resultType: 'complete',
          status: 'input_required',
          inputRequests: INPUT_REQUESTS,
        },
      },
      {
        name: 'completed',
        result: {
          ...TASK_BASE,
          resultType: 'complete',
          status: 'completed',
          result: CALL_TOOL_RESULT,
        },
      },
      {
        name: 'failed',
        result: {
          ...TASK_BASE,
          resultType: 'complete',
          status: 'failed',
          error: {
            code: -32_603,
            message: 'Tool execution failed',
            data: { retryable: false, requestId: 'request-123' },
          },
        },
      },
      {
        name: 'cancelled',
        result: {
          ...TASK_BASE,
          resultType: 'complete',
          status: 'cancelled',
          statusMessage: 'Cancelled by caller',
        },
      },
    ])('parses the $name status shape', ({ result }) => {
      expect(parseMcpGetTaskResult(result)).toEqual(result)
    })

    it('rejects a non-complete result discriminator', () => {
      expect(() =>
        parseMcpGetTaskResult({
          ...TASK_BASE,
          resultType: 'task',
          status: 'working',
        })
      ).toThrow()
    })

    it.each([
      { status: 'input_required', payload: {} },
      { status: 'completed', payload: {} },
      { status: 'failed', payload: {} },
    ])('rejects $status without its required payload', ({ status, payload }) => {
      expect(() =>
        parseMcpGetTaskResult({
          ...TASK_BASE,
          ...payload,
          resultType: 'complete',
          status,
        })
      ).toThrow()
    })

    it.each([
      {
        name: 'completed task with an error',
        result: {
          ...TASK_BASE,
          resultType: 'complete',
          status: 'completed',
          result: CALL_TOOL_RESULT,
          error: { code: -32_603, message: 'Contradiction' },
        },
      },
      {
        name: 'failed task with a result',
        result: {
          ...TASK_BASE,
          resultType: 'complete',
          status: 'failed',
          error: { code: -32_603, message: 'Failed' },
          result: CALL_TOOL_RESULT,
        },
      },
      {
        name: 'cancelled task with pending input',
        result: {
          ...TASK_BASE,
          resultType: 'complete',
          status: 'cancelled',
          inputRequests: INPUT_REQUESTS,
        },
      },
    ])('rejects a $name', ({ result }) => {
      expect(() => parseMcpGetTaskResult(result)).toThrow()
    })

    it('rejects an invalid nested CallToolResult', () => {
      expect(() =>
        parseMcpGetTaskResult({
          ...TASK_BASE,
          resultType: 'complete',
          status: 'completed',
          result: {
            content: [{ type: 'text' }],
          },
        })
      ).toThrow()
    })
  })

  describe('McpInputResponsesSchema', () => {
    it('parses sampling, roots, and elicitation responses', () => {
      expect(McpInputResponsesSchema.parse(INPUT_RESPONSES)).toEqual(INPUT_RESPONSES)
    })
  })

  describe('task acknowledgement schemas', () => {
    it.each([
      { operation: 'update', parse: parseMcpUpdateTaskResult },
      { operation: 'cancel', parse: parseMcpCancelTaskResult },
    ])('parses an empty $operation acknowledgement', ({ parse }) => {
      const result = {
        resultType: 'complete',
        _meta: { requestId: 'request-123' },
      }

      expect(parse(result)).toEqual(result)
    })

    it.each([
      { operation: 'update', parse: parseMcpUpdateTaskResult },
      { operation: 'cancel', parse: parseMcpCancelTaskResult },
    ])('rejects task state in a $operation acknowledgement', ({ parse }) => {
      expect(() =>
        parse({
          ...TASK_BASE,
          resultType: 'complete',
          status: 'working',
        })
      ).toThrow()
    })

    it.each([
      { operation: 'update', parse: parseMcpUpdateTaskResult },
      { operation: 'cancel', parse: parseMcpCancelTaskResult },
    ])('rejects a non-complete $operation acknowledgement', ({ parse }) => {
      expect(() => parse({ resultType: 'task' })).toThrow()
    })
  })

  describe('task notification schemas', () => {
    it('parses complete detailed task parameters', () => {
      const params = {
        ...TASK_BASE,
        status: 'input_required',
        inputRequests: INPUT_REQUESTS,
        _meta: { subscriptionId: 'subscription-123' },
      }

      expect(McpTaskStatusNotificationParamsSchema.parse(params)).toEqual(params)
      expect(parseMcpTaskStatusNotificationParams(params)).toEqual({
        ...params,
        resultType: 'complete',
      })
    })

    it('parses a complete JSON-RPC task notification', () => {
      const notification = {
        jsonrpc: '2.0',
        method: 'notifications/tasks',
        params: {
          ...TASK_BASE,
          status: 'completed',
          result: CALL_TOOL_RESULT,
        },
      }

      expect(McpTaskStatusNotificationSchema.parse(notification)).toEqual(notification)
    })

    it('rejects contradictory notification task state', () => {
      expect(() =>
        parseMcpTaskStatusNotificationParams({
          ...TASK_BASE,
          status: 'failed',
          error: { code: -32_603, message: 'Failed' },
          result: CALL_TOOL_RESULT,
        })
      ).toThrow()
    })
  })
})
