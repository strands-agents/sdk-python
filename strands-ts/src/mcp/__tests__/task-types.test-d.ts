import { describe, expectTypeOf, it } from 'vitest'

import { McpClient } from '../client.js'

import type { CallToolResult } from '@modelcontextprotocol/client'
import type { JSONValue } from '../../types/json.js'
import type { McpCallToolWithTaskResult, McpDirectCallToolResult } from '../task-types.js'

describe('MCP task public types', () => {
  it('preserves the source-compatible callTool return type', () => {
    expectTypeOf<McpClient['callTool']>().returns.resolves.toEqualTypeOf<JSONValue>()
  })

  it('keeps direct tool result extension fields available after narrowing', () => {
    expectTypeOf<McpDirectCallToolResult>().toMatchTypeOf<CallToolResult>()

    const result = undefined as unknown as McpCallToolWithTaskResult
    if (result.resultType !== 'task') {
      expectTypeOf(result['example.com/priority']).toEqualTypeOf<unknown>()
    }
  })
})
