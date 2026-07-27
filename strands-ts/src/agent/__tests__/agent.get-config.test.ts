import { describe, it, expect } from 'vitest'
import { Agent } from '../agent.js'
import type { AgentConfig } from '../agent.js'
import { TestModelProvider } from '../../__fixtures__/model-test-helpers.js'
import { createMockTool } from '../../__fixtures__/tool-helpers.js'

describe('Agent.getConfig', () => {
  it('returns the config the agent was constructed with', () => {
    const config: AgentConfig = {
      model: new TestModelProvider(),
      tools: [createMockTool('local_tool', () => 'ok')],
      name: 'tpl',
      description: 'the template',
      id: 'tpl-id',
      systemPrompt: 'be helpful',
      appState: { a: 1 },
      printer: false,
    }

    expect(new Agent(config).getConfig()).toBe(config)
  })

  it('returns an empty config when the agent was constructed without one', () => {
    expect(new Agent().getConfig()).toEqual({})
  })
})
