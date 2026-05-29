// --8<-- [start:basic_agent]
import { Agent, BedrockModel } from '@strands-agents/sdk'

// Good: Start simple
const agent = new Agent()
await agent.invoke('Hello, world!')

// Then show configuration
const configuredAgent = new Agent({
  model: new BedrockModel({ modelId: 'us.anthropic.claude-sonnet-4-20250514' }),
  systemPrompt: 'You are a helpful assistant.',
})
await configuredAgent.invoke("What's the weather like?")
// --8<-- [end:basic_agent]
