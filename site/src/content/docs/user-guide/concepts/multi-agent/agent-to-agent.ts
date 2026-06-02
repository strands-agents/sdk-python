// @ts-nocheck
// NOTE: Type-checking is disabled because the examples reference remote services not available at build time.

import { Agent, tool } from '@strands-agents/sdk'
import { A2AAgent } from '@strands-agents/sdk/a2a'
import { A2AExpressServer } from '@strands-agents/sdk/a2a/express'
import { z } from 'zod'

async function basicUsageExample() {
  // --8<-- [start:basic_usage]
  // Create an A2AAgent pointing to a remote A2A server
  const a2aAgent = new A2AAgent({ url: 'http://localhost:9000' })

  // Invoke it just like a regular Agent
  const result = await a2aAgent.invoke('Show me 10 ^ 6')
  console.log(result.lastMessage.content)
  // --8<-- [end:basic_usage]
}

async function streamingExample() {
  // --8<-- [start:streaming]
  const remoteAgent = new A2AAgent({ url: 'http://localhost:9000' })

  // stream() yields A2AStreamUpdateEvent for each protocol event,
  // then an AgentResultEvent with the final result
  const stream = remoteAgent.stream('Explain quantum computing')
  let next = await stream.next()
  while (!next.done) {
    console.log(next.value)
    next = await stream.next()
  }
  // Final result
  console.log(next.value)
  // --8<-- [end:streaming]
}

async function asToolExample() {
  // --8<-- [start:as_tool]
  const calculatorAgent = new A2AAgent({
    url: 'http://calculator-service:9000',
  })

  const calculate = tool({
    name: 'calculate',
    description: 'Perform a mathematical calculation.',
    inputSchema: z.object({
      expression: z.string().describe('The math expression to evaluate'),
    }),
    callback: async (input) => {
      const calcResult = await calculatorAgent.invoke(input.expression)
      return String(calcResult.lastMessage.content[0])
    },
  })

  const orchestrator = new Agent({
    systemPrompt: 'You are a helpful assistant. Use the calculate tool for math.',
    tools: [calculate],
  })
  // --8<-- [end:as_tool]
}

async function basicServerExample() {
  // --8<-- [start:basic_server]
  const agent = new Agent({
    systemPrompt: 'You are a calculator agent that can perform basic arithmetic.',
  })

  // Create and start the A2A server
  const server = new A2AExpressServer({
    agent,
    name: 'Calculator Agent',
    description: 'A calculator agent that can perform basic arithmetic operations.',
  })

  await server.serve()
  // --8<-- [end:basic_server]
}

async function serverConfigExample() {
  const agent = new Agent({
    systemPrompt: 'You are a helpful agent.',
  })

  // --8<-- [start:server_config]
  const server = new A2AExpressServer({
    agent,
    name: 'My Agent',
    description: 'A helpful agent',
    host: '0.0.0.0',
    port: 8080,
    version: '1.0.0',
    httpUrl: 'https://my-agent.example.com', // Public URL override
    skills: [
      { id: 'math', name: 'Math', description: 'Performs calculations', tags: [] },
    ],
  })

  await server.serve()
  // --8<-- [end:server_config]
}

async function expressMiddlewareExample() {
  const agent = new Agent({
    systemPrompt: 'You are a helpful agent.',
  })

  // --8<-- [start:express_middleware]
  const express = (await import('express')).default

  const server = new A2AExpressServer({
    agent,
    name: 'My Agent',
    description: 'A customizable agent',
  })

  // Get the A2A middleware as an Express Router
  const a2aRouter = server.createMiddleware()

  // Create your own Express app with custom routes/middleware
  const app = express()
  app.get('/health', (_req, res) => {
    res.json({ status: 'ok' })
  })
  app.use(a2aRouter)

  app.listen(9000, '127.0.0.1', () => {
    console.log('Server listening on http://127.0.0.1:9000')
  })
  // --8<-- [end:express_middleware]
}

async function abortExample() {
  const agent = new Agent({
    systemPrompt: 'You are a helpful agent.',
  })

  // --8<-- [start:abort_signal]
  const server = new A2AExpressServer({ agent, name: 'My Agent' })

  const controller = new AbortController()
  await server.serve({ signal: controller.signal })

  // Later, to stop the server:
  controller.abort()
  // --8<-- [end:abort_signal]
}

void basicUsageExample()
void streamingExample()
void asToolExample()
void basicServerExample()
void serverConfigExample()
void expressMiddlewareExample()
void abortExample()
