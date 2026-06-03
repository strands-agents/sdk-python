import { Agent, McpClient } from '@strands-agents/sdk'
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js'

async function mcpExample() {
  // --8<-- [start:mcp_strands]
  const mcpClient = new McpClient({
    transport: new StdioClientTransport({
      command: 'uvx',
      args: ['strands-agents-mcp-server'],
    }),
  })

  const agent = new Agent({ tools: [mcpClient] })
  await agent.invoke('How do I create a custom tool in Strands Agents?')

  await mcpClient.disconnect()
  // --8<-- [end:mcp_strands]
}
