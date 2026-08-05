import { Agent, BedrockModel, MemoryManager, FunctionTool, DocumentBlock } from '@strands-agents/sdk'
import type { JSONValue } from '@strands-agents/sdk'
import { BedrockKnowledgeBaseStore } from '@strands-agents/sdk/vended-memory-stores/bedrock-knowledge-base'

/**
 * Demonstrates native Bedrock Converse API citations with Knowledge Bases.
 *
 * Rather than relying on SDK-level config, this example builds a custom `FunctionTool` that
 * retrieves passages from a Knowledge Base and surfaces them as `DocumentBlock` objects with
 * `citations: { enabled: true }`. When the model receives these blocks in a tool result,
 * Bedrock's Converse API generates passage-level attribution citations automatically.
 *
 * This is the recommended "user-land" pattern — no SDK changes required. You can use the
 * existing `MemoryStore.getTools()` extension point or build a `FunctionTool` directly as
 * shown here.
 */
async function main() {
  // Replace with your actual Bedrock Knowledge Base ID
  const knowledgeBaseId = process.env.KNOWLEDGE_BASE_ID ?? 'EXAMPLE_KB_ID'

  const model = new BedrockModel()

  // 1. Create the knowledge base store (plain, no citation config needed)
  const store = new BedrockKnowledgeBaseStore({
    name: 'product_docs',
    description: 'Documentation and FAQs for our products',
    config: { knowledgeBaseId },
  })

  // 2. Build a custom retrieve tool that returns DocumentBlocks with citations enabled.
  //    This bridges Bedrock KB retrieval with native Converse API citations:
  //    - retrieve passages via the store's search method
  //    - wrap each passage in a DocumentBlock with `citations: { enabled: true }`
  //    - Converse inspects these blocks and generates chunk-level citation references
  const retrieveTool = new FunctionTool({
    name: 'retrieve_knowledge_base',
    description:
      'Search the knowledge base and return cited document passages. ' +
      'Results include source citations so the model can attribute each fact to its origin.',
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'The search query to retrieve relevant passages for' },
        maxResults: {
          type: 'integer',
          minimum: 1,
          description: 'Maximum number of passages to retrieve. Defaults to the store default.',
        },
      },
      required: ['query'],
    },
    callback: async (rawInput: unknown): Promise<JSONValue> => {
      const input = rawInput as { query: string; maxResults?: number }
      const results = await store.search(input.query, {
        ...(input.maxResults !== undefined && { maxSearchResults: input.maxResults }),
      })

      if (results.length === 0) {
        return 'No passages found for this query.' as unknown as JSONValue
      }

      // Each retrieved passage becomes a DocumentBlock with citations enabled.
      // FunctionTool._wrapInToolResult recognises DocumentBlock[] and passes them through
      // as a multi-block ToolResultBlock, which Bedrock Converse uses for native citations.
      return results.map((entry, index) => {
        const sourceUri =
          typeof entry.metadata?._sourceLocation === 'object' &&
          entry.metadata._sourceLocation !== null &&
          's3Location' in (entry.metadata._sourceLocation as object)
            ? ((entry.metadata._sourceLocation as { s3Location?: { uri?: string } }).s3Location?.uri ?? '')
            : ''
        const docName = sourceUri || `passage-${index + 1}`

        return new DocumentBlock({
          name: docName,
          format: 'txt',
          source: { text: entry.content },
          citations: { enabled: true },
        })
      }) as unknown as JSONValue
    },
  })

  // 3. Wire the store into MemoryManager for injection / extraction if desired.
  //    Disable the built-in search_memory tool so the agent uses our custom retrieve tool instead.
  const memoryManager = new MemoryManager({
    stores: [store],
    searchToolConfig: false,
  })

  // 4. Create the agent with the custom retrieve tool passed via `tools`
  const agent = new Agent({
    model,
    systemPrompt:
      'You are a helpful product support assistant. Use the knowledge base to answer questions accurately.',
    memoryManager,
    tools: [retrieveTool],
  })

  console.log('=== Bedrock Knowledge Base Native Citations Example ===\n')
  console.log(`Knowledge Base ID: ${knowledgeBaseId}`)

  if (knowledgeBaseId === 'EXAMPLE_KB_ID') {
    console.log(
      '\n[NOTE] Please set KNOWLEDGE_BASE_ID environment variable to run against a live knowledge base.\n'
    )
    return
  }

  const prompt = 'What is the return policy for defective items?'
  console.log(`User: ${prompt}\n`)

  const result = await agent.invoke(prompt)

  console.log(`Assistant: ${result.toString()}\n`)
}

await main().catch(console.error)
