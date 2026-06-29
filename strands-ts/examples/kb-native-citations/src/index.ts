import { Agent, BedrockModel, MemoryManager } from '@strands-agents/sdk'
import { BedrockKnowledgeBaseStore } from '@strands-agents/sdk/vended-memory-stores/bedrock-knowledge-base'

/**
 * Demonstrates using Bedrock Knowledge Bases with native citation support.
 *
 * When `citationDocumentBlocks` is set to `true`, the store provides a `retrieve_knowledge_base`
 * tool that surfaces retrieved passages as DocumentBlock objects with citations enabled.
 * This allows Bedrock Converse API to generate native citations attributing assertions
 * directly to specific chunk passages.
 */
async function main() {
  // Replace with your actual Bedrock Knowledge Base ID
  const knowledgeBaseId = process.env.KNOWLEDGE_BASE_ID ?? 'EXAMPLE_KB_ID'

  const model = new BedrockModel()

  // 1. Create the knowledge base store with citationDocumentBlocks enabled
  const store = new BedrockKnowledgeBaseStore({
    name: 'product_docs',
    description: 'Documentation and FAQs for our products',
    config: {
      knowledgeBaseId,
    },
    citationDocumentBlocks: true,
  })

  // 2. Wire it into MemoryManager
  // We disable the generic search_memory tool so the agent uses retrieve_knowledge_base directly
  const memoryManager = new MemoryManager({
    stores: [store],
    searchToolConfig: false,
  })

  // 3. Create the agent
  const agent = new Agent({
    model,
    systemPrompt: 'You are a helpful product support assistant. Use the knowledge base to answer questions accurately.',
    memoryManager,
  })

  console.log('=== Bedrock Knowledge Base Native Citations Example ===\n')
  console.log(`Knowledge Base ID: ${knowledgeBaseId}`)

  if (knowledgeBaseId === 'EXAMPLE_KB_ID') {
    console.log('\n[NOTE] Please set KNOWLEDGE_BASE_ID environment variable to run against a live knowledge base.\n')
    return
  }

  const prompt = 'What is the return policy for defective items?'
  console.log(`User: ${prompt}\n`)

  const result = await agent.invoke(prompt)

  console.log(`Assistant: ${result.toString()}\n`)
}

await main().catch(console.error)
