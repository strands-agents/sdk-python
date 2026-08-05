# Bedrock Knowledge Base Native Citations Example

Demonstrates how to use `BedrockKnowledgeBaseStore` with native Bedrock Converse API citations using a user-land `FunctionTool`.

## Overview

By default, knowledge base retrieval returns plain text snippets via the built-in `search_memory` tool. To enable native passage-level citations, this example builds a custom `FunctionTool` that:

1. Retrieves passages from the knowledge base via `store.search()`
2. Wraps each passage in a `DocumentBlock` with `citations: { enabled: true }`
3. Returns the blocks as the tool result

When Bedrock's Converse API sees `DocumentBlock`s in a tool result, it generates accurate, chunk-level citation references in its response — attributing each assertion to its source passage.

## Prerequisites

- Node.js 20+
- AWS credentials configured with access to Amazon Bedrock
- An existing Bedrock Knowledge Base

## Running the Example

```bash
# Install dependencies
npm install

# Run the example with your Knowledge Base ID
KNOWLEDGE_BASE_ID=your-kb-id npm start
```
