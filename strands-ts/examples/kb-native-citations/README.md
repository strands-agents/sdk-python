# Bedrock Knowledge Base Native Citations Example

Demonstrates how to configure and use `BedrockKnowledgeBaseStore` with native Bedrock Converse API citations.

## Overview

By default, knowledge base retrieval returns plain text snippets. By enabling `citationDocumentBlocks: true`, the SDK exposes a dedicated `retrieve_knowledge_base` tool to the agent. When called, it surfaces chunks as `DocumentBlock` objects with citations enabled (`{ citations: { enabled: true } }`), allowing Bedrock models to generate accurate, passage-level citations in their responses.

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
