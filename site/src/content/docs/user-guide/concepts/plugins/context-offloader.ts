import { Agent } from '@strands-agents/sdk'
import {
  ContextOffloader,
  InMemoryStorage,
  FileStorage,
  S3Storage,
} from '@strands-agents/sdk/vended-plugins/context-offloader'
import { bash } from '@strands-agents/sdk/vended-tools/bash'
import { fileEditor } from '@strands-agents/sdk/vended-tools/file-editor'

// =====================
// Disable Retrieval Tool
// =====================

{
  // --8<-- [start:disable_retrieval_tool]
  const agent = new Agent({
    tools: [bash, fileEditor],
    plugins: [
      new ContextOffloader({
        storage: new FileStorage('./artifacts'),
        includeRetrievalTool: false,
      }),
    ],
  })
  // --8<-- [end:disable_retrieval_tool]

  void agent
}

// =====================
// Getting Started
// =====================

{
  // --8<-- [start:getting_started]
  const agent = new Agent({
    plugins: [new ContextOffloader({ storage: new InMemoryStorage() })],
  })
  // --8<-- [end:getting_started]

  void agent
}

// =====================
// Custom Thresholds
// =====================

{
  // --8<-- [start:custom_thresholds]
  const agent = new Agent({
    plugins: [
      new ContextOffloader({
        storage: new InMemoryStorage(),
        maxResultTokens: 5_000,
        previewTokens: 2_000,
      }),
    ],
  })
  // --8<-- [end:custom_thresholds]

  void agent
}

// =====================
// In-Memory Storage
// =====================

{
  // --8<-- [start:in_memory_storage]
  const agent = new Agent({
    plugins: [
      new ContextOffloader({
        storage: new InMemoryStorage(),
      }),
    ],
  })
  // --8<-- [end:in_memory_storage]

  void agent
}

// =====================
// File Storage
// =====================

{
  // --8<-- [start:file_storage]
  const agent = new Agent({
    plugins: [
      new ContextOffloader({
        storage: new FileStorage('./artifacts'),
      }),
    ],
  })
  // --8<-- [end:file_storage]

  void agent
}

// =====================
// S3 Storage
// =====================

{
  // --8<-- [start:s3_storage]
  const agent = new Agent({
    plugins: [
      new ContextOffloader({
        storage: new S3Storage('my-agent-artifacts', {
          prefix: 'tool-results/',
        }),
      }),
    ],
  })
  // --8<-- [end:s3_storage]

  void agent
}
