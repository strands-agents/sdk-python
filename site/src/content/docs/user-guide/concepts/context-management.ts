import {
  Agent,
  ContextManager,
  Offload,
  SlidingWindowConversationManager,
} from '@strands-agents/sdk'
import { LocalFileStorage } from '@strands-agents/sdk/storage'

async function basic() {
  // --8<-- [start:basic]
  const agent = new Agent({
    contextManager: 'auto',
  })
  // --8<-- [end:basic]
}

async function agentic() {
  // --8<-- [start:agentic]
  const agent = new Agent({
    contextManager: 'agentic',
  })
  // --8<-- [end:agentic]
}

async function customConversationManager() {
  // --8<-- [start:custom_conversation_manager]
  // Your conversation manager is used;
  // ContextOffloader is still added automatically
  const agent = new Agent({
    contextManager: 'auto',
    conversationManager:
      new SlidingWindowConversationManager({
        windowSize: 30,
      }),
  })
  // --8<-- [end:custom_conversation_manager]
}

async function customStrategies() {
  // --8<-- [start:custom_strategies]
  const agent = new Agent({
    contextManager: new ContextManager({
      strategies: [
        // Truncate tool results over 2500 tokens
        Offload.truncate('toolResults')
          .when({ threshold: 2500 }),

        // Summarize oldest messages at 85% utilization
        Offload.summarize('*')
          .when({ utilization: 0.85 }),
      ],
    }),
  })
  // --8<-- [end:custom_strategies]
}

async function stash() {
  // --8<-- [start:stash]
  const agent = new Agent({
    contextManager: new ContextManager({
      strategies: [
        Offload.truncate('toolResults')
          .when({ threshold: 2500 }),
      ],
      stash: {
        storage: new LocalFileStorage(
          '/tmp/agent-stash'
        ),
      },
    }),
  })
  // --8<-- [end:stash]
}

async function disabled() {
  // --8<-- [start:disabled]
  const agent = new Agent({
    contextManager: false,
  })
  // --8<-- [end:disabled]
}
