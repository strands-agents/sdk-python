import { Agent, SlidingWindowConversationManager } from '@strands-agents/sdk'

async function basic() {
  // --8<-- [start:basic]
  const agent = new Agent({
    contextManager: 'auto',
  })
  // --8<-- [end:basic]
}

async function customCm() {
  // --8<-- [start:custom_cm]
  // Your conversation manager is used;
  // ContextOffloader is still added automatically
  const agent = new Agent({
    contextManager: 'auto',
    conversationManager: new SlidingWindowConversationManager({
      windowSize: 30,
    }),
  })
  // --8<-- [end:custom_cm]
}
