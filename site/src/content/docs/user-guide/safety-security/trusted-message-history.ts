import { Agent } from '@strands-agents/sdk'
import type { MessageData } from '@strands-agents/sdk'

async function clearTrailingToolUseExample(untrustedMessages: MessageData[]) {
  // --8<-- [start:clear_trailing_tool_use]
  function clearTrailingToolUse(messages: MessageData[]): MessageData[] {
    if (messages.length === 0) {
      return messages
    }
    const last = messages[messages.length - 1]
    const content = last.content.filter((block) => !('toolUse' in block))
    messages[messages.length - 1] = { ...last, content }
    return messages
  }

  // untrustedMessages came from a request body, queue, or shared store.
  const agent = new Agent()
  await agent.invoke(clearTrailingToolUse(untrustedMessages))
  // --8<-- [end:clear_trailing_tool_use]
}
