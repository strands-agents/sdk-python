import type { InvokeModelContext } from '../../middleware/stages.js'
import type { MiddlewareInputHandler } from '../../middleware/types.js'
import type { Model } from '../../models/model.js'
import { Message, TextBlock } from '../../types/messages.js'
import { DEFAULT_CONTEXT_WINDOW_LIMIT } from '../conversation-manager.js'

export function createTokenUsageMiddleware(model: Model): MiddlewareInputHandler<InvokeModelContext> {
  return async (context: InvokeModelContext): Promise<InvokeModelContext> => {
    const projectedInputTokens = context.projectedInputTokens
    if (projectedInputTokens === undefined) {
      return context
    }

    const contextWindowLimit = model.getConfig().contextWindowLimit ?? DEFAULT_CONTEXT_WINDOW_LIMIT
    const remaining = Math.max(0, contextWindowLimit - projectedInputTokens)
    const pct = ((projectedInputTokens / contextWindowLimit) * 100).toFixed(1)

    const statusText =
      `\n\n<context-status>\n` +
      `<used>${projectedInputTokens.toLocaleString()} / ${contextWindowLimit.toLocaleString()} tokens (${pct}%)</used>\n` +
      `<remaining>~${remaining.toLocaleString()} tokens</remaining>\n` +
      `<tools>summarize_context, truncate_context, pin</tools>\n` +
      `</context-status>`

    const messages = [...context.messages]
    const lastMessage = messages[messages.length - 1]
    if (!lastMessage) {
      return context
    }

    messages[messages.length - 1] = new Message({
      role: lastMessage.role,
      content: [...lastMessage.content, new TextBlock(statusText)],
      ...(lastMessage.metadata && { metadata: lastMessage.metadata }),
    })

    return { ...context, messages }
  }
}
