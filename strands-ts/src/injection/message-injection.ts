import { Message, TextBlock } from '../types/messages.js'
import type { MessageData } from '../types/messages.js'
import { logger } from '../logging/logger.js'
import { normalizeError } from '../errors.js'
import type { InvokeModelContext } from '../middleware/index.js'
import type { InjectionMiddlewareOptions, InjectionTrigger, InjectionContext } from './types.js'

/**
 * Builds an `InvokeModelStage` `Input` handler that folds {@link InjectionMiddlewareOptions.renderContent}'s
 * text into the latest user message, ephemerally — the model sees the augmented input for this one call
 * while the agent's durable history is never touched.
 *
 * Runs as an input-phase transformer (`(ctx) => ctx`): it gates on the resolved trigger, asks
 * `renderContent` for the text, and returns a context with the folded messages. Anything that skips —
 * the trigger not firing, `renderContent` returning empty, or any callback throwing — returns the
 * context unchanged so the model call proceeds (fail open). The injected text never enters durable
 * history because the input phase only rewrites the per-call context, not the agent's stored messages.
 *
 * @param opts - The trigger and `renderContent` callback the handler uses
 * @returns An `InvokeModelStage.Input` handler that returns a (possibly) folded context
 * @internal Delivery primitive. Reach injection through `ContextInjector` or `MemoryManager`.
 */
export function createInjectionMiddleware(
  opts: InjectionMiddlewareOptions
): (context: InvokeModelContext) => Promise<InvokeModelContext> {
  const trigger = resolveTrigger(opts.trigger)
  return async (context) => {
    const agent = context.agent
    const injectionContext: InjectionContext = {
      messages: context.messages.map((message) => message.toJSON()),
      appState: agent.appState,
      agent,
    }
    if (!trigger(injectionContext)) {
      return context
    }

    let text: string | undefined
    try {
      text = await opts.renderContent(injectionContext)
    } catch (error) {
      logger.warn(`reason=<${normalizeError(error).message}> | injection renderContent threw; skipping injection`)
      return context
    }
    if (!text?.trim()) {
      return context
    }

    return { ...context, messages: foldIntoLastUserMessage([...context.messages], text) }
  }
}

/**
 * Resolves an {@link InjectionTrigger} name or predicate into a single gate predicate over the
 * {@link InjectionContext}.
 *
 * `'userTurn'` maps to {@link isUserTurn} (over `ctx.messages`); `'everyTurn'` to an always-true gate;
 * a user-supplied predicate is wrapped so that a throw fails open (logs and skips injection rather than
 * aborting the model call).
 *
 * @param trigger - An {@link InjectionTrigger} name, a predicate, or `undefined` (defaults to `'userTurn'`)
 * @returns A predicate that, given the {@link InjectionContext}, returns whether to inject this call
 * @internal Delivery primitive. Reach injection through `ContextInjector` or `MemoryManager`.
 */
export function resolveTrigger(
  trigger: InjectionTrigger | ((context: InjectionContext) => boolean) | undefined
): (context: InjectionContext) => boolean {
  if (trigger === undefined || trigger === 'userTurn') {
    return (context) => isUserTurn(context.messages)
  }
  if (trigger === 'everyTurn') {
    return () => true
  }
  const predicate = trigger
  return (context) => {
    try {
      return predicate(context)
    } catch (error) {
      logger.warn(`reason=<${normalizeError(error).message}> | injection trigger threw; skipping injection`)
      return false
    }
  }
}

/**
 * Whether the latest message is a fresh user ask: a `user` message carrying no tool result. This is
 * the `'userTurn'` policy — it distinguishes a new chat ask from an autonomous tool-result turn.
 *
 * @param messages - The current conversation, as data
 * @returns `true` when the latest message is a plain user ask, otherwise `false`
 * @internal Delivery primitive. Reach injection through `ContextInjector` or `MemoryManager`.
 */
export function isUserTurn(messages: MessageData[]): boolean {
  const last = messages[messages.length - 1]
  return !!last && last.role === 'user' && !last.content.some((block) => 'toolResult' in block)
}

/**
 * Folds `text` into the most recent `user` message as a {@link TextBlock}, returning a NEW array. Other
 * messages are returned as-is.
 *
 * Folding into the existing user message (rather than inserting a standalone message) keeps role
 * alternation valid in both chat and the autonomous tool loop. The block is placed to keep the message
 * valid for the model:
 * - A plain user ask: the text is **prepended**, leaving the user's own ask in the recency slot — the
 *   last thing the model reads.
 * - A tool-result turn (the message carries a `ToolResultBlock`): the text is **appended**,
 *   because providers require the tool result to be the first content block in the turn that answers a
 *   tool use.
 *
 * {@link Message} fields are readonly, so the target is rebuilt as a new {@link Message}. When there is
 * no `user` message, the input array is returned unchanged.
 *
 * @param messages - The conversation to fold into
 * @param text - The text to fold into the most recent user message
 * @returns A new array with the folded message, or the input array when there is no user message
 * @internal Delivery primitive. Reach injection through `ContextInjector` or `MemoryManager`.
 */
export function foldIntoLastUserMessage(messages: Message[], text: string): Message[] {
  let targetIndex = -1
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]!.role === 'user') {
      targetIndex = i
      break
    }
  }
  if (targetIndex < 0) {
    return messages
  }

  const target = messages[targetIndex]!
  const injected = new TextBlock(text)
  // A tool result must stay the first block in the turn that answers a tool use, so append rather than
  // prepend when the target carries one.
  const hasToolResult = target.content.some((block) => block.type === 'toolResultBlock')
  const content = hasToolResult ? [...target.content, injected] : [injected, ...target.content]
  const folded = new Message({
    role: target.role,
    content,
    ...(target.metadata !== undefined && { metadata: target.metadata }),
  })

  const result = [...messages]
  result[targetIndex] = folded
  return result
}
