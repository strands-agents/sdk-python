import { Message, TextBlock } from '../types/messages.js'
import type { MessageData } from '../types/messages.js'
import { logger } from '../logging/logger.js'
import { normalizeError } from '../errors.js'
import type { InvokeModelContext } from '../middleware/index.js'
import type { InjectionMiddlewareOptions, InjectionTrigger } from './types.js'

/**
 * Whether the latest message is a fresh user ask: a `user` message carrying no tool result. This is
 * the `'userTurn'` policy — it distinguishes a new chat ask from an autonomous tool-result turn.
 *
 * @param messages - The current conversation, as data
 * @returns `true` when the latest message is a plain user ask, otherwise `false`
 */
export function isUserTurn(messages: MessageData[]): boolean {
  const last = messages[messages.length - 1]
  return !!last && last.role === 'user' && !last.content.some((block) => 'toolResult' in block)
}

/**
 * Resolves an {@link InjectionTrigger} name or predicate into a single gate predicate.
 *
 * `'userTurn'` maps to {@link isUserTurn}; `'everyTurn'` to an always-true gate; a user-supplied
 * predicate is wrapped so that a throw fails open (logs and skips injection rather than aborting the
 * model call).
 *
 * @param trigger - An {@link InjectionTrigger} name, a predicate, or `undefined` (defaults to `'userTurn'`)
 * @returns A predicate that, given the current messages, returns whether to inject this call
 */
export function resolveTrigger(
  trigger: InjectionTrigger | ((messages: MessageData[]) => boolean) | undefined
): (messages: MessageData[]) => boolean {
  if (trigger === undefined || trigger === 'userTurn') {
    return isUserTurn
  }
  if (trigger === 'everyTurn') {
    return () => true
  }
  const predicate = trigger
  return (messages) => {
    try {
      return predicate(messages)
    } catch (error) {
      logger.warn(`reason=<${normalizeError(error).message}> | injection trigger threw; skipping injection`)
      return false
    }
  }
}

/**
 * Folds `text` into the most recent `user` message as a leading {@link TextBlock}, ahead of that
 * message's own content (block-prepend), returning a NEW array. Other messages are returned as-is.
 *
 * Prepending into the existing user message (rather than inserting a standalone message) keeps role
 * alternation valid in both chat and the autonomous tool loop, and keeps the user's own ask in the
 * recency slot — the last thing the model reads. {@link Message} fields are readonly, so the target is
 * rebuilt as a new {@link Message}. When there is no `user` message, the input array is returned
 * unchanged.
 *
 * @param messages - The conversation to fold into
 * @param text - The text to prepend to the most recent user message
 * @returns A new array with the folded message, or the input array when there is no user message
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
  const folded = new Message({
    role: target.role,
    content: [new TextBlock(text), ...target.content],
    ...(target.metadata !== undefined && { metadata: target.metadata }),
  })

  const result = [...messages]
  result[targetIndex] = folded
  return result
}

/**
 * Builds an {@link InvokeModelStage} `Input` handler that folds {@link InjectionMiddlewareOptions.provide}'s
 * text into the latest user message, ephemerally — the model sees the augmented input for this one call
 * while the agent's durable history is never touched.
 *
 * Runs as an input-phase transformer (`(ctx) => ctx`): it gates on the resolved trigger, asks `provide`
 * for the text, and returns a context with the folded messages. Anything that skips — the trigger not
 * firing, `provide` returning empty, or any callback throwing — returns the context unchanged so the
 * model call proceeds (fail open). The injected text never enters durable history because the input
 * phase only rewrites the per-call context, not the agent's stored messages.
 *
 * @param opts - The trigger and `provide` callback the handler uses
 * @returns An `InvokeModelStage.Input` handler that returns a (possibly) folded context
 */
export function createInjectionMiddleware(
  opts: InjectionMiddlewareOptions
): (context: InvokeModelContext) => Promise<InvokeModelContext> {
  const trigger = resolveTrigger(opts.trigger)
  return async (context) => {
    const messages = context.messages.map((message) => message.toJSON())
    if (!trigger(messages)) {
      return context
    }

    let text: string | undefined
    try {
      text = await opts.provide(messages)
    } catch (error) {
      logger.warn(`reason=<${normalizeError(error).message}> | injection provide threw; skipping injection`)
      return context
    }
    if (!text?.trim()) {
      return context
    }

    return { ...context, messages: foldIntoLastUserMessage([...context.messages], text) }
  }
}
