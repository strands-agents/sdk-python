import type { MessageData } from '../types/messages.js'

/**
 * Determines when injection runs before a model call.
 *
 * - `'userTurn'`: only when the latest message is a fresh user ask (a `user` message with no tool
 *   result) — the common case for chat agents, where it keeps the user's ask the final message the
 *   model sees.
 * - `'everyTurn'`: before every model call, including mid-task tool-result turns — for autonomous
 *   agents that should consult injected context at each step.
 *
 * For finer control, pass a predicate as {@link InjectionConfig.trigger} instead.
 */
export type InjectionTrigger = 'userTurn' | 'everyTurn'

/**
 * Generic injection configuration shared by every injection consumer.
 *
 * Only the trigger is generic here. What text to inject (the query, result count, and rendering) is a
 * consumer concern — a memory consumer derives it from a search, while a fixed-text consumer (a
 * reminder, the date) needs none of that. Consumers extend this interface with their own knobs.
 */
export interface InjectionConfig {
  /**
   * When injection runs. An {@link InjectionTrigger} name selects a built-in policy; a predicate is
   * the escape hatch — it receives the current messages and returns whether to inject this call. A
   * predicate that throws fails open (injection is skipped, the model call proceeds).
   *
   * @defaultValue 'userTurn'
   */
  trigger?: InjectionTrigger | ((messages: MessageData[]) => boolean)
}

/**
 * Options for {@link createInjectionMiddleware}.
 *
 * The engine is text-in: it knows nothing about queries, search, or rendering. A consumer supplies a
 * single {@link InjectionMiddlewareOptions.provide} callback that returns the text to fold into the
 * conversation, and (optionally) a trigger that gates when to do so.
 */
export interface InjectionMiddlewareOptions {
  /**
   * When to inject. See {@link InjectionConfig.trigger}. Defaults to `'userTurn'`.
   */
  trigger?: InjectionTrigger | ((messages: MessageData[]) => boolean)
  /**
   * Returns the text to fold into the latest user message, or `undefined`/`''` to skip this call. A
   * callback that throws fails open (injection is skipped, the model call proceeds).
   */
  provide: (messages: MessageData[]) => Promise<string | undefined>
}
