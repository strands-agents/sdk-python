import type { Plugin } from '../../plugins/plugin.js'
import type { LocalAgent } from '../../types/agent.js'
import { InvokeModelStage } from '../../middleware/index.js'
import { createInjectionMiddleware } from '../../injection/message-injection.js'
import type { InjectionTrigger, InjectionContext } from '../../injection/types.js'

/** Configuration for the {@link ContextInjector} plugin. */
export interface ContextInjectorConfig {
  /**
   * Plugin name, for logging and duplicate detection. Defaults to `'strands:context-injector'`. Set a
   * distinct name when registering more than one injector so they can be told apart.
   */
  name?: string
  /**
   * When to inject. An {@link InjectionTrigger} name selects a built-in policy (`'userTurn'` —
   * default — or `'everyTurn'`); a predicate over the {@link InjectionContext} is the escape hatch. A
   * predicate that throws fails open (injection is skipped).
   *
   * @defaultValue 'userTurn'
   */
  trigger?: InjectionTrigger | ((context: InjectionContext) => boolean)
  /**
   * Renders the text to fold into the latest user message for this call, or `undefined`/`''` to skip.
   * Output is folded raw into the model input, so it is a stored-prompt-injection surface: escape any
   * attacker-influenced fields yourself (an `escapeXml` helper is exported alongside this plugin). A
   * callback that throws fails open (injection is skipped, the model call proceeds).
   */
  renderContent: (context: InjectionContext) => Promise<string | undefined>
  /**
   * Soft hint for the maximum tokens this provider's text should occupy. Reserved for shared-budget
   * coalescing across stacked injectors; not yet enforced.
   */
  maxTokens?: number
}

/**
 * Plugin that injects just-in-time context into the model input before each call.
 *
 * On a model call, the plugin asks {@link ContextInjectorConfig.renderContent} for text and folds it
 * into the most recent user message (ahead of the user's own content), gated by
 * {@link ContextInjectorConfig.trigger}. The injected text is ephemeral: it augments the model input
 * for that one call and never persists into the durable conversation or session.
 *
 * This is the public surface for the generic injection engine — `MemoryManager`'s `injection` config
 * is the same mechanism specialized to memory search.
 *
 * @example
 * ```typescript
 * import { Agent } from '@strands-agents/sdk'
 * import { ContextInjector } from '@strands-agents/sdk/vended-plugins/context-injector'
 *
 * const agent = new Agent({
 *   model,
 *   plugins: [new ContextInjector({ renderContent: async () => `<now>${new Date().toISOString()}</now>` })],
 * })
 * ```
 *
 * @remarks
 * **Stacking.** Multiple injectors may be registered. Each currently folds its own text into the user
 * message independently, in plugin-registration order, with no coalescing or shared budget — so N
 * injectors produce N prepends. A future revision will coalesce stacked injectors into one context
 * region and honor {@link ContextInjectorConfig.maxTokens} as a shared cap; until then, treat ordering
 * across injectors as registration order and keep individual outputs small.
 */
export class ContextInjector implements Plugin {
  readonly name: string

  private readonly _config: ContextInjectorConfig

  constructor(config: ContextInjectorConfig) {
    this.name = config.name ?? 'strands:context-injector'
    this._config = config
  }

  initAgent(agent: LocalAgent): void {
    const config = this._config
    agent.addMiddleware(
      InvokeModelStage.Input,
      createInjectionMiddleware({
        ...(config.trigger !== undefined && { trigger: config.trigger }),
        renderContent: config.renderContent,
      })
    )
  }
}
