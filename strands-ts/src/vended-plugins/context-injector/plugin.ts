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
   * Renders the text to inject for this call, or `undefined`/`''` to skip. The text reaches the model
   * verbatim, so it is a prompt-injection surface: escape any attacker-influenced fields yourself. A
   * callback that throws fails open (injection is skipped, the model call proceeds).
   */
  renderContent: (context: InjectionContext) => Promise<string | undefined>
}

/**
 * Plugin that injects just-in-time context into the model input before each call.
 *
 * Before each model call, the plugin asks {@link ContextInjectorConfig.renderContent} for text and
 * makes it available to the model for that call, gated by {@link ContextInjectorConfig.trigger}. The
 * injected text is ephemeral: it augments the model input for that one call and never persists into the
 * durable conversation or session.
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
 * Multiple injectors may be registered; each contributes its text independently, in plugin-registration
 * order.
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
