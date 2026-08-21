/**
 * Model routing primitives.
 *
 * A {@link ModelRouter} delegates every decision to its {@link RoutingStrategy}: once before the first
 * model call and again after an unclaimed failure. Returning `undefined` from the opening selection uses
 * the first candidate's concrete default model; returning it after a failure stops routing and preserves
 * the pending model error.
 *
 * A failure round uses each candidate at most once. A successful call starts a new round with that
 * candidate already counted as used. Nested routers are opaque candidates: they make one opening choice
 * with an empty attempt history and do not perform internal failover for the outer router.
 *
 * `agent.model` remains the first candidate's concrete model. Opening selection runs in model input
 * middleware after continuation input is incorporated, so the opening before-model hook reports that
 * default while {@link AfterModelCallEvent} reports the effective routed model. Subsystems
 * that inspect `agent.model`, including token estimation, also reason about the default model. Prefer
 * candidates with comparable context windows and tokenizers. If a candidate fails after emitting part
 * of a streamed response, consumers see that partial output followed by the replacement model's full
 * response.
 *
 * The API is provisional and may change before it is finalized.
 */
import { normalizeError } from '../../errors.js'
import { AfterInvocationEvent, AfterModelCallEvent, BeforeInvocationEvent } from '../../hooks/events.js'
import { HookOrder } from '../../hooks/types.js'
import { logger } from '../../logging/logger.js'
import { InvokeModelStage } from '../../middleware/stages.js'
import { Model } from '../model.js'
import { cloneSystemPrompt, type Message, type SystemPrompt } from '../../types/messages.js'
import { deepCopy } from '../../types/json.js'
import type { InvokeModelContext } from '../../middleware/stages.js'
import type { Plugin } from '../../plugins/plugin.js'
import type { ToolSpec } from '../../tools/types.js'
import type { InvocationState, LocalAgent } from '../../types/agent.js'
import { FallbackStrategy } from './fallback-strategy.js'
import type { CandidateInput, RoutingAttempt, RoutingContext, RoutingStrategy } from './strategy.js'

/** Construction data for a {@link RoutingCandidate}. */
export interface RoutingCandidateOptions {
  /** Concrete model or opaque nested router. */
  readonly model: Model | ModelRouter
  /** Optional strategy-facing name. */
  readonly name?: string
  /** Optional strategy-facing description. */
  readonly description?: string
}

/**
 * A model or opaque model group with an optional name and description.
 *
 * Base instances are frozen automatically. Subclasses must freeze themselves after initializing additional fields.
 */
export class RoutingCandidate {
  /** Concrete model or opaque nested router. */
  readonly model: Model | ModelRouter
  /** Optional strategy-facing name. */
  readonly name?: string
  /** Optional strategy-facing description. */
  readonly description?: string

  /**
   * Create an immutable routing candidate.
   *
   * @param options - Candidate model, name, and description
   */
  constructor(options: RoutingCandidateOptions) {
    this.model = options.model
    if (options.name !== undefined) this.name = options.name
    if (options.description !== undefined) this.description = options.description
    if (new.target === RoutingCandidate) Object.freeze(this)
  }
}

/** Options for constructing a {@link ModelRouter}. */
export interface ModelRouterOptions {
  /** Strategy responsible for every routing decision. */
  readonly strategy?: RoutingStrategy
  /** Maximum successful candidate switches per invocation. */
  readonly maxSwitches?: number
}

interface RoutingState {
  candidate: RoutingCandidate
  model: Model
  attempts: RoutingAttempt[]
  switchedTo: Set<RoutingCandidate>
  switches: number
}

const ROUTING_KEY_PREFIX = 'strands:modelRouting'
const OBJECT_IDS = new WeakMap<object, number>()
let nextObjectId = 1
/**
 * Routes each agent invocation among an immutable set of candidate models.
 *
 * The default {@link FallbackStrategy} prefers the candidate with the fewest recorded failures and
 * breaks ties by declaration order. `maxSwitches` bounds successful candidate changes per invocation.
 *
 * @example
 * ```typescript
 * const router = new ModelRouter([
 *   new RoutingCandidate({ model: primary, name: 'primary' }),
 *   new RoutingCandidate({ model: fallback, name: 'fallback' }),
 * ])
 * const agent = new Agent({ model: router })
 * ```
 */
export class ModelRouter implements Plugin {
  readonly name = 'strands:model-router'
  private readonly _candidates: readonly RoutingCandidate[]
  private readonly _strategy: RoutingStrategy
  private readonly _maxSwitches?: number
  private readonly _states = new WeakSet<RoutingState>()
  private readonly _attachedAgents = new WeakSet<LocalAgent>()
  private readonly _initializedAgents = new WeakSet<LocalAgent>()

  /**
   * Create a model router.
   *
   * @param models - Candidate models, nested routers, or candidate wrappers
   * @param options - Routing strategy and switch cap
   * @throws TypeError if models or the strategy are invalid
   * @throws Error if candidates are empty, duplicated, named alike, stateful, or `maxSwitches` is negative
   */
  constructor(models: readonly CandidateInput[], options: ModelRouterOptions = {}) {
    if (options.strategy !== undefined && !isRoutingStrategy(options.strategy)) {
      throw new TypeError('strategy must implement RoutingStrategy: a select(context) method')
    }
    if (options.maxSwitches !== undefined && options.maxSwitches < 0) {
      throw new Error('maxSwitches must be zero or greater')
    }

    const candidates = normalize(models)
    if (candidates.length === 0) throw new Error('ModelRouter requires at least one candidate model')
    rejectStateful(candidates)
    rejectDuplicates(candidates)

    this._candidates = Object.freeze(candidates)
    this._strategy = options.strategy ?? new FallbackStrategy()
    if (options.maxSwitches !== undefined) this._maxSwitches = options.maxSwitches
  }

  /** Normalized candidates in declaration order. */
  get candidates(): readonly RoutingCandidate[] {
    return this._candidates
  }

  /** First declared candidate resolved without consulting a strategy. */
  get defaultModel(): Model {
    const model = this._candidates[0]!.model
    return model instanceof ModelRouter ? model.defaultModel : model
  }

  /**
   * Attach this router as an agent's configured model.
   *
   * @internal
   * @param agent - Agent configured with this router
   */
  attachToAgent(agent: LocalAgent): void {
    this._attachedAgents.add(agent)
    this.initAgent(agent)
  }

  /**
   * Register routing middleware and lifecycle hooks.
   *
   * @param agent - Agent using this router as its model
   * @throws Error if attached as an ordinary plugin rather than as the model
   */
  initAgent(agent: LocalAgent): void {
    if (!this._attachedAgents.has(agent)) {
      throw new Error('ModelRouter must be passed through Agent({ model }), not plugins')
    }
    if (this._initializedAgents.has(agent)) return
    this._initializedAgents.add(agent)

    agent.addMiddleware(InvokeModelStage.Input, (context) => this._selectionMiddleware(context))
    agent.addHook(AfterModelCallEvent, (event) => this._onModelResult(event), { order: HookOrder.MODEL_ROUTING })
    agent.addHook(AfterInvocationEvent, (event) => this._clearState(event.agent, event.invocationState), {
      order: HookOrder.SDK_LAST,
    })
    agent.addHook(BeforeInvocationEvent, (event) => this._clearState(event.agent, event.invocationState), {
      order: HookOrder.SDK_FIRST,
    })
  }
  /**
   * Return the current routed model after hooks may have changed the selection.
   *
   * @internal
   * @param agent - Agent making the model call
   * @param invocationState - Live invocation state
   * @returns Current routed model, or `undefined` before selection
   */
  getRoutedModel(agent: LocalAgent, invocationState: InvocationState): Model | undefined {
    return this._getState(agent, invocationState)?.model
  }

  private async _selectionMiddleware(context: InvokeModelContext): Promise<InvokeModelContext> {
    let state = this._getState(context.agent, context.invocationState)
    if (state === undefined) {
      const routingContext = this._routingContext(
        [...context.messages],
        context.systemPrompt,
        [...context.toolSpecs],
        context.invocationState
      )
      state = await this._openAndCache(context.agent, context.invocationState, routingContext)
    }
    return { ...context, model: state.model } as InvokeModelContext
  }

  private async _openAndCache(
    agent: LocalAgent,
    invocationState: InvocationState,
    context: RoutingContext
  ): Promise<RoutingState> {
    const [candidate, model] = await this._open(context)
    const state: RoutingState = {
      candidate,
      model,
      attempts: [],
      switchedTo: new Set([candidate]),
      switches: 0,
    }
    this._states.add(state)
    invocationState[this._stateKey(agent)] = state
    return state
  }
  private async _open(context: RoutingContext): Promise<readonly [RoutingCandidate, Model]> {
    const candidate = await this._ask(context)
    if (candidate === undefined) {
      logger.info(`strategy=<${this._strategyName}> | strategy declined the opening choice, using the default model`)
      return [this._candidates[0]!, this.defaultModel]
    }

    logger.info(`strategy=<${this._strategyName}>, candidate=<${candidateLabel(candidate)}> | candidate selected`)
    return [candidate, await this._resolve(candidate, context)]
  }

  private async _ask(context: RoutingContext): Promise<RoutingCandidate | undefined> {
    return this._validated(await this._strategy.select(context), context)
  }

  private _validated(candidate: unknown, context: RoutingContext): RoutingCandidate | undefined {
    if (candidate === undefined) return undefined
    if (!(candidate instanceof RoutingCandidate)) {
      throw new TypeError(`strategy.select must return a RoutingCandidate or undefined; got ${typeName(candidate)}`)
    }
    if (!context.candidates.some((configured) => configured === candidate)) {
      throw new Error('strategy.select must return a candidate from context.candidates')
    }
    return candidate
  }

  private async _resolve(candidate: RoutingCandidate, context: RoutingContext): Promise<Model> {
    if (!(candidate.model instanceof ModelRouter)) return candidate.model
    return candidate.model._selectModel(
      candidate.model._routingContext(
        context.messages,
        context.systemPrompt,
        context.toolSpecs,
        context.invocationState,
        []
      )
    )
  }

  private async _selectModel(context: RoutingContext): Promise<Model> {
    const candidate = await this._ask(context)
    if (candidate === undefined) return this.defaultModel
    return this._resolve(candidate, context)
  }

  private get _strategyName(): string {
    return this._strategy.constructor.name
  }
  private async _onModelResult(event: AfterModelCallEvent): Promise<void> {
    const state = this._getState(event.agent, event.invocationState)
    if (state === undefined) return

    if (event.stopData !== undefined) {
      state.attempts.push(makeAttempt(state.candidate))
      state.switchedTo = new Set([state.candidate])
      return
    }
    if (event.retry || event.error === undefined) return

    state.attempts.push(makeAttempt(state.candidate, event.error))
    if (this._maxSwitches !== undefined && state.switches >= this._maxSwitches) {
      logger.warn(`maxSwitches=<${this._maxSwitches}> | switch cap reached, leaving the error to surface`)
      return
    }

    if (await this._advance(event, state)) event.retry = true
  }

  private async _advance(event: AfterModelCallEvent, state: RoutingState): Promise<boolean> {
    while (true) {
      const context = this._routingContext(
        event.agent.messages,
        event.agent.systemPrompt,
        event.agent.toolRegistry.list().map((tool) => tool.toolSpec),
        event.invocationState,
        state.attempts
      )

      let candidate: RoutingCandidate | undefined
      try {
        candidate = this._validated(await this._strategy.select(context), context)
      } catch (error) {
        logger.warn(
          `strategy=<${this._strategyName}>, error=<${String(error)}> | routing failed, leaving the error to surface`
        )
        return false
      }
      if (candidate === undefined) return false
      if (state.switchedTo.has(candidate)) {
        logger.info(`candidate=<${candidateLabel(candidate)}> | already used this round, leaving the error to surface`)
        return false
      }

      let model: Model
      try {
        model = await this._resolve(candidate, context)
      } catch (error) {
        const resolutionError = normalizeError(error)
        logger.warn(
          `candidate=<${candidateLabel(candidate)}>, error=<${resolutionError}> | candidate could not be resolved, asking again`
        )
        state.attempts.push(makeAttempt(candidate, resolutionError))
        state.switchedTo.add(candidate)
        continue
      }

      logger.info(
        `from_candidate=<${candidateLabel(state.candidate)}>, to_candidate=<${candidateLabel(candidate)}>, error=<${event.error?.constructor.name}> | model call failed, switching candidate`
      )
      state.candidate = candidate
      state.model = model
      state.switchedTo.add(candidate)
      state.switches += 1
      return true
    }
  }
  private _routingContext(
    messages: readonly Message[],
    systemPrompt: SystemPrompt | undefined,
    toolSpecs: readonly ToolSpec[],
    invocationState: InvocationState,
    attempts: readonly RoutingAttempt[] = []
  ): RoutingContext {
    const context: RoutingContext = {
      messages: messages.map((message) => message.clone()),
      ...(systemPrompt !== undefined && { systemPrompt: cloneSystemPrompt(systemPrompt) }),
      toolSpecs: deepCopy(toolSpecs) as unknown as ToolSpec[],
      candidates: this._candidates,
      invocationState,
      attempts: Object.freeze([...attempts]),
    }
    return Object.freeze(context)
  }

  private _clearState(agent: LocalAgent, invocationState: InvocationState): void {
    const key = this._stateKey(agent)
    if (this._getState(agent, invocationState) !== undefined) delete invocationState[key]
  }

  private _getState(agent: LocalAgent, invocationState: InvocationState): RoutingState | undefined {
    const value = invocationState[this._stateKey(agent)]
    return isObject(value) && this._states.has(value as RoutingState) ? (value as RoutingState) : undefined
  }

  private _stateKey(agent: LocalAgent): string {
    return `${ROUTING_KEY_PREFIX}:${objectId(agent).toString(16)}:${objectId(this).toString(16)}`
  }
}

function makeAttempt(candidate: RoutingCandidate, exception?: Error): RoutingAttempt {
  return Object.freeze({ candidate, ...(exception !== undefined && { exception }) })
}

function objectId(value: object): number {
  let identity = OBJECT_IDS.get(value)
  if (identity === undefined) {
    identity = nextObjectId
    nextObjectId += 1
    OBJECT_IDS.set(value, identity)
  }
  return identity
}
function normalize(models: unknown): RoutingCandidate[] {
  if (!Array.isArray(models)) throw new TypeError('models must be a sequence of candidates')
  return models.map(asCandidate)
}

function asCandidate(item: unknown): RoutingCandidate {
  const candidate = item instanceof RoutingCandidate ? item : new RoutingCandidate({ model: item as Model })
  if (!(candidate.model instanceof Model) && !(candidate.model instanceof ModelRouter)) {
    throw new TypeError(`candidate must be a Model or ModelRouter; got ${typeName(candidate.model)}`)
  }
  return candidate
}

function rejectStateful(candidates: readonly RoutingCandidate[]): void {
  for (const candidate of candidates) {
    if (candidate.model instanceof Model && candidate.model.stateful) {
      throw new Error(
        `candidate=<${candidateLabel(candidate)}> is stateful; routing among stateful models is not supported`
      )
    }
  }
}

function rejectDuplicates(candidates: readonly RoutingCandidate[]): void {
  const seenCandidates = new Set<RoutingCandidate>()
  const seenModels = new Set<Model>()
  const seenNames = new Set<string>()

  for (const candidate of candidates) {
    if (seenCandidates.has(candidate)) throw new Error('duplicate RoutingCandidate instance')
    seenCandidates.add(candidate)

    for (const model of reachableModels(candidate)) {
      if (seenModels.has(model)) {
        throw new Error(
          `candidate=<${candidateLabel(candidate)}> repeats a model already routed to; construct a separate instance so each candidate has its own health`
        )
      }
      seenModels.add(model)
    }

    if (candidate.name === undefined) continue
    if (seenNames.has(candidate.name)) throw new Error(`duplicate candidate name=<${candidate.name}>`)
    seenNames.add(candidate.name)
  }
}

function* reachableModels(candidate: RoutingCandidate): Generator<Model> {
  if (candidate.model instanceof ModelRouter) {
    for (const nested of candidate.model.candidates) yield* reachableModels(nested)
  } else {
    yield candidate.model
  }
}
function candidateLabel(candidate: RoutingCandidate): string {
  if (candidate.name !== undefined && candidate.name.length > 0) return candidate.name
  return candidate.model instanceof Model ? modelLabel(candidate.model) : candidate.model.constructor.name
}

function modelLabel(model: Model): string {
  const provider = model.constructor.name
  try {
    const modelId = model.getConfig()?.modelId
    return modelId ? `${provider}/${modelId}` : provider
  } catch {
    return provider
  }
}

function isRoutingStrategy(strategy: unknown): strategy is RoutingStrategy {
  return isObject(strategy) && 'select' in strategy && typeof strategy.select === 'function'
}

function isObject(value: unknown): value is object {
  return typeof value === 'object' && value !== null
}

function typeName(value: unknown): string {
  if (value === null) return 'null'
  if (value === undefined) return 'undefined'
  if (typeof value === 'object') return value.constructor?.name ?? 'object'
  return typeof value
}
