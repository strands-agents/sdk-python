import { InterventionHandler } from '../../interventions/handler.js'
import { proceed, deny } from '../../interventions/actions.js'
import type { InterventionAction } from '../../interventions/actions.js'
import type { BeforeToolCallEvent } from '../../hooks/events.js'
import type { OnError } from '../../interventions/handler.js'
import {
  isAuthorized,
  checkParsePolicySet,
  validate,
  type Entities,
  type CedarValueJson,
} from '@cedar-policy/cedar-wasm/nodejs'
import { readFileSync, existsSync } from 'node:fs'

/**
 * A {@link https://docs.cedarpolicy.com/syntax-entity.html | Cedar entity} identifier
 * consisting of a type and id.
 *
 * @example
 * ```typescript
 * const principal: CedarEntityUid = { type: 'User', id: 'alice@acme.com' }
 * const resource: CedarEntityUid = { type: 'Record', id: '42' }
 * ```
 */
export interface CedarEntityUid {
  type: string
  id: string
}

/**
 * A {@link https://docs.cedarpolicy.com/syntax-entity.html | Cedar entity} with
 * attributes and parent relationships.
 *
 * @example
 * ```typescript
 * const entity: CedarEntity = {
 *   uid: { type: 'User', id: 'alice' },
 *   attrs: { role: 'admin', department: 'engineering' },
 *   parents: [{ type: 'Role', id: 'admin' }],
 * }
 * ```
 */
export interface CedarEntity {
  uid: CedarEntityUid
  attrs: Record<string, CedarValueJson>
  parents: CedarEntityUid[]
}

/**
 * Maps tool names to Cedar resources. Either a static record mapping tool names
 * to entity type/key pairs, or a function for dynamic resolution.
 *
 * @example
 * ```typescript
 * // Static: tool "delete_record" resolves resource from input.record_id
 * const resolver: ResourceResolver = {
 *   delete_record: { key: 'record_id', type: 'Record' },
 * }
 *
 * // Dynamic: custom logic per tool
 * const resolver: ResourceResolver = (toolName, input) => {
 *   return { type: 'Record', id: String(input.id) }
 * }
 * ```
 */
export type ResourceResolver =
  | Record<string, { key: string; type: string }>
  | ((toolName: string, toolInput: Record<string, CedarValueJson>) => CedarEntityUid)

/**
 * Configuration for the {@link CedarAuthorization} intervention handler.
 *
 * @see {@link https://docs.cedarpolicy.com/syntax-policy.html | Cedar policy syntax}
 */
export interface CedarAuthorizationConfig {
  /**
   * Cedar policy text, or a path to a `.cedar` file on disk.
   *
   * @example
   * ```typescript
   * // Inline policy
   * { policies: 'permit(principal, action == Action::"search", resource);' }
   *
   * // File path
   * { policies: './policies/agent.cedar' }
   * ```
   */
  policies: string

  /**
   * Entity data as an array, or a path to a `.json` file on disk.
   * Entities define the principal/resource hierarchy for Cedar evaluation.
   */
  entities?: CedarEntity[] | string

  /**
   * Cedar schema text, or a path to a `.cedarschema` file on disk.
   * When provided, policies are validated against the schema at construction
   * time — catching type errors, unknown attributes, and invalid action names
   * before any tool call happens.
   *
   * Generate a schema from your tool definitions using the
   * {@link https://github.com/cedar-policy/cedar-for-agents | cedar-for-agents}
   * schema generator.
   *
   * TODO: Auto-generate schema from agent tool definitions via
   * `@cedar-policy/mcp-schema-generator-wasm` once published to npm.
   * This would enable type-checking context.input field access
   * (catching typos and type mismatches in policies at construction time).
   */
  schema?: string

  /**
   * Resolves the Cedar principal from the agent's invocationState.
   * Return `undefined` to deny the request (fail-closed).
   *
   * @example
   * ```typescript
   * principalResolver: (state) => {
   *   if (!state.user_id) return undefined
   *   return { type: 'User', id: String(state.user_id) }
   * }
   * ```
   */
  principalResolver: (invocationState: Record<string, CedarValueJson>) => CedarEntityUid | undefined

  /**
   * Maps tool calls to Cedar resources. When omitted, the resource is
   * unconstrained (`Resource::"agent"`). Use this to map tools to
   * domain-specific entities (e.g. `Record::"42"`).
   */
  resourceResolver?: ResourceResolver | undefined

  /**
   * Adds extra fields to the `context.session` object passed to Cedar.
   * Called on every tool invocation. Cannot overwrite built-in fields
   * (`hour_utc`, `call_count`, `environment`).
   */
  contextEnricher?:
    | ((context: { toolName: string; toolInput: Record<string, CedarValueJson> }) => Record<string, CedarValueJson>)
    | undefined

  /**
   * What to do when the handler throws during evaluation.
   * - `'throw'` (default) — rethrow the error
   * - `'deny'` — treat errors as denials (fail-closed)
   * - `'proceed'` — ignore errors and allow the tool call
   */
  onError?: OnError | undefined
}

/**
 * Cedar authorization intervention handler.
 *
 * Evaluates {@link https://cedarpolicy.com | Cedar} policies before each tool call
 * using {@link https://www.npmjs.com/package/@cedar-policy/cedar-wasm | @cedar-policy/cedar-wasm}.
 *
 * Uses the {@link https://github.com/cedar-policy/cedar-for-agents | cedar-for-agents}
 * schema generator conventions:
 * - One Cedar action per tool (e.g. `Action::"search"`)
 * - Resource is unconstrained by default (use `resourceResolver` for domain objects)
 * - Context is nested: `{ input: <tool args>, session: { hour_utc, call_count, ... } }`
 *
 * Rate-limit counters are scoped to the handler instance. If you share one
 * `CedarAuthorization` across multiple agents, they share counters (keyed by
 * `session_id` from invocationState). Call {@link resetSession} when a session
 * ends to free memory.
 *
 * @see {@link https://docs.cedarpolicy.com/syntax-policy.html | Cedar policy syntax}
 * @see {@link https://docs.cedarpolicy.com/syntax-entity.html | Cedar entity model}
 *
 * @example
 * ```typescript
 * import { CedarAuthorization } from '@strands-agents/sdk/vended-interventions/cedar'
 *
 * const cedar = new CedarAuthorization({
 *   policies: './policies/agent.cedar',
 *   entities: './policies/entities.json',
 *   principalResolver: (state) => {
 *     if (!state.user_id) return undefined
 *     return { type: 'User', id: String(state.user_id) }
 *   },
 * })
 *
 * const agent = new Agent({
 *   tools: [searchTool, deleteTool],
 *   interventions: [cedar],
 * })
 * ```
 */
export class CedarAuthorization extends InterventionHandler {
  readonly name = 'cedar-authorization'
  override readonly onError: OnError

  private _policies: string
  private _entities: CedarEntity[]
  private _schema: string | undefined
  private readonly _policySource: string
  private readonly _entitySource: CedarEntity[] | string | undefined
  private readonly _schemaSource: string | undefined
  private readonly _principalResolver: (invocationState: Record<string, CedarValueJson>) => CedarEntityUid | undefined
  private readonly _resourceResolver: ResourceResolver | undefined
  private readonly _contextEnricher: CedarAuthorizationConfig['contextEnricher']
  private readonly _callCounts = new Map<string, Map<string, number>>()
  private readonly _maxSessions = 1000

  constructor(config: CedarAuthorizationConfig) {
    super()
    this._policySource = config.policies
    this._entitySource = config.entities
    this._schemaSource = config.schema
    this._policies = loadPolicies(config.policies)
    this._entities = loadEntities(config.entities)
    this._schema = config.schema ? loadSchema(config.schema) : undefined
    this._principalResolver = config.principalResolver
    this._resourceResolver = config.resourceResolver
    this._contextEnricher = config.contextEnricher
    this.onError = config.onError ?? 'throw'

    validatePolicies(this._policies, this._schema)
  }

  override beforeToolCall(event: BeforeToolCallEvent): InterventionAction {
    const invocationState = event.invocationState as Record<string, CedarValueJson>
    const principal = this._principalResolver(invocationState)
    if (!principal) {
      return deny('No principal identity found in invocation state')
    }

    const sessionId = (invocationState.session_id as string | undefined) ?? '_default'
    const callCount = this._incrementCallCount(sessionId, event.toolUse.name)
    const toolInput = (event.toolUse.input ?? {}) as Record<string, CedarValueJson>
    const resource = this._resolveResource(event.toolUse.name, toolInput)
    const result = isAuthorized({
      principal,
      action: { type: 'Action', id: event.toolUse.name },
      resource,
      context: {
        input: toolInput,
        session: {
          ...(this._contextEnricher ? this._contextEnricher({ toolName: event.toolUse.name, toolInput }) : {}),
          hour_utc: new Date().getUTCHours(),
          call_count: callCount,
          environment: (invocationState.environment as string | undefined) ?? 'unknown',
        },
      },
      policies: { staticPolicies: this._policies },
      entities: this._entities as Entities,
    })

    if (result.type === 'failure') {
      return deny(`Cedar evaluation failed: ${result.errors.map((e) => e.message).join(', ')}`)
    }

    if (result.response.decision === 'deny') {
      const reasons = result.response.diagnostics.reason
      const errors = result.response.diagnostics.errors.map((e) => e.error.message)
      const details = [...reasons, ...errors].filter(Boolean)
      return deny(`Access denied by Cedar policy${details.length ? `: ${details.join(', ')}` : ''}`)
    }

    return proceed()
  }

  /**
   * Clears the rate-limit call counters for a given session.
   * Call this when a session ends to free memory.
   */
  resetSession(sessionId: string): void {
    this._callCounts.delete(sessionId)
  }

  /**
   * Reloads policies and entities from their original sources (file paths or inline).
   * Use this to pick up policy file changes at runtime without recreating the handler.
   *
   * @throws If the policy file no longer exists or contains invalid Cedar syntax.
   */
  reload(): void {
    const policies = loadPolicies(this._policySource)
    const entities = loadEntities(this._entitySource)
    const schema = this._schemaSource ? loadSchema(this._schemaSource) : undefined
    validatePolicies(policies, schema)
    this._policies = policies
    this._entities = entities
    this._schema = schema
  }

  private _resolveResource(toolName: string, toolInput: Record<string, CedarValueJson>): CedarEntityUid {
    if (!this._resourceResolver) {
      return { type: 'Resource', id: 'agent' }
    }
    if (typeof this._resourceResolver === 'function') {
      return this._resourceResolver(toolName, toolInput)
    }
    const mapping = this._resourceResolver[toolName]
    if (!mapping) {
      return { type: 'Resource', id: 'agent' }
    }
    const id = toolInput[mapping.key]
    return { type: mapping.type, id: String(id ?? toolName) }
  }

  private _incrementCallCount(sessionId: string, toolName: string): number {
    let session = this._callCounts.get(sessionId)
    if (!session) {
      if (this._callCounts.size >= this._maxSessions) {
        const oldest = this._callCounts.keys().next().value!
        this._callCounts.delete(oldest)
      }
      session = new Map()
      this._callCounts.set(sessionId, session)
    }
    const next = (session.get(toolName) ?? 0) + 1
    session.set(toolName, next)
    return next
  }
}

function validatePolicies(policies: string, schema?: string): void {
  const parseResult = checkParsePolicySet({ staticPolicies: policies })
  if (parseResult.type === 'failure') {
    const errors = parseResult.errors.map((e) => e.message).join(', ')
    throw new Error(`Invalid Cedar policy: ${errors}`)
  }

  if (schema) {
    const validationResult = validate({
      schema,
      policies: { staticPolicies: policies },
    })
    if (validationResult.type === 'failure') {
      const errors = validationResult.errors.map((e) => e.message).join(', ')
      throw new Error(`Cedar policy validation failed: ${errors}`)
    }
    if (validationResult.validationErrors.length > 0) {
      const errors = validationResult.validationErrors.map((e) => `${e.policyId}: ${e.error.message}`).join(', ')
      throw new Error(`Cedar policy validation failed: ${errors}`)
    }
  }
}

function loadSchema(schema: string): string {
  if (schema.endsWith('.cedarschema')) {
    if (!existsSync(schema)) {
      throw new Error(`Cedar schema file not found: ${schema}`)
    }
    return readFileSync(schema, 'utf-8')
  }
  return schema
}

function loadPolicies(policies: string): string {
  if (policies.endsWith('.cedar')) {
    if (!existsSync(policies)) {
      throw new Error(`Cedar policy file not found: ${policies}`)
    }
    return readFileSync(policies, 'utf-8')
  }
  return policies
}

function loadEntities(entities: CedarEntity[] | string | undefined): CedarEntity[] {
  if (!entities) return []
  if (typeof entities === 'string') {
    return JSON.parse(readFileSync(entities, 'utf-8')) as CedarEntity[]
  }
  return entities
}
