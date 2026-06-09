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
 * Minimal tool definition for schema generation. Matches MCP tool format.
 */
export interface ToolDefinition {
  name: string
  inputSchema?: { type: string; properties?: Record<string, CedarValueJson>; required?: string[] }
  description?: string
}

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
   * Tool definitions for automatic schema generation and request mapping.
   * When provided (and `@cedar-policy/mcp-schema-generator-wasm` is installed),
   * the handler auto-generates a Cedar schema from your tools and uses
   * `generateRequest()` for action/resource resolution at evaluation time.
   *
   * This enables compile-time policy validation against your actual tool
   * definitions — catching typos, type mismatches, and references to
   * nonexistent tools at startup.
   *
   * Accepts the same format as MCP `tools/list` responses.
   *
   * @example
   * ```typescript
   * { tools: [
   *   { name: 'search', inputSchema: { type: 'object', properties: { query: { type: 'string' } } } },
   *   { name: 'delete', inputSchema: { type: 'object', properties: { id: { type: 'string' } } } },
   * ]}
   * ```
   */
  tools?: ToolDefinition[]

  /**
   * Entity data as an array, or a path to a `.json` file on disk.
   *
   * **Most policies don't need this.** For role-based access, pass the role
   * via `invocationState` and check it in `context.session` instead:
   * ```typescript
   * contextEnricher: ({ invocationState }) => ({ role: String(invocationState.role) }),
   * ```
   * ```cedar
   * permit(principal, action, resource) when { context.session.role == "admin" };
   * ```
   *
   * Only use `entities` when you need Cedar's entity hierarchy — e.g.
   * `principal in Role::"admin"` with parent relationships, or static
   * attributes that don't change per-request.
   */
  entities?: CedarEntity[] | string

  /**
   * Cedar schema text, or a path to a `.cedarschema` file on disk.
   * When provided, policies are validated against the schema at construction
   * time — catching type errors, unknown attributes, and invalid action names
   * before any tool call happens.
   *
   * When `tools` is provided and `@cedar-policy/mcp-schema-generator-wasm` is
   * installed, the schema is auto-generated and this field is not needed.
   */
  schema?: string

  /**
   * Static principal identity. Use for single-user or CLI agents where the
   * identity is known upfront and doesn't change between invocations.
   *
   * When neither `principal` nor `principalResolver` is provided, defaults to
   * `User::"anonymous"` — policies can still permit actions for any principal.
   *
   * Mutually exclusive with `principalResolver`.
   *
   * @example
   * ```typescript
   * { principal: { type: 'User', id: 'alice@acme.com' } }
   * ```
   */
  principal?: CedarEntityUid

  /**
   * Dynamic principal resolver for multi-tenant agents. Called on every tool
   * invocation to extract the principal from `invocationState`.
   * Return `undefined` to deny the request (fail-closed).
   *
   * Mutually exclusive with `principal`.
   *
   * @example
   * ```typescript
   * principalResolver: (state) => {
   *   if (!state.user_id) return undefined
   *   return { type: 'User', id: String(state.user_id) }
   * }
   * ```
   */
  principalResolver?: ((invocationState: Record<string, unknown>) => CedarEntityUid | undefined) | undefined

  /**
   * Adds extra fields to the `context.session` object passed to Cedar.
   * Called on every tool invocation. Cannot overwrite built-in fields
   * (`hour_utc`, `call_count`).
   *
   * Use this to inject values from `invocationState` (e.g. environment,
   * tenant, department) into the Cedar context for policy evaluation.
   *
   * **Important**: If a policy references a field that doesn't exist in context,
   * Cedar skips that policy (it doesn't deny). For fail-closed behavior on
   * optional fields, use the allow-list pattern in your policies:
   * ```cedar
   * // SAFE: missing environment → no permit → deny
   * permit(principal, action, resource)
   * when { context.session has environment && context.session.environment != "production" };
   *
   * // Or: deny everything when field is absent
   * forbid(principal, action, resource)
   * unless { context.session has environment };
   * ```
   *
   * @see {@link https://docs.cedarpolicy.com/policies/syntax-operators.html | Cedar `has` operator}
   */
  contextEnricher?:
    | ((context: {
        toolName: string
        toolInput: Record<string, unknown>
        invocationState: Record<string, unknown>
      }) => Record<string, CedarValueJson>)
    | undefined

  /**
   * What to do when the handler throws during evaluation.
   * - `'throw'` (default) — rethrow the error
   * - `'deny'` — treat errors as denials (fail-closed)
   * - `'proceed'` — **dangerous: fail-open** — ignore errors and allow the tool call.
   *   Only use for non-critical observability-only deployments where blocking on
   *   auth errors is worse than allowing unauthenticated access.
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
 * - Resource is unconstrained by default
 * - Context is nested: `{ input: <tool args>, session: { hour_utc, call_count, ... } }`
 *
 * **`context.session.call_count` behavior:**
 * - Stored on `agent.appState` — persists across invocations when a session manager is configured
 * - Counts only **successful** (permitted) tool calls — denied attempts do not consume budget
 * - Scoped per-agent-instance — multiple agent instances or load-balanced workers have separate counters
 * - Resets when a new agent instance is created without restoring session state
 * - For hard distributed rate limiting, use an external store (Redis, etc.) via `contextEnricher`
 *
 * @see {@link https://docs.cedarpolicy.com/syntax-policy.html | Cedar policy syntax}
 * @see {@link https://docs.cedarpolicy.com/syntax-entity.html | Cedar entity model}
 *
 * @example
 * ```typescript
 * import { CedarAuthorization } from '@strands-agents/sdk/vended-interventions/cedar'
 *
 * // Simple: permit specific tools, deny everything else
 * const cedar = new CedarAuthorization({
 *   policies: `
 *     permit(principal, action == Action::"search", resource);
 *     permit(principal, action == Action::"read_file", resource);
 *   `,
 * })
 *
 * // Role-based: pass role via invocationState, check in context
 * const cedar = new CedarAuthorization({
 *   policies: `
 *     permit(principal, action, resource) when { context.session.role == "admin" };
 *     permit(principal, action == Action::"search", resource) when { context.session.role == "analyst" };
 *   `,
 *   principalResolver: (state) => {
 *     if (!state.user_id) return undefined
 *     return { type: 'User', id: String(state.user_id) }
 *   },
 *   contextEnricher: ({ invocationState }) => ({
 *     role: String(invocationState.role ?? 'none'),
 *   }),
 * })
 *
 * // With schema validation (requires @cedar-policy/mcp-schema-generator-wasm)
 * const cedar = new CedarAuthorization({
 *   policies: './policies/agent.cedar',
 *   tools: [searchTool, deleteTool],  // auto-generates schema from tool definitions
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
  private readonly _principal: CedarEntityUid | undefined
  private readonly _principalResolver:
    | ((invocationState: Record<string, unknown>) => CedarEntityUid | undefined)
    | undefined
  private readonly _contextEnricher: CedarAuthorizationConfig['contextEnricher']
  private readonly _tools: ToolDefinition[] | undefined
  private readonly _schemaGenerator: SchemaGenerator | undefined
  private readonly _callCounts = new Map<string, number>()
  private readonly _stateKey: string

  constructor(config: CedarAuthorizationConfig) {
    super()
    if (config.principal && config.principalResolver) {
      throw new Error('Provide either `principal` or `principalResolver`, not both')
    }
    this._policySource = config.policies
    this._entitySource = config.entities
    this._policies = loadPolicies(config.policies)
    this._entities = loadEntities(config.entities)
    this._tools = config.tools

    this._schemaGenerator = config.tools ? loadSchemaGenerator(true) : undefined
    if (config.schema) {
      this._schema = loadSchema(config.schema)
    } else if (this._schemaGenerator && config.tools) {
      this._schema = this._schemaGenerator.generateSchema(config.tools)
    } else {
      this._schema = undefined
    }

    if (config.principalResolver) {
      this._principal = undefined
    } else {
      this._principal = config.principal ?? { type: 'User', id: 'anonymous' }
    }
    this._principalResolver = config.principalResolver
    this._contextEnricher = config.contextEnricher
    this._stateKey = `cedar-authorization:${this.name}`
    this.onError = config.onError ?? 'throw'

    validatePolicies(this._policies, this._schema)
  }

  override beforeToolCall(event: BeforeToolCallEvent): InterventionAction {
    const invocationState = event.invocationState as Record<string, unknown>
    const principal = this._principal ?? this._principalResolver!(invocationState)
    if (!principal || !principal.type || !principal.id) {
      return deny('No principal identity found in invocation state')
    }

    const callCount = this._incrementCallCount(event.agent, event.toolUse.name)
    const toolInput = (event.toolUse.input ?? {}) as Record<string, unknown>

    let action: CedarEntityUid
    let resource: CedarEntityUid
    let entities: Entities

    if (this._schemaGenerator && this._tools) {
      const request = this._schemaGenerator.generateRequest(
        this._tools,
        event.toolUse.name,
        toolInput as Record<string, CedarValueJson>,
        principal
      )
      action = request.action
      resource = request.resource
      entities = [...(this._entities as Entities), ...(request.entities as Entities)]
    } else {
      action = { type: 'Action', id: event.toolUse.name }
      resource = { type: 'Resource', id: 'agent' }
      entities = this._entities as Entities
    }

    const result = isAuthorized({
      principal,
      action,
      resource,
      context: {
        input: toolInput as Record<string, CedarValueJson>,
        session: {
          ...(this._contextEnricher
            ? this._contextEnricher({ toolName: event.toolUse.name, toolInput, invocationState })
            : {}),
          hour_utc: new Date().getUTCHours(),
          call_count: callCount,
        },
      },
      policies: { staticPolicies: this._policies },
      entities,
    })

    if (result.type === 'failure') {
      this._decrementCallCount(event.agent, event.toolUse.name)
      return deny(`Cedar evaluation failed: ${result.errors.map((e) => e.message).join(', ')}`)
    }

    if (result.response.decision === 'deny') {
      this._decrementCallCount(event.agent, event.toolUse.name)
      const reasons = result.response.diagnostics.reason
      const errors = result.response.diagnostics.errors.map((e) => e.error.message)
      const details = [...reasons, ...errors].filter(Boolean)
      return deny(`Access denied by Cedar policy${details.length ? `: ${details.join(', ')}` : ''}`)
    }

    return proceed()
  }

  /**
   * Clears the rate-limit call counters stored on the agent's appState.
   */
  resetCallCounts(agent: { appState: { set: (key: string, value: unknown) => void } }): void {
    this._callCounts.clear()
    agent.appState.set(this._stateKey, {})
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
    const schema = this._schemaGenerator && this._tools ? this._schemaGenerator.generateSchema(this._tools) : undefined
    validatePolicies(policies, schema)
    this._policies = policies
    this._entities = entities
    this._schema = schema
  }

  private _incrementCallCount(
    agent: { appState: { set: (key: string, value: unknown) => void } },
    toolName: string
  ): number {
    const current = this._callCounts.get(toolName) ?? 0
    const next = current + 1
    this._callCounts.set(toolName, next)
    agent.appState.set(this._stateKey, Object.fromEntries(this._callCounts))
    return next
  }

  private _decrementCallCount(
    agent: { appState: { set: (key: string, value: unknown) => void } },
    toolName: string
  ): void {
    const current = this._callCounts.get(toolName) ?? 0
    if (current > 0) {
      this._callCounts.set(toolName, current - 1)
      agent.appState.set(this._stateKey, Object.fromEntries(this._callCounts))
    }
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
  let parsed: CedarEntity[]
  if (typeof entities === 'string') {
    parsed = JSON.parse(readFileSync(entities, 'utf-8')) as CedarEntity[]
  } else {
    parsed = entities
  }
  for (const entity of parsed) {
    if (!entity.uid || !entity.uid.type || !entity.uid.id) {
      throw new Error(`Invalid entity: each entity must have a uid with type and id`)
    }
  }
  return parsed
}

interface SchemaGenerator {
  generateSchema(tools: ToolDefinition[]): string
  generateRequest(
    tools: ToolDefinition[],
    toolName: string,
    toolInput: Record<string, CedarValueJson>,
    principal: CedarEntityUid
  ): { action: CedarEntityUid; resource: CedarEntityUid; entities: CedarEntity[] }
}

function loadSchemaGenerator(warn: boolean): SchemaGenerator | undefined {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const wasm = require('@cedar-policy/mcp-schema-generator-wasm') as {
      generateSchema: (stub: string, toolsJson: string, configJson?: string) => string
      generateRequest: (
        stub: string,
        toolsJson: string,
        inputJson: string,
        principalType: string,
        principalId: string,
        resourceType: string,
        resourceId: string,
        configJson?: string
      ) => string
    }

    const defaultStub = `
namespace Agent {
  @mcp_principal
  entity User;
  @mcp_resource
  entity Resource;
}
`

    return {
      generateSchema(tools: ToolDefinition[]): string {
        const config = JSON.stringify({ flattenNamespaces: true })
        const result = JSON.parse(wasm.generateSchema(defaultStub, JSON.stringify(tools), config)) as {
          schema: string | null
          error: string | null
          isOk: boolean
        }
        if (!result.isOk || !result.schema) {
          throw new Error(`Schema generation failed: ${result.error}`)
        }
        // Strip namespace wrapper so users can write unqualified action names
        return result.schema.replace(/^namespace\s+\w+\s*\{/, '').replace(/\}\s*$/, '')
      },

      generateRequest(
        tools: ToolDefinition[],
        toolName: string,
        toolInput: Record<string, CedarValueJson>,
        principal: CedarEntityUid
      ) {
        const input = JSON.stringify({ params: { tool: toolName, args: toolInput } })
        const config = JSON.stringify({ flattenNamespaces: true })
        const result = JSON.parse(
          wasm.generateRequest(
            defaultStub,
            JSON.stringify(tools),
            input,
            principal.type,
            principal.id,
            'Resource',
            'agent',
            config
          )
        ) as {
          action: string | null
          resource: string | null
          entitiesJson: string | null
          error: string | null
          isOk: boolean
        }
        if (!result.isOk) {
          throw new Error(`Request generation failed: ${result.error}`)
        }

        return {
          action: parseEntityUid(result.action!),
          resource: parseEntityUid(result.resource!),
          entities: result.entitiesJson ? (JSON.parse(result.entitiesJson) as CedarEntity[]) : [],
        }
      },
    }
  } catch {
    if (warn) {
      console.warn(
        'CedarAuthorization: `tools` provided but @cedar-policy/mcp-schema-generator-wasm is not installed. ' +
          'Schema validation and auto request generation are disabled. ' +
          'Install it: npm install @cedar-policy/mcp-schema-generator-wasm'
      )
    }
    return undefined
  }
}

function parseEntityUid(uid: string): CedarEntityUid {
  const match = uid.match(/(?:.*::)?([^:]+)::"([^"]+)"/)
  if (!match) return { type: 'Action', id: uid }
  return { type: match[1]!, id: match[2]! }
}
