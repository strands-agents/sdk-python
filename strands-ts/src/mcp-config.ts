import type { McpClientConfig, McpClientCredentials, McpClientOptions, McpTransport, TasksConfig } from './mcp.js'
import { logger } from './logging/index.js'

/**
 * Configuration for a single MCP server entry in a config file or object.
 *
 * Provide either `command` (stdio transport) or `url` (streamable-http/SSE), not both.
 * When `transport` is omitted, it is auto-detected from the fields present.
 */
export interface McpServerConfig {
  /** Command to spawn (stdio transport, supports `${VAR}` or `${env:VAR}` interpolation). */
  command?: string
  /** Arguments passed to the command (supports `${VAR}` or `${env:VAR}` interpolation). */
  args?: string[]
  /** Environment variables passed to the child process (supports `${VAR}` or `${env:VAR}` interpolation). */
  env?: Record<string, string>
  /** Working directory for the spawned process (supports `${VAR}` or `${env:VAR}` interpolation). */
  cwd?: string
  /** Server endpoint URL (streamable-http or SSE transport, supports `${VAR}` or `${env:VAR}` interpolation). */
  url?: string
  /** HTTP headers sent with every request (supports `${VAR}` or `${env:VAR}` interpolation). */
  headers?: Record<string, string>
  /** Explicit transport type. When omitted, auto-detected: `command` → stdio, `url` → streamable-http. */
  transport?: 'stdio' | 'sse' | 'streamable-http'
  /** Client credentials for OAuth machine-to-machine auth (streamable-http only). */
  auth?: McpClientCredentials
  /** When true, this server is skipped during loadServers. */
  disabled?: boolean
  /** When true, config or connection failures skip this server instead of throwing. */
  continueOnError?: boolean
  /** Task-augmented tool execution configuration (experimental). */
  tasksConfig?: TasksConfig
}

/**
 * Resolves an MCP servers config into an array of client configurations ready for instantiation.
 *
 * @param config - A file path to a JSON config, or a flat server map object.
 * @param defaults - Options applied to all clients unless overridden per-server.
 * @returns Resolved McpClientConfig array (one per enabled, successfully-resolved server).
 */
export async function resolveServerConfigs(
  config: string | Record<string, McpServerConfig>,
  defaults?: McpClientOptions
): Promise<McpClientConfig[]> {
  const servers = await loadServersObject(config)
  const results: McpClientConfig[] = []

  for (const [name, server] of Object.entries(servers)) {
    if (!server || typeof server !== 'object' || Array.isArray(server)) {
      throw new Error(`Server "${name}" must be an object, got ${Array.isArray(server) ? 'array' : typeof server}`)
    }

    if (server.disabled) continue

    const continueOnError = server.continueOnError ?? defaults?.continueOnError ?? false

    try {
      if (server.command && server.url && !server.transport) {
        throw new Error('Server config has both "command" and "url" — set "transport" explicitly or remove one')
      }

      const type = server.transport ?? (server.command ? 'stdio' : server.url ? 'streamable-http' : undefined)
      if (!type) throw new Error('Server config must include either "command" (stdio) or "url" (http)')

      let clientConfig: McpClientConfig
      switch (type) {
        case 'stdio':
          clientConfig = await buildStdioConfig(server)
          break
        case 'streamable-http':
          clientConfig = buildHttpConfig(server)
          break
        case 'sse':
          clientConfig = await buildSseConfig(server)
          break
        default: {
          const _exhaustive: never = type
          throw new Error(`Unsupported transport type: ${_exhaustive}`)
        }
      }

      results.push({ ...baseOptions(name, server, defaults), ...clientConfig })
    } catch (error) {
      if (!continueOnError) throw error
      logger.warn(`server=<${name}>, error=<${error}> | MCP server config failed, skipping (continueOnError)`)
    }
  }

  return results
}

async function buildStdioConfig(server: McpServerConfig): Promise<McpClientConfig> {
  if (!server.command) throw new Error('Stdio transport requires "command" field')
  const { StdioClientTransport, getDefaultEnvironment } = await import('@modelcontextprotocol/sdk/client/stdio.js')

  const opts: ConstructorParameters<typeof StdioClientTransport>[0] = {
    command: interpolateEnv(server.command),
  }
  if (server.args) opts.args = server.args.map(interpolateEnv)
  if (server.env) opts.env = { ...getDefaultEnvironment(), ...interpolateRecord(server.env) }
  if (server.cwd) opts.cwd = interpolateEnv(server.cwd)

  return { transport: new StdioClientTransport(opts) as McpTransport }
}

function buildHttpConfig(server: McpServerConfig): McpClientConfig {
  if (!server.url) throw new Error('Streamable HTTP transport requires "url" field')

  const config: McpClientConfig = { url: interpolateEnv(server.url) }
  if (server.headers) config.headers = interpolateRecord(server.headers)
  if (server.auth) {
    config.auth = {
      clientId: interpolateEnv(server.auth.clientId),
      clientSecret: interpolateEnv(server.auth.clientSecret),
      ...(server.auth.scopes && { scopes: server.auth.scopes.map(interpolateEnv) }),
    }
  }
  return config
}

async function buildSseConfig(server: McpServerConfig): Promise<McpClientConfig> {
  if (!server.url) throw new Error('SSE transport requires "url" field')
  if (server.auth)
    throw new Error('SSE transport does not support auth — use streamable-http or provide a pre-configured transport')

  const { SSEClientTransport } = await import('@modelcontextprotocol/sdk/client/sse.js')
  const headers = server.headers ? interpolateRecord(server.headers) : undefined

  return {
    transport: new SSEClientTransport(
      new URL(interpolateEnv(server.url)),
      headers ? { requestInit: { headers } } : undefined
    ) as McpTransport,
  }
}

function baseOptions(name: string, server: McpServerConfig, defaults?: McpClientOptions): McpClientOptions {
  const opts: McpClientOptions = { ...defaults, applicationName: defaults?.applicationName ?? name }
  if (server.continueOnError != null) opts.continueOnError = server.continueOnError
  if (server.tasksConfig != null) opts.tasksConfig = server.tasksConfig
  return opts
}

/**
 * Replaces `$\{VAR\}` and `$\{env:VAR\}` placeholders with their process.env values.
 * Throws if a referenced variable is not set.
 *
 * @example
 * ```typescript
 * interpolateEnv('Bearer $\{TOKEN\}')       // → 'Bearer ghp_abc123'
 * interpolateEnv('$\{env:HOME\}/config')    // → '/home/user/config'
 * ```
 */
function interpolateEnv(value: string): string {
  return value.replace(/\$\{(?:env:)?([^}]+)\}/g, (_, key: string) => {
    const resolved = process.env[key]
    if (resolved === undefined) throw new Error(`Environment variable "${key}" is not set`)
    return resolved
  })
}

/** Applies {@link interpolateEnv} to every value in a string record. */
function interpolateRecord(record: Record<string, string>): Record<string, string> {
  return Object.fromEntries(Object.entries(record).map(([k, v]) => [k, interpolateEnv(v)]))
}

async function loadServersObject(
  config: string | Record<string, McpServerConfig>
): Promise<Record<string, McpServerConfig>> {
  if (typeof config !== 'string') return config

  const { readFile } = await import('node:fs/promises')
  const { homedir } = await import('node:os')
  const { join } = await import('node:path')

  const filePath = config.startsWith('~/') ? join(homedir(), config.slice(2)) : config
  const parsed = JSON.parse(await readFile(filePath, 'utf-8'))
  const servers = parsed.mcpServers ?? parsed

  if (!servers || typeof servers !== 'object' || Array.isArray(servers)) {
    throw new Error(
      'MCP config must be a JSON object mapping server names to configs, e.g. { "my-server": { "command": "node" } }'
    )
  }

  return servers
}
