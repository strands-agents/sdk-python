/**
 * Environment-driven model selection for the `strands` CLI.
 *
 * The SDK's `OpenAIModel` natively supports OpenAI-compatible endpoints via
 * `clientConfig.baseURL`, so Groq and generic gateways need no provider
 * machinery of their own — only env-to-constructor mapping. API keys are read
 * but never logged or returned.
 */

import { Agent } from '../agent/agent.js'
import { OpenAIModel } from '../models/openai/index.js'
import type { Model } from '../models/model.js'

export type ProviderName = 'groq' | 'openai' | 'bedrock'

export class ProviderConfigError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ProviderConfigError'
  }
}

const GROQ_DEFAULT_BASE_URL = 'https://api.groq.com/openai/v1'
const GROQ_DEFAULT_MODEL = 'llama-3.3-70b-versatile'

export interface ResolvedModel {
  readonly provider: ProviderName
  readonly modelId: string | undefined
  readonly baseUrl: string | undefined
  /** The configured Model, or undefined for the SDK's Bedrock default. */
  readonly model: Model | undefined
}

function env(name: string): string | undefined {
  const value = process.env[name]
  return value && value.trim() !== '' ? value.trim() : undefined
}

function rawProviderName(): string | undefined {
  return env('STRANDS_PROVIDER')
}

/**
 * Picks the provider without constructing a model.
 * Explicit env (`STRANDS_PROVIDER`/`PROVIDER`) wins; otherwise the first
 * configured key auto-detects; otherwise the SDK's Bedrock default.
 */
export function detectProvider(): ProviderName {
  const raw = rawProviderName()?.toLowerCase()
  if (raw === 'groq' || raw === 'openai' || raw === 'bedrock') {
    return raw
  }
  if (raw !== undefined) {
    throw new ProviderConfigError(`Unknown provider "${raw}". Valid values: groq, openai, bedrock.`)
  }
  if (env('GROQ_API_KEY')) {
    return 'groq'
  }
  if (env('OPENAI_API_KEY')) {
    return 'openai'
  }
  return 'bedrock'
}

/** Builds the Model for the selected provider; throws when its key is missing. */
export function createModelFromEnv(provider: ProviderName = detectProvider()): ResolvedModel {
  if (provider === 'bedrock') {
    return { provider, modelId: undefined, baseUrl: undefined, model: undefined }
  }

  if (provider === 'groq') {
    const apiKey = env('GROQ_API_KEY')
    if (!apiKey) {
      throw new ProviderConfigError('Provider "groq" has no API key. Set GROQ_API_KEY.')
    }
    const baseUrl = env('GROQ_API_BASE_URL') ?? GROQ_DEFAULT_BASE_URL
    const modelId = env('GROQ_MODEL') ?? GROQ_DEFAULT_MODEL
    return {
      provider,
      modelId,
      baseUrl,
      model: new OpenAIModel({
        api: 'chat',
        apiKey,
        modelId,
        clientConfig: { baseURL: baseUrl },
      }),
    }
  }

  const apiKey = env('OPENAI_API_KEY')
  if (!apiKey) {
    throw new ProviderConfigError('Provider "openai" has no API key. Set OPENAI_API_KEY.')
  }
  const baseUrl = env('OPENAI_BASE_URL')
  const modelId = env('OPENAI_MODEL')
  return {
    provider,
    modelId: modelId ?? undefined,
    baseUrl,
    model: new OpenAIModel({
      apiKey,
      ...(modelId ? { modelId } : {}),
      ...(baseUrl ? { clientConfig: { baseURL: baseUrl } } : {}),
    }),
  }
}

/** Validates a provider name given via flag or env. */
export function resolveProviderName(raw: string): ProviderName {
  const value = raw.trim().toLowerCase()
  if (value === 'groq' || value === 'openai' || value === 'bedrock') {
    return value
  }
  throw new ProviderConfigError(`Unknown provider "${raw}". Valid values: groq, openai, bedrock.`)
}

/** Creates an Agent with the env-selected model and reports the choice. */
export function createAgentFromEnv(rawProvider?: string): { agent: Agent; resolved: ResolvedModel } {
  const provider = rawProvider ? resolveProviderName(rawProvider) : detectProvider()
  const resolved = createModelFromEnv(provider)
  const agent = resolved.model ? new Agent({ model: resolved.model, printer: false }) : new Agent({ printer: false })
  return { agent, resolved }
}

/**
 * Parses dotenv-format text: `KEY=VALUE` lines, `#` comments, blank lines,
 * optional surrounding quotes. Later duplicates overwrite earlier ones.
 */
export function parseEnvFile(text: string): Record<string, string> {
  const result: Record<string, string> = {}
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim()
    if (trimmed === '' || trimmed.startsWith('#')) {
      continue
    }
    const equals = trimmed.indexOf('=')
    if (equals <= 0) {
      continue
    }
    const key = trimmed.slice(0, equals).trim()
    let value = trimmed.slice(equals + 1).trim()
    if (
      value.length >= 2 &&
      ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'")))
    ) {
      value = value.slice(1, -1)
    }
    result[key] = value
  }
  return result
}

/**
 * Loads a `.env` file from the given directory (default: cwd) into
 * `process.env`. Variables already set in the real environment always win.
 * Uses a dynamic import so the module stays loadable outside Node.
 */
export async function loadDotEnv(directory?: string): Promise<void> {
  if (typeof process === 'undefined') {
    return
  }
  const { readFileSync, existsSync } = await import('node:fs')
  const { join } = await import('node:path')
  const cwd = directory ?? process.env.PWD ?? process.cwd()
  const path = join(cwd, '.env')
  if (!existsSync(path)) {
    return
  }
  for (const [key, value] of Object.entries(parseEnvFile(readFileSync(path, 'utf8')))) {
    if (process.env[key] === undefined) {
      process.env[key] = value
    }
  }
}
