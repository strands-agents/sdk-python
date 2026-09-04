import { describe, expect, it, vi, beforeEach } from 'vitest'
import { createModelFromEnv, detectProvider, parseEnvFile, ProviderConfigError, resolveProviderName } from '../env.js'
import { parseArgs } from '../bin.js'

// vi.stubEnv is auto-unstubbed after each test by the vitest config.
beforeEach(() => {
  vi.stubEnv('GROQ_API_KEY', undefined)
  vi.stubEnv('OPENAI_API_KEY', undefined)
  vi.stubEnv('STRANDS_PROVIDER', undefined)
  vi.stubEnv('PROVIDER', undefined)
})

describe('detectProvider', () => {
  it('defaults to bedrock when nothing is set', () => {
    expect(detectProvider()).toBe('bedrock')
  })

  it('auto-detects groq from its API key', () => {
    vi.stubEnv('GROQ_API_KEY', 'gsk_test')
    expect(detectProvider()).toBe('groq')
  })

  it('auto-detects openai when only its key is set', () => {
    vi.stubEnv('OPENAI_API_KEY', 'sk_test')
    expect(detectProvider()).toBe('openai')
  })

  it('prefers the explicit env provider over auto-detection', () => {
    vi.stubEnv('GROQ_API_KEY', 'gsk_test')
    vi.stubEnv('OPENAI_API_KEY', 'sk_test')
    vi.stubEnv('STRANDS_PROVIDER', 'openai')
    expect(detectProvider()).toBe('openai')
  })

  it('ignores the generic PROVIDER variable used by other tooling', () => {
    vi.stubEnv('PROVIDER', 'GROQ')
    expect(detectProvider()).toBe('bedrock')
  })

  it('rejects unknown STRANDS_PROVIDER names', () => {
    vi.stubEnv('STRANDS_PROVIDER', 'anthropic')
    expect(() => detectProvider()).toThrow(ProviderConfigError)
  })
})

describe('createModelFromEnv', () => {
  it('returns no model for bedrock (SDK default applies)', () => {
    const resolved = createModelFromEnv('bedrock')
    expect(resolved.provider).toBe('bedrock')
    expect(resolved.model).toBeUndefined()
  })

  it('builds a chat-completions model pointed at the Groq endpoint', () => {
    vi.stubEnv('GROQ_API_KEY', 'gsk_test')
    const resolved = createModelFromEnv('groq')
    expect(resolved.provider).toBe('groq')
    expect(resolved.modelId).toBe('llama-3.3-70b-versatile')
    expect(resolved.baseUrl).toBe('https://api.groq.com/openai/v1')
    expect(resolved.model).toBeDefined()
  })

  it('honors GROQ_MODEL and GROQ_API_BASE_URL overrides', () => {
    vi.stubEnv('GROQ_API_KEY', 'gsk_test')
    vi.stubEnv('GROQ_MODEL', 'moonshotai/kimi-k2')
    vi.stubEnv('GROQ_API_BASE_URL', 'https://proxy.example.com/v1')
    const resolved = createModelFromEnv('groq')
    expect(resolved.modelId).toBe('moonshotai/kimi-k2')
    expect(resolved.baseUrl).toBe('https://proxy.example.com/v1')
  })

  it('fails loud when the Groq key is missing', () => {
    expect(() => createModelFromEnv('groq')).toThrow(/GROQ_API_KEY/)
  })

  it('builds a model for openai without a base URL', () => {
    vi.stubEnv('OPENAI_API_KEY', 'sk_test')
    vi.stubEnv('OPENAI_MODEL', 'gpt-4o')
    const resolved = createModelFromEnv('openai')
    expect(resolved.modelId).toBe('gpt-4o')
    expect(resolved.baseUrl).toBeUndefined()
    expect(resolved.model).toBeDefined()
  })

  it('honors OPENAI_BASE_URL for generic OpenAI-compatible gateways', () => {
    vi.stubEnv('OPENAI_API_KEY', 'sk_test')
    vi.stubEnv('OPENAI_BASE_URL', 'https://gateway.example.com/v1')
    const resolved = createModelFromEnv('openai')
    expect(resolved.baseUrl).toBe('https://gateway.example.com/v1')
  })

  it('fails loud when the OpenAI key is missing', () => {
    expect(() => createModelFromEnv('openai')).toThrow(/OPENAI_API_KEY/)
  })
})

describe('resolveProviderName', () => {
  it('accepts case-insensitive names', () => {
    expect(resolveProviderName('  GROQ ')).toBe('groq')
  })

  it('rejects unknown names with the valid list', () => {
    expect(() => resolveProviderName('mistral')).toThrow(/groq, openai, bedrock/)
  })
})

describe('parseEnvFile', () => {
  it('parses key/value pairs and skips comments', () => {
    const parsed = parseEnvFile('# comment\nGROQ_API_KEY=gsk_1\n\nOPENAI_MODEL="gpt-4o"\nEMPTY=\nBAD LINE\n')
    expect(parsed).toMatchObject({ GROQ_API_KEY: 'gsk_1', OPENAI_MODEL: 'gpt-4o', EMPTY: '' })
    expect(Object.keys(parsed)).not.toContain('BAD LINE')
  })
})

describe('parseArgs --provider', () => {
  it('captures the provider flag value', () => {
    expect(parseArgs(['run', '--provider', 'groq', 'hello'])).toMatchObject({
      command: 'run',
      provider: 'groq',
      prompt: 'hello',
    })
  })
})
