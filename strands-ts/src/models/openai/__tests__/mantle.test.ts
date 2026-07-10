import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import OpenAI from 'openai'
import { isNode } from '../../../__fixtures__/environment.js'
import { OpenAIModel } from '../index.js'

vi.mock('openai', () => {
  const mockConstructor = vi.fn(function (this: unknown) {
    return {}
  })
  return {
    default: mockConstructor,
  }
})

const getTokenProviderMock = vi.fn()
vi.mock('@aws/bedrock-token-generator', () => ({
  getTokenProvider: (...args: unknown[]) => getTokenProviderMock(...args),
}))

const TEST_MODEL_ID = 'openai.gpt-oss-120b'
const TEST_TOKEN = 'bedrock-api-key-deadbeef&Version=1'

function lastApiKeySetter(): () => Promise<string> {
  const calls = (OpenAI as unknown as { mock: { calls: unknown[][] } }).mock.calls
  expect(calls.length).toBeGreaterThan(0)
  const options = calls[calls.length - 1]![0] as { apiKey: () => Promise<string> }
  expect(typeof options.apiKey).toBe('function')
  return options.apiKey
}

describe('OpenAIModel bedrockMantleConfig', () => {
  let provideTokenMock: ReturnType<typeof vi.fn>

  beforeEach(() => {
    vi.clearAllMocks()
    if (isNode) {
      // Mantle pathway shouldn't look at OPENAI_API_KEY — guard against
      // accidental env leakage by clearing it for the suite.
      vi.stubEnv('OPENAI_API_KEY', '')
      vi.stubEnv('AWS_REGION', '')
      vi.stubEnv('AWS_DEFAULT_REGION', '')
    }
    provideTokenMock = vi.fn().mockResolvedValue(TEST_TOKEN)
    getTokenProviderMock.mockReturnValue(provideTokenMock)
  })

  afterEach(() => {
    vi.clearAllMocks()
    if (isNode) {
      vi.unstubAllEnvs()
    }
  })

  describe('constructor wiring', () => {
    it('sets baseURL and installs async apiKey setter that mints a bearer token', async () => {
      new OpenAIModel({
        modelId: TEST_MODEL_ID,
        bedrockMantleConfig: { region: 'us-east-1' },
      })

      expect(OpenAI).toHaveBeenCalledWith(
        expect.objectContaining({
          baseURL: 'https://bedrock-mantle.us-east-1.api.aws/v1',
          apiKey: expect.any(Function),
        })
      )

      const apiKey = await lastApiKeySetter()()
      expect(apiKey).toBe(TEST_TOKEN)
      expect(getTokenProviderMock).toHaveBeenCalledWith({ region: 'us-east-1' })
    })

    it('forwards optional credentials and expiresInSeconds to getTokenProvider', async () => {
      const credentials = vi.fn()
      new OpenAIModel({
        modelId: TEST_MODEL_ID,
        bedrockMantleConfig: {
          region: 'us-west-2',
          credentials,
          expiresInSeconds: 900,
        },
      })

      await lastApiKeySetter()()

      expect(getTokenProviderMock).toHaveBeenCalledWith({
        region: 'us-west-2',
        credentials,
        expiresInSeconds: 900,
      })
    })

    it('mints a fresh token on every apiKey setter call', async () => {
      new OpenAIModel({
        modelId: TEST_MODEL_ID,
        bedrockMantleConfig: { region: 'us-east-1' },
      })

      const apiKey = lastApiKeySetter()
      await apiKey()
      await apiKey()
      await apiKey()

      // The token provider is created once and reused, but it is invoked per call.
      expect(getTokenProviderMock).toHaveBeenCalledTimes(1)
      expect(provideTokenMock).toHaveBeenCalledTimes(3)
    })

    it('merges with other clientConfig fields while overriding baseURL and apiKey', () => {
      const http = vi.fn()
      new OpenAIModel({
        modelId: TEST_MODEL_ID,
        clientConfig: {
          timeout: 42,
          fetch: http,
          defaultHeaders: { 'X-Trace-Id': 'abc' },
        },
        bedrockMantleConfig: { region: 'us-east-1' },
      })

      expect(OpenAI).toHaveBeenCalledWith(
        expect.objectContaining({
          baseURL: 'https://bedrock-mantle.us-east-1.api.aws/v1',
          apiKey: expect.any(Function),
          timeout: 42,
          fetch: http,
          defaultHeaders: { 'X-Trace-Id': 'abc' },
        })
      )
    })

    it('does not check OPENAI_API_KEY when bedrockMantleConfig is set', () => {
      // env vars are cleared in beforeEach — this would normally throw, but the
      // Mantle pathway has its own auth and must bypass the check.
      expect(
        () => new OpenAIModel({ modelId: TEST_MODEL_ID, bedrockMantleConfig: { region: 'us-east-1' } })
      ).not.toThrow()
    })

    it('works for api: "chat" as well as the default responses api', async () => {
      new OpenAIModel({
        api: 'chat',
        modelId: TEST_MODEL_ID,
        bedrockMantleConfig: { region: 'us-east-1' },
      })
      const apiKey = await lastApiKeySetter()()
      expect(apiKey).toBe(TEST_TOKEN)
    })
  })

  describe('validation', () => {
    it('throws when bedrockMantleConfig is combined with a pre-built client', () => {
      const client = {} as OpenAI
      expect(
        () =>
          new OpenAIModel({
            modelId: TEST_MODEL_ID,
            client,
            bedrockMantleConfig: { region: 'us-east-1' },
          })
      ).toThrow(/bedrockMantleConfig.*pre-built/)
    })

    it('throws when clientConfig.baseURL is set alongside bedrockMantleConfig', () => {
      expect(
        () =>
          new OpenAIModel({
            modelId: TEST_MODEL_ID,
            clientConfig: { baseURL: 'https://example.invalid' },
            bedrockMantleConfig: { region: 'us-east-1' },
          })
      ).toThrow(/baseURL/)
    })

    it('throws when clientConfig.apiKey is set alongside bedrockMantleConfig', () => {
      expect(
        () =>
          new OpenAIModel({
            modelId: TEST_MODEL_ID,
            clientConfig: { apiKey: 'sk-nope' },
            bedrockMantleConfig: { region: 'us-east-1' },
          })
      ).toThrow(/apiKey/)
    })

    it('throws when top-level apiKey is set alongside bedrockMantleConfig', () => {
      expect(
        () =>
          new OpenAIModel({
            modelId: TEST_MODEL_ID,
            apiKey: 'sk-nope',
            bedrockMantleConfig: { region: 'us-east-1' },
          })
      ).toThrow(/apiKey/)
    })
  })

  describe('region resolution', () => {
    it('throws when no region is available from config or env', () => {
      expect(() => new OpenAIModel({ modelId: TEST_MODEL_ID, bedrockMantleConfig: {} })).toThrow(
        /could not resolve an AWS region/
      )
    })

    if (isNode) {
      it('falls back to AWS_REGION env var', async () => {
        vi.stubEnv('AWS_REGION', 'eu-west-1')
        new OpenAIModel({ modelId: TEST_MODEL_ID, bedrockMantleConfig: {} })
        await lastApiKeySetter()()
        expect(OpenAI).toHaveBeenCalledWith(
          expect.objectContaining({ baseURL: 'https://bedrock-mantle.eu-west-1.api.aws/v1' })
        )
        expect(getTokenProviderMock).toHaveBeenCalledWith({ region: 'eu-west-1' })
      })

      it('falls back to AWS_DEFAULT_REGION when AWS_REGION is unset', async () => {
        vi.stubEnv('AWS_DEFAULT_REGION', 'ap-southeast-2')
        new OpenAIModel({ modelId: TEST_MODEL_ID, bedrockMantleConfig: {} })
        await lastApiKeySetter()()
        expect(getTokenProviderMock).toHaveBeenCalledWith({ region: 'ap-southeast-2' })
      })

      it('prefers explicit region over env vars', async () => {
        vi.stubEnv('AWS_REGION', 'eu-west-1')
        new OpenAIModel({ modelId: TEST_MODEL_ID, bedrockMantleConfig: { region: 'us-east-1' } })
        await lastApiKeySetter()()
        expect(getTokenProviderMock).toHaveBeenCalledWith({ region: 'us-east-1' })
      })
    }
  })

  describe('token minting errors', () => {
    it('wraps token provider failures with actionable context', async () => {
      provideTokenMock.mockRejectedValueOnce(new Error('no credentials in chain'))
      new OpenAIModel({ modelId: TEST_MODEL_ID, bedrockMantleConfig: { region: 'us-east-1' } })
      await expect(lastApiKeySetter()()).rejects.toThrow(/us-east-1/)
    })
  })
})
