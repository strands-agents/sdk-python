/**
 * Internal helpers for routing an {@link OpenAIModel} through Amazon Bedrock's
 * OpenAI-compatible "Mantle" endpoint.
 *
 * Converts a {@link BedrockMantleConfig} into the `baseURL` and `apiKey` the
 * OpenAI SDK consumes. Tokens are minted on demand via
 * `@aws/bedrock-token-generator` so long-running agents survive the bearer
 * token's maximum lifetime.
 *
 * `@aws/bedrock-token-generator` is declared as an optional peer dependency, so
 * the import is lazy: it happens the first time the OpenAI client's async
 * `apiKey` setter is invoked.
 */

import type { AwsCredentialIdentity, AwsCredentialIdentityProvider } from '@smithy/types'

const MANTLE_DOCS_URL = 'https://docs.aws.amazon.com/bedrock/latest/userguide/inference-openai.html'

/**
 * Async function that returns a freshly minted Bedrock Mantle bearer token.
 * Matches the shape returned by `@aws/bedrock-token-generator`'s
 * `getTokenProvider`.
 *
 * @internal
 */
export type TokenProvider = () => Promise<string>

/**
 * Config for routing an OpenAI-compatible client through Amazon Bedrock's
 * Mantle endpoint.
 *
 * When supplied to `OpenAIModel`, this config derives the OpenAI client's
 * `baseURL` and `apiKey`. It cannot be combined with a pre-built `client`,
 * a top-level `apiKey`, or `clientConfig.baseURL` / `clientConfig.apiKey`,
 * since those are derived from this config.
 */
export interface BedrockMantleConfig {
  /**
   * AWS region hosting the Bedrock Mantle endpoint. If omitted, resolved from
   * the `AWS_REGION` or `AWS_DEFAULT_REGION` environment variable. An error is
   * thrown if none resolve.
   */
  region?: string

  /**
   * AWS credentials forwarded to the bearer token generator. Accepts either a
   * static credential identity or a credential provider function (e.g. the
   * result of `fromNodeProviderChain()` from `@aws-sdk/credential-providers`).
   * When omitted, the token generator resolves credentials from the standard
   * AWS credential chain.
   */
  credentials?: AwsCredentialIdentity | AwsCredentialIdentityProvider

  /**
   * Bearer token lifetime in seconds, forwarded to the token generator.
   * Capped at 12 hours by AWS. When omitted, the generator's default applies.
   * @see https://docs.aws.amazon.com/bedrock/latest/userguide/inference-openai.html
   */
  expiresInSeconds?: number
}

/**
 * Resolves the AWS region for Mantle, preferring explicit config and falling
 * back to the standard AWS env vars.
 *
 * @internal
 */
export function resolveMantleRegion(config: BedrockMantleConfig): string {
  if (config.region) {
    return config.region
  }

  const envRegion = globalThis?.process?.env?.AWS_REGION || globalThis?.process?.env?.AWS_DEFAULT_REGION
  if (envRegion) {
    return envRegion
  }

  throw new Error(
    "could not resolve an AWS region for Bedrock Mantle. Pass 'region' in " +
      'bedrockMantleConfig or set AWS_REGION in the environment. ' +
      `See ${MANTLE_DOCS_URL} for supported regions.`
  )
}

/**
 * Builds the Mantle base URL for a region.
 *
 * @internal
 */
export function bedrockMantleBaseUrl(region: string): string {
  return `https://bedrock-mantle.${region}.api.aws/v1`
}

/**
 * Builds an async `apiKey` setter (matching the OpenAI SDK's `ApiKeySetter`
 * signature) that mints a fresh bearer token on every request.
 *
 * The `@aws/bedrock-token-generator` package is loaded lazily on first use so
 * applications that never touch the Mantle pathway don't need it installed.
 *
 * @internal
 */
export function createMantleApiKeySetter(config: BedrockMantleConfig, region: string): () => Promise<string> {
  let tokenProviderPromise: Promise<TokenProvider> | null = null

  const initProvider = async (): Promise<TokenProvider> => {
    const { getTokenProvider } = await loadTokenGenerator()
    return getTokenProvider({
      region,
      ...(config.credentials !== undefined ? { credentials: config.credentials } : {}),
      ...(config.expiresInSeconds !== undefined ? { expiresInSeconds: config.expiresInSeconds } : {}),
    })
  }

  return async (): Promise<string> => {
    if (tokenProviderPromise === null) {
      tokenProviderPromise = initProvider()
    }
    const provideToken = await tokenProviderPromise
    try {
      return await provideToken()
    } catch (cause) {
      throw new Error(
        `failed to mint Bedrock Mantle bearer token for region '${region}' | ` +
          'verify your AWS credentials and network connectivity',
        { cause }
      )
    }
  }
}

async function loadTokenGenerator(): Promise<typeof import('@aws/bedrock-token-generator')> {
  try {
    return await import('@aws/bedrock-token-generator')
  } catch (cause) {
    throw new Error(
      "bedrockMantleConfig requires the '@aws/bedrock-token-generator' package | " +
        "install it with: npm install '@aws/bedrock-token-generator'",
      { cause }
    )
  }
}
