/**
 * Drift detection for the Bedrock Mantle base-path table.
 *
 * Mantle serves each model from exactly one base path (`/v1` or `/openai/v1`) and rejects
 * the other with HTTP 400 `validation_error`. The path is a property of the individual
 * model and is *not* discoverable from the API: `GET /v1/models` reports `status` but not
 * routing, and there is no `/openai/v1/models`. So `OPENAI_PATH_MODEL_IDS` in `mantle.ts`
 * is a hand-maintained table, and it silently goes stale whenever Mantle onboards a model.
 *
 * This test closes that gap. It lists the live catalog and, for every model, probes the
 * path the SDK would *not* use. A model that answers on the unused path is mis-routed, so
 * the test fails naming the offending ids. That turns "Mantle onboarded a model and the SDK
 * sends it to the wrong route" from a 400 in a user's application into a CI failure.
 *
 * Failure means the table needs updating, not that the SDK is broken for existing models.
 * See https://github.com/strands-agents/harness-sdk/issues/3654.
 */

import { describe, expect, it } from 'vitest'
import { resolveMantleBasePath, createMantleApiKeySetter } from '$/sdk/models/openai/mantle.js'

import { bedrock } from '../../__fixtures__/model-providers.js'

const REGION = 'us-east-1'
const BASE = `https://bedrock-mantle.${REGION}.api.aws`
const TIMEOUT_MS = 30_000

// Models that answer on neither OpenAI-compatible base path. The Anthropic family is served
// from /anthropic/v1/messages (a different protocol, reached via AnthropicModel, not
// OpenAIModel), so it is out of scope for this table.
const NOT_OPENAI_COMPATIBLE_PREFIXES = ['anthropic.']

const mintToken = createMantleApiKeySetter({ region: REGION }, REGION)

/** POST to a Mantle route and return the HTTP status (0 on timeout: a hung route is "not served"). */
async function status(path: string, body: unknown, token: string): Promise<number> {
  try {
    const response = await globalThis.fetch(`${BASE}${path}`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(TIMEOUT_MS),
    })
    return response.status
  } catch {
    return 0
  }
}

/** Whether Mantle serves `modelId` from `basePath` on either API surface. */
async function serves(basePath: string, modelId: string, token: string): Promise<boolean> {
  const chat = await status(
    `${basePath}/chat/completions`,
    { model: modelId, messages: [{ role: 'user', content: 'hi' }], max_completion_tokens: 8 },
    token
  )
  if (chat === 200) return true

  const responses = await status(
    `${basePath}/responses`,
    { model: modelId, input: 'hi', max_output_tokens: 24 },
    token
  )
  return responses === 200
}

async function listModels(token: string): Promise<string[]> {
  const response = await globalThis.fetch(`${BASE}/v1/models`, {
    headers: { Authorization: `Bearer ${token}` },
    signal: AbortSignal.timeout(TIMEOUT_MS),
  })
  if (response.status === 401 || response.status === 403) {
    return []
  }
  const payload = (await response.json()) as { data: { id: string }[] }
  return payload.data.map((m) => m.id).sort()
}

describe.skipIf(bedrock.skip)('Bedrock Mantle base-path routing', () => {
  it(
    'routes every live Mantle model to the base path it is actually served from',
    async () => {
      const models = (await listModels(await mintToken())).filter(
        (id) => !NOT_OPENAI_COMPATIBLE_PREFIXES.some((prefix) => id.startsWith(prefix))
      )
      expect(models.length).toBeGreaterThan(0)

      const misrouted: Record<string, string> = {}
      // Probe only the path the SDK would *not* use: if that one answers, we are wrong.
      // Minting per model keeps long sweeps inside the token's lifetime.
      await Promise.all(
        models.map(async (modelId) => {
          const resolved = resolveMantleBasePath(modelId)
          const unused = resolved === '/v1' ? '/openai/v1' : '/v1'
          if (await serves(unused, modelId, await mintToken())) {
            misrouted[modelId] = resolved
          }
        })
      )

      expect(
        misrouted,
        'Mantle serves these models from the base path the SDK does not use. Update ' +
          'OPENAI_PATH_MODEL_IDS in strands-ts/src/models/openai/mantle.ts (and the Python ' +
          'mirror in strands-py/src/strands/models/_openai_bedrock.py)'
      ).toEqual({})
    },
    600_000
  )

  it.each(['xai.grok-4.3', 'google.gemma-4-31b', 'google.gemma-3-27b-it', 'openai.gpt-oss-120b'])(
    'serves %s from the resolved base path',
    async (modelId) => {
      const token = await mintToken()
      const models = await listModels(token)
      if (!models.includes(modelId)) return

      expect(await serves(resolveMantleBasePath(modelId), modelId, token)).toBe(true)
    },
    120_000
  )
})
