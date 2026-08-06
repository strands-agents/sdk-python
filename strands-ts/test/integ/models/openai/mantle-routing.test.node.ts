/**
 * Drift detection for the Bedrock Mantle base-path table.
 *
 * Mantle serves each model from exactly one base path (`/v1` or `/openai/v1`) and rejects
 * the other with HTTP 400 `validation_error`. The path is a property of the individual
 * model and is *not* discoverable from the API: `GET /v1/models` reports `status` but not
 * routing, and there is no `/openai/v1/models`. So `OPENAI_PATH_MODEL_IDS` in `mantle.ts`
 * is a hand-maintained table, and it silently goes stale whenever Mantle onboards a model.
 *
 * This test closes that gap. For every listed model it asserts the positive (the resolved
 * path really does serve the model) and only then checks the other path, so all three ways
 * the table can be wrong are distinguishable:
 *
 * - the resolved path is silent and the other one answers: mis-routed,
 * - neither path answers: the id is unroutable through `OpenAIModel` at all,
 * - the resolved path answers: correct, and a dual-served model needs no table edit.
 *
 * Only HTTP 200 ("served") and 400 ("this route does not serve the model") are treated as
 * answers. Any other status means *undetermined*, is retried, and fails the test if it
 * persists rather than being counted as "not served". That includes transient 429/5xx
 * responses and timeouts as well as permanent 401/403/404 access or entitlement gaps. A
 * detector that reports "no drift" when it means "could not tell" is worse than none,
 * because the "Verified against the us-east-1 catalog" note in the table would look
 * CI-backed when it is not. The #3654 reporter hit exactly such a transient
 * `internal_server_error`.
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

// Statuses that answer "does this route serve this model": 200 yes, 400 no. Everything
// else is inconclusive and must not be read as "no".
const DEFINITIVE = [200, 400]
const ATTEMPTS = 3

// Models that answer on neither OpenAI-compatible base path. The Anthropic family is served
// from /anthropic/v1/messages (a different protocol, reached via AnthropicModel, not
// OpenAIModel), so it is out of scope for this table.
const NOT_OPENAI_COMPATIBLE_PREFIXES = ['anthropic.']

const mintToken = createMantleApiKeySetter({ region: REGION }, REGION)

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms))

/** POST to a Mantle route and return the HTTP status (0 on timeout or transport error). */
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
    // Transport failure is undetermined, not "not served".
    return 0
  }
}

/** `status` retried with backoff until it answers 200/400, or the last status seen. */
async function statusSettled(path: string, body: unknown, token: string): Promise<number> {
  let latest = 0
  for (let attempt = 0; attempt < ATTEMPTS; attempt++) {
    latest = await status(path, body, token)
    if (DEFINITIVE.includes(latest)) return latest
    if (attempt < ATTEMPTS - 1) {
      await sleep(2 ** attempt * 1000)
    }
  }
  return latest
}

/**
 * Whether Mantle serves `modelId` from `basePath`.
 *
 * Returns `true` if either API surface answers 200, `false` only if both definitively
 * reject with 400, and `null` when a surface never settled (the route could not be
 * determined and the caller must not treat that as "not served").
 */
async function serves(basePath: string, modelId: string, token: string): Promise<boolean | null> {
  const surfaces: [string, unknown][] = [
    ['chat/completions', { model: modelId, messages: [{ role: 'user', content: 'hi' }], max_completion_tokens: 8 }],
    ['responses', { model: modelId, input: 'hi', max_output_tokens: 24 }],
  ]

  let determined = true
  for (const [surface, body] of surfaces) {
    const settled = await statusSettled(`${basePath}/${surface}`, body, token)
    if (settled === 200) return true
    if (settled !== 400) determined = false
  }
  return determined ? false : null
}

/**
 * List the live Mantle catalog, or `null` if this account cannot list it.
 *
 * `null` means "cannot run locally" rather than "no models": an account may be able to mint
 * a token but lack `bedrock-mantle:ListModels`. CI treats the same permission gap as a
 * failure so the drift detector cannot silently switch off.
 */
async function listModels(token: string): Promise<string[] | null> {
  const response = await globalThis.fetch(`${BASE}/v1/models`, {
    headers: { Authorization: `Bearer ${token}` },
    signal: AbortSignal.timeout(TIMEOUT_MS),
  })
  if (response.status === 401 || response.status === 403) {
    if (globalThis.process?.env?.CI === 'true' || globalThis.process?.env?.GITHUB_ACTIONS === 'true') {
      throw new Error(`account lacks bedrock-mantle:ListModels (${response.status})`)
    }
    return null
  }
  if (!response.ok) {
    throw new Error(`GET /v1/models failed with HTTP ${response.status}: ${await response.text()}`)
  }
  const payload = (await response.json()) as { data?: { id: string }[] }
  if (!payload.data) {
    throw new Error(`GET /v1/models returned no 'data' array: ${JSON.stringify(payload)}`)
  }
  return payload.data.map((m) => m.id).sort()
}

describe.skipIf(bedrock.skip)('Bedrock Mantle base-path routing', () => {
  it('routes every live Mantle model to the base path it is actually served from', async (ctx) => {
    const catalog = await listModels(await mintToken())
    if (catalog === null) {
      ctx.skip('account lacks bedrock-mantle:ListModels')
      return
    }

    const models = catalog.filter((id) => !NOT_OPENAI_COMPATIBLE_PREFIXES.some((prefix) => id.startsWith(prefix)))
    expect(models.length).toBeGreaterThan(0)

    const verdicts: Record<string, 'ok' | 'misrouted' | 'unserved' | 'undetermined'> = {}
    const resolvedFor: Record<string, string> = {}

    // Probe the resolved path first, so a correctly routed model costs the same as before
    // and a model served from *both* paths classifies as ok rather than sending a
    // maintainer to edit a table that cannot satisfy it. Minting per model keeps long
    // sweeps inside the token's lifetime.
    await Promise.all(
      models.map(async (modelId) => {
        const resolved = resolveMantleBasePath(modelId)
        const other = resolved === '/v1' ? '/openai/v1' : '/v1'
        resolvedFor[modelId] = resolved

        const onResolved = await serves(resolved, modelId, await mintToken())
        if (onResolved === true) {
          verdicts[modelId] = 'ok'
          return
        }

        const onOther = await serves(other, modelId, await mintToken())
        if (onOther === true) {
          verdicts[modelId] = 'misrouted'
        } else if (onResolved === null || onOther === null) {
          verdicts[modelId] = 'undetermined'
        } else {
          verdicts[modelId] = 'unserved'
        }
      })
    )

    const ids = (verdict: string): Record<string, string> =>
      Object.fromEntries(
        Object.entries(verdicts)
          .filter(([, outcome]) => outcome === verdict)
          .map(([modelId]) => [modelId, resolvedFor[modelId]!])
      )

    // Asserted before the two below so an inconclusive probe cannot be mistaken for a
    // clean sweep: an undetermined model is a gap in coverage, not a pass.
    expect(
      ids('undetermined'),
      'Mantle never returned a definitive 200/400 for these models, so their routing could ' +
        'not be verified (transient 429/5xx/timeout, or permanent 401/403/404; check model entitlement)'
    ).toEqual({})

    expect(
      ids('misrouted'),
      'Mantle serves these models from the base path the SDK does not use. Update ' +
        'OPENAI_PATH_MODEL_IDS in strands-ts/src/models/openai/mantle.ts (and the Python ' +
        'mirror in strands-py/src/strands/models/_openai_bedrock.py)'
    ).toEqual({})

    expect(
      ids('unserved'),
      'Mantle lists these models but serves them from neither OpenAI-compatible base path, ' +
        'so OpenAIModel cannot reach them at all. They likely speak another protocol (as ' +
        'anthropic.* does via /anthropic/v1/messages) and need adding to ' +
        'NOT_OPENAI_COMPATIBLE_PREFIXES'
    ).toEqual({})
  }, 600_000)

  it.for(['xai.grok-4.3', 'google.gemma-4-31b', 'google.gemma-3-27b-it', 'openai.gpt-oss-120b'])(
    'serves %s from the resolved base path',
    { timeout: 240_000 },
    async (modelId, ctx) => {
      const token = await mintToken()
      const models = await listModels(token)
      if (models === null) {
        ctx.skip('account lacks bedrock-mantle:ListModels')
        return
      }
      if (!models.includes(modelId)) {
        ctx.skip(`${modelId} is not in the ${REGION} catalog`)
        return
      }

      expect(await serves(resolveMantleBasePath(modelId), modelId, token)).toBe(true)
    }
  )
})
