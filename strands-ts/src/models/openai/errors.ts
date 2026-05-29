/**
 * Shared error classification for the OpenAI model provider.
 *
 * @internal
 */

/**
 * Error message patterns that indicate context window overflow.
 *
 * @see https://platform.openai.com/docs/guides/error-codes
 */
const CONTEXT_WINDOW_OVERFLOW_PATTERNS = [
  'maximum context length',
  'context_length_exceeded',
  'too many tokens',
  'context length',
  'Input is too long for requested model',
  'input length and `max_tokens` exceed context limit',
  'too many total text bytes',
]

/**
 * Error patterns that indicate rate limiting.
 *
 * @see https://platform.openai.com/docs/guides/error-codes
 */
const RATE_LIMIT_PATTERNS = ['rate_limit_exceeded', 'rate limit', 'too many requests']

export type OpenAIErrorKind = 'contextOverflow' | 'throttling'

/**
 * Classifies an OpenAI SDK error.
 *
 * @internal
 */
export function classifyOpenAIError(err: Error & { status?: number; code?: string }): OpenAIErrorKind | undefined {
  const message = err.message?.toLowerCase() ?? ''
  const code = err.code?.toLowerCase() ?? ''

  if (err.status === 429 || code === 'rate_limit_exceeded' || RATE_LIMIT_PATTERNS.some((p) => message.includes(p))) {
    return 'throttling'
  }

  if (
    code === 'context_length_exceeded' ||
    CONTEXT_WINDOW_OVERFLOW_PATTERNS.some((pattern) => message.includes(pattern.toLowerCase()))
  ) {
    return 'contextOverflow'
  }

  return undefined
}
