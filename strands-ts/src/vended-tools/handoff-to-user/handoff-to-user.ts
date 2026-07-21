import { tool } from '../../tools/tool-factory.js'
import { z } from 'zod'
import type { JSONValue } from '../../types/json.js'
import {
  HANDOFF_TO_USER_DESCRIPTION,
  INTERRUPT_NAME,
  MAX_OPTIONS_COUNT,
  MAX_OPTION_LENGTH,
  MAX_QUESTION_LENGTH,
  type HandoffAnswer,
  type HandoffQuestion,
} from './types.js'

/**
 * Zod schema for the tool's input. The schema is the single source of truth
 * for the validated input shape and its size caps; the exported
 * `HandoffToUserInput` type is derived from it via `z.input` (the pre-parse
 * shape, so `allow_free_text` remains optional in the caller-facing type).
 *
 * Zod runs before the callback, so oversized or malformed inputs never reach
 * the interrupt path.
 */
export const handoffToUserInputSchema = z
  .object({
    question: z
      .string()
      .min(1, 'question must be a non-empty string')
      .max(MAX_QUESTION_LENGTH, `question length exceeds maximum allowed length (${MAX_QUESTION_LENGTH})`)
      .describe('The question to ask the user.'),
    options: z
      .array(
        z
          .string()
          .max(MAX_OPTION_LENGTH, `options entry length exceeds maximum allowed length (${MAX_OPTION_LENGTH})`)
          .refine((o) => o.trim().length > 0, 'options entries must be non-empty')
      )
      .min(1, 'options must contain at least one entry when provided')
      .max(MAX_OPTIONS_COUNT, `options count exceeds maximum allowed count (${MAX_OPTIONS_COUNT})`)
      .optional()
      .describe(`Optional multiple-choice options (up to ${MAX_OPTIONS_COUNT} entries).`),
    allow_free_text: z
      .boolean()
      .optional()
      .describe('Whether the consumer should accept a free-text answer (defaults to true).'),
  })
  .refine((v) => v.options !== undefined || v.allow_free_text !== false, {
    message: 'handoff must accept either options or free text; got neither',
  })

/**
 * Input accepted by the `handoffToUser` tool, derived from the Zod schema so
 * the type and validation cannot drift.
 */
export type HandoffToUserInput = z.input<typeof handoffToUserInputSchema>

/**
 * Validate the request beyond what Zod covers and build the interrupt payload.
 *
 * Zod already checks length caps; we additionally enforce uniqueness of options,
 * which the schema alone cannot express without pushing the validator surface up.
 */
function buildReason(input: z.infer<typeof handoffToUserInputSchema>): HandoffQuestion {
  const { question, options, allow_free_text } = input

  if (question.trim().length === 0) {
    throw new Error('question must be a non-empty string')
  }

  let normalizedOptions: string[] | null = null
  if (options !== undefined) {
    // Compare options on their trimmed value so `["yes", "yes "]` is rejected
    // as a duplicate — the schema-level non-empty refinement already treats
    // such entries as equivalent (both are non-empty after trim).
    const seen = new Set<string>()
    for (const opt of options) {
      const stripped = opt.trim()
      if (seen.has(stripped)) {
        throw new Error(`options duplicates an earlier entry: ${JSON.stringify(opt)}`)
      }
      seen.add(stripped)
    }
    normalizedOptions = [...options]
  }

  return {
    question,
    options: normalizedOptions,
    allow_free_text: allow_free_text ?? true,
  }
}

/**
 * Normalize the resume response into a well-shaped HandoffAnswer.
 *
 * - A bare string becomes an object of the form &#123; answer: string &#125;.
 * - An object with a string 'answer' (and optionally string 'chose') passes through.
 * - Anything else throws: the tool result should be well-typed, not opaque.
 */
function coerceResponse(response: JSONValue): HandoffAnswer {
  // The same 4096-char budget bounds the question going out and the answer
  // coming back — one direction is user prompt, the other is user answer, both
  // get plumbed into model context, so a single per-turn text budget is enough.
  if (typeof response === 'string') {
    if (response.length > MAX_QUESTION_LENGTH) {
      throw new Error(
        `handoff response 'answer' length (${response.length}) exceeds maximum allowed length (${MAX_QUESTION_LENGTH})`
      )
    }
    return { answer: response }
  }
  if (response !== null && typeof response === 'object' && !Array.isArray(response)) {
    const { answer, chose } = response as { answer?: unknown; chose?: unknown }
    if (typeof answer !== 'string') {
      throw new Error("handoff response 'answer' must be a string")
    }
    if (answer.length > MAX_QUESTION_LENGTH) {
      throw new Error(
        `handoff response 'answer' length (${answer.length}) exceeds maximum allowed length (${MAX_QUESTION_LENGTH})`
      )
    }
    const result: HandoffAnswer = { answer }
    if (chose !== undefined) {
      if (typeof chose !== 'string') {
        throw new Error("handoff response 'chose' must be a string when provided")
      }
      if (chose.length > MAX_OPTION_LENGTH) {
        throw new Error(
          `handoff response 'chose' length (${chose.length}) exceeds maximum allowed length (${MAX_OPTION_LENGTH})`
        )
      }
      result.chose = chose
    }
    return result
  }
  throw new Error(
    "handoff response must be a string or an object with an 'answer' key; " +
      `got ${response === null ? 'null' : typeof response}`
  )
}

/**
 * handoffToUser: pause the agent to ask the user a structured question.
 *
 * A thin shim over context.interrupt. Validates the request, then raises
 * a single interrupt whose 'reason' carries a HandoffQuestion. The SDK
 * halts the agent; the caller resumes with the user's response, which is
 * normalized into a HandoffAnswer and returned as the tool result.
 *
 * No console I/O and no UI: rendering the question and collecting the answer
 * belong to the consumer (a chat frontend, Slack, custom CLI intervention).
 * When nothing is listening, the SDK's default interrupt handling applies.
 *
 * @example
 * ```typescript
 * import { Agent } from '@strands-agents/sdk'
 * import { handoffToUser } from '@strands-agents/sdk/vended-tools/handoff-to-user'
 *
 * const agent = new Agent({ tools: [handoffToUser] })
 * const result = await agent.invoke('Ask me which environment to deploy to.')
 *
 * if (result.stopReason === 'interrupt') {
 *   const interrupt = result.interrupts[0]
 *   // interrupt.reason = { question, options, allow_free_text }
 *   const resumed = await agent.invoke([
 *     new InterruptResponseContent({
 *       interruptId: interrupt.id,
 *       response: { answer: 'prod', chose: 'prod' },
 *     }),
 *   ])
 * }
 * ```
 */
export const handoffToUser = tool({
  name: 'handoff_to_user',
  description: HANDOFF_TO_USER_DESCRIPTION,
  inputSchema: handoffToUserInputSchema,
  callback: (input, context): HandoffAnswer => {
    if (!context) {
      throw new Error('Tool context is required for handoffToUser')
    }
    const reason = buildReason(input)
    // context.interrupt throws InterruptError on the first call (halting the
    // agent) and returns the user's response on resume.
    const response = context.interrupt<JSONValue>({
      name: INTERRUPT_NAME,
      reason: reason as unknown as JSONValue,
    })
    return coerceResponse(response)
  },
})
