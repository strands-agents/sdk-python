/**
 * Type definitions and shared constants for the `handoffToUser` tool.
 */

/**
 * Model-facing description for the `handoffToUser` tool.
 */
export const HANDOFF_TO_USER_DESCRIPTION =
  'Hand off to the user with a structured question when you cannot proceed without ' +
  'human input. Emits an interrupt carrying the question, optional multiple-choice ' +
  'options, and whether free-text answers are accepted. The agent pauses; on resume ' +
  "the user's answer is returned as the tool result. Use sparingly and only when the " +
  'next step genuinely depends on information only the user can supply.'

/**
 * Maximum length of the question string (in characters).
 */
export const MAX_QUESTION_LENGTH = 4096

/**
 * Maximum number of multiple-choice options.
 */
export const MAX_OPTIONS_COUNT = 20

/**
 * Maximum length of each option string (in characters).
 */
export const MAX_OPTION_LENGTH = 256

/**
 * Fixed interrupt name emitted by the tool. Consumers pattern-match on this to
 * decide whether to render a handoff UI vs. some other interrupt (e.g. HITL
 * approval).
 */
export const INTERRUPT_NAME = 'strands:handoff-to-user'

/**
 * The structured payload carried on the interrupt's `reason` field.
 *
 * Consumers reading the interrupt (custom UI, HITL handler) see
 * `interrupt.reason` shaped like this.
 */
export interface HandoffQuestion {
  /** The question the agent is asking the user. */
  question: string

  /** Multiple-choice options, or `null` for a free-text question. */
  options: string[] | null

  /**
   * Whether a free-text answer is acceptable. Ignored when `options` is
   * provided; the consumer decides whether to also allow free text alongside
   * a choice.
   */
  allow_free_text: boolean
}

/**
 * The shape the consumer resumes with. `answer` is required; `chose` is
 * optional and, when present, reports which option the consumer selected.
 * Bare-string resume responses are coerced into an object of the form
 * &#123; answer: string &#125;.
 */
export interface HandoffAnswer {
  /** The human's response as free text (the primary field, always present). */
  answer: string

  /**
   * The option string the consumer reports as selected. The tool does **not**
   * validate this against the emitted `options` list — a HITL consumer may
   * return any string here. Callers that need canonical matching should
   * compare `chose` against their own copy of `options` themselves.
   */
  chose?: string
}

// The input type for the tool is derived from the Zod schema in
// `handoff-to-user.ts` (via `z.input` — the pre-parse shape) and re-exported
// from `index.ts`. Keeping the schema as the single source of truth prevents
// a hand-written interface here from silently drifting when a field is added
// or renamed.
