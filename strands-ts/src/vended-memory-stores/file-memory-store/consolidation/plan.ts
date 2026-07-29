/**
 * The consolidation plan: its schema, and the guards that bound a plan's size.
 *
 * A plan is untrusted model output. This module is where it becomes a typed value — everything
 * downstream (validation, execution) relies on the shape the schema here proves.
 *
 * @internal
 */

import { z } from 'zod'
import { logger } from '../../../logging/logger.js'
import { DEFAULT_MAX_GENERATED_BYTES, encoder, resolveCanonicalKey } from '../internal.js'

/**
 * Cap on model-generated reason/summary text written into the consolidation changelog.
 * Keeps plan metadata concise without truncating useful context.
 *
 * @internal
 */
const MAX_PLAN_TEXT_LENGTH = 500

/**
 * Cap on any single string value inside a logged plan, set to {@link MAX_PLAN_TEXT_LENGTH} so the
 * only field it truncates is the one the schema does not already bound.
 *
 * `reason` and `summary` are schema-capped at that same length, so they reach the log whole — and
 * `reason` is the model's own account of why an action exists, which is worth having in full when
 * diagnosing a rejection. Action `content` is what makes a plan large; a truncated body still shows
 * the frontmatter and opening line, and the appended length reports how much was dropped, which is
 * itself the diagnostic when a plan is rejected for size.
 */
const MAX_LOGGED_STRING_LENGTH = MAX_PLAN_TEXT_LENGTH

/**
 * Cap on a whole logged plan payload. A plan can hold up to `maxActionsPerPlan` actions (default
 * 1000), so per-string truncation alone still leaves the total unbounded — this bounds the
 * accumulation across actions rather than the size of any one value.
 *
 * Chosen, not derived. Serialized actions run roughly 96 chars for a delete, 120 for a move, and 159
 * for a merge, so this holds about 40, 32, or 24 of them respectively — the first page of a plan,
 * which is where a systematic mistake shows itself (every path pointing at one directory, every
 * merge naming the same source). It is not sized to hold a whole plan, because the errors this
 * payload accompanies already identify the offending action: a `ZodError` names the failing path,
 * and {@link validatePlan} names every rejected path in its message. The payload corroborates those
 * errors rather than replacing them.
 *
 * Raise it if plans are routinely rejected for reasons the first page does not explain.
 */
const MAX_LOGGED_PLAN_LENGTH = 4000

/**
 * Clip `text` to `limit`, appending how many characters were dropped.
 *
 * The dropped count is load-bearing, not decoration: it distinguishes a value that merely reached
 * the cap from one that blew far past it, which is often the whole diagnosis when a plan is rejected
 * for size. Shared by every log-bounding path here so the truncation marker reads identically
 * whether a single `content` value or a whole plan payload was clipped.
 */
function clipWithCount(text: string, limit: number): string {
  return text.length > limit ? `${text.slice(0, limit)}…(+${text.length - limit} chars)` : text
}

/**
 * Render an untrusted plan-shaped value as a bounded single-line string for a log field.
 *
 * Log payloads must be bounded at the point of construction rather than gated on log level. The
 * {@link Logger} interface exposes no way to ask whether a level is enabled, so a caller cannot
 * know if a payload will be discarded, and an injected logger's `debug` is a real function even
 * when configured to drop the record — the string gets built either way. Bounding the value here
 * means the cost is the same whether the record is kept or dropped.
 *
 * Truncation happens inside the `JSON.stringify` replacer, so an oversized `content` never becomes
 * part of the output string at all; the total cap then bounds a plan made large by action count.
 *
 * Accepts `unknown` because the pre-schema call site logs raw model output, whose shape is not yet
 * proven. Never throws: a value that cannot be serialized logs as a placeholder, since failing to
 * build a diagnostic must not fail the run.
 *
 * @internal
 */
export function summarizeForLog(value: unknown): string {
  let json: string | undefined
  try {
    json = JSON.stringify(value, (_key, entry: unknown) =>
      typeof entry === 'string' ? clipWithCount(entry, MAX_LOGGED_STRING_LENGTH) : entry
    )
  } catch {
    // Circular references and BigInt both make stringify throw
    return '<unserializable>'
  }
  // stringify returns undefined for undefined and for a function value
  if (json === undefined) return String(value)
  return clipWithCount(json, MAX_LOGGED_PLAN_LENGTH)
}

/**
 * Bound a plain string destined for a log field, on the same reasoning as {@link summarizeForLog}.
 *
 * Validation accumulates one message per offending action, so the joined error grows with the
 * plan's action count and needs the same cap the plan payload gets.
 *
 * @internal
 */
export function truncateForLog(text: string): string {
  return clipWithCount(text, MAX_LOGGED_PLAN_LENGTH)
}

/**
 * Cap on model-generated content per action (characters). Derived from the default maxGeneratedBytes
 * budget so a single oversized action fails at parse time rather than after a multi-MB warn/revise
 * cycle. The byte-level cap in _consolidate still applies to the plan as a whole.
 *
 * Note: z.string().max() counts characters while the budget is in UTF-8 bytes; for ASCII-dominant
 * content they are equal, and the schema cap is an upper bound — the byte-level guard remains
 * authoritative.
 *
 * @internal
 */
const MAX_ACTION_CONTENT_LENGTH = DEFAULT_MAX_GENERATED_BYTES

/**
 * Schema for a consolidation plan, used both as the planner's structured-output contract and as the
 * parse gate on what it returns.
 *
 * @internal
 */
export const ConsolidationPlanSchema = z.object({
  actions: z.array(
    z.discriminatedUnion('action', [
      z.object({
        action: z.literal('merge'),
        sources: z.array(z.string()),
        target: z.string(),
        content: z.string().max(MAX_ACTION_CONTENT_LENGTH),
        reason: z.string().max(MAX_PLAN_TEXT_LENGTH),
      }),
      z.object({
        action: z.literal('update'),
        path: z.string(),
        content: z.string().max(MAX_ACTION_CONTENT_LENGTH),
        reason: z.string().max(MAX_PLAN_TEXT_LENGTH),
      }),
      z.object({
        action: z.literal('delete'),
        path: z.string(),
        reason: z.string().max(MAX_PLAN_TEXT_LENGTH),
      }),
      z.object({
        action: z.literal('move'),
        from: z.string(),
        to: z.string(),
        reason: z.string().max(MAX_PLAN_TEXT_LENGTH),
      }),
    ])
  ),
  summary: z.string().max(MAX_PLAN_TEXT_LENGTH),
})

/** A validated consolidation plan. @internal */
export type ConsolidationPlan = z.infer<typeof ConsolidationPlanSchema>

/** A single action within a {@link ConsolidationPlan}. @internal */
export type ConsolidationAction = ConsolidationPlan['actions'][number]

/**
 * Extract and validate the plan from a raw agent result.
 *
 * Runs the untrusted model output through the schema so everything downstream can rely on the
 * plan's shape being correct, then bounds the action count. The count guard throws rather than
 * routing into the revise-retry: an oversized plan is an abuse/runaway signal, not a fixable
 * mistake, and feeding it back to the model would re-incur the same unbounded cost.
 *
 * @throws Error when the result carries no structured output
 * @throws ZodError when the structured output does not match {@link ConsolidationPlanSchema}
 * @throws Error when the plan's action count exceeds `maxActionsPerPlan`
 *
 * @internal
 */
export function extractPlan(result: { structuredOutput?: unknown }, maxActionsPerPlan: number): ConsolidationPlan {
  if (!result.structuredOutput) {
    throw new Error('Model did not return structured output — cannot produce a consolidation plan')
  }
  // Log before parsing so a plan rejected by the schema or the action-count guard is still
  // inspectable — the thrown errors carry no plan body. The payload is bounded rather than gated on
  // whether debug is enabled, which the Logger interface gives no way to ask; see summarizeForLog.
  logger.debug(`plan=<${summarizeForLog(result.structuredOutput)}> | raw consolidation plan returned by planner`)
  const plan = ConsolidationPlanSchema.parse(result.structuredOutput)
  if (plan.actions.length > maxActionsPerPlan) {
    throw new Error(
      `Consolidation plan exceeds action limit: ${plan.actions.length} actions (maxActionsPerPlan: ${maxActionsPerPlan})`
    )
  }
  return plan
}

/**
 * Total UTF-8 bytes of model-generated content across a plan's write actions.
 *
 * Bounds planner output volume independently of the action count: a plan within the action limit
 * can still carry a few very large writes. Move actions write a snapshot copy of their source, so
 * their byte contribution equals the source content length — without this, moves report 0 bytes
 * and can amplify a single source past the cap.
 *
 * @param plan - The validated consolidation plan
 * @param files - The snapshot map from readAllFiles, used to measure move source content
 *
 * @internal
 */
export function generatedByteSize(plan: ConsolidationPlan, files: Map<string, string>): number {
  let bytes = 0
  for (const action of plan.actions) {
    if (action.action === 'merge' || action.action === 'update') {
      bytes += encoder.encode(action.content).byteLength
    } else if (action.action === 'move') {
      // A move writes the source's content to the new target — count it so N moves from one
      // large source cannot escape the generated-bytes cap
      const canonicalFrom = resolveCanonicalKey(files, action.from)
      const sourceContent = canonicalFrom !== undefined ? files.get(canonicalFrom) : undefined
      if (sourceContent !== undefined) {
        bytes += encoder.encode(sourceContent).byteLength
      }
    }
    bytes += encoder.encode(action.reason).byteLength
  }
  bytes += encoder.encode(plan.summary).byteLength
  return bytes
}
