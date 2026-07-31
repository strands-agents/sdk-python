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

/**
 * Caps on an untrusted plan-derived payload — the plan echoed back in the revise prompt, and the
 * same plan in a diagnostic log field.
 *
 * Set by what the revise prompt needs, since that is the binding constraint — the model must re-emit
 * the actions it is told to keep unchanged, so a cap tight enough to make a compact log line would
 * break the revise path on ordinary plans. The per-string cap only stops one pathological `content`
 * from consuming the whole allowance; the total cap bounds a plan made large by action count instead.
 */
const MAX_PAYLOAD_LENGTH = 32 * 1024
const MAX_PAYLOAD_STRING_LENGTH = MAX_PAYLOAD_LENGTH / 8

/**
 * Clip `text` to `limit`, appending how many characters were dropped.
 *
 * The dropped count is load-bearing, not decoration: it distinguishes a value that merely reached
 * the cap from one that blew far past it, which is often the whole diagnosis when a plan is rejected
 * for size. Shared across every bounding path — the revise-prompt payload, a log field, and the
 * changelog — so the truncation marker reads identically wherever a value is clipped.
 *
 * @internal
 */
export function clipWithCount(text: string, limit: number): string {
  return text.length > limit ? `${text.slice(0, limit)}…(+${text.length - limit} chars)` : text
}

/**
 * Render an untrusted plan-shaped value as a bounded single-line string, for the revise prompt or a
 * log field.
 *
 * Per-string truncation happens inside the `JSON.stringify` replacer, so an oversized `content` never
 * becomes part of the output at all. The `…(+N chars)` marker tells the model a value was clipped so
 * it re-emits it rather than treating the truncation as intended content.
 *
 * Bounded at construction rather than gated on log level — {@link Logger} cannot report whether a
 * level is enabled, so the string gets built either way. Accepts `unknown` because the pre-schema
 * call site logs raw model output. Never throws: an unserializable value renders as a placeholder,
 * since failing to build a diagnostic must not fail the run.
 *
 * @internal
 */
export function summarizePayload(value: unknown): string {
  let json: string | undefined
  try {
    json = JSON.stringify(value, (_key, entry: unknown) =>
      typeof entry === 'string' ? clipWithCount(entry, MAX_PAYLOAD_STRING_LENGTH) : entry
    )
  } catch {
    // Circular references and BigInt both make stringify throw
    return '<unserializable>'
  }
  // stringify returns undefined for undefined and for a function value
  if (json === undefined) return String(value)
  return clipWithCount(json, MAX_PAYLOAD_LENGTH)
}

/**
 * Bound a plain string — a joined validation error — for the revise prompt or a log field.
 *
 * Validation emits one message per offending action, so the joined error grows with the plan's action
 * count and needs the same cap the plan payload gets.
 *
 * @internal
 */
export function truncatePayload(text: string): string {
  return clipWithCount(text, MAX_PAYLOAD_LENGTH)
}

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
        content: z.string(),
        reason: z.string(),
      }),
      z.object({
        action: z.literal('update'),
        path: z.string(),
        content: z.string(),
        reason: z.string(),
      }),
      z.object({
        action: z.literal('delete'),
        path: z.string(),
        reason: z.string(),
      }),
      z.object({
        action: z.literal('move'),
        from: z.string(),
        to: z.string(),
        reason: z.string(),
      }),
    ])
  ),
  summary: z.string(),
})

/** A validated consolidation plan. @internal */
export type ConsolidationPlan = z.infer<typeof ConsolidationPlanSchema>

/** A single action within a {@link ConsolidationPlan}. @internal */
export type ConsolidationAction = ConsolidationPlan['actions'][number]

/**
 * Extract and validate the plan from a raw agent result.
 *
 * Runs the untrusted model output through the schema so everything downstream can rely on the plan's
 * shape, then bounds the action count. That guard throws rather than routing into the revise-retry —
 * an oversized plan is a runaway signal, and echoing it back would re-incur the same cost.
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
  // inspectable — the thrown errors carry no plan body
  logger.debug(`plan=<${summarizePayload(result.structuredOutput)}> | raw consolidation plan returned by planner`)
  const plan = ConsolidationPlanSchema.parse(result.structuredOutput)
  if (plan.actions.length > maxActionsPerPlan) {
    throw new Error(
      `Consolidation plan exceeds action limit: ${plan.actions.length} actions (maxActionsPerPlan: ${maxActionsPerPlan})`
    )
  }
  return plan
}
