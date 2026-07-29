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
 * Sentinel captured at module load: the default logger's debug is a no-op, so we skip the
 * expensive JSON.stringify when the user has not injected a custom logger with real debug output.
 * Comparing by reference is reliable because configureLogging replaces the entire logger object.
 */
const INITIAL_DEBUG = logger.debug

/**
 * Cap on model-generated reason/summary text written into the consolidation changelog.
 * Keeps plan metadata concise without truncating useful context.
 *
 * @internal
 */
const MAX_PLAN_TEXT_LENGTH = 500

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
  // inspectable — the thrown errors carry no plan body. Guard the eager JSON.stringify behind a
  // no-op check so the potentially multi-MB string is never built when debug discards it (the
  // default logger's debug is a no-op). Any custom logger that implements debug will still see it.
  if (logger.debug !== INITIAL_DEBUG) {
    logger.debug(`plan=<${JSON.stringify(result.structuredOutput)}> | raw consolidation plan returned by planner`)
  }
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
