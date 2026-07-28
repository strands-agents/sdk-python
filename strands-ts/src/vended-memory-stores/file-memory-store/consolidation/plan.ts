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
import { encoder } from '../internal.js'

/**
 * Cap on model-generated reason/summary text written into the consolidation changelog.
 * Keeps plan metadata concise without truncating useful context.
 *
 * @internal
 */
const MAX_PLAN_TEXT_LENGTH = 500

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
        reason: z.string().max(MAX_PLAN_TEXT_LENGTH),
      }),
      z.object({
        action: z.literal('update'),
        path: z.string(),
        content: z.string(),
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
  // inspectable — the thrown errors carry no plan body
  logger.debug(`plan=<${JSON.stringify(result.structuredOutput)}> | raw consolidation plan returned by planner`)
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
 * can still carry a few very large writes.
 *
 * @internal
 */
export function generatedByteSize(plan: ConsolidationPlan): number {
  let bytes = 0
  for (const action of plan.actions) {
    if (action.action === 'merge' || action.action === 'update') {
      bytes += encoder.encode(action.content).byteLength
    }
    bytes += encoder.encode(action.reason).byteLength
  }
  bytes += encoder.encode(plan.summary).byteLength
  return bytes
}
