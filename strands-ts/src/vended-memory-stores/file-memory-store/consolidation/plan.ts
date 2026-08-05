/**
 * The consolidation plan: its schema, and the guards that bound a plan's size.
 *
 * A plan is untrusted model output. This module is where it becomes a typed value — everything
 * downstream (validation, execution) relies on the shape the schema here proves.
 *
 * @internal
 */

import { z } from 'zod'
import { ConsolidationError, StructuredOutputError } from '../../../errors.js'
import { logger } from '../../../logging/logger.js'

/**
 * Clip `text` to `limit`, appending a `…(+N chars)` marker so a clipped value is distinguishable from one that fit.
 *
 * @internal
 */
export function clipWithCount(text: string, limit: number): string {
  return text.length > limit ? `${text.slice(0, limit)}…(+${text.length - limit} chars)` : text
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
 * shape, then bounds the action count. An oversized plan is a runaway signal, so that guard throws.
 *
 * @throws StructuredOutputError when the result carries no structured output
 * @throws ZodError when the structured output does not match {@link ConsolidationPlanSchema}
 * @throws ConsolidationError when the plan's action count exceeds `maxActionsPerPlan`
 *
 * @internal
 */
export function extractPlan(result: { structuredOutput?: unknown }, maxActionsPerPlan: number): ConsolidationPlan {
  if (!result.structuredOutput) {
    throw new StructuredOutputError('Model did not return structured output — cannot produce a consolidation plan')
  }
  // Log before parsing so a plan rejected by the schema or the action-count guard is still
  // inspectable — the thrown errors carry no plan body
  logger.debug(`plan=<${JSON.stringify(result.structuredOutput)}> | raw consolidation plan returned by planner`)
  const plan = ConsolidationPlanSchema.parse(result.structuredOutput)
  if (plan.actions.length > maxActionsPerPlan) {
    throw new ConsolidationError(
      `Consolidation plan exceeds action limit: ${plan.actions.length} actions (maxActionsPerPlan: ${maxActionsPerPlan})`
    )
  }
  return lowercasePlanPaths(plan)
}

/**
 * Lowercase every path a plan names so it matches the store's lowercased keys (see
 * {@link FileMemoryStore.add}). With both sides lowercased, path identity is plain string equality —
 * validation and execution never have to resolve casing.
 */
function lowercasePlanPaths(plan: ConsolidationPlan): ConsolidationPlan {
  const actions = plan.actions.map((action) => {
    switch (action.action) {
      case 'merge':
        return {
          ...action,
          sources: action.sources.map((source) => source.toLowerCase()),
          target: action.target.toLowerCase(),
        }
      case 'update':
        return { ...action, path: action.path.toLowerCase() }
      case 'delete':
        return { ...action, path: action.path.toLowerCase() }
      case 'move':
        return { ...action, from: action.from.toLowerCase(), to: action.to.toLowerCase() }
    }
  })
  return { ...plan, actions }
}
