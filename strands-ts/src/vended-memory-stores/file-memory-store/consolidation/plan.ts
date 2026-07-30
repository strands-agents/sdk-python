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
import { DEFAULT_MAX_INPUT_BYTES, encoder, resolveCanonicalKey } from '../internal.js'

/**
 * Caps on an untrusted plan-derived payload — the plan echoed back in the revise prompt, and the
 * same plan in a diagnostic log field. Sized as a fraction of the input the planner was given: the
 * retry re-sends that input anyway, so the echo stays a bounded share of a cost already paid rather
 * than scaling with whatever the model generated.
 *
 * Set by what the revise prompt needs, since that is the binding constraint — the model must re-emit
 * the actions it is told to keep unchanged, so a cap tight enough to make a compact log line would
 * break the revise path on ordinary plans. The per-string cap only stops one pathological `content`
 * from consuming the whole budget; the total cap bounds a plan made large by action count instead.
 */
const MAX_PAYLOAD_LENGTH = DEFAULT_MAX_INPUT_BYTES / 4
const MAX_PAYLOAD_STRING_LENGTH = MAX_PAYLOAD_LENGTH / 8

/**
 * Clip `text` to `limit`, appending how many characters were dropped.
 *
 * The dropped count is load-bearing, not decoration: it distinguishes a value that merely reached
 * the cap from one that blew far past it, which is often the whole diagnosis when a plan is rejected
 * for size. Shared by both bounding paths so the truncation marker reads identically whether a
 * single `content` value or a whole plan payload was clipped.
 */
function clipWithCount(text: string, limit: number): string {
  return text.length > limit ? `${text.slice(0, limit)}…(+${text.length - limit} chars)` : text
}

/**
 * Render an untrusted plan-shaped value as a bounded single-line string, for the revise prompt or a
 * log field.
 *
 * Per-string truncation happens inside the `JSON.stringify` replacer, so an oversized `content` never
 * becomes part of the output string at all. The `…(+N chars)` marker tells the model a value was
 * clipped so it re-emits that value rather than treating the truncation as the intended content.
 *
 * Bounding happens here, at construction, rather than being gated on log level: the {@link Logger}
 * interface exposes no way to ask whether a level is enabled, and an injected logger's `debug` is a
 * real function even when configured to drop the record — the string gets built either way.
 *
 * Accepts `unknown` because the pre-schema call site logs raw model output, whose shape is not yet
 * proven. Never throws: a value that cannot be serialized renders as a placeholder, since failing to
 * build a diagnostic must not fail the run.
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
  logger.debug(`plan=<${summarizePayload(result.structuredOutput)}> | raw consolidation plan returned by planner`)
  const plan = ConsolidationPlanSchema.parse(result.structuredOutput)
  if (plan.actions.length > maxActionsPerPlan) {
    throw new Error(
      `Consolidation plan exceeds action limit: ${plan.actions.length} actions (maxActionsPerPlan: ${maxActionsPerPlan})`
    )
  }
  return plan
}

/**
 * Total UTF-8 bytes of model-generated text across a plan — write content, paths, reasons, summary.
 *
 * Bounds planner output volume independently of the action count: a plan within the action limit
 * can still carry a few very large writes. Move actions write a snapshot copy of their source, so
 * their byte contribution equals the source content length — without this, moves report 0 bytes
 * and can amplify a single source past the cap.
 *
 * Paths, reasons, and the summary count too: nothing else bounds them, and a delete-only plan
 * generates no content at all. Every field is measured here rather than capped with a schema `.max()`
 * because a schema violation comes back as a tool *error result* that stays in history and re-drives
 * the agent loop — the model retries against a prompt carrying the oversized value, growing the
 * payload each turn until the turn limit trips. Measuring throws out of the run on the first turn.
 *
 * @param plan - The validated consolidation plan
 * @param files - The snapshot map from readAllFiles, used to measure move source content
 *
 * @internal
 */
export function generatedByteSize(plan: ConsolidationPlan, files: Map<string, string>): number {
  const size = (text: string): number => encoder.encode(text).byteLength

  let bytes = 0
  for (const action of plan.actions) {
    bytes += size(action.reason)
    switch (action.action) {
      case 'merge':
        bytes += size(action.content) + size(action.target)
        for (const source of action.sources) bytes += size(source)
        break
      case 'update':
        bytes += size(action.content) + size(action.path)
        break
      case 'delete':
        bytes += size(action.path)
        break
      case 'move': {
        // A move writes the source's content to the new target — count it so N moves from one
        // large source cannot escape the generated-bytes cap
        const canonicalFrom = resolveCanonicalKey(files, action.from)
        const sourceContent = canonicalFrom !== undefined ? files.get(canonicalFrom) : undefined
        bytes += size(action.from) + size(action.to) + (sourceContent !== undefined ? size(sourceContent) : 0)
        break
      }
    }
  }
  bytes += encoder.encode(plan.summary).byteLength
  return bytes
}
