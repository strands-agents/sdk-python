import { logger } from '../logging/logger.js'
import { asTokenCount } from './usage.js'
import type { Role, StopReason } from '../types/messages.js'
import type { JSONValue } from '../types/json.js'
import type { Citation, CitationGeneratedContent } from '../types/citations.js'

/**
 * ModelStreamEvent types for Model interactions.
 *
 * This module follows a pattern where "Data" interfaces define the structure
 * for objects, while corresponding classes extend those interfaces with additional
 * functionality and type discrimination.
 */

/**
 * Union type representing all possible streaming events from a model provider.
 * This is a discriminated union where each event has a unique type field.
 *
 * This allows for type-safe event handling using switch statements.
 */
export type ModelStreamEvent =
  | ModelMessageStartEventData
  | ModelContentBlockStartEventData
  | ModelContentBlockDeltaEventData
  | ModelContentBlockStopEventData
  | ModelMessageStopEventData
  | ModelMetadataEventData
  | ModelRedactionEventData

/** Set of all ModelStreamEvent type discriminators. */
const modelStreamEventTypes: ReadonlySet<string> = new Set<ModelStreamEvent['type']>([
  'modelMessageStartEvent',
  'modelContentBlockStartEvent',
  'modelContentBlockDeltaEvent',
  'modelContentBlockStopEvent',
  'modelMessageStopEvent',
  'modelMetadataEvent',
  'modelRedactionEvent',
])

/**
 * Type guard to check if an event with a type discriminator is a ModelStreamEvent.
 * @param event - The event to check
 * @returns true if the event is a ModelStreamEvent
 */
export function isModelStreamEvent(event: { type: string }): event is ModelStreamEvent {
  return modelStreamEventTypes.has(event.type)
}

/**
 * Data for a message start event.
 */
export interface ModelMessageStartEventData {
  /**
   * Discriminator for message start events.
   */
  type: 'modelMessageStartEvent'

  /**
   * The role of the message being started.
   */
  role: Role
}

/**
 * Event emitted when a new message starts in the stream.
 */
export class ModelMessageStartEvent implements ModelMessageStartEventData {
  /**
   * Discriminator for message start events.
   */
  readonly type = 'modelMessageStartEvent' as const

  /**
   * The role of the message being started.
   */
  readonly role: Role

  constructor(data: ModelMessageStartEventData) {
    this.role = data.role
  }
}

/**
 * Data for a content block start event.
 */
export interface ModelContentBlockStartEventData {
  /**
   * Discriminator for content block start events.
   */
  type: 'modelContentBlockStartEvent'

  /**
   * Information about the content block being started.
   * Only present for tool use blocks.
   */
  start?: ContentBlockStart
}

/**
 * Event emitted when a new content block starts in the stream.
 */
export class ModelContentBlockStartEvent implements ModelContentBlockStartEventData {
  /**
   * Discriminator for content block start events.
   */
  readonly type = 'modelContentBlockStartEvent' as const

  /**
   * Information about the content block being started.
   * Only present for tool use blocks.
   */
  readonly start?: ContentBlockStart

  constructor(data: ModelContentBlockStartEventData) {
    if (data.start !== undefined) {
      this.start = data.start
    }
  }
}

/**
 * Data for a content block delta event.
 */
export interface ModelContentBlockDeltaEventData {
  /**
   * Discriminator for content block delta events.
   */
  type: 'modelContentBlockDeltaEvent'

  /**
   * The incremental content update.
   */
  delta: ContentBlockDelta
}

/**
 * Event emitted when there is new content in a content block.
 */
export class ModelContentBlockDeltaEvent implements ModelContentBlockDeltaEventData {
  /**
   * Discriminator for content block delta events.
   */
  readonly type = 'modelContentBlockDeltaEvent' as const

  /**
   * Index of the content block being updated.
   */
  readonly contentBlockIndex?: number

  /**
   * The incremental content update.
   */
  readonly delta: ContentBlockDelta

  constructor(data: ModelContentBlockDeltaEventData) {
    this.delta = data.delta
  }
}

/**
 * Data for a content block stop event.
 */
export interface ModelContentBlockStopEventData {
  /**
   * Discriminator for content block stop events.
   */
  type: 'modelContentBlockStopEvent'
}

/**
 * Event emitted when a content block completes.
 */
export class ModelContentBlockStopEvent implements ModelContentBlockStopEventData {
  /**
   * Discriminator for content block stop events.
   */
  readonly type = 'modelContentBlockStopEvent' as const

  constructor(_data: ModelContentBlockStopEventData) {}
}

/**
 * Data for a message stop event.
 */
export interface ModelMessageStopEventData {
  /**
   * Discriminator for message stop events.
   */
  type: 'modelMessageStopEvent'

  /**
   * Reason why generation stopped.
   */
  stopReason: StopReason

  /**
   * Additional provider-specific response fields.
   */
  additionalModelResponseFields?: JSONValue
}

/**
 * Event emitted when the message completes.
 */
export class ModelMessageStopEvent implements ModelMessageStopEventData {
  /**
   * Discriminator for message stop events.
   */
  readonly type = 'modelMessageStopEvent' as const

  /**
   * Reason why generation stopped.
   */
  readonly stopReason: StopReason

  /**
   * Additional provider-specific response fields.
   */
  readonly additionalModelResponseFields?: JSONValue

  constructor(data: ModelMessageStopEventData) {
    this.stopReason = data.stopReason
    if (data.additionalModelResponseFields !== undefined) {
      this.additionalModelResponseFields = data.additionalModelResponseFields
    }
  }
}

/**
 * Data for a metadata event.
 */
export interface ModelMetadataEventData {
  /**
   * Discriminator for metadata events.
   */
  type: 'modelMetadataEvent'

  /**
   * Token usage information.
   */
  usage?: Usage

  /**
   * Performance metrics.
   */
  metrics?: Metrics

  /**
   * Trace information for observability.
   */
  trace?: unknown
}

/**
 * Event containing metadata about the stream.
 * Includes usage statistics, performance metrics, and trace information.
 */
export class ModelMetadataEvent implements ModelMetadataEventData {
  /**
   * Discriminator for metadata events.
   */
  readonly type = 'modelMetadataEvent' as const

  /**
   * Token usage information.
   */
  readonly usage?: Usage

  /**
   * Performance metrics.
   */
  readonly metrics?: Metrics

  /**
   * Trace information for observability.
   */
  readonly trace?: unknown

  constructor(data: ModelMetadataEventData) {
    if (data.usage !== undefined) {
      this.usage = data.usage
    }
    if (data.metrics !== undefined) {
      this.metrics = data.metrics
    }
    if (data.trace !== undefined) {
      this.trace = data.trace
    }
  }
}

/**
 * Information about input content redaction.
 * Does not include redactedContent since the original input is already available
 * in the messages array from BeforeModelCallEvent.
 */
export interface RedactInputContent {
  /**
   * The content to replace the redacted input with.
   */
  replaceContent: string
}

/**
 * Information about output content redaction.
 * May include the original content if captured during streaming.
 */
export interface RedactOutputContent {
  /**
   * The original content that was blocked by guardrails.
   * May not be available for all providers.
   */
  redactedContent?: string

  /**
   * The content to replace the redacted output with.
   */
  replaceContent: string
}

/**
 * Data for a redact event.
 * Emitted when guardrails block content and redaction is enabled.
 */
export interface ModelRedactionEventData {
  /**
   * Discriminator for redact events.
   */
  type: 'modelRedactionEvent'

  /**
   * Input redaction information (when input is blocked).
   */
  inputRedaction?: RedactInputContent

  /**
   * Output redaction information (when output is blocked).
   */
  outputRedaction?: RedactOutputContent
}

/**
 * Event emitted when guardrails block content and trigger redaction.
 */
export class ModelRedactionEvent implements ModelRedactionEventData {
  /**
   * Discriminator for redact events.
   */
  readonly type = 'modelRedactionEvent' as const

  /**
   * Input redaction information (when input is blocked).
   */
  readonly inputRedaction?: RedactInputContent

  /**
   * Output redaction information (when output is blocked).
   */
  readonly outputRedaction?: RedactOutputContent

  constructor(data: ModelRedactionEventData) {
    if (data.inputRedaction !== undefined) {
      this.inputRedaction = data.inputRedaction
    }
    if (data.outputRedaction !== undefined) {
      this.outputRedaction = data.outputRedaction
    }
  }
}

/**
 * Information about a content block that is starting.
 * Currently only represents tool use starts.
 */
export type ContentBlockStart = ToolUseStart

/**
 * Information about a tool use that is starting.
 */
export interface ToolUseStart {
  /**
   * Discriminator for tool use start.
   */
  type: 'toolUseStart'

  /**
   * The name of the tool being used.
   */
  name: string

  /**
   * Unique identifier for this tool use.
   */
  toolUseId: string

  /**
   * Reasoning signature from thinking models (e.g., Gemini).
   * Must be preserved and sent back to the model for multi-turn tool use.
   */
  reasoningSignature?: string
}

/**
 * A delta (incremental chunk) of content within a content block.
 * Can be text, tool use input, or reasoning content.
 *
 * This is a discriminated union for type-safe delta handling.
 */
export type ContentBlockDelta = TextDelta | ToolUseInputDelta | ReasoningContentDelta | CitationsDelta

/**
 * Text delta within a content block.
 * Represents incremental text content from the model.
 */
export interface TextDelta {
  /**
   * Discriminator for text delta.
   */
  type: 'textDelta'

  /**
   * Incremental text content.
   */
  text: string
}

/**
 * Tool use input delta within a content block.
 * Represents incremental tool input being generated.
 */
export interface ToolUseInputDelta {
  /**
   * Discriminator for tool use input delta.
   */
  type: 'toolUseInputDelta'

  /**
   * Partial JSON string representing the tool input.
   */
  input: string
}

/**
 * Reasoning content delta within a content block.
 * Represents incremental reasoning or thinking content.
 */
export interface ReasoningContentDelta {
  /**
   * Discriminator for reasoning delta.
   */
  type: 'reasoningContentDelta'

  /**
   * Incremental reasoning text.
   */
  text?: string

  /**
   * Incremental signature data.
   */
  signature?: string

  /**
   * Incremental redacted content data.
   */
  redactedContent?: Uint8Array
}

/**
 * Citations content delta within a content block.
 * Represents a citations content block from the model.
 */
export interface CitationsDelta {
  /**
   * Discriminator for citations content delta.
   */
  type: 'citationsDelta'

  /**
   * Array of citations linking generated content to source locations.
   */
  citations: Citation[]

  /**
   * The generated content associated with these citations.
   */
  content: CitationGeneratedContent[]
}

/**
 * Token usage statistics for a model invocation.
 *
 * The counts are **disjoint**: every billed token falls into exactly one of `inputTokens`,
 * `outputTokens`, `cacheReadInputTokens`, or `cacheWriteInputTokens`. Providers disagree on
 * this — some report the full prompt in their input field with cache tokens broken out as a
 * subset of it — so each adapter normalizes via `normalizeUsage` from `models/usage.ts`.
 *
 * The invariant every adapter upholds:
 *
 * ```
 * inputTokens + outputTokens + cacheReadInputTokens + cacheWriteInputTokens === totalTokens
 * ```
 *
 * which makes cost a plain weighted sum:
 *
 * ```
 * cost = inputTokens * inputRate
 *      + cacheReadInputTokens * cacheReadRate    // typically ~0.1x inputRate
 *      + cacheWriteInputTokens * cacheWriteRate  // typically 1.25x-2x inputRate
 *      + outputTokens * outputRate
 * ```
 */
export interface Usage {
  /**
   * Number of **net new** prompt tokens — neither read from nor written to the prompt cache.
   * Excludes `cacheReadInputTokens` and `cacheWriteInputTokens`; add all three for the full prompt size.
   */
  inputTokens: number

  /**
   * Number of tokens in the output (completion), including any reasoning tokens.
   */
  outputTokens: number

  /**
   * Total tokens billed for the request, across all four counters.
   */
  totalTokens: number

  /**
   * Number of prompt tokens served from the cache, billed at a discount. Disjoint from `inputTokens`.
   */
  cacheReadInputTokens?: number

  /**
   * Number of prompt tokens written to the cache, billed at a premium. Disjoint from `inputTokens`.
   */
  cacheWriteInputTokens?: number

  /**
   * Number of tokens spent on internal reasoning. Unlike the cache counters this is a **subset**
   * of `outputTokens` (already billed and counted there), so it must not be added to the total.
   */
  reasoningOutputTokens?: number
}

/**
 * Performance metrics for a model invocation.
 */
export interface Metrics {
  /**
   * Latency in milliseconds.
   */
  latencyMs: number

  /**
   * Time to first byte in milliseconds.
   * Latency from sending the model request to receiving the first content chunk.
   */
  timeToFirstByteMs?: number
}

/**
 * Returns the full prompt size, including tokens served from or written to the cache.
 *
 * `Usage.inputTokens` counts only net new tokens, but cached tokens occupy the model's context
 * window and are part of the prompt that was sent, so anything reasoning about prompt size
 * (rather than cost) needs all three counters.
 *
 * @param usage - The usage reported by the model
 * @returns The total number of prompt tokens sent to the model
 *
 * @internal
 */
export function promptTokenCount(usage: Usage): number {
  return (
    asTokenCount(usage.inputTokens) +
    asTokenCount(usage.cacheReadInputTokens) +
    asTokenCount(usage.cacheWriteInputTokens)
  )
}

/**
 * Returns how much of the context window a model call consumed, prompt plus generated output.
 *
 * Cached tokens occupy the context window like any other, so the whole prompt counts, not just the
 * net new tokens `inputTokens` reports.
 *
 * Usage recorded before cache tokens became separate counters counted them inside `inputTokens`.
 * Such a payload is only sometimes distinguishable from a current one, so no attempt is made here:
 * it reads high by the size of its cache hit, which is the safe direction — it compacts a
 * conversation early rather than overflowing the window — and it corrects itself after the first
 * model call of a resumed session.
 *
 * @param usage - The usage reported by the model
 * @returns The number of context-window tokens the call accounted for
 *
 * @internal
 */
export function contextTokenCount(usage: Usage): number {
  return promptTokenCount(usage) + asTokenCount(usage.outputTokens)
}

/** The token counters {@link Usage} declares, in the order they are reported. @internal */
const USAGE_COUNTERS = [
  'inputTokens',
  'outputTokens',
  'totalTokens',
  'cacheReadInputTokens',
  'cacheWriteInputTokens',
  'reasoningOutputTokens',
] as const

/**
 * Reads one counter out of a payload, whatever shape it arrived in.
 *
 * @param payload - The usage to read, which need not be an object
 * @param field - The name of the counter to read
 * @returns The raw value reported for the counter, or `undefined` when it is absent or unreadable
 *
 * @internal
 */
function usageCount(payload: unknown, field: string): unknown {
  // Session state is user-editable JSON, so anything that is not an object of counts is read as
  // no usage at all. An accessor that throws reads as absent rather than failing the invocation
  // the counts merely describe.
  if (typeof payload !== 'object' || payload === null) {
    return undefined
  }
  try {
    return (payload as Record<string, unknown>)[field]
  } catch {
    return undefined
  }
}

/**
 * Reads a running total this SDK computed, which has no ceiling a reported count would have.
 *
 * @internal
 */
function accumulated(count: number | undefined): number {
  return typeof count === 'number' && Number.isInteger(count) && count >= 0 ? count : asTokenCount(count)
}

/**
 * Reads the counters {@link Usage} declares out of an arbitrary payload.
 *
 * A model implementation and a persisted session both reach the SDK as untyped data, so every
 * count is coerced. The result is a new object, leaving the caller's payload alone and keeping
 * keys the contract does not declare out of the SDK's own type.
 *
 * @param payload - The usage to read, which need not be an object
 * @returns The declared counters as non-negative integers, with absent ones left absent
 *
 * @internal
 */
function coerceUsageCounters(payload: unknown): Usage {
  const usage: Usage = {
    inputTokens: asTokenCount(usageCount(payload, 'inputTokens')),
    outputTokens: asTokenCount(usageCount(payload, 'outputTokens')),
    totalTokens: asTokenCount(usageCount(payload, 'totalTokens')),
  }
  for (const key of ['cacheReadInputTokens', 'cacheWriteInputTokens', 'reasoningOutputTokens'] as const) {
    // A counter reported as null is skipped, so an absent one stays absent rather than reading as
    // a real zero.
    const raw = usageCount(payload, key)
    if (raw != null) {
      usage[key] = asTokenCount(raw)
    }
  }
  return usage
}

/**
 * Reads a model's reported usage into well-formed counters.
 *
 * A custom {@link Model} implementation can report a count as anything, and an unusable one is
 * carried no further: arithmetic on a string silently concatenates, and a non-number is dropped
 * outright by the OpenTelemetry counters. Coercing a count can still satisfy the disjointness
 * invariant, so {@link warnIfUsageInconsistent} would not see it, and it is reported here instead.
 *
 * @param reported - The usage reported by the model
 * @returns The same counters as non-negative integers, with absent ones left absent
 *
 * @internal
 */
export function readReportedUsage(reported: unknown): Usage {
  // Every counter is read once, into a snapshot both the coercion and the check below read from.
  // Reading one twice would report the difference between the two reads as an unusable count.
  const reportedCounts: Record<string, unknown> = {}
  for (const key of USAGE_COUNTERS) {
    reportedCounts[key] = usageCount(reported, key)
  }
  const usage = coerceUsageCounters(reportedCounts)
  const unusable = USAGE_COUNTERS.filter((key) => {
    const raw = reportedCounts[key]
    return raw != null && usage[key] !== raw
  })
  if (unusable.length > 0) {
    // The coerced counters are logged rather than the payload, which a model can report as
    // something that does not serialize at all, such as a BigInt or a self-referencing object.
    logger.warn(
      `fields=<${[...unusable].sort().join(', ')}>, usage=<${JSON.stringify(usage)}> | ` +
        `model reported token counts that are not usable numbers`
    )
  }
  return usage
}

/**
 * Logs when a model's usage violates the {@link Usage} contract.
 *
 * `Usage` requires its four billed counters to be disjoint and to sum to `totalTokens`. The
 * built-in adapters normalize to this via `normalizeUsage`; a custom {@link Model} implementation
 * reporting cache tokens as a subset of `inputTokens` would silently inflate cost calculations,
 * so surface it instead.
 *
 * @param usage - The usage reported by the model
 *
 * @internal
 */
export function warnIfUsageInconsistent(usage: Usage): void {
  const counted =
    asTokenCount(usage.inputTokens) +
    asTokenCount(usage.outputTokens) +
    asTokenCount(usage.cacheReadInputTokens) +
    asTokenCount(usage.cacheWriteInputTokens)
  const totalTokens = asTokenCount(usage.totalTokens)
  if (counted !== totalTokens) {
    logger.warn(
      `counted_tokens=<${counted}>, total_tokens=<${totalTokens}> | model usage does not ` +
        `satisfy input + output + cacheRead + cacheWrite == total ` +
        `| derived cost may be inaccurate`
    )
  }
}

/**
 * Repairs a `totalTokens` persisted before cache tokens became separate counters.
 *
 * A session written by an earlier version reports `totalTokens` as only the input and output
 * counts, so resuming it and adding live counts yields a total that no combination of counters
 * accounts for.
 *
 * Only `totalTokens` is repaired, and only where the cache count exceeds `inputTokens`: a subset
 * cannot be larger than the set containing it, so that case proves the counters are already
 * disjoint and the missing total is recoverable by addition. Whether `inputTokens` itself contained
 * the cache count is never recoverable, so it is left as persisted rather than guessed at.
 *
 * @param usage - The persisted usage to repair
 * @returns A new usage carrying the repaired counters
 *
 * @internal
 */
export function repairPersistedUsage(usage: unknown): Usage {
  const repaired = coerceUsageCounters(usage)

  const cacheTokens = (repaired.cacheReadInputTokens ?? 0) + (repaired.cacheWriteInputTokens ?? 0)
  if (cacheTokens > repaired.inputTokens && repaired.totalTokens === repaired.inputTokens + repaired.outputTokens) {
    repaired.totalTokens += cacheTokens
    logger.debug(`usage=<${JSON.stringify(repaired)}> | repaired a total persisted before the disjoint token contract`)
  }
  return repaired
}

/**
 * Accumulates token usage from a source into a target, mutating the target in place.
 *
 * @param target - Usage object to accumulate into
 * @param source - Usage object to add from
 *
 * @internal
 */
export function accumulateUsage(target: Usage, source: Usage): void {
  // The required counters are read defensively because restored session state reaches this
  // function as an unvalidated cast; a missing counter would otherwise poison the target with NaN.
  target.inputTokens = accumulated(target.inputTokens) + asTokenCount(source.inputTokens)
  target.outputTokens = accumulated(target.outputTokens) + asTokenCount(source.outputTokens)
  target.totalTokens = accumulated(target.totalTokens) + asTokenCount(source.totalTokens)
  if (source.cacheReadInputTokens !== undefined) {
    target.cacheReadInputTokens = accumulated(target.cacheReadInputTokens) + asTokenCount(source.cacheReadInputTokens)
  }
  if (source.cacheWriteInputTokens !== undefined) {
    target.cacheWriteInputTokens =
      accumulated(target.cacheWriteInputTokens) + asTokenCount(source.cacheWriteInputTokens)
  }
  if (source.reasoningOutputTokens !== undefined) {
    target.reasoningOutputTokens =
      accumulated(target.reasoningOutputTokens) + asTokenCount(source.reasoningOutputTokens)
  }
}

/**
 * Creates a Usage object with all counters zeroed.
 *
 * @returns A Usage object with zeroed counters
 *
 * @internal
 */
export function createEmptyUsage(): Usage {
  return {
    inputTokens: 0,
    outputTokens: 0,
    totalTokens: 0,
  }
}
