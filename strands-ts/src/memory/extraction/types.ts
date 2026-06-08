import type { JSONValue } from '../../types/json.js'
import type { MessageData, ContentBlockData } from '../../types/messages.js'
import type { Model } from '../../models/model.js'
import type { LocalAgent } from '../../types/agent.js'

// `keyof` over a union yields only the keys every member shares (none, here). The conditional makes
// it distribute over each member instead, collecting the union of their individual keys.
type DistributedKeyof<Union> = Union extends unknown ? keyof Union : never

/**
 * Content block kinds that {@link MessageFilter} can exclude before messages reach an
 * {@link Extractor} or the no-extractor passthrough.
 *
 * Derived from the SDK's {@link ContentBlockData} union: every member's discriminator key is its
 * kind (`{ toolUse: ... }` → `'toolUse'`, `{ text: string }` → `'text'`), so this tracks new block
 * types automatically instead of drifting. Matches the runtime `_blockKind` used to filter.
 */
export type MemoryContentBlockType = DistributedKeyof<ContentBlockData>

/**
 * Filters content blocks out of messages before extraction.
 *
 * Blocks whose kind is in {@link exclude} are stripped; a message left with no content is dropped
 * entirely. Defaults to excluding tool traffic (`toolUse` / `toolResult`), which is rarely useful as
 * long-term memory and adds noise.
 */
export interface MessageFilter {
  /** Content block kinds to strip before extraction. */
  exclude: MemoryContentBlockType[]
}

/** Default filter: drop tool-call traffic, keep everything else (text, reasoning, media). */
export const DEFAULT_MESSAGE_FILTER: MessageFilter = {
  exclude: ['toolUse', 'toolResult'],
}

/**
 * A discrete entry produced by an {@link Extractor}, ready to be written to a store via its `add`.
 */
export interface ExtractedEntry {
  /** The textual content of the entry. */
  content: string
  /** Optional metadata to associate with the entry. */
  metadata?: Record<string, JSONValue>
}

/**
 * Context passed to {@link Extractor.extract}.
 *
 * Lets the manager hand an extractor a fallback model without the extractor having to be wired to
 * the agent directly. {@link ModelExtractor} uses its own configured model when set, else
 * {@link defaultModel}.
 */
export interface ExtractorContext {
  /** The agent's model, supplied so an extractor can default to it. */
  defaultModel?: Model
}

/**
 * Transforms conversation messages into discrete, searchable entries.
 *
 * Implementations distill raw turns into facts worth remembering. Optional on a store's
 * {@link ExtractionConfig}: when absent, the manager passes messages straight to the store (see the
 * no-extractor passthrough on {@link MemoryStore}), which is the right path for backends that
 * extract server-side.
 */
export interface Extractor {
  /**
   * Extract entries from a batch of messages.
   *
   * @param messages - The filtered messages to extract from
   * @param context - Optional context (e.g. a fallback model)
   * @returns The entries to write to the store
   */
  extract(messages: MessageData[], context?: ExtractorContext): Promise<ExtractedEntry[]>
}

/**
 * Context handed to {@link ExtractionTrigger.attach} so a trigger can wire itself into the agent
 * lifecycle and signal when extraction should run for its store.
 */
export interface ExtractionTriggerContext {
  /** The agent the trigger attaches its hooks to. */
  agent: LocalAgent
  /**
   * Signal that extraction should run for this trigger's store. **Non-blocking and fire-and-forget**:
   * it dispatches the extraction (filter, optional model call, store write) onto the background and
   * returns immediately, so calling it from a lifecycle hook never blocks the agent loop. Processes
   * only messages added since the store's high-water mark (so repeated fires never re-process the
   * same messages); per-store runs are serialized. Errors are handled by the manager (logged,
   * non-fatal). To await completion (graceful shutdown, tests), use {@link MemoryManager.flush}.
   */
  fire: () => void
}

/**
 * Controls when a store's {@link ExtractionConfig} runs.
 *
 * A trigger is a self-attaching value object: {@link attach} wires whatever agent hooks the trigger
 * needs and calls {@link ExtractionTriggerContext.fire} when extraction should happen. Extend this
 * class for custom triggering logic.
 */
export abstract class ExtractionTrigger {
  /** Stable identifier for this trigger kind, used in logging. */
  abstract readonly name: string

  /**
   * Wire this trigger into the agent lifecycle.
   *
   * Called once per store during {@link MemoryManager} initialization. Register hooks on
   * `context.agent` and call `context.fire()` when extraction should run.
   *
   * @param context - The agent to attach to and the fire callback bound to this trigger's store
   */
  abstract attach(context: ExtractionTriggerContext): void
}

/** Normalizes the `trigger` field (a single trigger or an array) to an array. */
export function normalizeTriggers(trigger: ExtractionTrigger | ExtractionTrigger[]): ExtractionTrigger[] {
  return Array.isArray(trigger) ? trigger : [trigger]
}

/**
 * Per-store automatic-extraction configuration.
 *
 * Lives on a store (via {@link MemoryStoreConfig}) so different stores can extract on different
 * schedules and in different styles. {@link trigger} decides *when*; {@link extractor} decides *how*
 * (omit it to pass raw messages straight to the store); {@link filter} prunes content blocks first.
 */
export interface ExtractionConfig {
  /**
   * When to run extraction. A single trigger or an array; an empty array is rejected at
   * construction. Multiple triggers compose (extraction runs whenever any of them fires).
   */
  trigger: ExtractionTrigger | ExtractionTrigger[]
  /**
   * How to turn messages into entries. When set, the store must implement `add` (entries are written
   * to it). When omitted, the manager hands the filtered messages straight to the store's
   * `addMessages` (which the store must then implement) — so backends that extract server-side need
   * no client-side extractor.
   */
  extractor?: Extractor
  /**
   * Content blocks to strip before extraction. Defaults to {@link DEFAULT_MESSAGE_FILTER}
   * (excludes `toolUse` / `toolResult`).
   */
  filter?: MessageFilter
}
