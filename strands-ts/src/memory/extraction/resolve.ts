import type { MemoryStore } from '../types.js'
import { IntervalTrigger } from './triggers.js'
import { ModelExtractor } from './model-extractor.js'
import {
  DEFAULT_MEMORY_MESSAGE_FILTER,
  type ExtractionConfig,
  type ExtractionTrigger,
  type Extractor,
  type MemoryMessageFilter,
} from './types.js'

/** Default cadence when an {@link ExtractionConfig} omits its `trigger`: extract every N turns. */
export const DEFAULT_EXTRACTION_TRIGGER_TURNS = 5

/**
 * An {@link ExtractionConfig} with every field resolved to a concrete value, ready to drive
 * extraction. Produced by {@link resolveExtractionConfig} so the {@link MemoryManager} and
 * {@link ExtractionCoordinator} never have to re-apply defaults or normalize shapes.
 */
export interface ResolvedExtractionConfig {
  /** Normalized to an array (a single trigger is wrapped). Never empty for a resolved config. */
  triggers: ExtractionTrigger[]
  /** The extractor to distill facts, or `undefined` for the raw-message `addMessages` passthrough. */
  extractor?: Extractor
  /** The content-block filter applied before extraction. */
  filter: MemoryMessageFilter
}

/**
 * Resolves a store's `extraction` setting into a {@link ResolvedExtractionConfig}, applying defaults.
 *
 * The single place the `boolean | ExtractionConfig` shorthand is interpreted: `false`/omitted is off
 * (returns `undefined`), `true` enables all defaults, an {@link ExtractionConfig} defaults its unset
 * fields. Defaults are: trigger every {@link DEFAULT_EXTRACTION_TRIGGER_TURNS} turns (an explicit empty
 * array is left empty for the {@link MemoryManager} to reject); a capability-based extractor (add-only
 * stores get a {@link ModelExtractor}; stores with `addMessages` get the no-extractor passthrough, with
 * no model call); and {@link DEFAULT_MEMORY_MESSAGE_FILTER}.
 *
 * @param extraction - The store's `extraction` setting
 * @param store - The store, inspected for its write sinks to pick the default extractor
 * @returns The resolved config, or `undefined` when extraction is disabled
 */
export function resolveExtractionConfig(
  extraction: boolean | ExtractionConfig | undefined,
  store: Pick<MemoryStore, 'add' | 'addMessages'>
): ResolvedExtractionConfig | undefined {
  if (!extraction) {
    return undefined
  }
  const config: ExtractionConfig = extraction === true ? {} : extraction

  const triggers =
    config.trigger === undefined
      ? [new IntervalTrigger({ turns: DEFAULT_EXTRACTION_TRIGGER_TURNS })]
      : Array.isArray(config.trigger)
        ? config.trigger
        : [config.trigger]

  let extractor = config.extractor
  if (extractor === undefined) {
    // A store with addMessages (or both sinks) takes the passthrough - no implicit model call. Only an
    // add-only store defaults to distilling facts with a ModelExtractor.
    const hasAdd = typeof store.add === 'function'
    const hasAddMessages = typeof store.addMessages === 'function'
    if (hasAdd && !hasAddMessages) {
      extractor = new ModelExtractor()
    }
  }

  return {
    triggers,
    ...(extractor !== undefined && { extractor }),
    filter: config.filter ?? DEFAULT_MEMORY_MESSAGE_FILTER,
  }
}
