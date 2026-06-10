export { MemoryManager } from './memory-manager.js'
export type {
  MemoryEntry,
  MemoryStore,
  MemoryStoreConfig,
  SearchOptions,
  AddMessagesContext,
  MemorySearchOptions,
  MemoryAddOptions,
  MemoryToolConfig,
  MemoryAddToolConfig,
  MemoryManagerConfig,
} from './types.js'

export { ExtractionTrigger } from './extraction/types.js'
export { InvocationTrigger, IntervalTrigger } from './extraction/triggers.js'
export type { IntervalTriggerOptions } from './extraction/triggers.js'
export { ModelExtractor } from './extraction/model-extractor.js'
export type { ModelExtractorOptions } from './extraction/model-extractor.js'
export type {
  ExtractionConfig,
  Extractor,
  ExtractorContext,
  ExtractionResult,
  ExtractionTriggerContext,
  MemoryMessageFilter,
  MemoryContentBlockType,
} from './extraction/types.js'
