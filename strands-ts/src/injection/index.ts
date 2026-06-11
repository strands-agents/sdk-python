/**
 * Generic context-injection engine.
 *
 * Only the configuration types are public. The delivery primitives (`createInjectionMiddleware`,
 * `foldIntoLastUserMessage`, `isUserTurn`, `resolveTrigger`) are internal plumbing: consumers reach
 * injection through a {@link ContextInjector} plugin (or, for memory, `MemoryManager`'s `injection`
 * config), not by wiring middleware themselves.
 */
export type { InjectionConfig, InjectionTrigger, InjectionContext } from './types.js'
