/**
 * Checkpoint system for durable agent execution.
 *
 * **Experimental** — this module is experimental and subject to change in future
 * revisions without notice.
 *
 * A {@link Checkpoint} is a pause-point marker emitted at agent cycle boundaries.
 * It captures the position (which boundary fired) and the cycle index. It does
 * **not** capture conversation state — pair with a `SessionManager` for
 * cross-process state continuity.
 *
 * Positions per ReAct cycle:
 * - `after_model`: model returned tool use; tools have not run yet.
 * - `after_tools`: tools finished; the next model call has not happened yet.
 *
 * Per-tool granularity within a cycle is the tool executor's responsibility.
 *
 * Usage (mirrors interrupts):
 * - Pause: `AgentResult` with `stopReason === 'checkpoint'` and `checkpoint` populated.
 * - Resume: pass back `{ checkpointResume: { checkpoint: ckpt.toJSON() } }`.
 *
 * Precedence:
 * - Interrupt takes precedence over checkpoint: an interrupt during a
 *   checkpointing cycle returns `stopReason === 'interrupt'` and skips `after_tools`.
 * - Cancel takes precedence over checkpoint: a cancel signal at either boundary
 *   returns `stopReason === 'cancelled'`.
 *
 * Notes:
 * - Checkpoints are only emitted on tool-use cycles. A turn with no tool calls
 *   emits no checkpoint; use a `SessionManager` for durability of every turn.
 * - Metrics reset per invocation; aggregate yourself if needed.
 * - `BeforeInvocationEvent` / `AfterInvocationEvent` fire on every resume,
 *   same as interrupts.
 */

import { logger } from '../logging/logger.js'
import type { JSONValue } from '../types/json.js'
import { CheckpointError } from '../errors.js'

/**
 * Current checkpoint schema version. Bumped when the serialized shape changes
 * in a way that is not backward compatible; {@link Checkpoint.fromJSON} rejects
 * checkpoints carrying any other version.
 *
 * @experimental
 */
export const CHECKPOINT_SCHEMA_VERSION = '1.0'

/**
 * Which cycle boundary a checkpoint was emitted at.
 * - `after_model`: model returned tool use; tools have not run yet.
 * - `after_tools`: tools finished; the next model call has not happened yet.
 *
 * @experimental
 */
export type CheckpointPosition = 'after_model' | 'after_tools'

/**
 * Serialized form of a {@link Checkpoint}, as produced by {@link Checkpoint.toJSON}
 * and accepted by {@link Checkpoint.fromJSON}.
 *
 * @experimental
 */
export interface CheckpointData {
  position: CheckpointPosition
  cycleIndex?: number
  snapshot?: Record<string, JSONValue>
  appData?: Record<string, JSONValue>
  schemaVersion?: string
}

/**
 * Resume payload passed back to a checkpointing agent to continue a durable run.
 * The whole invocation argument is this object; the agent consumes it, restores
 * its cycle position, and continues without appending any new input messages.
 *
 * @experimental
 *
 * @example
 * ```typescript
 * const result = await agent.invoke({ checkpointResume: { checkpoint: saved } })
 * ```
 */
export interface CheckpointResumeContent {
  checkpointResume: { checkpoint: CheckpointData }
}

/**
 * Pause-point marker. Treat as opaque — pass it back to resume.
 *
 * @experimental
 */
export class Checkpoint {
  /** Which boundary fired (`after_model` or `after_tools`). */
  readonly position: CheckpointPosition

  /** ReAct loop cycle (0-based). */
  readonly cycleIndex: number

  /** Reserved for forward extensibility. Not read by the SDK; round-trips through serialization. */
  readonly snapshot: Record<string, JSONValue>

  /** Reserved for caller metadata. Not read by the SDK; round-trips through serialization. */
  readonly appData: Record<string, JSONValue>

  /**
   * Schema version. Always set to {@link CHECKPOINT_SCHEMA_VERSION}; not accepted
   * from the constructor so it cannot be forged. Rejects incompatible checkpoints
   * on {@link Checkpoint.fromJSON}.
   */
  readonly schemaVersion: string = CHECKPOINT_SCHEMA_VERSION

  constructor(data: {
    position: CheckpointPosition
    cycleIndex?: number
    snapshot?: Record<string, JSONValue>
    appData?: Record<string, JSONValue>
  }) {
    this.position = data.position
    this.cycleIndex = data.cycleIndex ?? 0
    this.snapshot = data.snapshot ?? {}
    this.appData = data.appData ?? {}
  }

  /**
   * Serializes the checkpoint for persistence.
   */
  toJSON(): Required<CheckpointData> {
    return {
      position: this.position,
      cycleIndex: this.cycleIndex,
      snapshot: this.snapshot,
      appData: this.appData,
      schemaVersion: this.schemaVersion,
    }
  }

  /**
   * Reconstructs a checkpoint from a value produced by {@link Checkpoint.toJSON}.
   * Round-trips cleanly: `Checkpoint.fromJSON(ckpt.toJSON())` type-checks and
   * returns an equivalent checkpoint.
   *
   * @param data - Serialized checkpoint data.
   * @throws CheckpointError If `schemaVersion` does not match the current version.
   */
  static fromJSON(data: CheckpointData): Checkpoint {
    const version = data.schemaVersion ?? ''
    if (version !== CHECKPOINT_SCHEMA_VERSION) {
      throw new CheckpointError(
        `Checkpoints with schema version ${JSON.stringify(version)} are not compatible ` +
          `with current version ${CHECKPOINT_SCHEMA_VERSION}.`
      )
    }

    const knownKeys = new Set(['position', 'cycleIndex', 'snapshot', 'appData', 'schemaVersion'])
    const unknownKeys = Object.keys(data).filter((key) => !knownKeys.has(key))
    if (unknownKeys.length > 0) {
      logger.warn(`unknown_keys=<${unknownKeys.join(', ')}> | ignoring unknown fields in checkpoint data`)
    }

    return new Checkpoint({
      position: data.position,
      ...(data.cycleIndex !== undefined && { cycleIndex: data.cycleIndex }),
      ...(data.snapshot !== undefined && { snapshot: data.snapshot }),
      ...(data.appData !== undefined && { appData: data.appData }),
    })
  }
}
