import { afterEach, describe, expect, it, vi } from 'vitest'
import { CHECKPOINT_SCHEMA_VERSION, Checkpoint, type CheckpointData } from '../checkpoint.js'
import { CheckpointError } from '../../errors.js'
import { logger } from '../../logging/logger.js'

describe('Checkpoint serialization', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('round-trips through toJSON/fromJSON', () => {
    const checkpoint = new Checkpoint({ position: 'after_model', cycleIndex: 1 })

    const restored = Checkpoint.fromJSON(checkpoint.toJSON())

    expect(restored.position).toBe('after_model')
    expect(restored.cycleIndex).toBe(1)
    expect(restored.schemaVersion).toBe(CHECKPOINT_SCHEMA_VERSION)
  })

  it('always sets schemaVersion to the current constant', () => {
    const checkpoint = new Checkpoint({ position: 'after_tools' })
    expect(checkpoint.schemaVersion).toBe(CHECKPOINT_SCHEMA_VERSION)
  })

  it('defaults cycleIndex to 0', () => {
    const checkpoint = new Checkpoint({ position: 'after_model' })
    expect(checkpoint.cycleIndex).toBe(0)
  })

  it('throws CheckpointError on schema version mismatch', () => {
    const data = { ...new Checkpoint({ position: 'after_model' }).toJSON(), schemaVersion: '0.0' }
    expect(() => Checkpoint.fromJSON(data)).toThrow(CheckpointError)
    expect(() => Checkpoint.fromJSON(data)).toThrow(/not compatible with current version/)
  })

  it('throws CheckpointError when schemaVersion is missing', () => {
    const data: CheckpointData = { position: 'after_model', cycleIndex: 0 }
    expect(() => Checkpoint.fromJSON(data)).toThrow(CheckpointError)
    expect(() => Checkpoint.fromJSON(data)).toThrow(/not compatible with current version/)
  })

  it('warns and ignores unknown fields', () => {
    const warnSpy = vi.spyOn(logger, 'warn').mockImplementation(() => {})
    const data = { ...new Checkpoint({ position: 'after_tools' }).toJSON(), unknownFutureField: 'something' }

    const restored = Checkpoint.fromJSON(data)

    expect(restored.position).toBe('after_tools')
    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('unknownFutureField'))
    // The unknown field is ignored, not carried onto the reconstructed checkpoint.
    expect(restored).not.toHaveProperty('unknownFutureField')
    expect(restored.toJSON()).not.toHaveProperty('unknownFutureField')
  })

  it('ignores legacy snapshot/appData fields from older serialized checkpoints', () => {
    // Checkpoints serialized before snapshot/appData were removed must still
    // deserialize: the now-unknown keys are warned about and dropped.
    const warnSpy = vi.spyOn(logger, 'warn').mockImplementation(() => {})
    const data = {
      position: 'after_tools',
      cycleIndex: 2,
      snapshot: { messages: [] },
      appData: { workflowId: 'wf-123' },
      schemaVersion: CHECKPOINT_SCHEMA_VERSION,
    } as unknown as CheckpointData

    const restored = Checkpoint.fromJSON(data)

    expect(restored.position).toBe('after_tools')
    expect(restored.cycleIndex).toBe(2)
    expect(restored).not.toHaveProperty('snapshot')
    expect(restored).not.toHaveProperty('appData')
    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('snapshot'))
  })
})
