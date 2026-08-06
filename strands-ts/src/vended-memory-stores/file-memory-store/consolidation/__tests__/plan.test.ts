import { describe, it, expect } from 'vitest'
import { ZodError } from 'zod'
import { extractPlan } from '../plan.js'
import { ConsolidationError, StructuredOutputError } from '../../../../errors.js'
import type { JSONValue } from '../../../../types/json.js'

/** Wrap a plan-shaped object as the `structuredOutput` field of an agent result. */
function result(structuredOutput: JSONValue): { structuredOutput?: unknown } {
  return { structuredOutput }
}

describe('extractPlan', () => {
  describe('structured-output gate', () => {
    it('throws StructuredOutputError when no structured output is present', () => {
      expect(() => extractPlan({}, 100)).toThrow(StructuredOutputError)
    })

    it('throws ZodError when the structured output does not match the schema', () => {
      expect(() => extractPlan(result({ not_a_plan: true }), 100)).toThrow(ZodError)
    })

    it('throws ZodError when an action has an unknown discriminator', () => {
      const plan = { actions: [{ action: 'rename', path: 'facts/a.md' }], summary: 'x' }
      expect(() => extractPlan(result(plan), 100)).toThrow(ZodError)
    })

    it('parses a well-formed plan into a typed value', () => {
      const plan = {
        actions: [{ action: 'delete', path: 'facts/a.md', reason: 'stale' }],
        summary: 'pruned one file',
      }
      expect(extractPlan(result(plan), 100)).toEqual(plan)
    })

    it('accepts a plan with an empty action list', () => {
      const plan = { actions: [], summary: 'nothing to do' }
      expect(extractPlan(result(plan), 100)).toEqual(plan)
    })
  })

  describe('action-count guard', () => {
    const threeDeletes = {
      actions: [
        { action: 'delete', path: 'facts/a.md', reason: 'x' },
        { action: 'delete', path: 'facts/b.md', reason: 'x' },
        { action: 'delete', path: 'facts/c.md', reason: 'x' },
      ],
      summary: 'x',
    }

    it('throws ConsolidationError when the action count exceeds the limit', () => {
      expect(() => extractPlan(result(threeDeletes), 2)).toThrow(ConsolidationError)
      expect(() => extractPlan(result(threeDeletes), 2)).toThrow(
        'Consolidation plan exceeds action limit: 3 actions (maxActionsPerPlan: 2)'
      )
    })

    it('accepts a plan exactly at the limit (the guard is > not >=)', () => {
      expect(() => extractPlan(result(threeDeletes), 3)).not.toThrow()
    })
  })

  describe('path lowercasing', () => {
    it('lowercases every path across all action shapes so identity is plain equality', () => {
      const plan = {
        actions: [
          {
            action: 'merge',
            sources: ['Facts/A.md', 'Facts/B.md'],
            target: 'Facts/Merged.md',
            content: 'c',
            reason: 'r',
          },
          { action: 'update', path: 'Facts/C.md', content: 'c', reason: 'r' },
          { action: 'delete', path: 'Facts/D.md', reason: 'r' },
          { action: 'move', from: 'Facts/E.md', to: 'Ops/E.md', reason: 'r' },
        ],
        summary: 's',
      }

      expect(extractPlan(result(plan), 100).actions).toEqual([
        {
          action: 'merge',
          sources: ['facts/a.md', 'facts/b.md'],
          target: 'facts/merged.md',
          content: 'c',
          reason: 'r',
        },
        { action: 'update', path: 'facts/c.md', content: 'c', reason: 'r' },
        { action: 'delete', path: 'facts/d.md', reason: 'r' },
        { action: 'move', from: 'facts/e.md', to: 'ops/e.md', reason: 'r' },
      ])
    })

    it('leaves content and reason untouched while lowercasing paths', () => {
      const plan = {
        actions: [{ action: 'update', path: 'Facts/A.md', content: 'KEEP CASE', reason: 'Keep Case' }],
        summary: 's',
      }

      const [action] = extractPlan(result(plan), 100).actions
      expect(action).toEqual({ action: 'update', path: 'facts/a.md', content: 'KEEP CASE', reason: 'Keep Case' })
    })
  })
})
