import { describe, it, expect, vi } from 'vitest'
import { handoffToUser } from '../handoff-to-user.js'
import { INTERRUPT_NAME, MAX_OPTION_LENGTH, MAX_OPTIONS_COUNT, MAX_QUESTION_LENGTH } from '../types.js'
import type { ToolContext } from '../../../tools/tool.js'
import type { JSONValue } from '../../../types/json.js'
import { InterruptError, Interrupt } from '../../../interrupt.js'
import { createMockAgent } from '../../../__fixtures__/agent-helpers.js'

/**
 * Build a ToolContext whose `interrupt` records every call. If a preloaded
 * response is present, it is returned; otherwise `interrupt` throws an
 * InterruptError just like the real SDK does on the halting path.
 */
function makeContext(preloadedResponse?: JSONValue): {
  context: ToolContext
  calls: Array<{ name: string; reason: unknown }>
} {
  const calls: Array<{ name: string; reason: unknown }> = []
  const interrupt = vi.fn(<T>(params: { name: string; reason?: JSONValue }): T => {
    calls.push({ name: params.name, reason: params.reason })
    if (preloadedResponse !== undefined) {
      return preloadedResponse as T
    }
    throw new InterruptError(
      new Interrupt({
        id: `tool:test:${params.name}`,
        name: params.name,
        ...(params.reason !== undefined && { reason: params.reason }),
        source: 'tool',
      })
    )
  })
  const agent = createMockAgent()
  const context: ToolContext = {
    toolUse: { name: 'handoff_to_user', toolUseId: 'test-tool-use-id', input: {} },
    agent,
    invocationState: {},
    interrupt: interrupt as ToolContext['interrupt'],
  }
  return { context, calls }
}

describe('handoffToUser', () => {
  describe('input validation (Zod schema)', () => {
    it('rejects an empty question', async () => {
      const { context } = makeContext()
      await expect(handoffToUser.invoke({ question: '' }, context)).rejects.toThrow(/non-empty/i)
    })

    it('rejects an oversized question', async () => {
      const { context } = makeContext()
      const q = 'x'.repeat(MAX_QUESTION_LENGTH + 1)
      await expect(handoffToUser.invoke({ question: q }, context)).rejects.toThrow(/maximum/i)
    })

    it('rejects too many options', async () => {
      const { context } = makeContext()
      const options = Array.from({ length: MAX_OPTIONS_COUNT + 1 }, (_, i) => `opt${i}`)
      await expect(handoffToUser.invoke({ question: 'q', options }, context)).rejects.toThrow(/options count/i)
    })

    it('rejects an oversized option entry', async () => {
      const { context } = makeContext()
      const options = ['ok', 'x'.repeat(MAX_OPTION_LENGTH + 1)]
      await expect(handoffToUser.invoke({ question: 'q', options }, context)).rejects.toThrow()
    })

    it('rejects an empty option entry', async () => {
      const { context } = makeContext()
      await expect(handoffToUser.invoke({ question: 'q', options: ['', 'b'] }, context)).rejects.toThrow()
    })

    it('rejects an empty options list', async () => {
      const { context } = makeContext()
      await expect(handoffToUser.invoke({ question: 'q', options: [] }, context)).rejects.toThrow(/at least one/i)
    })

    it('rejects duplicate options', async () => {
      const { context } = makeContext()
      await expect(handoffToUser.invoke({ question: 'q', options: ['a', 'a'] }, context)).rejects.toThrow(/duplicates/i)
    })

    it('rejects duplicate options that differ only by surrounding whitespace', async () => {
      const { context } = makeContext()
      await expect(handoffToUser.invoke({ question: 'q', options: ['yes', 'yes '] }, context)).rejects.toThrow(
        /duplicates/i
      )
    })

    it('rejects whitespace-only option entries', async () => {
      const { context } = makeContext()
      await expect(handoffToUser.invoke({ question: 'q', options: ['   ', 'b'] }, context)).rejects.toThrow()
    })

    it('rejects a question with no answer channel (no options, allow_free_text=false)', async () => {
      const { context } = makeContext()
      await expect(handoffToUser.invoke({ question: 'Q?', allow_free_text: false }, context)).rejects.toThrow(
        /options or free text/i
      )
    })

    it('rejects a boolean question (guards against future string coercion)', async () => {
      const { context } = makeContext()
      // The Python side rejects a bool for `question` because bool is not str;
      // this test asserts the Zod schema does not silently coerce a boolean
      // into "true"/"false", so the two sides stay in lockstep.
      await expect(handoffToUser.invoke({ question: true as unknown as string }, context)).rejects.toThrow()
    })

    it('rejects a boolean option entry (guards against future string coercion)', async () => {
      const { context } = makeContext()
      await expect(
        handoffToUser.invoke({ question: 'q', options: [true as unknown as string, 'b'] }, context)
      ).rejects.toThrow()
    })
  })

  describe('interrupt emission', () => {
    it('emits an interrupt with the documented event shape for a free-text question', async () => {
      const { context, calls } = makeContext()
      await expect(handoffToUser.invoke({ question: "What's your name?" }, context)).rejects.toBeInstanceOf(
        InterruptError
      )
      expect(calls).toHaveLength(1)
      expect(calls[0]).toEqual({
        name: INTERRUPT_NAME,
        reason: {
          question: "What's your name?",
          options: null,
          allow_free_text: true,
        },
      })
    })

    it('passes options and allow_free_text through on the reason payload', async () => {
      const { context, calls } = makeContext()
      await expect(
        handoffToUser.invoke(
          { question: 'Which env?', options: ['dev', 'staging', 'prod'], allow_free_text: false },
          context
        )
      ).rejects.toBeInstanceOf(InterruptError)
      expect(calls[0]!.reason).toEqual({
        question: 'Which env?',
        options: ['dev', 'staging', 'prod'],
        allow_free_text: false,
      })
    })

    it('halts execution via InterruptError so the SDK agent loop pauses', async () => {
      const { context } = makeContext()
      let raised: unknown = null
      try {
        await handoffToUser.invoke({ question: 'Q?' }, context)
      } catch (err) {
        raised = err
      }
      expect(raised).toBeInstanceOf(InterruptError)
      expect((raised as InterruptError).interrupts[0]!.name).toBe(INTERRUPT_NAME)
    })
  })

  describe('response coercion on resume', () => {
    it('wraps a bare string as { answer }', async () => {
      const { context } = makeContext('Alice')
      const result = await handoffToUser.invoke({ question: 'Name?' }, context)
      expect(result).toEqual({ answer: 'Alice' })
    })

    it('passes a well-shaped object through', async () => {
      const { context } = makeContext({ answer: 'prod', chose: 'prod' })
      const result = await handoffToUser.invoke({ question: 'Env?', options: ['dev', 'prod'] }, context)
      expect(result).toEqual({ answer: 'prod', chose: 'prod' })
    })

    it('rejects a non-string answer', async () => {
      const { context } = makeContext({ answer: 42 } as unknown as JSONValue)
      await expect(handoffToUser.invoke({ question: 'Q?' }, context)).rejects.toThrow(/answer/i)
    })

    it('rejects a non-string chose', async () => {
      const { context } = makeContext({ answer: 'ok', chose: 5 } as unknown as JSONValue)
      await expect(handoffToUser.invoke({ question: 'Q?' }, context)).rejects.toThrow(/chose/i)
    })

    it('rejects a numeric response entirely', async () => {
      const { context } = makeContext(42 as unknown as JSONValue)
      await expect(handoffToUser.invoke({ question: 'Q?' }, context)).rejects.toThrow(/string or an object/i)
    })

    it('rejects a bare-string answer that exceeds the size cap', async () => {
      const { context } = makeContext('x'.repeat(MAX_QUESTION_LENGTH + 1))
      await expect(handoffToUser.invoke({ question: 'Q?' }, context)).rejects.toThrow(/maximum/i)
    })

    it('rejects an oversized answer inside a dict', async () => {
      const { context } = makeContext({ answer: 'x'.repeat(MAX_QUESTION_LENGTH + 1) })
      await expect(handoffToUser.invoke({ question: 'Q?' }, context)).rejects.toThrow(/maximum/i)
    })

    it('rejects an oversized chose inside a dict', async () => {
      const { context } = makeContext({ answer: 'ok', chose: 'x'.repeat(MAX_OPTION_LENGTH + 1) })
      await expect(handoffToUser.invoke({ question: 'Q?' }, context)).rejects.toThrow(/maximum/i)
    })
  })
})
