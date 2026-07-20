import { describe, expect, it, vi } from 'vitest'
import { JsonBlock, ReasoningBlock, TextBlock, ToolResultBlock, ToolUseBlock } from '../../types/messages.js'
import { ModelError } from '../../errors.js'
import { HarnessStreamDecoder } from '../stream-decoder.js'
import type { AgentCoreHarnessEventData } from '../events.js'
import type { DecodedHarnessTurn } from '../stream-decoder.js'

/** Builds non-error Harness events used by decoder tests. */
const event = {
  messageStart: (role: 'assistant' | 'user' = 'assistant'): AgentCoreHarnessEventData =>
    ({ messageStart: { role } }) as AgentCoreHarnessEventData,
  textDelta: (text: string): AgentCoreHarnessEventData =>
    ({ contentBlockDelta: { delta: { text } } }) as AgentCoreHarnessEventData,
  toolUseStart: (toolUseId: string, name: string): AgentCoreHarnessEventData =>
    ({ contentBlockStart: { start: { toolUse: { toolUseId, name } } } }) as AgentCoreHarnessEventData,
  toolUseDelta: (input: string): AgentCoreHarnessEventData =>
    ({ contentBlockDelta: { delta: { toolUse: { input } } } }) as AgentCoreHarnessEventData,
  toolResultStart: (toolUseId: string, status: 'success' | 'error'): AgentCoreHarnessEventData =>
    ({ contentBlockStart: { start: { toolResult: { toolUseId, status } } } }) as AgentCoreHarnessEventData,
  toolResultTextDelta: (text: string): AgentCoreHarnessEventData =>
    ({ contentBlockDelta: { delta: { toolResult: [{ text }] } } }) as AgentCoreHarnessEventData,
  toolResultJsonDelta: (json: unknown): AgentCoreHarnessEventData =>
    ({ contentBlockDelta: { delta: { toolResult: [{ json }] } } }) as AgentCoreHarnessEventData,
  reasoningTextDelta: (text: string): AgentCoreHarnessEventData =>
    ({ contentBlockDelta: { delta: { reasoningContent: { text } } } }) as AgentCoreHarnessEventData,
  reasoningSignatureDelta: (signature: string): AgentCoreHarnessEventData =>
    ({ contentBlockDelta: { delta: { reasoningContent: { signature } } } }) as AgentCoreHarnessEventData,
  reasoningRedactedDelta: (redactedContent: Uint8Array): AgentCoreHarnessEventData =>
    ({ contentBlockDelta: { delta: { reasoningContent: { redactedContent } } } }) as AgentCoreHarnessEventData,
  bareContentBlockStart: (): AgentCoreHarnessEventData =>
    ({ contentBlockStart: { start: undefined } }) as AgentCoreHarnessEventData,
  contentBlockStop: (): AgentCoreHarnessEventData => ({ contentBlockStop: {} }) as AgentCoreHarnessEventData,
  messageStop: (stopReason?: string): AgentCoreHarnessEventData =>
    ({ messageStop: { stopReason } }) as AgentCoreHarnessEventData,
  metadata: (inputTokens = 100, outputTokens = 20, latencyMs = 1500): AgentCoreHarnessEventData =>
    ({
      metadata: {
        usage: { inputTokens, outputTokens, totalTokens: inputTokens + outputTokens },
        metrics: { latencyMs },
      },
    }) as AgentCoreHarnessEventData,
}

/** Decodes a complete sequence of Harness events. */
function decode(...events: AgentCoreHarnessEventData[]): DecodedHarnessTurn {
  const decoder = new HarnessStreamDecoder()
  events.forEach((streamEvent) => decoder.accept(streamEvent))
  return decoder.complete()
}

describe('HarnessStreamDecoder', () => {
  describe('complete', () => {
    it.each([
      ['end_turn', 'endTurn'],
      ['tool_use', 'toolUse'],
      ['tool_result', 'toolResult'],
      ['stop_sequence', 'stopSequence'],
      ['content_filtered', 'contentFiltered'],
      ['model_context_window_exceeded', 'modelContextWindowExceeded'],
      ['max_iterations_exceeded', 'limitTurns'],
      ['max_output_tokens_exceeded', 'limitOutputTokens'],
      ['timeout_exceeded', 'timeoutExceeded'],
      ['interrupted', 'interrupt'],
      ['partial_turn', 'pauseTurn'],
    ])('maps Harness stop reason %s to %s', (harnessStopReason, expectedStopReason) => {
      expect(decode(event.messageStop(harnessStopReason)).stopReason).toBe(expectedStopReason)
    })

    it('preserves an unknown Harness stop reason in SDK casing', () => {
      const warning = vi.spyOn(console, 'warn').mockImplementation(() => {})

      expect(decode(event.messageStop('future_stop_reason')).stopReason).toBe('futureStopReason')
      expect(warning).toHaveBeenCalledWith(
        'stop_reason=<future_stop_reason>, fallback=<futureStopReason> | unknown stop reason, converting to camelCase'
      )

      warning.mockRestore()
    })

    it.each([undefined, '', '   '])('rejects a missing or empty Harness stop reason %#', (stopReason) => {
      expect(() => decode(event.messageStop(stopReason))).toThrow(
        new ModelError('Harness messageStop event is missing a non-empty stopReason')
      )
    })

    it('rejects a stream without a completed message', () => {
      const decoder = new HarnessStreamDecoder()
      decoder.accept(event.messageStart())
      decoder.accept(event.textDelta('partial'))

      expect(() => decoder.complete()).toThrow(new ModelError('Stream ended without completing a message'))
    })

    it('rejects an incomplete final message in a multi-message stream', () => {
      const decoder = new HarnessStreamDecoder()
      ;[
        event.messageStart(),
        event.messageStop('tool_result'),
        event.messageStart(),
        event.textDelta('partial'),
      ].forEach((streamEvent) => decoder.accept(streamEvent))

      expect(() => decoder.complete()).toThrow(new ModelError('Stream ended without completing a message'))
    })

    it('accumulates text and reasoning blocks', () => {
      const result = decode(
        event.messageStart(),
        event.bareContentBlockStart(),
        event.reasoningTextDelta('Let me think. '),
        event.reasoningTextDelta('The answer is 4.'),
        event.reasoningSignatureDelta('sig-'),
        event.reasoningSignatureDelta('abc'),
        event.reasoningRedactedDelta(new Uint8Array([1, 2])),
        event.reasoningRedactedDelta(new Uint8Array([3, 4])),
        event.contentBlockStop(),
        event.textDelta('4'),
        event.contentBlockStop(),
        event.messageStop('end_turn')
      )

      expect(result.message.content).toStrictEqual([
        new ReasoningBlock({
          text: 'Let me think. The answer is 4.',
          signature: 'sig-abc',
          redactedContent: new Uint8Array([1, 2, 3, 4]),
        }),
        new TextBlock('4'),
      ])
    })

    it('accumulates text and JSON tool-result content', () => {
      const textResult = decode(
        event.messageStart(),
        event.toolResultStart('tu-1', 'success'),
        event.toolResultTextDelta('{"stdout":"ok"}'),
        event.contentBlockStop(),
        event.messageStop('tool_result')
      )
      const jsonResult = decode(
        event.messageStart(),
        event.toolResultStart('tu-2', 'success'),
        event.toolResultJsonDelta({ stdout: 'ok' }),
        event.contentBlockStop(),
        event.messageStop('tool_result')
      )

      expect([textResult.message.content, jsonResult.message.content]).toStrictEqual([
        [
          new ToolResultBlock({
            toolUseId: 'tu-1',
            status: 'success',
            content: [new TextBlock('{"stdout":"ok"}')],
          }),
        ],
        [
          new ToolResultBlock({
            toolUseId: 'tu-2',
            status: 'success',
            content: [new JsonBlock({ json: { stdout: 'ok' } })],
          }),
        ],
      ])
    })

    it('keeps only the final message from a multi-message stream', () => {
      const result = decode(
        event.messageStart(),
        event.toolUseStart('tu-1', 'vended_shell'),
        event.contentBlockStop(),
        event.messageStop('tool_use'),
        event.messageStart(),
        event.textDelta('Here is the summary.'),
        event.contentBlockStop(),
        event.messageStop('end_turn')
      )

      expect(result).toMatchObject({
        message: { role: 'assistant', content: [new TextBlock('Here is the summary.')] },
        stopReason: 'endTurn',
        assistantMessageCount: 2,
      })
    })

    it('counts only completed assistant messages', () => {
      expect(
        decode(
          event.messageStart(),
          event.messageStop('tool_use'),
          event.messageStart('user'),
          event.messageStop('tool_result'),
          event.messageStart(),
          event.messageStop('end_turn')
        ).assistantMessageCount
      ).toBe(2)
    })

    it.each([
      [
        'message role',
        { messageStart: { role: undefined } },
        "Harness messageStart event has invalid role 'undefined'",
      ],
      [
        'tool-use ID',
        { contentBlockStart: { start: { toolUse: { toolUseId: undefined, name: 'weather' } } } },
        'Harness tool-use start event is missing a non-empty toolUseId',
      ],
      [
        'tool-use name',
        { contentBlockStart: { start: { toolUse: { toolUseId: 'tu-1', name: '' } } } },
        'Harness tool-use start event is missing a non-empty name',
      ],
      [
        'tool-result ID',
        { contentBlockStart: { start: { toolResult: { toolUseId: undefined, status: 'success' } } } },
        'Harness tool-result start event is missing a non-empty toolUseId',
      ],
    ])('rejects a malformed %s at the stream boundary', (_field, malformedEvent, message) => {
      const decoder = new HarnessStreamDecoder()
      expect(() => decoder.accept(malformedEvent as AgentCoreHarnessEventData)).toThrow(new ModelError(message))
    })

    it('scopes tool-input parse failures to the retained message', () => {
      const recovered = decode(
        event.messageStart(),
        event.toolUseStart('tu-1', 'vended_tool'),
        event.toolUseDelta('{"partial":'),
        event.contentBlockStop(),
        event.messageStop('tool_use'),
        event.messageStart(),
        event.textDelta('Recovered.'),
        event.contentBlockStop(),
        event.messageStop('end_turn')
      )
      const malformed = decode(
        event.messageStart(),
        event.toolUseStart('tu-2', 'vended_tool'),
        event.toolUseDelta('{"partial":'),
        event.contentBlockStop(),
        event.messageStop('tool_use')
      )

      expect(recovered).not.toHaveProperty('toolInputParseError')
      expect(malformed).toMatchObject({
        message: {
          content: [new ToolUseBlock({ toolUseId: 'tu-2', name: 'vended_tool', input: {} })],
        },
        toolInputParseError: expect.any(SyntaxError),
      })
    })

    it('captures usage and latency metadata', () => {
      expect(decode(event.messageStop('end_turn'), event.metadata())).toMatchObject({
        stopReason: 'endTurn',
        usage: { inputTokens: 100, outputTokens: 20, totalTokens: 120 },
        latestUsage: { inputTokens: 100, outputTokens: 20, totalTokens: 120 },
        latencyMs: 1500,
      })
    })

    it('accumulates metadata while retaining the latest model-turn usage', () => {
      expect(
        decode(event.messageStop('end_turn'), event.metadata(50, 5, 100), event.metadata(200, 30, 900))
      ).toMatchObject({
        usage: { inputTokens: 250, outputTokens: 35, totalTokens: 285 },
        latestUsage: { inputTokens: 200, outputTokens: 30, totalTokens: 230 },
        latencyMs: 1000,
      })
    })
  })

  describe('partialMessage', () => {
    it('returns content accumulated before cancellation', () => {
      const decoder = new HarnessStreamDecoder()
      decoder.accept(event.messageStart())
      decoder.accept(event.textDelta('partial'))

      expect(decoder.partialMessage().content).toStrictEqual([new TextBlock('partial')])
    })
  })
})
