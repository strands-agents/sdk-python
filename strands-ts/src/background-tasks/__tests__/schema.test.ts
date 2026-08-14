import { describe, expect, it } from 'vitest'
import type { ToolSpec } from '../../tools/types.js'
import { addBackgroundSelection, stripBackgroundSelection } from '../schema.js'

function transform(inputSchema?: unknown) {
  return addBackgroundSelection({
    name: 'work',
    description: 'Perform work.',
    ...(inputSchema !== undefined && {
      inputSchema: inputSchema as NonNullable<ToolSpec['inputSchema']>,
    }),
  })
}

describe('background task schema transformation', () => {
  it('supports absent, empty, and direct object schemas without mutating them', () => {
    const direct = {
      type: 'object',
      properties: {
        value: {
          type: 'object',
          properties: {
            _background: { type: 'string' },
            nestedReference: { $ref: '#/$defs/value' },
          },
        },
      },
      required: ['value'],
      additionalProperties: false,
    }
    const absent = transform()
    const empty = transform({})
    const object = transform(direct)

    expect(absent).toEqual({
      compatible: true,
      toolSpec: expect.objectContaining({
        inputSchema: expect.objectContaining({
          type: 'object',
          properties: expect.objectContaining({
            _background: {
              type: 'boolean',
              description:
                'Run this tool call in the background. Acknowledgement is immediate; continue without waiting or polling. The final result will be delivered automatically at a later Agent boundary.',
            },
          }),
        }),
      }),
    })
    expect(empty).toEqual({
      compatible: true,
      toolSpec: expect.objectContaining({
        inputSchema: expect.objectContaining({
          type: 'object',
          properties: expect.objectContaining({ _background: expect.objectContaining({ type: 'boolean' }) }),
        }),
      }),
    })
    expect(object).toEqual({
      compatible: true,
      toolSpec: expect.objectContaining({
        inputSchema: expect.objectContaining({
          ...direct,
          properties: expect.objectContaining({
            value: direct.properties.value,
            _background: expect.objectContaining({ type: 'boolean' }),
          }),
        }),
      }),
    })
    expect(direct).toEqual({
      type: 'object',
      properties: {
        value: {
          type: 'object',
          properties: {
            _background: { type: 'string' },
            nestedReference: { $ref: '#/$defs/value' },
          },
        },
      },
      required: ['value'],
      additionalProperties: false,
    })
  })

  it('rejects incompatible root schemas', () => {
    const cases = [
      [{ type: 'string' }, "root schema type must be 'object'"],
      [{ oneOf: [{ type: 'object' }] }, "unsupported root schema keyword 'oneOf'"],
      [{ additionalProperties: { type: 'string' } }, 'schema-valued additionalProperties is not supported'],
      [{ required: ['value', 'value'] }, 'root schema required must contain unique property names'],
      [{ required: ['value', 1] }, 'root schema required must contain unique property names'],
      [{ properties: { _background: { type: 'boolean' } } }, "schema already defines reserved property '_background'"],
    ] as const

    for (const [schema, reason] of cases) {
      expect(transform(schema)).toEqual({ compatible: false, reason })
    }
  })

  it('strips the selector from a copy and rejects malformed selector values', () => {
    const input = { value: 'x', _background: true }

    expect(stripBackgroundSelection(input)).toEqual({
      input: { value: 'x' },
      selected: true,
    })
    expect(input).toEqual({ value: 'x', _background: true })
    expect(() => stripBackgroundSelection({ value: 'x', _background: 'true' })).toThrow(
      "'_background' must be a boolean"
    )
  })
})
