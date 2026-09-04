import { describe, expect, it } from 'vitest'
import chalk from 'chalk'
import { Agent } from '../../agent/agent.js'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { collectGenerator } from '../../__fixtures__/model-test-helpers.js'
import { createMockTool } from '../../__fixtures__/tool-helpers.js'
import { CliPrinter } from '../cli-printer.js'
import { createTheme, stripAnsi } from '../theme.js'

// Chalk disables colors when stdout is not a TTY, which is the case under
// vitest. Force a color level so styled-output assertions are deterministic.
chalk.level = 1

function makePrinter(plain: boolean, outputs: string[], errors: string[] = []): CliPrinter {
  return new CliPrinter({
    theme: createTheme({ plain }),
    appender: (text: string): void => {
      outputs.push(text)
    },
    errorAppender: (text: string): void => {
      errors.push(text)
    },
    spinner: false,
  })
}

describe('CliPrinter', () => {
  it('streams agent text output', async () => {
    const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Hello world' })
    const outputs: string[] = []
    const printer = makePrinter(true, outputs)

    const agent = new Agent({ model, printer: false })
    ;(agent as unknown as { _printer: CliPrinter })._printer = printer

    await collectGenerator(agent.stream('Test'))

    const allOutput = stripAnsi(outputs.join(''))
    expect(allOutput).toContain('Hello world')
    expect(allOutput).toContain('stop reason: endTurn')
  })

  it('announces tool calls and results with status and content', async () => {
    const tool = createMockTool('calculator', () => 'The answer is 42')
    const model = new MockMessageModel()
      .addTurn({ type: 'toolUseBlock', name: 'calculator', toolUseId: 'tool-1', input: { a: 2 } })
      .addTurn({ type: 'textBlock', text: 'done' })
    const outputs: string[] = []
    const printer = makePrinter(true, outputs)

    const agent = new Agent({ model, printer: false, tools: [tool] })
    ;(agent as unknown as { _printer: CliPrinter })._printer = printer

    await collectGenerator(agent.stream('What is 2?'))

    const allOutput = stripAnsi(outputs.join(''))
    expect(allOutput).toContain('Tool #1: calculator')
    expect(allOutput).toContain('{"a":2}')
    expect(allOutput).toContain('✓ Tool completed')
    expect(allOutput).toContain('The answer is 42')
  })

  it('reports failed tool runs as errors', async () => {
    const tool = createMockTool('boom', () => {
      throw new Error('exploded')
    })
    const model = new MockMessageModel()
      .addTurn({ type: 'toolUseBlock', name: 'boom', toolUseId: 'tool-1', input: {} })
      .addTurn({ type: 'textBlock', text: 'handled' })
    const outputs: string[] = []
    const printer = makePrinter(true, outputs)

    const agent = new Agent({ model, printer: false, tools: [tool] })
    ;(agent as unknown as { _printer: CliPrinter })._printer = printer

    await collectGenerator(agent.stream('Break things'))

    const allOutput = stripAnsi(outputs.join(''))
    expect(allOutput).toContain('✗ Tool failed')
  })

  it('writes errors to the error sink, not the output sink', () => {
    const outputs: string[] = []
    const errors: string[] = []
    const printer = makePrinter(true, outputs, errors)

    printer.error('something went wrong')

    expect(stripAnsi(errors.join(''))).toContain('something went wrong')
    expect(outputs).toStrictEqual([])
  })

  it('renders handoff lines with source and targets', () => {
    const outputs: string[] = []
    const printer = makePrinter(true, outputs)

    printer.handoff('researcher', ['writer', 'reviewer'])

    const line = stripAnsi(outputs.join(''))
    expect(line).toContain('handoff:')
    expect(line).toContain('researcher')
    expect(line).toContain('writer, reviewer')
  })

  it('emits no ANSI escape codes in plain mode', async () => {
    const model = new MockMessageModel().addTurn({ type: 'textBlock', text: '# Heading\n\nPlain **bold** text' })
    const outputs: string[] = []
    const printer = makePrinter(true, outputs)

    const agent = new Agent({ model, printer: false })
    ;(agent as unknown as { _printer: CliPrinter })._printer = printer

    await collectGenerator(agent.stream('Test'))

    expect(outputs.join('')).not.toMatch(new RegExp(String.fromCharCode(27)))
  })

  it('emits styled output when not in plain mode', async () => {
    const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Answer' })
    const outputs: string[] = []
    const printer = makePrinter(false, outputs)

    const agent = new Agent({ model, printer: false })
    ;(agent as unknown as { _printer: CliPrinter })._printer = printer

    await collectGenerator(agent.stream('Test'))

    expect(outputs.join('')).toContain(String.fromCharCode(27))
  })
})
