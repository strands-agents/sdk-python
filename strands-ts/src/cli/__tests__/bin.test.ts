import { describe, expect, it } from 'vitest'
import { parseArgs } from '../bin.js'

describe('parseArgs', () => {
  it('defaults to REPL mode', () => {
    expect(parseArgs([])).toMatchObject({ command: 'repl', plain: false, spinner: true, json: false })
  })

  it('detects the run command and joins the prompt', () => {
    expect(parseArgs(['run', 'hello', 'world'])).toMatchObject({ command: 'run', prompt: 'hello world' })
  })

  it('recognizes plain and no-spinner flags', () => {
    expect(parseArgs(['--plain', '--no-spinner'])).toMatchObject({ plain: true, spinner: false })
  })

  it('JSON mode implies plain output', () => {
    expect(parseArgs(['--json'])).toMatchObject({ json: true, plain: true })
  })
})
