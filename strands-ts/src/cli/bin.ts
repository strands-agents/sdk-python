/**
 * `strands` command-line entry point.
 *
 * Usage:
 *   strands                    interactive REPL (default model)
 *   strands run "prompt"       one-shot invocation
 *   strands --help
 *
 * Flags:
 *   --plain          disable colors and Markdown styling
 *   --no-spinner     disable progress spinners
 *   --json           emit stream events as newline-delimited JSON (implies plain)
 */
import readline from 'node:readline'
import { Agent } from '../agent/agent.js'
import type { AgentStreamEvent } from '../types/agent.js'
import type { MultiAgentStreamEvent } from '../multiagent/events.js'
import { CliPrinter } from './cli-printer.js'
import { createTheme } from './theme.js'

interface CliArgs {
  command: 'repl' | 'run'
  prompt?: string
  plain: boolean
  spinner: boolean
  json: boolean
}

const HELP = `strands - run Strands agents from the terminal

Usage:
  strands                     start an interactive session
  strands run "prompt"        run a single prompt and exit

Flags:
  --plain        disable colors and Markdown styling
  --no-spinner   disable progress spinners
  --json         emit stream events as newline-delimited JSON
  --help         show this help
`

function parseArgs(argv: string[]): CliArgs {
  const args: CliArgs = { command: 'repl', plain: false, spinner: true, json: false }
  const positional: string[] = []
  for (const arg of argv) {
    switch (arg) {
      case '--plain':
        args.plain = true
        break
      case '--no-spinner':
        args.spinner = false
        break
      case '--json':
        args.json = true
        args.plain = true
        break
      default:
        positional.push(arg)
        break
    }
  }
  if (positional[0] === 'run') {
    args.command = 'run'
    args.prompt = positional.slice(1).join(' ')
  }
  return args
}

function printHelp(): void {
  process.stdout.write(HELP)
}

function attachPrinter(agent: Agent, printer: CliPrinter): void {
  ;(agent as unknown as { _printer: CliPrinter })._printer = printer
}

async function runOnce(agent: Agent, printer: CliPrinter, prompt: string): Promise<void> {
  printer.userInput(prompt)
  for await (const event of agent.stream(prompt)) {
    printer.processEvent(event)
  }
}

type AnyStreamEvent = AgentStreamEvent | MultiAgentStreamEvent

function isHandoff(event: AnyStreamEvent): event is Extract<MultiAgentStreamEvent, { type: 'multiAgentHandoffEvent' }> {
  return event.type === 'multiAgentHandoffEvent'
}

async function streamToPrinter(
  stream: AsyncGenerator<AnyStreamEvent, unknown, undefined>,
  printer: CliPrinter
): Promise<void> {
  for await (const event of stream) {
    if (isHandoff(event)) {
      printer.handoff(event.source, event.targets)
      continue
    }
    printer.processEvent(event as AgentStreamEvent)
  }
}

async function runRepl(agent: Agent, printer: CliPrinter): Promise<void> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: '❯ ',
  })
  printer.system('strands REPL — type a prompt, or Ctrl+C to exit')
  rl.prompt()
  for await (const line of rl) {
    const prompt = line.trim()
    if (prompt === '') {
      rl.prompt()
      continue
    }
    printer.userInput(prompt)
    try {
      for await (const event of agent.stream(prompt)) {
        printer.processEvent(event)
      }
    } catch (error) {
      printer.error(error instanceof Error ? error.message : String(error))
    }
    rl.prompt()
  }
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2)
  if (argv.includes('--help') || argv.includes('-h')) {
    printHelp()
    return
  }
  const args = parseArgs(argv)
  const theme = createTheme({ plain: args.plain })
  const printer = new CliPrinter({ theme, spinner: args.spinner })
  const agent = new Agent({ printer: false })
  attachPrinter(agent, printer)

  if (args.json) {
    for await (const event of agent.stream(args.command === 'run' ? (args.prompt ?? '') : '')) {
      process.stdout.write(`${JSON.stringify(event)}\n`)
    }
    return
  }

  if (args.command === 'run') {
    if (!args.prompt) {
      printer.error('the run command requires a prompt: strands run "prompt"')
      process.exitCode = 1
      return
    }
    await runOnce(agent, printer, args.prompt)
    return
  }

  await runRepl(agent, printer)
}

const invokedDirectly = typeof process !== 'undefined' && process.argv?.[1]?.replace(/\\/g, '/').endsWith('cli/bin.js')

if (invokedDirectly) {
  main().catch((error: unknown) => {
    process.stderr.write(`✗ ${error instanceof Error ? error.message : String(error)}\n`)
    process.exitCode = 1
  })
}

export { parseArgs, streamToPrinter, main }
