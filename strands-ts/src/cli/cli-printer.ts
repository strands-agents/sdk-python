import type { Printer } from '../agent/printer.js'
import type { AgentStreamEvent } from '../types/agent.js'
import type { ModelStreamEvent } from '../models/streaming.js'
import type { BeforeToolCallEvent, BeforeToolsEvent, ToolResultEvent } from '../hooks/events.js'
import { createTheme, type Theme } from './theme.js'
import { renderMarkdown } from './markdown.js'
import { Spinner } from './spinner.js'

/**
 * Terminal printer with color, Markdown rendering, and spinners.
 *
 * Implements the SDK's `Printer` interface so it can be attached to an Agent
 * in place of the default `AgentPrinter`. Honors plain mode (no TTY,
 * `NO_COLOR`, `STRANDS_PLAIN=1`) by degrading to raw text.
 */

export interface CliPrinterOptions {
  /** Color theme. Defaults to a theme auto-detected from the environment. */
  theme?: Theme
  /** Output sink. Defaults to `process.stdout` (or `console.log` in browsers). */
  appender?: (text: string) => void
  /** Error sink. Defaults to `process.stderr` (or `console.error` in browsers). */
  errorAppender?: (text: string) => void
  /** Enable spinners for tool execution. Defaults to true (no-op off TTY). */
  spinner?: boolean
  /** Render agent text as Markdown. Defaults to true. */
  markdown?: boolean
}

const MAX_RESULT_LENGTH = 2000

function defaultAppender(): (text: string) => void {
  if (typeof process !== 'undefined' && process.stdout?.write) {
    return (text: string): void => {
      process.stdout.write(text)
    }
  }
  return (text: string): void => console.log(text)
}

function defaultErrorAppender(): (text: string) => void {
  if (typeof process !== 'undefined' && process.stderr?.write) {
    return (text: string): void => {
      process.stderr.write(text)
    }
  }
  return (text: string): void => console.error(text)
}

function truncate(text: string, maxLength: number): string {
  if (text.length <= maxLength) {
    return text
  }
  return `${text.slice(0, maxLength)}… (+${text.length - maxLength} chars)`
}

export class CliPrinter implements Printer {
  private readonly _appender: (text: string) => void
  private readonly _errorAppender: (text: string) => void
  private readonly _theme: Theme
  private readonly _markdown: boolean
  private readonly _spinner: Spinner
  private _inReasoningBlock = false
  private _needReasoningIndent = false
  private _toolCount = 0
  private _textBuffer: string[] = []

  constructor(options?: CliPrinterOptions) {
    this._theme = options?.theme ?? createTheme()
    this._appender = options?.appender ?? defaultAppender()
    this._errorAppender = options?.errorAppender ?? defaultErrorAppender()
    this._markdown = options?.markdown ?? true
    this._spinner = new Spinner({
      enabled: options?.spinner ?? true,
      write: (text: string): void => this._appender(text),
    })
  }

  public write(content: string): void {
    this._appender(content)
  }

  /** Writes a line to the error sink (stderr), styled as an error. */
  public error(message: string): void {
    this._spinner.stop()
    this._errorAppender(`✗ ${this._theme.error(message)}\n`)
  }

  /** Writes a styled warning line. */
  public warning(message: string): void {
    this._appender(`⚠ ${this._theme.warning(message)}\n`)
  }

  /** Writes a styled success line. */
  public success(message: string): void {
    this._appender(`✓ ${this._theme.success(message)}\n`)
  }

  /** Writes a dim system/status line. */
  public system(message: string): void {
    this._appender(this._theme.system(message) + '\n')
  }

  /** Renders and writes an echoed user prompt. */
  public userInput(prompt: string): void {
    this._appender(`${this._theme.user('❯ You')} ${prompt}\n\n`)
  }

  /** Announces a multi-agent handoff. */
  public handoff(source: string, targets: string[]): void {
    this._spinner.stop()
    this._appender(
      `${this._theme.system('→ handoff:')} ${this._theme.agent(source)} ${this._theme.dim('→')} ${this._theme.agent(targets.join(', '))}\n`
    )
  }

  public processEvent(event: AgentStreamEvent): void {
    switch (event.type) {
      case 'beforeInvocationEvent':
        this._startInvocation()
        break
      case 'modelStreamUpdateEvent':
        this._handleModelStreamEvent(event.event)
        break
      case 'beforeToolsEvent':
        this._handleBeforeTools(event)
        break
      case 'beforeToolCallEvent':
        this._handleBeforeToolCall(event)
        break
      case 'toolResultEvent':
        this._handleToolResult(event)
        break
      case 'agentResultEvent':
        this._handleAgentResult(event)
        break
      case 'interruptEvent':
        this.warning('execution paused for human input')
        break
      default:
        break
    }
  }

  private _startInvocation(): void {
    this._toolCount = 0
    if (this._theme.plain) {
      return
    }
    if (this._spinner.isEnabled()) {
      this._spinner.start('Thinking…')
    } else {
      this._appender(this._theme.system('● Thinking…') + '\n')
    }
  }

  private _handleModelStreamEvent(event: ModelStreamEvent): void {
    switch (event.type) {
      case 'modelContentBlockDeltaEvent':
        this._handleContentBlockDelta(event)
        break
      case 'modelContentBlockStartEvent':
        this._handleContentBlockStart(event)
        break
      case 'modelContentBlockStopEvent':
        this._handleContentBlockStop()
        break
      default:
        break
    }
  }

  private _handleContentBlockDelta(event: Extract<ModelStreamEvent, { type: 'modelContentBlockDeltaEvent' }>): void {
    const { delta } = event
    if (delta.type === 'textDelta') {
      this._spinner.stop()
      if (delta.text && delta.text.length > 0) {
        this._textBuffer.push(delta.text)
      }
    } else if (delta.type === 'reasoningContentDelta') {
      if (!this._inReasoningBlock) {
        this._inReasoningBlock = true
        this._needReasoningIndent = true
        this.write('\n' + this._theme.system('💭 Reasoning:') + '\n')
      }
      if (delta.text && delta.text.length > 0) {
        this._writeReasoningText(delta.text)
      }
    }
  }

  private _writeReasoningText(text: string): void {
    let output = ''
    for (const char of text) {
      if (this._needReasoningIndent && char !== '\n') {
        output += '   '
        this._needReasoningIndent = false
      }
      output += char
      if (char === '\n') {
        this._needReasoningIndent = true
      }
    }
    this.write(output)
  }

  private _handleContentBlockStart(event: Extract<ModelStreamEvent, { type: 'modelContentBlockStartEvent' }>): void {
    if (event.start?.type === 'toolUseStart') {
      this._spinner.stop()
      this.write('\n  ' + this._theme.system(`⏳ ${event.start.name}`) + '\n')
    }
  }

  private _handleContentBlockStop(): void {
    this._flushText()
    if (this._inReasoningBlock) {
      if (!this._needReasoningIndent) {
        this.write('\n')
      }
      this._inReasoningBlock = false
      this._needReasoningIndent = false
    }
  }

  /** Renders buffered agent text as Markdown (or raw in plain mode) and writes it. */
  private _flushText(): void {
    if (this._textBuffer.length === 0) {
      return
    }
    const text = this._textBuffer.join('')
    this._textBuffer = []
    const rendered = this._markdown ? renderMarkdown(text, { theme: this._theme }) : text
    this.write(rendered)
  }

  private _handleBeforeTools(event: BeforeToolsEvent): void {
    if (event.cancel) {
      this._spinner.stop()
      this.write(this._theme.warning('🚫 All tools denied') + '\n')
    }
  }

  private _handleBeforeToolCall(event: BeforeToolCallEvent): void {
    this._toolCount++
    this._spinner.stop()
    const args = truncate(JSON.stringify(event.toolUse.input ?? {}), 80)
    const denied = event.cancel ? ` ${this._theme.error('(denied)')}` : ''
    this.write(
      `\n${this._theme.tool(`🔧 Tool #${this._toolCount}: ${event.toolUse.name}`)} ${this._theme.dim(args)}${denied}\n`
    )
    if (!event.cancel && this._spinner.isEnabled()) {
      this._spinner.start(`Running ${event.toolUse.name}…`)
    }
  }

  private _handleToolResult(event: ToolResultEvent): void {
    this._spinner.stop()
    const text = this._resultText(event)
    if (event.result.status === 'success') {
      this.write(this._theme.toolResult('  ✓ Tool completed') + '\n')
    } else if (event.result.status === 'error') {
      this.write(this._theme.error('  ✗ Tool failed') + '\n')
    }
    if (text !== '') {
      const rendered =
        this._markdown && event.result.status === 'success' ? renderMarkdown(text, { theme: this._theme }) : text
      this.write(this._indent(rendered, '    ') + '\n')
    }
  }

  private _resultText(event: ToolResultEvent): string {
    const parts: string[] = []
    for (const block of event.result.content ?? []) {
      if ('text' in block && typeof block.text === 'string') {
        parts.push(block.text)
      }
    }
    return truncate(parts.join('\n'), MAX_RESULT_LENGTH)
  }

  private _indent(text: string, prefix: string): string {
    return text
      .split('\n')
      .map((line) => (line.trim() === '' ? line : prefix + line))
      .join('\n')
  }

  private _handleAgentResult(event: Extract<AgentStreamEvent, { type: 'agentResultEvent' }>): void {
    this._spinner.stop()
    this._flushText()
    this.write(`\n${this._theme.system(`stop reason: ${event.result.stopReason}`)}\n`)
  }
}
