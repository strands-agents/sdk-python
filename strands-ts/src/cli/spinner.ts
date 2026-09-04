/**
 * Spinner for long-running operations. Renders a braille spinner with a label
 * on a TTY; in plain mode or when stdout is not a TTY (piped, JSON output) the
 * spinner is inert and final status is printed as a plain line.
 */

const FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
const INTERVAL_MS = 80

export interface SpinnerOptions {
  enabled: boolean
  write: (text: string) => void
}

export class Spinner {
  private readonly _enabled: boolean
  private readonly _write: (text: string) => void
  private _timer: ReturnType<typeof globalThis.setInterval> | undefined
  private _frame = 0
  private _text = ''
  private _active = false

  constructor(options: SpinnerOptions) {
    this._enabled = options.enabled && isInteractive()
    this._write = options.write
  }

  /** Starts the spinner with the given label. No-op when disabled. */
  isEnabled(): boolean {
    return this._enabled
  }

  start(text: string): void {
    if (!this._enabled) {
      this._text = text
      return
    }
    this._text = text
    this._active = true
    this._timer = globalThis.setInterval((): void => {
      this._frame = (this._frame + 1) % FRAMES.length
      this._render()
    }, INTERVAL_MS)

    this._render()
  }

  private _render(): void {
    this._write(`\r${FRAMES[this._frame]} ${this._text}`)
  }

  /** Stops the spinner without printing a status line. */
  stop(): void {
    if (this._timer !== undefined) {
      globalThis.clearInterval(this._timer)
      this._timer = undefined
    }
    if (this._active) {
      this._write('\r' + ' '.repeat(this._text.length + 2) + '\r')
      this._active = false
    }
    this._text = ''
  }

  /** Stops the spinner and prints a success line. */
  succeed(text: string): void {
    const label = this._text
    this.stop()
    if (label !== '') {
      this._write(`✓ ${text}\n`)
    }
  }

  /** Stops the spinner and prints a failure line. */
  fail(text: string): void {
    const label = this._text
    this.stop()
    if (label !== '') {
      this._write(`✗ ${text}\n`)
    }
  }
}

function isInteractive(): boolean {
  return typeof process !== 'undefined' && process.stdout?.isTTY === true
}
