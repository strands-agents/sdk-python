import chalk from 'chalk'

/**
 * Semantic color theme for terminal output.
 *
 * Chalk automatically disables ANSI codes when stdout is not a TTY or when
 * `NO_COLOR`/`TERM=dumb` is set, so piped output stays clean. `plain: true`
 * (or the `STRANDS_PLAIN` environment variable) forces raw text regardless.
 */
export interface ThemeOptions {
  /** Force unstyled output even on a TTY. Defaults to `STRANDS_PLAIN=1`. */
  plain?: boolean
}

export interface Theme {
  /** Section headings (e.g. agent names). */
  heading(text: string): string
  /** User-entered text. */
  user(text: string): string
  /** Agent-attributed text. */
  agent(text: string): string
  /** Tool names and call announcements. */
  tool(text: string): string
  /** Successful tool results. */
  toolResult(text: string): string
  /** Inline code spans and fenced code blocks. */
  code(text: string): string
  /** Error messages. */
  error(text: string): string
  /** Warning messages. */
  warning(text: string): string
  /** Success confirmations. */
  success(text: string): string
  /** System/status lines (low emphasis). */
  system(text: string): string
  /** De-emphasized helper text. */
  dim(text: string): string
  /** Bold emphasis. */
  bold(text: string): string
  /** Markdown links, rendered as `text (url)`. */
  link(text: string, url: string): string
  /** True when all styling is disabled. */
  readonly plain: boolean
}

// eslint-disable-next-line no-control-regex -- intentional match on ANSI escape sequences
const ansiPattern = /\u001b\[[0-9;]*m/g

/** Removes ANSI escape sequences from a string. */
export function stripAnsi(text: string): string {
  return text.replace(ansiPattern, '')
}

function isPlainMode(plain?: boolean): boolean {
  if (plain !== undefined) {
    return plain
  }
  return typeof process !== 'undefined' && process.env?.STRANDS_PLAIN === '1'
}

const identity = (text: string): string => text

/** Creates a theme; returns a pass-through theme in plain mode. */
export function createTheme(options?: ThemeOptions): Theme {
  const plain = isPlainMode(options?.plain)

  if (plain) {
    return {
      plain,
      heading: identity,
      user: identity,
      agent: identity,
      tool: identity,
      toolResult: identity,
      code: identity,
      error: identity,
      warning: identity,
      success: identity,
      system: identity,
      dim: identity,
      bold: identity,
      link: (text: string, url: string): string => (text === url ? text : `${text} (${url})`),
    }
  }

  return {
    plain,
    heading: (text: string): string => chalk.bold.underline(text),
    user: (text: string): string => chalk.bold.cyan(text),
    agent: (text: string): string => chalk.bold.magenta(text),
    tool: (text: string): string => chalk.yellow(text),
    toolResult: (text: string): string => chalk.green(text),
    code: (text: string): string => chalk.cyan(text),
    error: (text: string): string => chalk.bold.red(text),
    warning: (text: string): string => chalk.bold.yellow(text),
    success: (text: string): string => chalk.bold.green(text),
    system: (text: string): string => chalk.dim(text),
    dim: (text: string): string => chalk.dim(text),
    bold: (text: string): string => chalk.bold(text),
    link: (text: string, url: string): string => `${chalk.blue.underline(text)} ${chalk.dim(`(${url})`)}`,
  }
}
