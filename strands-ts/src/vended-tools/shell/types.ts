/**
 * Type definitions for the sandbox-routed shell tool.
 *
 * The Shell* errors extend the Bash* errors from `../bash/types.js` so that
 * callers who caught the pre-rename error types keep working.
 */

import { BashTimeoutError, BashSessionError } from '../bash/types.js'

export const SANDBOX_SHELL_DESCRIPTION =
  'Executes shell commands. Each call runs in a fresh shell; ' +
  'state such as variables and the working directory does not persist across calls.'

/**
 * Error thrown when a sandbox-routed shell command exceeds its timeout.
 *
 * Extends {@link BashTimeoutError} so that callers who caught the previous error
 * type keep working; new code should catch this instead.
 */
export class ShellTimeoutError extends BashTimeoutError {
  constructor(message: string) {
    super(message)
    this.name = 'ShellTimeoutError'
  }
}

/**
 * Error thrown when a sandbox-routed shell command fails.
 *
 * Extends {@link BashSessionError} so that callers who caught the previous error
 * type keep working; new code should catch this instead.
 */
export class ShellExecutionError extends BashSessionError {
  constructor(message: string) {
    super(message)
    this.name = 'ShellExecutionError'
  }
}
