/**
 * Type definitions for the bash and sandbox-routed shell tools.
 */

export const SANDBOX_SHELL_DESCRIPTION =
  'Executes shell commands. Each call runs in a fresh shell; ' +
  'state such as variables and the working directory does not persist across calls.'

/**
 * @deprecated SANDBOX_BASH_DESCRIPTION is deprecated and will be removed in v2.0.0. Use SANDBOX_SHELL_DESCRIPTION instead. The tool routes commands through the sandbox, which runs sh or the remote login shell rather than bash specifically.
 */
export const SANDBOX_BASH_DESCRIPTION = SANDBOX_SHELL_DESCRIPTION

/**
 * Input parameters for execute operation.
 */
export interface ExecuteInput {
  /**
   * Operation mode, must be 'execute'.
   */
  mode: 'execute'

  /**
   * The bash command to execute.
   */
  command: string

  /**
   * Timeout in seconds for the command execution.
   * Defaults to 120 seconds.
   */
  timeout?: number
}

/**
 * Input parameters for restart operation.
 */
export interface RestartInput {
  /**
   * Operation mode, must be 'restart'.
   */
  mode: 'restart'
}

/**
 * Union type of all valid bash tool inputs.
 */
export type BashInput = ExecuteInput | RestartInput

/**
 * Output format for bash command execution.
 */
export interface BashOutput {
  /**
   * Standard output from the command.
   */
  output: string

  /**
   * Standard error from the command.
   * Empty string if no errors occurred.
   */
  error: string

  /**
   * Allow indexing with string keys for JSONValue compatibility.
   */
  [key: string]: string
}

/**
 * Error thrown when a bash command exceeds its timeout.
 */
export class BashTimeoutError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'BashTimeoutError'
  }
}

/**
 * Error thrown when a bash session encounters an error.
 */
export class BashSessionError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'BashSessionError'
  }
}

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
