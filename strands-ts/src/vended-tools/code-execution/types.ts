/**
 * Type definitions for the code_execution tool.
 */

export const CODE_EXECUTION_DESCRIPTION =
  'Executes source code through a configured sandbox and returns stdout, stderr, ' +
  'the exit code, and wall-clock elapsed milliseconds. Each call is a fresh ' +
  'interpreter invocation; state does not persist across calls. Requires an ' +
  'isolating sandbox to be configured on the agent.'

/**
 * Default interpreter used when the factory is not passed a `language`.
 * The tool executes the SDK's own language; `node` is TypeScript/JavaScript's convention.
 */
export const DEFAULT_LANGUAGE = 'node'

/**
 * Default upper bound on the source-code size accepted from the model (bytes,
 * UTF-8). Keeps a runaway prompt from stuffing the sandbox with megabytes of
 * source before the interpreter ever runs.
 */
export const DEFAULT_MAX_CODE_BYTES = 100_000

/**
 * Default upper bound on the stdout/stderr the tool returns to the model
 * (bytes, UTF-8). Anything past this is dropped and a truncation marker is
 * appended so the model knows it happened.
 */
export const DEFAULT_MAX_OUTPUT_BYTES = 100_000

/**
 * Default execution timeout in seconds; passed through to the sandbox, which
 * owns the actual kill.
 */
export const DEFAULT_TIMEOUT_SECONDS = 60

/**
 * Marker appended when stdout/stderr is trimmed to `maxOutputBytes`.
 */
export const TRUNCATION_MARKER = '\n... [truncated]'

/**
 * Result of a code_execution call.
 */
export interface CodeExecutionOutput {
  /**
   * Standard output captured from the interpreter. May be truncated with a
   * trailing marker if the sandbox produced more than `maxOutputBytes` bytes.
   */
  stdout: string
  /**
   * Standard error captured from the interpreter. Truncated on the same terms
   * as `stdout`.
   */
  stderr: string
  /**
   * Exit code from the interpreter. `0` indicates success.
   */
  exitCode: number
  /**
   * Wall-clock time in milliseconds from the call entering the sandbox to the
   * sandbox returning a result.
   */
  elapsedMs: number

  /**
   * Allow indexing with string keys for JSONValue compatibility.
   */
  [key: string]: string | number
}

/**
 * Thrown when the caller-supplied `code` exceeds the configured `maxCodeBytes`.
 */
export class CodeSizeExceededError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'CodeSizeExceededError'
  }
}

/**
 * Thrown when the tool refuses to execute because no isolating sandbox is
 * configured on the agent.
 */
export class SandboxNotConfiguredError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'SandboxNotConfiguredError'
  }
}
