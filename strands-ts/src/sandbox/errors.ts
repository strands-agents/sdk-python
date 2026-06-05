/**
 * Error types for sandbox command and code execution.
 *
 * These are runtime throwables raised by sandbox execution; consumers can
 * branch on them via `instanceof` to distinguish timeouts from aborts.
 */

/**
 * Thrown by sandbox execution when the configured `timeout` elapses.
 */
export class SandboxTimeoutError extends Error {
  constructor(seconds: number) {
    super(`Execution timed out after ${seconds} seconds`)
    this.name = 'SandboxTimeoutError'
  }
}

/**
 * Thrown by sandbox execution when the abort signal fires.
 */
export class SandboxAbortError extends Error {
  constructor() {
    super('Execution aborted')
    this.name = 'SandboxAbortError'
  }
}
