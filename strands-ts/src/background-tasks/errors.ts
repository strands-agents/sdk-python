/** Raised when a background task ID is not found. */
export class BackgroundTaskNotFoundError extends Error {
  /**
   * @param taskId - ID that could not be found.
   */
  constructor(taskId: string) {
    super(`Background task '${taskId}' was not found`)
    this.name = 'BackgroundTaskNotFoundError'
  }
}

/** Raised when waiting for Background Tasks to become idle exceeds its timeout. */
export class BackgroundTasksTimeoutError extends Error {
  /** Timeout supplied to `Agent.backgroundTasks.wait()`, in milliseconds. */
  readonly timeoutMs: number

  /**
   * @param timeoutMs - Timeout supplied to the wait operation, in milliseconds.
   */
  constructor(timeoutMs: number) {
    super(`Background Tasks wait timed out after ${timeoutMs}ms`)
    this.name = 'BackgroundTasksTimeoutError'
    this.timeoutMs = timeoutMs
  }
}
