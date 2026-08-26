/** Raised when a background task ID is not found. @internal */
export class BackgroundTaskNotFoundError extends Error {
  /**
   * @param taskId - ID that could not be found.
   */
  constructor(taskId: string) {
    super(`Background task '${taskId}' was not found`)
    this.name = 'BackgroundTaskNotFoundError'
  }
}
