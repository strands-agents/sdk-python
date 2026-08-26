/** Background task lifecycle status. @internal */
export type BackgroundTaskStatus = 'queued' | 'working' | 'input_required' | 'completed' | 'failed' | 'cancelled'

/** Background task failure category. @internal */
export type BackgroundTaskFailureType = 'toolError' | 'executionError' | 'timeout'
