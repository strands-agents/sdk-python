import type { TaskStatus } from './types.js'

/** Returns whether execution has permanently stopped. @internal */
export function isInProcessTaskTerminalStatus(status: TaskStatus): boolean {
  return status === 'completed' || status === 'failed' || status === 'cancelled'
}
