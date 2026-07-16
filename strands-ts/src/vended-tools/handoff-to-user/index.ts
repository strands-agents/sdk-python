/**
 * handoff-to-user tool: pause the agent to ask the user a structured question,
 * shimmed on top of the SDK's interrupt primitive.
 */

export { handoffToUser } from './handoff-to-user.js'
export type { HandoffToUserInput } from './handoff-to-user.js'
export {
  HANDOFF_TO_USER_DESCRIPTION,
  INTERRUPT_NAME,
  MAX_OPTIONS_COUNT,
  MAX_OPTION_LENGTH,
  MAX_QUESTION_LENGTH,
} from './types.js'
export type { HandoffAnswer, HandoffQuestion } from './types.js'
