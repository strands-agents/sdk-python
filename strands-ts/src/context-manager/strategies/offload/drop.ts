/**
 * Drop strategy — removes matching content from the context window entirely.
 *
 * @internal
 */

import { logger } from '../../../logging/logger.js'
import { Message, TextBlock, ToolResultBlock } from '../../../types/messages.js'
import type { ContentBlock } from '../../../types/messages.js'
import type { LocalAgent } from '../../../types/agent.js'
import type { ContextStrategy } from '../../types.js'
import { BaseOffloadStrategy, DROPPED_MARKER, type OffloadConditions, type OffloadTarget } from './base.js'

export class DropStrategy extends BaseOffloadStrategy {
  readonly name = 'offload:drop'

  when(conditions: OffloadConditions): ContextStrategy {
    return new DropStrategy(this._target, conditions)
  }

  protected override _makeRemovalMarker(count: number): string {
    return `[Dropped: ${count} ${count === 1 ? 'message' : 'messages'}]`
  }

  protected async _replaceBlock(
    block: TextBlock | ToolResultBlock,
    _tokens: number,
    message: Message,
    _agent: LocalAgent
  ): Promise<ContentBlock | null> {
    if (block instanceof ToolResultBlock) {
      logger.debug(`toolUseId=<${block.toolUseId}> | dropped tool result from context window`)
      return new ToolResultBlock({
        toolUseId: block.toolUseId,
        status: block.status,
        content: [new TextBlock(DROPPED_MARKER)],
      })
    }
    logger.debug(`trackingId=<${message.trackingId}> | dropped text block from context window`)
    return new TextBlock(DROPPED_MARKER)
  }
}
