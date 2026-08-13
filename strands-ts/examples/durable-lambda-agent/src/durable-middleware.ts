import { ExecuteToolStage, InvokeModelStage, Message, ToolResultBlock } from '@strands-agents/sdk'

import type { DurableContext, DurableLoggingContext } from '@aws/durable-execution-sdk-js'
import type {
  Agent,
  AgentStreamEvent,
  ExecuteToolContext,
  ExecuteToolResult,
  InvokeModelContext,
  InvokeModelResult,
  MessageData,
} from '@strands-agents/sdk'

interface RegisterOptions {
  crashAfterFirstTool?: boolean
}

/**
 * Persists model and tool stage results with Lambda durable execution.
 *
 * Strands checkpoint boundaries are handled by the invocation loop in the
 * Lambda handler. These durable steps restore the state that CheckpointData
 * intentionally does not contain.
 *
 * @param agent - Agent whose model and tool stages should be persisted.
 * @param context - Durable Lambda invocation context.
 * @param options - Optional failure injection used by the recovery demo.
 * @returns A function that removes both middleware handlers.
 */
export function registerDurableMiddleware(
  agent: Agent,
  context: DurableContext,
  options: RegisterOptions = {}
): () => void {
  let modelCallIndex = 0
  let toolCallIndex = 0
  let loggingContext: DurableLoggingContext | undefined

  context.configureLogger({
    customLogger: {
      log: (level, ...parameters) => console.log(`[${level}]`, ...parameters),
      info: (...parameters) => console.info(...parameters),
      warn: (...parameters) => console.warn(...parameters),
      error: (...parameters) => console.error(...parameters),
      debug: (...parameters) => console.debug(...parameters),
      configureDurableLoggingContext: (nextLoggingContext) => {
        loggingContext = nextLoggingContext
      },
    },
    modeAware: false,
  })

  const removeModelMiddleware = agent.addMiddleware(
    InvokeModelStage,
    async function* (
      stageContext: InvokeModelContext,
      next
    ): AsyncGenerator<AgentStreamEvent, InvokeModelResult, undefined> {
      // An afterModel checkpoint resume invokes the model again in the TypeScript
      // SDK, so this index tracks physical model calls rather than ReAct cycles.
      const callIndex = modelCallIndex
      modelCallIndex += 1
      const stepName = `invoke-model:call-${callIndex}`
      const events: AgentStreamEvent[] = []

      const persisted = await context.step<{ message: MessageData; stopReason: string }>(stepName, async () => {
        const result = await drainGenerator(
          () => next(stageContext),
          (event) => events.push(event)
        )
        return {
          message: result.result.message.toJSON(),
          stopReason: result.result.stopReason,
        }
      })

      yield* events
      return {
        result: {
          message: Message.fromMessageData(persisted.message),
          stopReason: persisted.stopReason as InvokeModelResult['result']['stopReason'],
        },
      }
    }
  )

  const removeToolMiddleware = agent.addMiddleware(
    ExecuteToolStage,
    async function* (
      stageContext: ExecuteToolContext,
      next
    ): AsyncGenerator<AgentStreamEvent, ExecuteToolResult, undefined> {
      const currentToolIndex = toolCallIndex
      toolCallIndex += 1
      const stepName = `tool:${stageContext.toolUse.name}:${stageContext.toolUse.toolUseId}`
      const events: AgentStreamEvent[] = []
      const persisted = await context.step<ReturnType<ToolResultBlock['toJSON']>>(stepName, async () => {
        const result = await drainGenerator(
          () => next(stageContext),
          (event) => events.push(event)
        )
        return result.result.toJSON()
      })

      yield* events

      if (options.crashAfterFirstTool === true && currentToolIndex === 0) {
        const failureStepName = `failure-after-tool:${stageContext.toolUse.toolUseId}`
        await context.step<boolean>(
          failureStepName,
          async (stepContext) => {
            const attempt = loggingContext?.getDurableLogData().attempt ?? 1
            if (attempt === 1) {
              stepContext.logger.warn(`step=<${failureStepName}>, attempt=<${attempt}> | injecting failure`)
              throw new Error('Injected failure after the first tool completed')
            }

            stepContext.logger.info(`step=<${failureStepName}>, attempt=<${attempt}> | continuing after retry`)
            return true
          },
          {
            retryStrategy: (_error, attemptCount) => ({
              shouldRetry: attemptCount < 2,
              delay: { seconds: 1 },
            }),
          }
        )
      }

      return { result: ToolResultBlock.fromJSON(persisted) }
    }
  )

  return () => {
    removeToolMiddleware()
    removeModelMiddleware()
  }
}

async function drainGenerator<TEvent, TResult>(
  generatorFactory: () => AsyncGenerator<TEvent, TResult, undefined>,
  onEvent: (event: TEvent) => void
): Promise<TResult> {
  const generator = generatorFactory()
  let nextResult = await generator.next()
  while (!nextResult.done) {
    onEvent(nextResult.value)
    nextResult = await generator.next()
  }
  return nextResult.value
}
