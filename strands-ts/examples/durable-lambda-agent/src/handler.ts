import { createHash } from 'node:crypto'

import { withDurableExecution } from '@aws/durable-execution-sdk-js'
import { Agent, BedrockModel, SessionManager } from '@strands-agents/sdk'
import { Checkpoint } from '@strands-agents/sdk/experimental'
import { S3Storage } from '@strands-agents/sdk/storage'

import { registerDurableMiddleware } from './durable-middleware.js'
import { DurableStorage } from './durable-storage.js'
import { buildTools } from './tools.js'

import type { DurableContext } from '@aws/durable-execution-sdk-js'
import type { InvokeArgs } from '@strands-agents/sdk'
import type { CheckpointData } from '@strands-agents/sdk/experimental'

interface AgentEvent {
  prompt?: string
  sessionId?: string
  simulateRestart?: boolean
  crashAfterFirstTool?: boolean
}

interface AgentOutput {
  stopReason: string
  text: string
}

const DEFAULT_MODEL_ID = 'global.anthropic.claude-sonnet-4-6'
const DEFAULT_PROMPT = 'Plan my trip to Seattle.'
const SYSTEM_PROMPT = [
  'You are a trip planner. The user will name a city.',
  'Call get_weather with that city, then call book_flight to that city.',
  'After both tools succeed, respond with one short sentence.',
].join(' ')

async function invokeToCompletion(agent: Agent, prompt: string, context: DurableContext): Promise<AgentOutput> {
  let args: InvokeArgs = prompt

  while (true) {
    const result = await agent.invoke(args)
    if (result.stopReason !== 'checkpoint') {
      const text = result.lastMessage.content.map((block) => (block.type === 'textBlock' ? block.text : '')).join('')
      return { stopReason: result.stopReason, text }
    }

    if (result.checkpoint === undefined) {
      throw new Error('Agent returned stopReason=checkpoint without checkpoint data')
    }

    const checkpointData = result.checkpoint.toJSON()
    const stepName = `agent-checkpoint:cycle-${checkpointData.cycleIndex}:${checkpointData.position}`
    const persistedCheckpoint = await context.step<Required<CheckpointData>>(stepName, async (stepContext) => {
      stepContext.logger.info(`checkpoint=<${JSON.stringify(checkpointData)}> | persisting Strands checkpoint`)
      return checkpointData
    })
    const checkpoint = Checkpoint.fromJSON(persistedCheckpoint)

    args = {
      checkpointResume: {
        checkpoint: checkpoint.toJSON(),
      },
    }
  }
}

function defaultSessionId(context: DurableContext): string {
  const digest = createHash('sha256').update(context.executionContext.durableExecutionArn).digest('hex')
  return `execution_${digest}`
}

async function handler(event: AgentEvent, context: DurableContext): Promise<AgentOutput> {
  const prompt = event.prompt ?? DEFAULT_PROMPT
  const sessionId = event.sessionId ?? defaultSessionId(context)
  const sessionBucketName = process.env.SESSION_BUCKET_NAME
  if (sessionBucketName === undefined || sessionBucketName.length === 0) {
    throw new Error('SESSION_BUCKET_NAME is required')
  }

  context.logger.info('agent invocation started', { prompt, sessionId })

  const sessionManager = new SessionManager({
    sessionId,
    storage: new DurableStorage(
      context,
      new S3Storage(sessionBucketName, {
        prefix: 'durable-agent',
      })
    ),
    saveLatestOn: 'invocation',
  })
  const agent = new Agent({
    id: 'durable-agent',
    model: new BedrockModel({ modelId: process.env.BEDROCK_MODEL_ID ?? DEFAULT_MODEL_ID }),
    tools: buildTools(),
    systemPrompt: SYSTEM_PROMPT,
    printer: false,
    toolExecutor: 'sequential',
    checkpointing: true,
    sessionManager,
  })

  registerDurableMiddleware(agent, context, {
    crashAfterFirstTool: event.crashAfterFirstTool === true,
  })

  const output = await invokeToCompletion(agent, prompt, context)
  if (event.simulateRestart === true) {
    await context.wait('replay-trigger', { seconds: 1 })
  }

  context.logger.info('agent invocation completed', { stopReason: output.stopReason, sessionId })
  return output
}

export const lambdaHandler = withDurableExecution(handler)
