import { describe, expect, it, vi } from 'vitest'
import { z } from 'zod'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { Agent } from '../../agent/agent.js'
import type { McpClient } from '../../mcp/client.js'
import { ExecuteToolStage } from '../../middleware/index.js'
import type { ToolExecutionInput, ToolExecutorOptions } from '../../tools/executors/executor.js'
import { FunctionTool } from '../../tools/function-tool.js'
import { McpTool } from '../../tools/mcp-tool.js'
import { SequentialToolExecutor } from '../../tools/executors/sequential.js'
import { tool } from '../../tools/tool-factory.js'
import type { AgentStreamEvent } from '../../types/agent.js'
import { ImageBlock } from '../../types/media.js'
import type { ToolResultBlock, ToolUseBlock } from '../../types/messages.js'
import { BACKGROUND_TASKS_STATE_KEY } from '../in-process-task-manager.js'
import { toBackgroundTask, type StoredBackgroundTask } from '../record.js'
import type { BackgroundTask, BackgroundTasksConfig } from '../types.js'

class RecordingExecutor extends SequentialToolExecutor {
  readonly batchSizes: number[] = []

  override async *execute(
    options: ToolExecutorOptions,
    input: ToolExecutionInput
  ): AsyncGenerator<AgentStreamEvent, void, undefined> {
    this.batchSizes.push(input.toolUseBlocks.length)
    return yield* super.execute(options, input)
  }
}

function backgroundTasks(mode: 'agentic' | 'always'): BackgroundTasksConfig {
  return { timeout: 5_000, [mode]: ['*'] }
}

function taskRecords(agent: Agent): BackgroundTask[] {
  const tasks = agent.appState.get(BACKGROUND_TASKS_STATE_KEY) as
    Record<string, { record: StoredBackgroundTask }> | undefined
  return Object.values(tasks ?? {}).map(({ record }) => toBackgroundTask(record))
}

interface BackgroundDelivery {
  readonly toolUse: ToolUseBlock
  readonly toolResult: ToolResultBlock
}

function backgroundDeliveries(agent: Agent): readonly BackgroundDelivery[] {
  const toolUses: ToolUseBlock[] = []
  const toolResults = new Map<string, ToolResultBlock>()
  for (const message of agent.messages) {
    for (const block of message.content) {
      if (block.type === 'toolUseBlock' && block.name === 'strands_background_task_result') {
        toolUses.push(block)
      } else if (block.type === 'toolResultBlock') {
        toolResults.set(block.toolUseId, block)
      }
    }
  }
  return toolUses.map((toolUse) => {
    const toolResult = toolResults.get(toolUse.toolUseId)
    if (!toolResult) throw new Error(`delivery result '${toolUse.toolUseId}' is missing`)
    return { toolUse, toolResult }
  })
}

async function waitForTaskRecord(agent: Agent, taskId: string): Promise<BackgroundTask> {
  let task: BackgroundTask | undefined
  await vi.waitFor(() => {
    task = taskRecords(agent).find((record) => record.taskId === taskId)
    expect(task?.status).toMatch(/^(paused|completed|failed|cancelled)$/)
  })
  return task!
}

describe('Background Tasks tool parity', () => {
  it('routes the original mixed batch and detached singleton through the configured executor', async () => {
    const executor = new RecordingExecutor()
    const backgroundCallback = vi.fn(() => 'background')
    const foregroundCallback = vi.fn(() => 'foreground')
    const backgroundTool = tool({
      name: 'background',
      description: 'Background work.',
      inputSchema: z.object({ value: z.string() }),
      callback: backgroundCallback,
    })
    const foregroundTool = tool({
      name: 'foreground',
      description: 'Foreground work.',
      inputSchema: z.object({ value: z.string() }),
      callback: foregroundCallback,
    })
    const background = { always: [backgroundTool], never: [foregroundTool], timeout: 5_000 }
    const model = new MockMessageModel()
      .addTurn([
        {
          type: 'toolUseBlock',
          name: 'background',
          toolUseId: 'background-1',
          input: { value: 'x' },
        },
        {
          type: 'toolUseBlock',
          name: 'foreground',
          toolUseId: 'foreground-1',
          input: { value: 'y' },
        },
      ])
      .addTurn({ type: 'textBlock', text: 'admitted' })
      .addTurn({ type: 'textBlock', text: 'delivered' })
    const agent = new Agent({
      model,
      tools: [backgroundTool, foregroundTool],
      backgroundTasks: background,
      toolExecutor: executor,
      printer: false,
    })
    await agent.invoke('run')

    expect(executor.batchSizes).toEqual([2, 1])
    expect(backgroundCallback).toHaveBeenCalledTimes(1)
    expect(foregroundCallback).toHaveBeenCalledTimes(1)
  })

  it('cancels an MCP tool with stripped input through the task signal', async () => {
    const background: BackgroundTasksConfig = {
      timeout: 5_000,
      waitForCompletion: false,
      agentic: ['*'],
    }
    let signalSeen: AbortSignal | undefined
    let finishCall!: () => void
    const callFinished = new Promise<void>((resolve) => {
      finishCall = resolve
    })
    const callTool = vi.fn(async (_tool, input, options) => {
      expect(input).toEqual({ value: 'x' })
      const signal = options.signal
      if (!signal) throw new Error('Expected MCP cancellation signal')
      signalSeen = signal
      try {
        await new Promise<void>((resolve) => {
          if (signal.aborted) {
            resolve()
          } else {
            signal.addEventListener('abort', () => resolve(), { once: true })
          }
        })
        return { content: [{ type: 'text', text: 'remote stopped' }] }
      } finally {
        finishCall()
      }
    })
    const remote = new McpTool({
      name: 'remote',
      description: 'Remote work.',
      inputSchema: {
        type: 'object',
        properties: { value: { type: 'string' } },
        required: ['value'],
      },
      client: { callTool } as unknown as McpClient,
    })
    const model = new MockMessageModel()
      .addTurn({
        type: 'toolUseBlock',
        name: 'remote',
        toolUseId: 'remote-1',
        input: { value: 'x', _background: true },
      })
      .addTurn({ type: 'textBlock', text: 'admitted' })
      .addTurn({ type: 'textBlock', text: 'delivered' })
    const agent = new Agent({
      model,
      tools: [remote],
      backgroundTasks: background,
      printer: false,
    })
    await agent.invoke('run')
    await vi.waitFor(() => expect(callTool).toHaveBeenCalledTimes(1))
    const [running] = taskRecords(agent)

    await agent.tool.strands_manage_background_task!.invoke(
      { mode: 'cancel', taskId: running!.taskId },
      { recordDirectToolCall: false }
    )
    await callFinished

    expect(signalSeen?.aborted).toBe(true)
    expect(await waitForTaskRecord(agent, running!.taskId)).toEqual(expect.objectContaining({ status: 'cancelled' }))
  })

  it('preserves detached middleware failures', async () => {
    const background = backgroundTasks('always')
    const callback = vi.fn(() => 'unused')
    const model = new MockMessageModel()
      .addTurn({
        type: 'toolUseBlock',
        name: 'work',
        toolUseId: 'work-1',
        input: { value: 'x' },
      })
      .addTurn({ type: 'textBlock', text: 'admitted' })
      .addTurn({ type: 'textBlock', text: 'failure delivered' })
    const agent = new Agent({
      model,
      tools: [
        tool({
          name: 'work',
          description: 'Background work.',
          inputSchema: z.object({ value: z.string() }),
          callback,
        }),
      ],
      backgroundTasks: background,
      printer: false,
    })
    // eslint-disable-next-line require-yield
    agent.addMiddleware(ExecuteToolStage, async function* () {
      throw new Error('middleware failed')
    })

    await agent.invoke('run')
    const [delivery] = backgroundDeliveries(agent)
    expect(taskRecords(agent)).toEqual([])
    expect(delivery?.toolUse.input).toEqual(
      expect.objectContaining({
        status: 'failed',
        error: {
          type: 'toolError',
          message: 'middleware failed',
        },
      })
    )
    expect(delivery?.toolResult.content[1]).toEqual(expect.objectContaining({ text: 'middleware failed' }))
    expect(callback).not.toHaveBeenCalled()
  })

  it('preserves tool error content through persistence and delivery', async () => {
    const background = backgroundTasks('always')
    const model = new MockMessageModel()
      .addTurn({
        type: 'toolUseBlock',
        name: 'work',
        toolUseId: 'work-1',
        input: {},
      })
      .addTurn({ type: 'textBlock', text: 'admitted' })
      .addTurn({ type: 'textBlock', text: 'failure delivered' })
    const agent = new Agent({
      model,
      tools: [
        tool({
          name: 'work',
          description: 'Background work.',
          inputSchema: z.object({}),
          callback: () => {
            throw new Error('retry with a smaller value')
          },
        }),
      ],
      backgroundTasks: background,
      printer: false,
    })

    await agent.invoke('run')
    const [delivery] = backgroundDeliveries(agent)
    if (!delivery) throw new Error('background task delivery is missing')
    const taskId = delivery.toolUse.toolUseId

    expect(taskRecords(agent)).toEqual([])
    expect(delivery.toolUse.input).toEqual(
      expect.objectContaining({
        status: 'failed',
        error: {
          type: 'toolError',
          message: 'retry with a smaller value',
        },
      })
    )
    expect(delivery.toolResult.toJSON()).toEqual({
      toolResult: {
        toolUseId: taskId,
        status: 'error',
        content: [
          {
            text: [
              'Background task failed.',
              '',
              `Task ID: ${taskId}`,
              'Tool: work',
              'Status: failed',
              'Error type: toolError',
              'Reason: retry with a smaller value',
              '',
              'The tool error follows.',
            ].join('\n'),
          },
          { text: 'Error: retry with a smaller value' },
        ],
      },
    })
  })

  it('preserves multimodal output through persistence and delivery', async () => {
    const background = backgroundTasks('always')
    const media = new FunctionTool({
      name: 'media',
      description: 'Return an image.',
      inputSchema: { type: 'object', properties: {} },
      callback: () =>
        new ImageBlock({
          format: 'png',
          source: { bytes: new Uint8Array([1, 2, 3]) },
        }) as never,
    })
    const model = new MockMessageModel()
      .addTurn({
        type: 'toolUseBlock',
        name: 'media',
        toolUseId: 'media-1',
        input: {},
      })
      .addTurn({ type: 'textBlock', text: 'admitted' })
      .addTurn({ type: 'textBlock', text: 'delivered' })
    const agent = new Agent({
      model,
      tools: [media],
      backgroundTasks: background,
      printer: false,
    })
    await agent.invoke('run')

    const [delivery] = backgroundDeliveries(agent)
    if (!delivery) throw new Error('background task delivery is missing')
    expect(taskRecords(agent)).toEqual([])
    expect(delivery.toolUse.input).toEqual(expect.objectContaining({ status: 'completed', toolName: 'media' }))
    expect(delivery.toolResult.content[1]?.toJSON()).toEqual({
      image: {
        format: 'png',
        source: { bytes: 'AQID' },
      },
    })
    expect(delivery.toolResult.content[1]).toEqual(expect.objectContaining({ type: 'imageBlock' }))
  })

  it('backgrounds an AgentAsTool like any other tool', async () => {
    const innerCallback = vi.fn(() => 'inner complete')
    const inner = tool({
      name: 'inner',
      description: 'Nested work.',
      inputSchema: z.object({ value: z.string() }),
      callback: innerCallback,
    })
    const childModel = new MockMessageModel()
      .addTurn({
        type: 'toolUseBlock',
        name: 'inner',
        toolUseId: 'inner-1',
        input: { value: 'x' },
      })
      .addTurn({ type: 'textBlock', text: 'child complete' })
    const child = new Agent({
      id: 'child-agent',
      name: 'child',
      model: childModel,
      tools: [inner],
      printer: false,
    })
    const parentBackground = backgroundTasks('always')
    const parentModel = new MockMessageModel()
      .addTurn({
        type: 'toolUseBlock',
        name: 'child',
        toolUseId: 'child-1',
        input: { input: 'run child' },
      })
      .addTurn({ type: 'textBlock', text: 'admitted' })
      .addTurn({ type: 'textBlock', text: 'delivered' })
    const parent = new Agent({
      id: 'parent-agent',
      model: parentModel,
      tools: [child.asTool()],
      backgroundTasks: parentBackground,
      printer: false,
    })
    await parent.invoke('run')

    expect(innerCallback).toHaveBeenCalledTimes(1)
    expect(taskRecords(parent)).toEqual([])
    expect(backgroundDeliveries(parent)[0]?.toolUse.input).toEqual(
      expect.objectContaining({ status: 'completed', toolName: 'child' })
    )
    expect(backgroundDeliveries(parent)[0]?.toolResult.content[1]).toEqual(
      expect.objectContaining({ text: 'child complete' })
    )
  })
})
