import { metrics as otelMetrics, type Attributes, type Counter, type Histogram } from '@opentelemetry/api'
import { getServiceName } from '../telemetry/utils.js'
import type { BackgroundTask } from './types.js'

/** Emits bounded-cardinality background task metrics. */
export class BackgroundTaskTelemetry {
  private readonly _admitted: Counter
  private readonly _execution: Counter
  private readonly _queueDuration: Histogram
  private readonly _executionDuration: Histogram
  private readonly _failure: Counter
  private readonly _cancellation: Counter
  private readonly _terminal: Counter

  constructor() {
    const meter = otelMetrics.getMeter(getServiceName())
    this._admitted = meter.createCounter('gen_ai.agent.background_task.admitted.count', {
      description: 'Number of background tasks durably admitted',
    })
    this._execution = meter.createCounter('gen_ai.agent.background_task.execution.count', {
      description: 'Number of physical background task executions started',
    })
    this._queueDuration = meter.createHistogram('gen_ai.agent.background_task.queue.duration', {
      description: 'Time background task executions spend queued in milliseconds',
      unit: 'ms',
    })
    this._executionDuration = meter.createHistogram('gen_ai.agent.background_task.execution.duration', {
      description: 'Duration of physical background task executions in milliseconds',
      unit: 'ms',
    })
    this._failure = meter.createCounter('gen_ai.agent.background_task.failure.count', {
      description: 'Number of failed background task attempts',
    })
    this._cancellation = meter.createCounter('gen_ai.agent.background_task.cancellation.count', {
      description: 'Number of background task cancellation requests accepted',
    })
    this._terminal = meter.createCounter('gen_ai.agent.background_task.terminal.count', {
      description: 'Number of background tasks committed to a terminal status',
    })
  }

  recordAdmission(toolName: string): void {
    this._admitted.add(1, toolAttributes(toolName))
  }

  recordExecutionStarted(options: {
    toolName: string
    attempt: number
    resumed: boolean
    queueDuration: number
  }): void {
    const attributes: Attributes = {
      ...toolAttributes(options.toolName),
      'background_task.attempt': options.attempt,
      'background_task.resumed': options.resumed,
    }
    this._execution.add(1, attributes)
    this._queueDuration.record(Math.max(0, options.queueDuration), attributes)
  }

  recordExecutionFinished(options: {
    toolName: string
    outcome: 'completed' | 'paused' | 'failed' | 'cancelled' | 'executionError'
    duration: number
  }): void {
    this._executionDuration.record(Math.max(0, options.duration), {
      ...toolAttributes(options.toolName),
      'background_task.outcome': options.outcome,
    })
  }

  recordFailure(toolName: string, failureType: string): void {
    this._failure.add(1, {
      ...toolAttributes(toolName),
      'background_task.failure.type': failureType,
    })
  }

  recordCancellation(toolName: string): void {
    this._cancellation.add(1, toolAttributes(toolName))
  }

  recordTerminal(
    toolName: string,
    status: Extract<BackgroundTask['status'], 'completed' | 'failed' | 'cancelled'>
  ): void {
    this._terminal.add(1, {
      ...toolAttributes(toolName),
      'background_task.status': status,
    })
  }
}

function toolAttributes(toolName: string): Attributes {
  return { 'gen_ai.tool.name': toolName }
}
