import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { metrics as otelMetrics, type Meter as OtelMeter } from '@opentelemetry/api'
import { MockMeter } from '../../__fixtures__/mock-meter.js'
import { BackgroundTaskTelemetry } from '../telemetry.js'

describe('background task telemetry', () => {
  let mockMeter: MockMeter

  beforeEach(() => {
    mockMeter = new MockMeter()
    vi.spyOn(otelMetrics, 'getMeter').mockReturnValue(mockMeter as unknown as OtelMeter)
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('emits admission, duration, attempt, failure, timeout, cancellation, and terminal metrics', () => {
    const telemetry = new BackgroundTaskTelemetry()

    telemetry.recordAdmission('search')
    telemetry.recordExecutionStarted({ toolName: 'search', attempt: 2, resumed: false, queueDuration: 125 })
    telemetry.recordExecutionFinished({ toolName: 'search', outcome: 'failed', duration: 450 })
    telemetry.recordFailure('search', 'timeout')
    telemetry.recordCancellation('search')
    telemetry.recordTerminal('search', 'failed')

    expect(mockMeter.getCounter('gen_ai.agent.background_task.admitted.count')?.sum).toBe(1)
    expect(mockMeter.getCounter('gen_ai.agent.background_task.execution.count')?.dataPoints).toEqual([
      {
        value: 1,
        attributes: {
          'gen_ai.tool.name': 'search',
          'background_task.attempt': 2,
          'background_task.resumed': false,
        },
      },
    ])
    expect(mockMeter.getHistogram('gen_ai.agent.background_task.queue.duration')?.dataPoints[0]?.value).toBe(125)
    expect(mockMeter.getHistogram('gen_ai.agent.background_task.execution.duration')?.dataPoints).toEqual([
      {
        value: 450,
        attributes: {
          'gen_ai.tool.name': 'search',
          'background_task.outcome': 'failed',
        },
      },
    ])
    expect(mockMeter.getCounter('gen_ai.agent.background_task.failure.count')?.dataPoints[0]?.attributes).toEqual({
      'gen_ai.tool.name': 'search',
      'background_task.failure.type': 'timeout',
    })
    expect(mockMeter.getCounter('gen_ai.agent.background_task.cancellation.count')?.dataPoints[0]?.attributes).toEqual({
      'gen_ai.tool.name': 'search',
    })
    expect(mockMeter.getCounter('gen_ai.agent.background_task.terminal.count')?.dataPoints[0]?.attributes).toEqual({
      'gen_ai.tool.name': 'search',
      'background_task.status': 'failed',
    })
  })
})
