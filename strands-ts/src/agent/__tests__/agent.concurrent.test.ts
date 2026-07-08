import { describe, expect, it } from 'vitest'
import { Agent } from '../agent.js'
import {
  AfterToolCallEvent,
  AfterToolsEvent,
  BeforeToolCallEvent,
  BeforeToolsEvent,
  ToolResultEvent,
  ToolStreamUpdateEvent,
} from '../../hooks/index.js'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { MockPlugin } from '../../__fixtures__/mock-plugin.js'
import { Message, TextBlock, ToolResultBlock } from '../../types/messages.js'
import { Tool, ToolStreamEvent, type ToolContext, type ToolStreamGenerator } from '../../tools/tool.js'
import type { ToolSpec } from '../../tools/types.js'

/**
 * A tool whose `stream()` suspends until `release()` is called. Lets tests
 * drive concurrency deterministically without wall-clock sleeps.
 *
 * `started` resolves as soon as the agent enters the tool's `stream()`, so
 * tests can await "both tools in flight" without polling. The tool also
 * honors `ctx.agent.cancelSignal`: aborting the signal resolves the gate and
 * marks `observations.cancelled = true`.
 */
class GatedTool extends Tool {
  name: string
  description: string
  toolSpec: ToolSpec

  readonly started: Promise<void>
  readonly observations = { started: false, cancelled: false, completed: false }

  private _signalStarted!: () => void
  private readonly _releaser: Promise<void>
  private _release!: () => void

  constructor(name: string) {
    super()
    this.name = name
    this.description = `Gated tool ${name}`
    this.toolSpec = { name, description: this.description, inputSchema: { type: 'object', properties: {} } }
    this.started = new Promise<void>((resolve) => (this._signalStarted = resolve))
    this._releaser = new Promise<void>((resolve) => (this._release = resolve))
  }

  release(): void {
    this._release()
  }

  // eslint-disable-next-line require-yield
  async *stream(ctx: ToolContext): ToolStreamGenerator {
    this.observations.started = true
    this._signalStarted()

    await new Promise<void>((resolve) => {
      void this._releaser.then(resolve)
      ctx.agent.cancelSignal.addEventListener(
        'abort',
        () => {
          this.observations.cancelled = true
          resolve()
        },
        { once: true }
      )
    })

    this.observations.completed = true
    return new ToolResultBlock({
      toolUseId: ctx.toolUse.toolUseId,
      status: 'success',
      content: [new TextBlock(`${this.name} done`)],
    })
  }
}

/**
 * A streaming tool whose `emit(data)` yields a `ToolStreamEvent` and resolves
 * only after the agent has fully dispatched it; `complete()` terminates the
 * stream. Tests can drive exact interleaving between tools without timers.
 */
class GatedStreamingTool extends Tool {
  name: string
  description: string
  toolSpec: ToolSpec

  private readonly _queue: { cmd: { type: 'emit'; data: unknown } | { type: 'complete' }; ack: () => void }[] = []
  private _notify: (() => void) | null = null

  constructor(name: string) {
    super()
    this.name = name
    this.description = `Gated streaming tool ${name}`
    this.toolSpec = { name, description: this.description, inputSchema: { type: 'object', properties: {} } }
  }

  async emit(data: unknown): Promise<void> {
    return this._send({ type: 'emit', data })
  }

  async complete(): Promise<void> {
    return this._send({ type: 'complete' })
  }

  private _send(cmd: { type: 'emit'; data: unknown } | { type: 'complete' }): Promise<void> {
    return new Promise<void>((ack) => {
      this._queue.push({ cmd, ack })
      this._notify?.()
      this._notify = null
    })
  }

  async *stream(ctx: ToolContext): ToolStreamGenerator {
    while (true) {
      while (this._queue.length === 0) {
        await new Promise<void>((resolve) => (this._notify = resolve))
      }
      const { cmd, ack } = this._queue.shift()!
      if (cmd.type === 'complete') {
        ack()
        return new ToolResultBlock({
          toolUseId: ctx.toolUse.toolUseId,
          status: 'success',
          content: [new TextBlock(`${this.name} done`)],
        })
      }
      yield new ToolStreamEvent({ data: cmd.data })
      ack()
    }
  }
}

function twoToolTurn(): MockMessageModel {
  return new MockMessageModel()
    .addTurn([
      { type: 'toolUseBlock', name: 'toolA', toolUseId: 'a', input: {} },
      { type: 'toolUseBlock', name: 'toolB', toolUseId: 'b', input: {} },
    ])
    .addTurn({ type: 'textBlock', text: 'Done' })
}

describe('Agent concurrent tool execution', () => {
  it('runs tools concurrently by default', async () => {
    const toolA = new GatedTool('toolA')
    const toolB = new GatedTool('toolB')
    const agent = new Agent({
      model: twoToolTurn(),
      tools: [toolA, toolB],
      // no toolExecutor — relies on the concurrent default
      printer: false,
    })

    const invocation = agent.invoke('Go')
    // Both tools reach their stream() before either is released — proves
    // concurrency without relying on wall-clock overlap.
    await Promise.all([toolA.started, toolB.started])
    expect(toolA.observations.completed).toBe(false)
    expect(toolB.observations.completed).toBe(false)
    toolA.release()
    toolB.release()
    await invocation
    expect(toolA.observations.completed).toBe(true)
    expect(toolB.observations.completed).toBe(true)
  })

  it('runs tools sequentially when toolExecutor is sequential', async () => {
    const toolA = new GatedTool('toolA')
    const toolB = new GatedTool('toolB')
    const agent = new Agent({
      model: twoToolTurn(),
      tools: [toolA, toolB],
      toolExecutor: 'sequential',
      printer: false,
    })

    const invocation = agent.invoke('Go')
    await toolA.started
    // B has not started — sequential executor is still blocked on A.
    expect(toolB.observations.started).toBe(false)
    toolA.release()
    await toolB.started
    toolB.release()
    await invocation
  })

  it('preserves per-tool event ordering while interleaving across tools', async () => {
    const toolA = new GatedStreamingTool('toolA')
    const toolB = new GatedStreamingTool('toolB')
    const plugin = new MockPlugin()
    const agent = new Agent({
      model: twoToolTurn(),
      tools: [toolA, toolB],
      toolExecutor: 'concurrent',
      printer: false,
      plugins: [plugin],
    })

    const invocation = agent.invoke('Go')
    // Drive explicit A,B,A,B,A,B interleaving.
    await toolA.emit({ tool: 'toolA', step: 0 })
    await toolB.emit({ tool: 'toolB', step: 0 })
    await toolA.emit({ tool: 'toolA', step: 1 })
    await toolB.emit({ tool: 'toolB', step: 1 })
    await toolA.emit({ tool: 'toolA', step: 2 })
    await toolB.emit({ tool: 'toolB', step: 2 })
    await toolA.complete()
    await toolB.complete()
    await invocation

    // Reduce MockPlugin's invocations to the per-tool lifecycle events we care about.
    type Entry = { kind: string; toolUseId?: string; tool?: string }
    const events: Entry[] = plugin.invocations
      .map((e): Entry | null => {
        if (e instanceof BeforeToolCallEvent) return { kind: 'before', toolUseId: e.toolUse.toolUseId }
        if (e instanceof AfterToolCallEvent) return { kind: 'after', toolUseId: e.toolUse.toolUseId }
        if (e instanceof ToolResultEvent) return { kind: 'result', toolUseId: e.result.toolUseId }
        if (e instanceof ToolStreamUpdateEvent) {
          const data = e.event.data as { tool?: string } | undefined
          return data?.tool !== undefined ? { kind: 'stream', tool: data.tool } : { kind: 'stream' }
        }
        return null
      })
      .filter((e): e is Entry => e !== null)

    // Per-tool subsequence shape: [before, stream*, after, result].
    for (const toolUseId of ['a', 'b']) {
      const subseq = events.filter(
        (e) => e.toolUseId === toolUseId || (e.kind === 'stream' && e.tool === (toolUseId === 'a' ? 'toolA' : 'toolB'))
      )
      const kinds = subseq.map((e) => e.kind)
      expect(kinds[0]).toBe('before')
      expect(kinds.slice(-2)).toEqual(['after', 'result'])
      for (const k of kinds.slice(1, -2)) {
        expect(k).toBe('stream')
      }
    }

    // Cross-tool interleaving: collapse consecutive same-tool stream events
    // into runs. Strictly sequential execution produces 2 runs (A,A,A,B,B,B);
    // anything > 2 means the stream alternated at least once.
    const streamTools = events.filter((e) => e.kind === 'stream').map((e) => e.tool)
    const runs = streamTools.reduce<(string | undefined)[]>((acc, t) => {
      if (acc.length === 0 || acc[acc.length - 1] !== t) acc.push(t)
      return acc
    }, [])
    expect(runs.length).toBeGreaterThan(2)
  })

  it('retries one tool independently from the other', async () => {
    let retriesA = 0
    const toolA = new GatedTool('toolA')
    const toolB = new GatedTool('toolB')
    const agent = new Agent({
      model: twoToolTurn(),
      tools: [toolA, toolB],
      toolExecutor: 'concurrent',
      printer: false,
    })

    const beforeCalls: string[] = []
    agent.addHook(BeforeToolCallEvent, (e) => void beforeCalls.push(e.toolUse.name))
    agent.addHook(AfterToolCallEvent, (e) => {
      if (e.toolUse.name === 'toolA' && retriesA === 0) {
        retriesA++
        e.retry = true
      }
    })

    const invocation = agent.invoke('Go')
    // Release both gates; on retry A re-enters with an already-resolved
    // releaser and completes immediately.
    await Promise.all([toolA.started, toolB.started])
    toolA.release()
    toolB.release()
    await invocation

    expect(beforeCalls.filter((n) => n === 'toolA')).toHaveLength(2)
    expect(beforeCalls.filter((n) => n === 'toolB')).toHaveLength(1)
  })

  it('cancels all tools when BeforeToolsEvent.cancel is set (concurrent mode)', async () => {
    const toolA = new GatedTool('toolA')
    const toolB = new GatedTool('toolB')
    const agent = new Agent({
      model: twoToolTurn(),
      tools: [toolA, toolB],
      toolExecutor: 'concurrent',
      printer: false,
    })

    agent.addHook(BeforeToolsEvent, (e) => {
      e.cancel = 'hook cancelled'
    })

    let afterMessage: Message | undefined
    agent.addHook(AfterToolsEvent, (e) => {
      afterMessage = e.message
    })

    await agent.invoke('Go')

    // No tool ever ran.
    expect(toolA.observations.started).toBe(false)
    expect(toolB.observations.started).toBe(false)
    expect(afterMessage!.content).toHaveLength(2)
    const r0 = afterMessage!.content[0] as ToolResultBlock
    const r1 = afterMessage!.content[1] as ToolResultBlock
    expect(r0.status).toBe('error')
    expect(r1.status).toBe('error')
    expect(r0.toolUseId).toBe('a')
    expect(r1.toolUseId).toBe('b')
  })

  it('cancels all tools when agent is cancelled before launch (concurrent mode)', async () => {
    const toolA = new GatedTool('toolA')
    const toolB = new GatedTool('toolB')
    const agent = new Agent({
      model: twoToolTurn(),
      tools: [toolA, toolB],
      toolExecutor: 'concurrent',
      printer: false,
    })

    agent.addHook(BeforeToolsEvent, () => {
      agent.cancel()
    })

    await agent.invoke('Go')
    expect(toolA.observations.started).toBe(false)
    expect(toolB.observations.started).toBe(false)
  })

  it('cooperative mid-flight cancel — tools honor cancelSignal and exit', async () => {
    const toolA = new GatedTool('toolA')
    const toolB = new GatedTool('toolB')
    const agent = new Agent({
      model: twoToolTurn(),
      tools: [toolA, toolB],
      toolExecutor: 'concurrent',
      printer: false,
    })

    // Cancel deterministically once both tools have entered their gates.
    void Promise.all([toolA.started, toolB.started]).then(() => agent.cancel())

    await agent.invoke('Go')

    expect(toolA.observations.cancelled).toBe(true)
    expect(toolB.observations.cancelled).toBe(true)
  })

  it('handles a throwing tool without affecting siblings', async () => {
    const toolA = new GatedTool('toolA')
    const toolB = new GatedTool('toolB')
    const agent = new Agent({
      model: twoToolTurn(),
      tools: [toolA, toolB],
      toolExecutor: 'concurrent',
      printer: false,
    })

    // A throwing tool.stream is caught by executeTool's own try/catch and
    // normalized to an error ToolResultBlock, so the race loop never sees the
    // rejection. This test verifies that normalization path keeps the sibling
    // unaffected in concurrent mode. The race loop's `kind: 'throw'` fallback
    // is a defensive backstop for generator-level rejections that escape
    // executeTool entirely — not expected in normal operation and not exercised
    // here.
    const results: ToolResultBlock[] = []
    agent.addHook(AfterToolsEvent, (e) => {
      for (const b of e.message.content) {
        if (b.type === 'toolResultBlock') results.push(b)
      }
    })

    // eslint-disable-next-line require-yield
    toolA.stream = async function* () {
      throw new Error('boom')
    }

    const invocation = agent.invoke('Go')
    await toolB.started
    toolB.release()
    await invocation

    const [a, b] = results.sort((x, y) => x.toolUseId.localeCompare(y.toolUseId))
    expect(a!.status).toBe('error')
    expect(b!.status).toBe('success')
  })

  it('handles a hallucinated tool name in a batch without affecting siblings', async () => {
    const toolA = new GatedTool('toolA')
    const agent = new Agent({
      model: new MockMessageModel()
        .addTurn([
          { type: 'toolUseBlock', name: 'toolA', toolUseId: 'a', input: {} },
          { type: 'toolUseBlock', name: 'unknownTool', toolUseId: 'b', input: {} },
        ])
        .addTurn({ type: 'textBlock', text: 'Done' }),
      tools: [toolA],
      toolExecutor: 'concurrent',
      printer: false,
    })

    let afterMessage: Message | undefined
    agent.addHook(AfterToolsEvent, (e) => {
      afterMessage = e.message
    })

    const invocation = agent.invoke('Go')
    await toolA.started
    toolA.release()
    await invocation

    expect(afterMessage!.content).toHaveLength(2)
    const blocks = afterMessage!.content as ToolResultBlock[]
    expect(blocks.find((r) => r.toolUseId === 'a')!.status).toBe('success')
    expect(blocks.find((r) => r.toolUseId === 'b')!.status).toBe('error')
  })

  it('preserves source order of tool results in AfterToolsEvent.message', async () => {
    const toolA = new GatedTool('toolA')
    const toolB = new GatedTool('toolB')
    const agent = new Agent({
      model: twoToolTurn(),
      tools: [toolA, toolB],
      toolExecutor: 'concurrent',
      printer: false,
    })

    // Deterministically complete B before A.
    let resolveBDone: () => void = () => {}
    const bDone = new Promise<void>((resolve) => (resolveBDone = resolve))
    agent.addHook(ToolResultEvent, (e) => {
      if (e.result.toolUseId === 'b') resolveBDone()
    })
    let afterMessage: Message | undefined
    agent.addHook(AfterToolsEvent, (e) => {
      afterMessage = e.message
    })

    const invocation = agent.invoke('Go')
    await Promise.all([toolA.started, toolB.started])
    toolB.release()
    await bDone
    toolA.release()
    await invocation

    const blocks = afterMessage!.content as ToolResultBlock[]
    expect(blocks.map((b) => b.toolUseId)).toEqual(['a', 'b'])
  })

  it('AfterToolsEvent.message contains completed results when consumer breaks mid-stream', async () => {
    const toolA = new GatedTool('toolA')
    const toolB = new GatedTool('toolB') // never released
    const agent = new Agent({
      model: twoToolTurn(),
      tools: [toolA, toolB],
      toolExecutor: 'concurrent',
      printer: false,
    })

    agent.addHook(BeforeToolCallEvent, (e) => {
      if (e.toolUse.name === 'toolA') toolA.release()
    })

    let afterToolsMessage: Message | undefined
    agent.addHook(AfterToolsEvent, (e) => {
      afterToolsMessage = e.message
    })

    let toolResultsSeen = 0
    for await (const event of agent.stream('Go')) {
      if (event.type === 'toolResultEvent') {
        toolResultsSeen++
        if (toolResultsSeen === 1) {
          // Cancel so toolB (still parked on its gate) observes cancelSignal
          // and exits cooperatively — otherwise gen.return() stays blocked on
          // a suspended await.
          agent.cancel()
          break
        }
      }
    }

    expect(afterToolsMessage).toBeDefined()
    const blocks = afterToolsMessage!.content.filter((b): b is ToolResultBlock => b.type === 'toolResultBlock')
    expect(blocks.length).toBeGreaterThanOrEqual(1)
    expect(blocks.some((b) => b.toolUseId === 'a')).toBe(true)
  })

  it('pre-launch agent.cancel() during BeforeToolsEvent produces "Tool execution cancelled" (concurrent)', async () => {
    const toolA = new GatedTool('toolA')
    const toolB = new GatedTool('toolB')
    const agent = new Agent({
      model: twoToolTurn(),
      tools: [toolA, toolB],
      toolExecutor: 'concurrent',
      printer: false,
    })

    agent.addHook(BeforeToolsEvent, () => {
      agent.cancel()
    })

    let afterMessage: Message | undefined
    agent.addHook(AfterToolsEvent, (e) => {
      afterMessage = e.message
    })

    await agent.invoke('Go')

    expect(toolA.observations.started).toBe(false)
    expect(toolB.observations.started).toBe(false)
    const blocks = afterMessage!.content as ToolResultBlock[]
    expect(blocks).toHaveLength(2)
    for (const b of blocks) {
      expect((b.content[0] as TextBlock).text).toBe('Tool execution cancelled')
    }
  })

  it('closes in-flight generators and includes fallback results when consumer breaks', async () => {
    const toolA = new GatedTool('toolA')
    const toolB = new GatedTool('toolB') // never released
    const agent = new Agent({
      model: twoToolTurn(),
      tools: [toolA, toolB],
      toolExecutor: 'concurrent',
      printer: false,
    })

    agent.addHook(BeforeToolCallEvent, (e) => {
      if (e.toolUse.name === 'toolA') toolA.release()
    })

    let afterToolsMessage: Message | undefined
    agent.addHook(AfterToolsEvent, (e) => {
      afterToolsMessage = e.message
    })

    let toolResultsSeen = 0
    for await (const event of agent.stream('Go')) {
      if (event.type === 'toolResultEvent') {
        toolResultsSeen++
        if (toolResultsSeen === 1) {
          // Cancel so toolB (still parked on its gate) observes cancelSignal
          // and exits cooperatively — otherwise gen.return() stays blocked on
          // a suspended await.
          agent.cancel()
          break
        }
      }
    }

    // AfterToolsEvent.message should have entries for both tools:
    // toolA completed normally, toolB gets a fallback "interrupted" result.
    expect(afterToolsMessage).toBeDefined()
    const blocks = afterToolsMessage!.content as ToolResultBlock[]
    expect(blocks).toHaveLength(2)
    expect(blocks.map((b) => b.toolUseId)).toEqual(['a', 'b'])
    expect(blocks.find((b) => b.toolUseId === 'a')!.status).toBe('success')
    expect(blocks.find((b) => b.toolUseId === 'b')!.status).toBe('error')
  })
})
