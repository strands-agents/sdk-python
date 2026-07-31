import { createRequire } from 'node:module'

import { describe, expect, it, vi } from 'vitest'
import {
  SandboxFilesystemNotADirectoryError,
  SandboxFilesystemNotFoundError,
  SandboxTimeoutError as ModalSandboxTimeoutError,
} from 'modal'
import type {
  ContainerProcess,
  FileInfo as ModalFileInfo,
  ModalReadStream,
  Sandbox as ModalClientSandbox,
  SandboxExecParams,
} from 'modal'

import { ModalSandbox } from '../modal.js'
import { SandboxAbortError, SandboxPathNotFoundError, SandboxTimeoutError } from '../../../sandbox/errors.js'
import { SANDBOX_BASH_DESCRIPTION } from '../../../vended-tools/bash/types.js'

const modalCommonJs = createRequire(import.meta.url)('modal') as typeof import('modal')

interface ControlledProcess {
  process: ContainerProcess<string>
  complete(exitCode: number): void
}

function makeReadable(chunks: string[], close = true): ModalReadStream<string> {
  const stream = new ReadableStream<string>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(chunk)
      }
      if (close) {
        controller.close()
      }
    },
  })

  return Object.assign(stream, {
    async readText(): Promise<string> {
      const reader = stream.getReader()
      const output: string[] = []
      try {
        while (true) {
          const { value, done } = await reader.read()
          if (done) {
            return output.join('')
          }
          output.push(value)
        }
      } finally {
        reader.releaseLock()
      }
    },
    async readBytes(): Promise<Uint8Array> {
      return new TextEncoder().encode(await this.readText())
    },
  })
}

function makeStdin(): ContainerProcess<string>['stdin'] {
  return {
    close: vi.fn().mockResolvedValue(undefined),
  } as unknown as ContainerProcess<string>['stdin']
}

function makeProcess(stdout: string[], stderr: string[], exitCode: number): ContainerProcess<string> {
  return {
    stdin: makeStdin(),
    stdout: makeReadable(stdout),
    stderr: makeReadable(stderr),
    wait: vi.fn().mockResolvedValue(exitCode),
  } as unknown as ContainerProcess<string>
}

function makeRejectedProcess(error: Error): ContainerProcess<string> {
  return {
    stdin: makeStdin(),
    stdout: makeReadable([]),
    stderr: makeReadable([]),
    wait: vi.fn().mockRejectedValue(error),
  } as unknown as ContainerProcess<string>
}

function makeControlledProcess(initialStdout?: string): ControlledProcess {
  let stdoutController: ReadableStreamDefaultController<string> | undefined
  let stderrController: ReadableStreamDefaultController<string> | undefined
  let resolveWait: ((exitCode: number) => void) | undefined
  const stdout = new ReadableStream<string>({
    start(controller) {
      stdoutController = controller
      if (initialStdout !== undefined) {
        controller.enqueue(initialStdout)
      }
    },
  })
  const stderr = new ReadableStream<string>({
    start(controller) {
      stderrController = controller
    },
  })
  const wait = new Promise<number>((resolve) => {
    resolveWait = resolve
  })

  return {
    process: {
      stdin: makeStdin(),
      stdout,
      stderr,
      wait: vi.fn(() => wait),
    } as unknown as ContainerProcess<string>,
    complete(exitCode: number): void {
      stdoutController?.close()
      stderrController?.close()
      resolveWait?.(exitCode)
    },
  }
}

function makeFilesystem() {
  return {
    readBytes: vi.fn(async (_path: string): Promise<Uint8Array> => new Uint8Array()),
    writeBytes: vi.fn(async (_content: Uint8Array, _path: string): Promise<void> => {}),
    remove: vi.fn(async (_path: string): Promise<void> => {}),
    stat: vi.fn(async (_path: string): Promise<ModalFileInfo> => {
      return { name: 'file', path: '/file', type: 'file', size: 0 } as ModalFileInfo
    }),
    listFiles: vi.fn(async (_path: string): Promise<ModalFileInfo[]> => []),
  }
}

function makeRemote(
  exec: (command: string[], params?: SandboxExecParams) => Promise<ContainerProcess>,
  filesystem = makeFilesystem()
): ModalClientSandbox {
  return {
    sandboxId: 'sb-test-123',
    exec,
    filesystem,
  } as unknown as ModalClientSandbox
}

describe('ModalSandbox', () => {
  it('does not register process handlers when imported', async () => {
    const events = ['beforeExit', 'exit', 'SIGINT', 'SIGTERM'] as const
    const listenersBeforeImport = events.map((event) => process.listenerCount(event))

    vi.resetModules()
    await import('../modal.js')

    expect(events.map((event) => process.listenerCount(event))).toEqual(listenersBeforeImport)
  })

  describe('constructor', () => {
    it('stores the Modal Sandbox and absolute working directory', () => {
      const remote = makeRemote(async () => makeProcess([], [], 0))
      const sandbox = new ModalSandbox({ sandbox: remote, workingDir: '/workspace' })

      expect({ modalSandbox: sandbox.modalSandbox, workingDir: sandbox.workingDir }).toEqual({
        modalSandbox: remote,
        workingDir: '/workspace',
      })
    })

    it('rejects a relative working directory', () => {
      const remote = makeRemote(async () => makeProcess([], [], 0))

      expect(() => new ModalSandbox({ sandbox: remote, workingDir: 'workspace' })).toThrow(
        'workingDir must be absolute'
      )
    })
  })

  describe('executeStreaming', () => {
    it('streams both output channels and returns the complete result', async () => {
      const remoteProcess = makeProcess(['hello', '\n'], ['warning\n'], 7)
      const exec = vi.fn(async (_command: string[], _params?: SandboxExecParams) => remoteProcess)
      const sandbox = new ModalSandbox({ sandbox: makeRemote(exec), workingDir: '/workspace' })

      const events = []
      for await (const event of sandbox.executeStreaming('echo hello', {
        cwd: '/override',
        env: { FOO: 'bar', STRANDS_MODAL_COMMAND_ID: 'caller-value' },
        timeout: 1.2,
      })) {
        events.push(event)
      }

      expect(events.filter((event) => event.type === 'streamChunk')).toEqual(
        expect.arrayContaining([
          { type: 'streamChunk', data: 'hello', streamType: 'stdout' },
          { type: 'streamChunk', data: '\n', streamType: 'stdout' },
          { type: 'streamChunk', data: 'warning\n', streamType: 'stderr' },
        ])
      )
      expect(events.at(-1)).toEqual({
        type: 'executionResult',
        exitCode: 7,
        stdout: 'hello\n',
        stderr: 'warning\n',
        outputFiles: [],
      })

      const [command, params] = exec.mock.calls[0]!
      expect(command.slice(0, 3)).toEqual(['sh', '-c', expect.stringContaining('setsid --fork --wait')])
      expect(command[2]).toContain('export "$command_tag_key=$command_id"')
      expect(command.slice(-5)).toEqual([
        expect.stringMatching(/^\/tmp\/\.strands-modal\/.+\.pid$/),
        expect.stringMatching(/^\/tmp\/\.strands-modal\/.+\.cancel$/),
        'echo hello',
        expect.stringMatching(/^STRANDS_MODAL_COMMAND_[0-9A-F]{32}$/),
        expect.stringMatching(/^[0-9a-f-]+$/),
      ])
      expect(params).toEqual({
        mode: 'text',
        stdout: 'pipe',
        stderr: 'pipe',
        workdir: '/override',
        env: { FOO: 'bar', STRANDS_MODAL_COMMAND_ID: 'caller-value' },
        timeoutMs: 3000,
      })
      expect(remoteProcess.stdin.close).toHaveBeenCalledOnce()
    })

    it('rejects invalid environment variable names before starting a process', async () => {
      const exec = vi.fn(async () => makeProcess([], [], 0))
      const sandbox = new ModalSandbox({ sandbox: makeRemote(exec) })

      await expect(sandbox.execute('true', { env: { 'BAD-NAME': 'value' } })).rejects.toThrow(
        'Invalid environment variable name'
      )
      expect(exec).not.toHaveBeenCalled()
    })

    it.each([0, -1, Number.NaN, Number.POSITIVE_INFINITY])(
      'rejects a non-positive or non-finite timeout before starting a process',
      async (timeout) => {
        const exec = vi.fn(async () => makeProcess([], [], 0))
        const sandbox = new ModalSandbox({ sandbox: makeRemote(exec) })

        await expect(sandbox.execute('true', { timeout })).rejects.toThrow(
          'timeout must be a finite positive number of seconds'
        )
        expect(exec).not.toHaveBeenCalled()
      }
    )

    it('translates a Modal process timeout', async () => {
      const controlled = makeControlledProcess()
      const exec = vi.fn(async (command: string[]) => {
        if (command[3] === 'strands-modal-terminate') {
          controlled.complete(137)
          return makeProcess([], [], 0)
        }
        return makeRejectedProcess(new ModalSandboxTimeoutError())
      })
      const sandbox = new ModalSandbox({ sandbox: makeRemote(exec) })

      await expect(sandbox.execute('sleep 60', { timeout: 5 })).rejects.toBeInstanceOf(SandboxTimeoutError)
      expect(exec).toHaveBeenCalledTimes(2)
    })

    it('translates a Modal process timeout from the CommonJS package build', async () => {
      const controlled = makeControlledProcess()
      const exec = vi.fn(async (command: string[]) => {
        if (command[3] === 'strands-modal-terminate') {
          controlled.complete(137)
          return makeProcess([], [], 0)
        }
        return makeRejectedProcess(new modalCommonJs.SandboxTimeoutError())
      })
      const sandbox = new ModalSandbox({ sandbox: makeRemote(exec) })

      await expect(sandbox.execute('sleep 60', { timeout: 5 })).rejects.toBeInstanceOf(SandboxTimeoutError)
      expect(exec).toHaveBeenCalledTimes(2)
    })

    it('terminates the command when closing stdin fails', async () => {
      const closeError = new Error('stdin unavailable')
      const remoteProcess = makeControlledProcess()
      vi.mocked(remoteProcess.process.stdin.close).mockRejectedValue(closeError)
      const exec = vi.fn(async (command: string[]) => {
        if (command[3] === 'strands-modal-terminate') {
          remoteProcess.complete(143)
          return makeProcess([], [], 0)
        }
        return remoteProcess.process
      })
      const sandbox = new ModalSandbox({ sandbox: makeRemote(exec) })

      await expect(sandbox.execute('cat')).rejects.toBe(closeError)
      expect(exec).toHaveBeenCalledTimes(2)
    })

    it('kills the command process group when aborted', async () => {
      const controlled = makeControlledProcess()
      const exec = vi.fn(async (command: string[], _params?: SandboxExecParams) => {
        if (command[3] === 'strands-modal-terminate') {
          controlled.complete(143)
          return makeProcess([], [], 0)
        }
        return controlled.process
      })
      const sandbox = new ModalSandbox({ sandbox: makeRemote(exec) })
      const controller = new AbortController()

      const execution = sandbox.execute('sleep 60', { signal: controller.signal })
      await vi.waitFor(() => expect(exec).toHaveBeenCalledTimes(1))
      controller.abort()

      await expect(execution).rejects.toBeInstanceOf(SandboxAbortError)
      expect(exec.mock.calls[1]![0]).toEqual([
        'sh',
        '-c',
        expect.stringContaining('kill -TERM'),
        'strands-modal-terminate',
        exec.mock.calls[0]![0][4],
        exec.mock.calls[0]![0][5],
        exec.mock.calls[0]![0][7],
        exec.mock.calls[0]![0][8],
        '0',
      ])
      expect(exec.mock.calls[1]![0][2]).toContain('find_tagged_pids')
      expect(exec.mock.calls[1]![1]).toEqual({
        stdout: 'ignore',
        stderr: 'pipe',
        timeoutMs: 10_000,
      })
    })

    it('cancels a command whose launch has not returned', async () => {
      let resolveLaunch: ((process: ContainerProcess<string>) => void) | undefined
      const launch = new Promise<ContainerProcess<string>>((resolve) => {
        resolveLaunch = resolve
      })
      const exec = vi.fn(async (command: string[]) => {
        if (command[3] === 'strands-modal-terminate') {
          return makeProcess([], [], 0)
        }
        return launch
      })
      const sandbox = new ModalSandbox({ sandbox: makeRemote(exec) })
      const controller = new AbortController()

      const execution = sandbox.execute('sleep 60', { signal: controller.signal })
      await vi.waitFor(() => expect(exec).toHaveBeenCalledTimes(1))
      controller.abort()

      await expect(execution).rejects.toBeInstanceOf(SandboxAbortError)
      expect(exec).toHaveBeenCalledTimes(2)
      expect(exec.mock.calls[1]![0][2]).toContain(': > "$cancel_file"')
      expect(exec.mock.calls[1]![0].at(-1)).toBe('1')
      resolveLaunch?.(makeProcess([], [], 143))
      await vi.waitFor(() => expect(exec).toHaveBeenCalledTimes(3))
      expect(exec.mock.calls[2]![0].at(-1)).toBe('0')
    })

    it('cleans the retained cancel marker when a cancelled launch rejects later', async () => {
      let rejectRemoteLaunch: ((error: Error) => void) | undefined
      const launch = new Promise<ContainerProcess<string>>((_resolve, reject) => {
        rejectRemoteLaunch = reject
      })
      const exec = vi.fn(async (command: string[]) => {
        if (command[3] === 'strands-modal-terminate') {
          return makeProcess([], [], 0)
        }
        return launch
      })
      const sandbox = new ModalSandbox({ sandbox: makeRemote(exec) })
      const controller = new AbortController()

      const execution = sandbox.execute('sleep 60', { signal: controller.signal })
      await vi.waitFor(() => expect(exec).toHaveBeenCalledTimes(1))
      controller.abort()

      await expect(execution).rejects.toBeInstanceOf(SandboxAbortError)
      expect(exec.mock.calls[1]![0].at(-1)).toBe('1')
      rejectRemoteLaunch?.(new Error('launch failed'))
      await vi.waitFor(() => expect(exec).toHaveBeenCalledTimes(3))
      expect(exec.mock.calls[2]![0].at(-1)).toBe('0')
    })

    it('waits for marker preservation before cleaning up a rejected late launch', async () => {
      let rejectRemoteLaunch: ((error: Error) => void) | undefined
      let resolveTerminationLaunch: ((process: ContainerProcess<string>) => void) | undefined
      const launch = new Promise<ContainerProcess<string>>((_resolve, reject) => {
        rejectRemoteLaunch = reject
      })
      const terminationLaunch = new Promise<ContainerProcess<string>>((resolve) => {
        resolveTerminationLaunch = resolve
      })
      let terminationAttempts = 0
      const exec = vi.fn(async (command: string[]) => {
        if (command[3] !== 'strands-modal-terminate') {
          return launch
        }
        terminationAttempts += 1
        return terminationAttempts === 1 ? terminationLaunch : makeProcess([], [], 0)
      })
      const sandbox = new ModalSandbox({ sandbox: makeRemote(exec) })
      const controller = new AbortController()

      const execution = sandbox.execute('sleep 60', { signal: controller.signal })
      await vi.waitFor(() => expect(exec).toHaveBeenCalledTimes(1))
      controller.abort()
      await vi.waitFor(() => expect(exec).toHaveBeenCalledTimes(2))
      rejectRemoteLaunch?.(new Error('launch failed'))
      await Promise.resolve()

      expect(exec).toHaveBeenCalledTimes(2)
      resolveTerminationLaunch?.(makeProcess([], [], 0))
      await expect(execution).rejects.toBeInstanceOf(SandboxAbortError)
      await vi.waitFor(() => expect(exec).toHaveBeenCalledTimes(3))
      expect(exec.mock.calls[2]![0].at(-1)).toBe('0')
    })

    it('cleans the marker again when preservation outlives the client deadline', async () => {
      vi.useFakeTimers()
      try {
        let rejectRemoteLaunch: ((error: Error) => void) | undefined
        let resolveTerminationLaunch: ((process: ContainerProcess<string>) => void) | undefined
        const launch = new Promise<ContainerProcess<string>>((_resolve, reject) => {
          rejectRemoteLaunch = reject
        })
        const terminationLaunch = new Promise<ContainerProcess<string>>((resolve) => {
          resolveTerminationLaunch = resolve
        })
        let terminationAttempts = 0
        const exec = vi.fn(async (command: string[]) => {
          if (command[3] !== 'strands-modal-terminate') {
            return launch
          }
          terminationAttempts += 1
          return terminationAttempts === 1 ? terminationLaunch : makeProcess([], [], 0)
        })
        const sandbox = new ModalSandbox({ sandbox: makeRemote(exec) })
        const controller = new AbortController()

        const execution = sandbox.execute('sleep 60', { signal: controller.signal })
        const rejection = execution.catch((reason: unknown) => reason)
        await vi.advanceTimersByTimeAsync(0)
        controller.abort()
        await vi.advanceTimersByTimeAsync(12_000)

        await expect(rejection).resolves.toBeInstanceOf(SandboxAbortError)
        rejectRemoteLaunch?.(new Error('launch failed'))
        await vi.advanceTimersByTimeAsync(0)
        expect(exec).toHaveBeenCalledTimes(3)

        resolveTerminationLaunch?.(makeProcess([], [], 0))
        await vi.advanceTimersByTimeAsync(0)
        expect(exec).toHaveBeenCalledTimes(4)
        expect(exec.mock.calls[3]![0].at(-1)).toBe('0')
      } finally {
        vi.useRealTimers()
      }
    })

    it('retries termination when a late launch outlives a failed cancellation request', async () => {
      let resolveLaunch: ((process: ContainerProcess<string>) => void) | undefined
      const launch = new Promise<ContainerProcess<string>>((resolve) => {
        resolveLaunch = resolve
      })
      const controlled = makeControlledProcess()
      let terminationAttempts = 0
      const exec = vi.fn(async (command: string[]) => {
        if (command[3] !== 'strands-modal-terminate') {
          return launch
        }
        terminationAttempts += 1
        if (terminationAttempts === 1) {
          throw new Error('transport unavailable')
        }
        controlled.complete(143)
        return makeProcess([], [], 0)
      })
      const sandbox = new ModalSandbox({ sandbox: makeRemote(exec) })
      const controller = new AbortController()

      const execution = sandbox.execute('sleep 60', { signal: controller.signal })
      await vi.waitFor(() => expect(exec).toHaveBeenCalledTimes(1))
      controller.abort()

      const error = await execution.catch((reason: unknown) => reason)
      expect(error).toBeInstanceOf(SandboxAbortError)
      expect((error as Error).cause).toMatchObject({
        message: 'Failed to terminate command in Modal Sandbox',
      })
      resolveLaunch?.(controlled.process)
      await vi.waitFor(() => expect(exec).toHaveBeenCalledTimes(3))
    })

    it('bounds cancellation when the termination launch does not return', async () => {
      vi.useFakeTimers()
      try {
        const controlled = makeControlledProcess()
        let terminationAttempts = 0
        const exec = vi.fn(async (command: string[]) => {
          if (command[3] !== 'strands-modal-terminate') {
            return controlled.process
          }
          terminationAttempts += 1
          if (terminationAttempts === 1) {
            return new Promise<ContainerProcess<string>>(() => {})
          }
          controlled.complete(143)
          return makeProcess([], [], 0)
        })
        const sandbox = new ModalSandbox({ sandbox: makeRemote(exec) })
        const controller = new AbortController()

        const execution = sandbox.execute('sleep 60', { signal: controller.signal })
        await vi.advanceTimersByTimeAsync(0)
        controller.abort()
        const rejection = execution.catch((reason: unknown) => reason)
        await vi.advanceTimersByTimeAsync(12_000)

        const error = await rejection
        expect(error).toBeInstanceOf(SandboxAbortError)
        expect((error as Error).cause).toMatchObject({
          message: 'Failed to terminate command in Modal Sandbox',
        })
        expect(exec).toHaveBeenCalledTimes(3)
      } finally {
        vi.useRealTimers()
      }
    })

    it('cancels a command best-effort when its launch rejects', async () => {
      const launchError = new Error('launch response unavailable')
      const exec = vi.fn(async (command: string[]) => {
        if (command[3] === 'strands-modal-terminate') {
          return makeProcess([], [], 0)
        }
        throw launchError
      })
      const sandbox = new ModalSandbox({ sandbox: makeRemote(exec) })

      await expect(sandbox.execute('touch /tmp/output')).rejects.toBe(launchError)
      expect(exec).toHaveBeenCalledTimes(2)
      expect(exec.mock.calls[1]![0].slice(-5)).toEqual([
        exec.mock.calls[0]![0][4],
        exec.mock.calls[0]![0][5],
        exec.mock.calls[0]![0][7],
        exec.mock.calls[0]![0][8],
        '0',
      ])
    })

    it('reports both the launch and cleanup failures when cancellation cannot be confirmed', async () => {
      const launchError = new Error('launch response unavailable')
      const terminationError = new Error('transport unavailable')
      const exec = vi.fn(async (command: string[]) => {
        if (command[3] === 'strands-modal-terminate') {
          throw terminationError
        }
        throw launchError
      })
      const sandbox = new ModalSandbox({ sandbox: makeRemote(exec) })

      const error = await sandbox.execute('touch /tmp/output').catch((reason: unknown) => reason)

      expect(error).toBe(launchError)
      expect(error).toMatchObject({
        cause: {
          message: 'Failed to terminate command in Modal Sandbox',
          cause: terminationError,
        },
      })
      expect(exec).toHaveBeenCalledTimes(2)
    })

    it('preserves a typed launch timeout when cleanup also fails', async () => {
      const terminationError = new Error('transport unavailable')
      const exec = vi.fn(async (command: string[]) => {
        if (command[3] === 'strands-modal-terminate') {
          throw terminationError
        }
        throw new ModalSandboxTimeoutError()
      })
      const sandbox = new ModalSandbox({ sandbox: makeRemote(exec) })

      const error = await sandbox.execute('sleep 60', { timeout: 5 }).catch((reason: unknown) => reason)

      expect(error).toBeInstanceOf(SandboxTimeoutError)
      expect(error).toMatchObject({
        cause: {
          message: 'Failed to terminate command in Modal Sandbox',
          cause: terminationError,
        },
      })
    })

    it('kills the command process group when the timeout elapses', async () => {
      const controlled = makeControlledProcess()
      const exec = vi.fn(async (command: string[]) => {
        if (command[3] === 'strands-modal-terminate') {
          controlled.complete(143)
          return makeProcess([], [], 0)
        }
        return controlled.process
      })
      const sandbox = new ModalSandbox({ sandbox: makeRemote(exec) })

      await expect(sandbox.execute('sleep 60', { timeout: 0.01 })).rejects.toBeInstanceOf(SandboxTimeoutError)
      expect(exec).toHaveBeenCalledTimes(2)
    })

    it('times out when the process exits but an output stream stalls', async () => {
      const exec = vi.fn(async (command: string[]) => {
        if (command[3] === 'strands-modal-terminate') {
          return makeProcess([], [], 0)
        }
        return {
          stdin: makeStdin(),
          stdout: makeReadable([], false),
          stderr: makeReadable([], false),
          wait: vi.fn().mockResolvedValue(0),
        } as unknown as ContainerProcess<string>
      })
      const sandbox = new ModalSandbox({ sandbox: makeRemote(exec) })

      await expect(sandbox.execute('true', { timeout: 0.01 })).rejects.toBeInstanceOf(SandboxTimeoutError)
      expect(exec).toHaveBeenCalledTimes(2)
    })

    it('bounds local stream cleanup when provider cancellation stalls', async () => {
      vi.useFakeTimers()
      try {
        let resolveWait: ((exitCode: number) => void) | undefined
        const wait = new Promise<number>((resolve) => {
          resolveWait = resolve
        })
        const makeStalledStream = (): ReadableStream<string> =>
          new ReadableStream<string>({
            cancel: () => new Promise<void>(() => {}),
          })
        const remoteProcess = {
          stdin: makeStdin(),
          stdout: makeStalledStream(),
          stderr: makeStalledStream(),
          wait: vi.fn(() => wait),
        } as unknown as ContainerProcess<string>
        const exec = vi.fn(async (command: string[]) => {
          if (command[3] === 'strands-modal-terminate') {
            resolveWait?.(143)
            return makeProcess([], [], 0)
          }
          return remoteProcess
        })
        const sandbox = new ModalSandbox({ sandbox: makeRemote(exec) })
        const controller = new AbortController()

        const execution = sandbox.execute('sleep 60', { signal: controller.signal })
        await vi.advanceTimersByTimeAsync(0)
        controller.abort()
        const rejection = execution.catch((reason: unknown) => reason)
        await vi.advanceTimersByTimeAsync(1_000)

        await expect(rejection).resolves.toBeInstanceOf(SandboxAbortError)
      } finally {
        vi.useRealTimers()
      }
    })

    it('terminates the command when a provider stream fails', async () => {
      const streamError = new Error('stream unavailable')
      const exec = vi.fn(async (command: string[]) => {
        if (command[3] === 'strands-modal-terminate') {
          return makeProcess([], [], 0)
        }
        return {
          stdin: makeStdin(),
          stdout: new ReadableStream<string>({
            start(controller) {
              controller.error(streamError)
            },
          }),
          stderr: makeReadable([], false),
          wait: vi.fn(() => new Promise<number>(() => {})),
        } as unknown as ContainerProcess<string>
      })
      const sandbox = new ModalSandbox({ sandbox: makeRemote(exec) })

      await expect(sandbox.execute('true')).rejects.toBe(streamError)
      expect(exec).toHaveBeenCalledTimes(2)
    })

    it('kills the command when streaming stops before completion', async () => {
      const controlled = makeControlledProcess('started\n')
      const exec = vi.fn(async (command: string[]) => {
        if (command[3] === 'strands-modal-terminate') {
          controlled.complete(143)
          return makeProcess([], [], 0)
        }
        return controlled.process
      })
      const sandbox = new ModalSandbox({ sandbox: makeRemote(exec) })
      const iterator = sandbox.executeStreaming('sleep 60')[Symbol.asyncIterator]()

      await expect(iterator.next()).resolves.toEqual({
        done: false,
        value: { type: 'streamChunk', data: 'started\n', streamType: 'stdout' },
      })
      await iterator.return?.()

      expect(exec).toHaveBeenCalledTimes(2)
    })

    it('reports a termination failure when streaming stops before completion', async () => {
      const controlled = makeControlledProcess('started\n')
      const exec = vi.fn(async (command: string[]) => {
        if (command[3] === 'strands-modal-terminate') {
          throw new Error('transport unavailable')
        }
        return controlled.process
      })
      const sandbox = new ModalSandbox({ sandbox: makeRemote(exec) })
      const stream = sandbox.executeStreaming('sleep 60')

      await expect(stream.next()).resolves.toEqual({
        done: false,
        value: { type: 'streamChunk', data: 'started\n', streamType: 'stdout' },
      })
      await expect(stream.return()).rejects.toThrow('Failed to terminate command in Modal Sandbox')
      expect(exec).toHaveBeenCalledTimes(2)
    })

    it('does not start a process for an already-aborted signal', async () => {
      const exec = vi.fn(async () => makeProcess([], [], 0))
      const sandbox = new ModalSandbox({ sandbox: makeRemote(exec) })
      const controller = new AbortController()
      controller.abort()

      await expect(sandbox.execute('true', { signal: controller.signal })).rejects.toBeInstanceOf(SandboxAbortError)
      expect(exec).not.toHaveBeenCalled()
    })
  })

  describe('file operations', () => {
    it('uses Modal filesystem APIs with binary content and mapped metadata', async () => {
      const filesystem = makeFilesystem()
      const bytes = new Uint8Array([0, 127, 255])
      filesystem.readBytes.mockResolvedValue(bytes)
      filesystem.listFiles.mockResolvedValue([
        {
          name: 'src',
          path: '/workspace/src',
          type: 'directory',
          size: 4096,
        } as ModalFileInfo,
        {
          name: 'index.ts',
          path: '/workspace/index.ts',
          type: 'file',
          size: 42,
        } as ModalFileInfo,
      ])
      const sandbox = new ModalSandbox({
        sandbox: makeRemote(async () => makeProcess([], [], 0), filesystem),
        workingDir: '/workspace',
      })

      await expect(sandbox.readFile('data.bin')).resolves.toEqual(bytes)
      await sandbox.writeFile('nested/data.bin', bytes)
      await sandbox.removeFile('old.bin')
      await expect(sandbox.listFiles('.')).resolves.toEqual([
        { name: 'src', isDir: true, size: 4096 },
        { name: 'index.ts', isDir: false, size: 42 },
      ])
      expect(filesystem.readBytes).toHaveBeenCalledWith('/workspace/data.bin')
      expect(filesystem.writeBytes).toHaveBeenCalledWith(bytes, '/workspace/nested/data.bin')
      expect(filesystem.stat).toHaveBeenCalledWith('/workspace/old.bin')
      expect(filesystem.remove).toHaveBeenCalledWith('/workspace/old.bin')
      expect(filesystem.listFiles).toHaveBeenCalledWith('/workspace')
    })

    it.each([
      new SandboxFilesystemNotFoundError('missing'),
      new SandboxFilesystemNotADirectoryError('not a directory'),
      new modalCommonJs.SandboxFilesystemNotFoundError('missing'),
      new modalCommonJs.SandboxFilesystemNotADirectoryError('not a directory'),
    ])('translates an unavailable directory into SandboxPathNotFoundError', async (error) => {
      const filesystem = makeFilesystem()
      filesystem.listFiles.mockRejectedValue(error)
      const sandbox = new ModalSandbox({
        sandbox: makeRemote(async () => makeProcess([], [], 0), filesystem),
        workingDir: '/workspace',
      })

      await expect(sandbox.listFiles('missing')).rejects.toBeInstanceOf(SandboxPathNotFoundError)
    })

    it('discovers and caches the Modal default directory for relative paths', async () => {
      const filesystem = makeFilesystem()
      const exec = vi.fn(async () => makeProcess(['/home/modal\n'], [], 0))
      const sandbox = new ModalSandbox({ sandbox: makeRemote(exec, filesystem) })

      await sandbox.readFile('one.txt')
      await sandbox.readFile('two.txt')

      expect(exec).toHaveBeenCalledTimes(1)
      expect(filesystem.readBytes.mock.calls).toEqual([['/home/modal/one.txt'], ['/home/modal/two.txt']])
    })

    it('retries default directory discovery after a transient failure', async () => {
      const filesystem = makeFilesystem()
      let discoveryAttempts = 0
      const exec = vi.fn(async (command: string[]) => {
        if (command[3] === 'strands-modal-terminate') {
          return makeProcess([], [], 0)
        }
        discoveryAttempts += 1
        if (discoveryAttempts === 1) {
          throw new Error('transport unavailable')
        }
        return makeProcess(['/home/modal\n'], [], 0)
      })
      const sandbox = new ModalSandbox({ sandbox: makeRemote(exec, filesystem) })

      await expect(sandbox.readFile('one.txt')).rejects.toThrow('transport unavailable')
      await expect(sandbox.readFile('two.txt')).resolves.toEqual(new Uint8Array())

      expect(exec).toHaveBeenCalledTimes(3)
      expect(filesystem.readBytes).toHaveBeenCalledWith('/home/modal/two.txt')
    })

    it('rejects an invalid default directory returned by Modal', async () => {
      const filesystem = makeFilesystem()
      const exec = vi.fn(async () => makeProcess(['\n'], [], 0))
      const sandbox = new ModalSandbox({ sandbox: makeRemote(exec, filesystem) })

      await expect(sandbox.readFile('file.txt')).rejects.toThrow('returned an invalid working directory')
      expect(filesystem.readBytes).not.toHaveBeenCalled()
    })

    it('does not remove a directory through the file API', async () => {
      const filesystem = makeFilesystem()
      filesystem.stat.mockResolvedValue({
        name: 'output',
        path: '/workspace/output',
        type: 'directory',
        size: 4096,
      } as ModalFileInfo)
      const sandbox = new ModalSandbox({
        sandbox: makeRemote(async () => makeProcess([], [], 0), filesystem),
        workingDir: '/workspace',
      })

      await expect(sandbox.removeFile('output')).rejects.toThrow('it is a directory')
      expect(filesystem.remove).not.toHaveBeenCalled()
    })
  })

  describe('getTools', () => {
    it('vends sandbox-routed tools with the Modal resource context', () => {
      const sandbox = new ModalSandbox({
        sandbox: makeRemote(async () => makeProcess([], [], 0)),
        workingDir: '/workspace',
      })
      const tools = sandbox.getTools()

      expect(tools.map((tool) => tool.name)).toEqual(['sandbox_file_editor', 'sandbox_bash'])
      expect(tools.find((tool) => tool.name === 'sandbox_bash')?.description).toContain(SANDBOX_BASH_DESCRIPTION)
      expect(tools.find((tool) => tool.name === 'sandbox_bash')?.description).toContain('Modal Sandbox "sb-test-123"')
    })
  })
})
