/**
 * Modal sandbox adapter.
 */

import { posix as path } from 'node:path'

import type { ContainerProcess, Sandbox as ModalEsmSandbox, SandboxExecParams } from 'modal'

import type { ExecuteOptions } from '../../sandbox/base.js'
import { SandboxAbortError, SandboxPathNotFoundError, SandboxTimeoutError } from '../../sandbox/errors.js'
import { PosixShellSandbox, validateEnvKeys } from '../../sandbox/posix-shell.js'
import type { ExecutionResult, FileInfo, StreamChunk, StreamType } from '../../sandbox/types.js'
import type { Tool } from '../../tools/tool.js'
import { makeBash } from '../../vended-tools/bash/make-bash.js'
import { SANDBOX_BASH_DESCRIPTION } from '../../vended-tools/bash/types.js'
import { makeFileEditor, DEFAULT_FILE_EDITOR_DESCRIPTION } from '../../vended-tools/file-editor/index.js'

const CONTROL_DIRECTORY = '/tmp/.strands-modal'
const COMMAND_TAG_KEY_PREFIX = 'STRANDS_MODAL_COMMAND_'
const TERMINATION_PROCESS_TIMEOUT_MS = 10_000
const TERMINATION_CLIENT_TIMEOUT_MS = 12_000
const STREAM_CLEANUP_TIMEOUT_MS = 1_000
const MODAL_SANDBOX_TIMEOUT_ERROR_NAME = 'SandboxTimeoutError'
const MODAL_MISSING_DIRECTORY_ERROR_NAMES = new Set([
  'SandboxFilesystemNotFoundError',
  'SandboxFilesystemNotADirectoryError',
])

type ModalClientSandbox = ModalEsmSandbox | import('modal', { with: { 'resolution-mode': 'require' } }).Sandbox

interface TerminationOperation {
  result: Promise<void>
  settled: Promise<void>
  isSettled(): boolean
}

// Each command gets a session so cancellation can target that process tree without
// terminating the caller-owned Modal Sandbox.
const COMMAND_WRAPPER = `set -u
pid_file=$1
cancel_file=$2
command=$3
command_tag_key=$4
command_id=$5
mkdir -p "${CONTROL_DIRECTORY}"
[ ! -e "$cancel_file" ] || {
  rm -f "$cancel_file"
  exit 143
}
exec setsid --fork --wait sh -c '
  pid_file=$1
  cancel_file=$2
  command=$3
  command_tag_key=$4
  command_id=$5
  [ ! -e "$cancel_file" ] || {
    rm -f "$cancel_file"
    exit 143
  }
  printf "%s\\n" "$$" > "$pid_file"
  trap '\\''rm -f "$pid_file" "$cancel_file"'\\'' EXIT
  export "$command_tag_key=$command_id"
  sh -c "$command"
  status=$?
  exit "$status"
' strands-modal-command "$pid_file" "$cancel_file" "$command" "$command_tag_key" "$command_id"`

const TERMINATE_PROCESS_GROUP = `set -u
pid_file=$1
cancel_file=$2
command_tag_key=$3
command_id=$4
preserve_cancel=$5
command_tag="$command_tag_key=$command_id"
mkdir -p "${CONTROL_DIRECTORY}"
: > "$cancel_file"
attempt=0
while [ ! -r "$pid_file" ] && [ "$attempt" -lt 100 ]; do
  sleep 0.01
  attempt=$((attempt + 1))
done

pid=""
if [ -r "$pid_file" ]; then
  pid=$(cat "$pid_file")
  case "$pid" in
    ""|*[!0-9]*) pid="" ;;
  esac
fi

find_tagged_pids() {
  for env_file in /proc/[0-9]*/environ; do
    [ -r "$env_file" ] || continue
    tr "\\000" "\\n" < "$env_file" 2>/dev/null | grep -Fqx "$command_tag" || continue
    tagged_pid=\${env_file#/proc/}
    printf "%s\\n" "\${tagged_pid%/environ}"
  done
}

signal_tagged_groups() {
  signal=$1
  for tagged_pid in $(find_tagged_pids); do
    stat=$(cat "/proc/$tagged_pid/stat" 2>/dev/null) || continue
    stat_tail=\${stat##*) }
    set -- $stat_tail
    process_group=$3
    case "$process_group" in
      ""|*[!0-9]*) continue ;;
    esac
    if [ "$signal" = "TERM" ]; then
      kill -TERM "-$process_group" 2>/dev/null || kill -TERM "$tagged_pid" 2>/dev/null || true
    else
      kill -KILL "-$process_group" 2>/dev/null || kill -KILL "$tagged_pid" 2>/dev/null || true
    fi
  done
}

if [ -n "$pid" ]; then
  kill -TERM "-$pid" 2>/dev/null || true
fi
signal_tagged_groups TERM

attempt=0
while [ "$attempt" -lt 20 ]; do
  alive=0
  if [ -n "$pid" ] && kill -0 "-$pid" 2>/dev/null; then
    alive=1
  fi
  for tagged_pid in $(find_tagged_pids); do
    if kill -0 "$tagged_pid" 2>/dev/null; then
      alive=1
      break
    fi
  done
  [ "$alive" -eq 1 ] || break
  sleep 0.05
  attempt=$((attempt + 1))
done

if [ -n "$pid" ]; then
  kill -KILL "-$pid" 2>/dev/null || true
fi
signal_tagged_groups KILL
rm -f "$pid_file"
if [ "$preserve_cancel" != "1" ]; then
  rm -f "$cancel_file"
fi`

/**
 * Options for constructing a {@link ModalSandbox}.
 *
 * @experimental
 */
export interface ModalSandboxOptions {
  /** Running Modal Sandbox to use; the caller retains lifecycle ownership. */
  sandbox: ModalClientSandbox
  /** Absolute command and relative-file directory; omitted uses the Modal default. */
  workingDir?: string
}

/**
 * Execute commands and access files in a caller-owned Modal Sandbox.
 *
 * This integration is experimental and may change between minor releases.
 * The Modal image must provide a POSIX shell, Linux `/proc`, and the
 * `util-linux` implementation of `setsid` with `--fork` and `--wait`.
 * Per-command cancellation is not a security boundary against commands that
 * deliberately evade process cleanup. Run hostile code in a dedicated Modal
 * Sandbox and terminate the entire resource afterward.
 *
 * @experimental
 *
 * @example
 * ```typescript
 * import { Agent } from '@strands-agents/sdk'
 * import { ModalSandbox } from '@strands-agents/sdk/experimental/sandbox/modal'
 * import { ModalClient } from 'modal'
 *
 * const modal = new ModalClient()
 * const app = await modal.apps.fromName('my-app', { createIfMissing: true })
 * const image = modal.images
 *   .fromRegistry('node:22-slim')
 *   .dockerfileCommands(['RUN apt-get update && apt-get install -y util-linux'])
 * const remote = await modal.sandboxes.create(app, image)
 *
 * try {
 *   const agent = new Agent({
 *     sandbox: new ModalSandbox({ sandbox: remote, workingDir: '/root' }),
 *   })
 *   await agent.invoke('Use the shell to print the Node.js version')
 * } finally {
 *   await remote.terminate()
 * }
 * ```
 */
export class ModalSandbox extends PosixShellSandbox {
  readonly modalSandbox: ModalClientSandbox
  readonly workingDir: string | undefined
  private readonly _client: ModalEsmSandbox
  private _defaultWorkingDir: Promise<string> | undefined

  constructor(options: ModalSandboxOptions) {
    super()
    if (options.workingDir !== undefined && !path.isAbsolute(options.workingDir)) {
      throw new Error(`ModalSandbox workingDir must be absolute: ${options.workingDir}`)
    }
    this.modalSandbox = options.sandbox
    this._client = options.sandbox as ModalEsmSandbox
    this.workingDir = options.workingDir
  }

  async *executeStreaming(
    command: string,
    options?: ExecuteOptions
  ): AsyncGenerator<StreamChunk | ExecutionResult, void, undefined> {
    if (options?.signal?.aborted) {
      throw new SandboxAbortError()
    }
    if (options?.env) {
      validateEnvKeys(options.env)
    }
    if (options?.timeout !== undefined && (!Number.isFinite(options.timeout) || options.timeout <= 0)) {
      throw new RangeError('ModalSandbox timeout must be a finite positive number of seconds')
    }

    const controlPath = `${CONTROL_DIRECTORY}/${crypto.randomUUID()}`
    const pidFile = `${controlPath}.pid`
    const cancelFile = `${controlPath}.cancel`
    let commandTagKey: string
    do {
      commandTagKey = `${COMMAND_TAG_KEY_PREFIX}${crypto.randomUUID().replaceAll('-', '').toUpperCase()}`
    } while (options?.env !== undefined && commandTagKey in options.env)
    const commandId = crypto.randomUUID()
    let process: ContainerProcess<string> | undefined
    let processExited = false
    let executionFinished = false
    let terminationReason: SandboxAbortError | SandboxTimeoutError | undefined
    let terminationCompleted = false
    let terminationFailure: unknown
    let terminationTask: Promise<void> | undefined
    let terminationOperation: TerminationOperation | undefined
    let timeoutHandle: ReturnType<typeof setTimeout> | undefined
    let resolveActivity: (() => void) | undefined
    let executionError: unknown

    const notifyActivity = (): void => {
      resolveActivity?.()
      resolveActivity = undefined
    }

    const startTermination = (preserveCancelMarker = false): void => {
      if (terminationTask) {
        return
      }
      const operation = this._startProcessGroupTermination(
        pidFile,
        cancelFile,
        commandTagKey,
        commandId,
        preserveCancelMarker
      )
      terminationOperation = operation
      terminationTask = operation.result
      void terminationTask.then(
        () => {
          terminationCompleted = true
          notifyActivity()
        },
        (error: unknown) => {
          terminationCompleted = true
          terminationFailure = error
          notifyActivity()
        }
      )
    }

    const waitForTermination = async (): Promise<void> => {
      try {
        await terminationTask
      } catch (error) {
        throw new Error('Failed to terminate command in Modal Sandbox', { cause: error })
      }
    }

    const throwTerminationReason = async (): Promise<never> => {
      try {
        await terminationTask
      } catch (error) {
        Object.defineProperty(terminationReason!, 'cause', {
          configurable: true,
          value: new Error('Failed to terminate command in Modal Sandbox', { cause: error }),
        })
      }
      throw terminationReason!
    }

    let rejectLaunch: ((reason: Error) => void) | undefined
    const launchCancellation = new Promise<never>((_resolve, reject) => {
      rejectLaunch = reject
    })

    const requestTermination = (reason: SandboxAbortError | SandboxTimeoutError): void => {
      if (executionFinished) {
        return
      }
      terminationReason ??= reason
      startTermination(process === undefined)
      rejectLaunch?.(terminationReason)
    }

    const onAbort = (): void => requestTermination(new SandboxAbortError())
    options?.signal?.addEventListener('abort', onAbort, { once: true })

    if (options?.timeout !== undefined) {
      timeoutHandle = setTimeout(
        () => requestTermination(new SandboxTimeoutError(options.timeout!)),
        Math.max(0, options.timeout * 1000)
      )
    }

    const streamChunks: StreamChunk[] = []
    const readers: ReadableStreamDefaultReader<string>[] = []
    const outputTasks: Promise<void>[] = []
    let pendingOperations = 0
    let providerError: unknown
    let exitCode: number | undefined
    let stdout = ''
    let stderr = ''

    const waitForActivity = async (): Promise<void> => {
      if (streamChunks.length > 0 || pendingOperations === 0 || providerError !== undefined || terminationCompleted) {
        return
      }
      await new Promise<void>((resolve) => {
        resolveActivity = resolve
      })
    }

    const pump = (stream: ReadableStream<string>, streamType: StreamType): Promise<void> => {
      const reader = stream.getReader()
      readers.push(reader)
      pendingOperations += 1
      return (async (): Promise<void> => {
        try {
          while (true) {
            const { value, done } = await reader.read()
            if (done) {
              return
            }
            if (value.length > 0) {
              if (streamType === 'stdout') {
                stdout += value
              } else {
                stderr += value
              }
              streamChunks.push({ type: 'streamChunk', data: value, streamType })
              notifyActivity()
            }
          }
        } catch (error) {
          providerError ??= error
          startTermination()
        } finally {
          reader.releaseLock()
          pendingOperations -= 1
          notifyActivity()
        }
      })()
    }

    try {
      const execParams: SandboxExecParams & { mode: 'text' } = {
        mode: 'text',
        stdout: 'pipe',
        stderr: 'pipe',
      }
      const cwd = options?.cwd ?? this.workingDir
      if (cwd !== undefined) {
        execParams.workdir = cwd
      }
      if (options?.env !== undefined) {
        execParams.env = options.env
      }
      if (options?.timeout !== undefined && Number.isFinite(options.timeout)) {
        // Modal requires whole-second process timeouts. Keep it behind the Strands
        // timer so the adapter controls the public timeout and error semantics.
        execParams.timeoutMs = Math.max(1000, Math.ceil(options.timeout + 1) * 1000)
      }

      const launchTask = this._client
        .exec(
          ['sh', '-c', COMMAND_WRAPPER, 'strands-modal', pidFile, cancelFile, command, commandTagKey, commandId],
          execParams
        )
        .then((remoteProcess) => {
          process = remoteProcess
          return remoteProcess
        })
      try {
        process = await Promise.race([launchTask, launchCancellation])
      } catch (error) {
        if (terminationReason) {
          void launchTask.then(
            async (lateProcess) => {
              await terminationTask?.catch(() => {})
              const cleanupAgainAfterOriginal = terminationOperation !== undefined && !terminationOperation.isSettled()
              await this._startProcessGroupTermination(
                pidFile,
                cancelFile,
                commandTagKey,
                commandId,
                false
              ).result.catch(() => {})
              const settlingProcess = this._settleCancelledProcess(lateProcess)
              if (cleanupAgainAfterOriginal) {
                await terminationOperation?.settled
                await this._startProcessGroupTermination(
                  pidFile,
                  cancelFile,
                  commandTagKey,
                  commandId,
                  false
                ).result.catch(() => {})
              }
              await settlingProcess
            },
            async () => {
              await terminationTask?.catch(() => {})
              const cleanupAgainAfterOriginal = terminationOperation !== undefined && !terminationOperation.isSettled()
              await this._startProcessGroupTermination(
                pidFile,
                cancelFile,
                commandTagKey,
                commandId,
                false
              ).result.catch(() => {})
              if (cleanupAgainAfterOriginal) {
                await terminationOperation?.settled
                await this._startProcessGroupTermination(
                  pidFile,
                  cancelFile,
                  commandTagKey,
                  commandId,
                  false
                ).result.catch(() => {})
              }
            }
          )
          await throwTerminationReason()
        }
        startTermination()
        const executionLaunchError = this._translateExecutionError(error, options?.timeout)
        try {
          await waitForTermination()
        } catch (terminationError) {
          if (executionLaunchError instanceof Error) {
            const cause =
              executionLaunchError.cause === undefined
                ? terminationError
                : new AggregateError(
                    [executionLaunchError.cause, terminationError],
                    'Command launch and cleanup both failed'
                  )
            Object.defineProperty(executionLaunchError, 'cause', {
              configurable: true,
              value: cause,
            })
            throw executionLaunchError
          }
          throw new AggregateError(
            [executionLaunchError, terminationError],
            'Modal Sandbox command launch failed and cleanup could not be confirmed',
            { cause: terminationError }
          )
        }
        throw executionLaunchError
      }
      if (terminationReason) {
        startTermination()
      }

      outputTasks.push(pump(process.stdout, 'stdout'), pump(process.stderr, 'stderr'))
      pendingOperations += 1
      const stdinTask = process.stdin
        .close()
        .catch((error: unknown) => {
          providerError ??= error
          startTermination()
        })
        .finally(() => {
          pendingOperations -= 1
          notifyActivity()
        })
      pendingOperations += 1
      const waitTask = process
        .wait()
        .then((code) => {
          exitCode = code
          processExited = true
        })
        .catch((error: unknown) => {
          providerError ??= error
          startTermination()
        })
        .finally(() => {
          pendingOperations -= 1
          notifyActivity()
        })

      while (pendingOperations > 0 || streamChunks.length > 0) {
        while (streamChunks.length > 0) {
          yield streamChunks.shift()!
        }
        if (providerError !== undefined || terminationCompleted) {
          break
        }
        await waitForActivity()
      }

      if (providerError === undefined && terminationTask === undefined) {
        await Promise.all([...outputTasks, stdinTask, waitTask])
      }
      if (terminationReason) {
        await throwTerminationReason()
      }
      await waitForTermination()
      if (providerError !== undefined) {
        throw this._translateExecutionError(providerError, options?.timeout)
      }
      if (exitCode === undefined) {
        throw new Error('Modal process did not return an exit code')
      }
      executionFinished = true

      yield {
        type: 'executionResult',
        exitCode,
        stdout,
        stderr,
        outputFiles: [],
      } satisfies ExecutionResult
    } catch (error) {
      executionError = error
      throw error
    } finally {
      if (timeoutHandle !== undefined) {
        clearTimeout(timeoutHandle)
      }
      options?.signal?.removeEventListener('abort', onAbort)

      let reportTerminationFailure = false
      if (process && !processExited) {
        if (terminationFailure === undefined) {
          startTermination()
          if (executionError === undefined) {
            reportTerminationFailure = true
          } else {
            await terminationTask?.catch(() => {})
          }
        } else {
          void this._startProcessGroupTermination(pidFile, cancelFile, commandTagKey, commandId, false).result.catch(
            () => {}
          )
        }
      }
      let streamCleanupTimeout: ReturnType<typeof setTimeout> | undefined
      const streamCleanup = Promise.allSettled([...readers.map((reader) => reader.cancel()), ...outputTasks])
      const streamCleanupDeadline = new Promise<void>((resolve) => {
        streamCleanupTimeout = setTimeout(resolve, STREAM_CLEANUP_TIMEOUT_MS)
      })
      try {
        await Promise.race([streamCleanup, streamCleanupDeadline])
      } finally {
        clearTimeout(streamCleanupTimeout)
      }
      if (reportTerminationFailure) {
        await waitForTermination()
      }
    }
  }

  override async readFile(filePath: string): Promise<Uint8Array> {
    return this._client.filesystem.readBytes(await this._resolvePath(filePath))
  }

  override async writeFile(filePath: string, content: Uint8Array): Promise<void> {
    await this._client.filesystem.writeBytes(content, await this._resolvePath(filePath))
  }

  override async removeFile(filePath: string): Promise<void> {
    const resolvedPath = await this._resolvePath(filePath)
    const entry = await this._client.filesystem.stat(resolvedPath)
    if (entry.type === 'directory') {
      throw new Error(`Failed to remove file because it is a directory: ${filePath}`)
    }
    await this._client.filesystem.remove(resolvedPath)
  }

  override async listFiles(directoryPath: string): Promise<FileInfo[]> {
    try {
      const entries = await this._client.filesystem.listFiles(await this._resolvePath(directoryPath))
      return entries.map((entry) => ({
        name: entry.name,
        isDir: entry.type === 'directory',
        size: entry.size,
      }))
    } catch (error) {
      if (error instanceof Error && MODAL_MISSING_DIRECTORY_ERROR_NAMES.has(error.name)) {
        throw new SandboxPathNotFoundError(directoryPath)
      }
      throw error
    }
  }

  override getTools(): Tool[] {
    const cwd = this.workingDir ? ` Working directory: ${this.workingDir}.` : ''
    return [
      makeFileEditor(this, {
        name: 'sandbox_file_editor',
        description: `${DEFAULT_FILE_EDITOR_DESCRIPTION} Files are in Modal Sandbox "${this.modalSandbox.sandboxId}".`,
      }),
      makeBash(this, {
        name: 'sandbox_bash',
        description: `${SANDBOX_BASH_DESCRIPTION} Runs in Modal Sandbox "${this.modalSandbox.sandboxId}".${cwd}`,
      }),
    ]
  }

  private async _resolvePath(filePath: string): Promise<string> {
    if (path.isAbsolute(filePath)) {
      return path.normalize(filePath)
    }
    return path.resolve(await this._getDefaultWorkingDir(), filePath)
  }

  private async _getDefaultWorkingDir(): Promise<string> {
    if (this.workingDir !== undefined) {
      return this.workingDir
    }
    this._defaultWorkingDir ??= this.execute('pwd').then((result) => {
      if (result.exitCode !== 0) {
        throw new Error(result.stderr || 'Failed to determine Modal Sandbox working directory')
      }
      const workingDir = result.stdout.trim()
      if (!path.isAbsolute(workingDir)) {
        throw new Error(`Modal Sandbox returned an invalid working directory: ${workingDir}`)
      }
      return workingDir
    })
    const discovery = this._defaultWorkingDir
    try {
      return await discovery
    } catch (error) {
      if (this._defaultWorkingDir === discovery) {
        this._defaultWorkingDir = undefined
      }
      throw error
    }
  }

  private _startProcessGroupTermination(
    pidFile: string,
    cancelFile: string,
    commandTagKey: string,
    commandId: string,
    preserveCancelMarker: boolean
  ): TerminationOperation {
    const remoteTermination = (async (): Promise<void> => {
      const process = await this._client.exec(
        [
          'sh',
          '-c',
          TERMINATE_PROCESS_GROUP,
          'strands-modal-terminate',
          pidFile,
          cancelFile,
          commandTagKey,
          commandId,
          preserveCancelMarker ? '1' : '0',
        ],
        {
          stdout: 'ignore',
          stderr: 'pipe',
          timeoutMs: TERMINATION_PROCESS_TIMEOUT_MS,
        }
      )
      const [exitCode, stderr] = await Promise.all([process.wait(), process.stderr.readText(), process.stdin.close()])
      if (exitCode !== 0) {
        throw new Error(stderr || `Failed to terminate Modal process group (exit code ${exitCode})`)
      }
    })()
    let remoteSettled = false
    const settled = remoteTermination.then(
      () => {
        remoteSettled = true
      },
      () => {
        remoteSettled = true
      }
    )
    const result = (async (): Promise<void> => {
      let timeoutHandle: ReturnType<typeof setTimeout> | undefined
      const deadline = new Promise<never>((_resolve, reject) => {
        timeoutHandle = setTimeout(
          () => reject(new Error('Timed out while terminating command in Modal Sandbox')),
          TERMINATION_CLIENT_TIMEOUT_MS
        )
      })

      try {
        await Promise.race([remoteTermination, deadline])
      } finally {
        clearTimeout(timeoutHandle)
      }
    })()
    return { result, settled, isSettled: () => remoteSettled }
  }

  private async _settleCancelledProcess(process: ContainerProcess<string>): Promise<void> {
    await Promise.allSettled([process.wait(), process.stdin.close(), process.stdout.cancel(), process.stderr.cancel()])
  }

  private _translateExecutionError(error: unknown, timeout: number | undefined): unknown {
    if (timeout !== undefined && error instanceof Error && error.name === MODAL_SANDBOX_TIMEOUT_ERROR_NAME) {
      return new SandboxTimeoutError(timeout)
    }
    return error
  }
}
