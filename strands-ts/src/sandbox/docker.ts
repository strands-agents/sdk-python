/**
 * Docker sandbox — executes commands inside a Docker container via `docker exec`.
 */

import { randomUUID } from 'crypto'
import path from 'path'
import type { ExecuteOptions } from './base.js'
import { PosixShellSandbox, shellQuote } from './posix-shell.js'
import { streamProcess } from './stream-process.js'
import type { ExecutionResult, StreamChunk } from './types.js'

// Host paths that bypass container isolation when bind-mounted. Covers system
// directories, runtime sockets, and user data that would expose the host to
// LLM-generated commands running inside the container.
const DANGEROUS_MOUNTS = [
  '/',
  '/boot',
  '/dev',
  '/etc',
  '/home',
  '/lib',
  '/lib64',
  '/proc',
  '/root',
  '/run',
  '/sys',
  '/tmp',
  '/usr',
  '/var/run',
]

// Windows drive roots (C:, D:) are flagged; POSIX paths are checked against DANGEROUS_MOUNTS.
// Docker volume format: host:container[:options]. Windows paths (C:\...) have a colon after
// the drive letter, so the separator search starts at index 2 to avoid splitting on it.
function isHostPathDangerous(vol: string): boolean {
  const sep = vol.indexOf(':', /^[a-zA-Z]:[/\\]/.test(vol) ? 2 : 0)
  let hostPath = (sep > 0 ? vol.slice(0, sep) : vol).replace(/[/\\]+$/, '') || '/'
  if (hostPath.startsWith('/')) hostPath = path.posix.normalize(hostPath).replace(/\/+$/, '') || '/'
  return /^[a-zA-Z]:$/.test(hostPath) || DANGEROUS_MOUNTS.some((d) => hostPath === d || hostPath.startsWith(d + '/'))
}

async function dockerCmd(args: string[]): Promise<ExecutionResult> {
  for await (const chunk of streamProcess('docker', args, {
    enoentMessage: 'docker is not installed or not on PATH',
  })) {
    if (chunk.type === 'executionResult') return chunk
  }
  throw new Error(`docker command did not produce a result: docker ${args.join(' ')}`)
}

/**
 * Options for constructing a {@link DockerSandbox}.
 */
export interface DockerSandboxOptions {
  /** Docker image to use (e.g., `"python:3.12"`, `"node:20-alpine"`). */
  image: string
  /**
   * Working directory inside the container. Defaults to `"/tmp"`.
   *
   * `/tmp` is used because the default non-root user (`1000:1000`) with `--cap-drop ALL`
   * cannot create or chown directories. `/tmp` is world-writable on every standard base image.
   * If you specify a custom path, it must already exist in the image and be writable by `user`.
   */
  workingDir?: string
  /** Container name. Auto-generated if not provided. */
  name?: string
  /**
   * Volume mounts in `"host:container"` format.
   *
   * By default, mounts exposing sensitive host paths (`/`, `/etc`, `/proc`, `/sys`,
   * `/var/run`, etc.) throw at construction time. Set {@link allowDangerousMounts}
   * to bypass validation.
   */
  volumes?: string[]
  /** Environment variables to set in the container. */
  env?: Record<string, string>
  /** Memory limit (e.g., `"512m"`, `"2g"`). */
  memory?: string
  /** CPU limit (e.g., `1.5` for one and a half cores). */
  cpus?: number
  /** Maximum number of PIDs in the container. Prevents fork bombs. */
  pidsLimit?: number
  /** Docker network mode. Use `"none"` to disable network access. */
  network?: string
  /**
   * User to run as inside the container. Defaults to `"1000:1000"` (non-root).
   * Pass `"root"` to run as root.
   */
  user?: string
  /**
   * Allow mounting sensitive host paths.
   *
   * When `false` (default), volumes exposing dangerous host paths throw at construction time.
   * When `true`, all volume mounts are permitted without validation.
   */
  allowDangerousMounts?: boolean
  /**
   * Allow privilege escalation inside the container.
   *
   * When `false` (default), applies `--cap-drop ALL` and `--security-opt no-new-privileges`
   * to prevent setuid escalation and drop all Linux capabilities.
   */
  allowPrivilegeEscalation?: boolean
}

/**
 * Execute commands inside a Docker container.
 *
 * The container is created on {@link start} and destroyed on {@link stop}.
 * All sandbox operations route through `docker exec`.
 */
export class DockerSandbox extends PosixShellSandbox {
  readonly image: string
  readonly workingDir: string
  private readonly _name: string
  private readonly _volumes: string[]
  private readonly _env: Record<string, string>
  private readonly _memory: string | undefined
  private readonly _cpus: number | undefined
  private readonly _pidsLimit: number | undefined
  private readonly _network: string | undefined
  private readonly _user: string
  private readonly _allowPrivilegeEscalation: boolean
  private _running = false

  constructor(options: DockerSandboxOptions) {
    super()
    this.image = options.image
    // /tmp is world-writable on all base images — works with non-root user + cap-drop ALL.
    this.workingDir = options.workingDir ?? '/tmp'
    this._name = options.name ?? `strands-sandbox-${randomUUID()}`
    this._volumes = options.volumes ?? []
    if (!options.allowDangerousMounts) {
      for (const vol of this._volumes) {
        if (isHostPathDangerous(vol)) {
          throw new Error(
            `Volume "${vol}" mounts a sensitive host path. ` + 'Set allowDangerousMounts: true to bypass validation.'
          )
        }
      }
    }
    this._env = options.env ?? {}
    this._memory = options.memory
    this._cpus = options.cpus
    this._pidsLimit = options.pidsLimit
    this._network = options.network
    // Non-root by default to limit blast radius of container escapes and in-container misbehavior.
    this._user = options.user ?? '1000:1000'
    this._allowPrivilegeEscalation = options.allowPrivilegeEscalation ?? false
  }

  /**
   * Create and start the Docker container.
   *
   * @throws If Docker is not available or the container fails to start.
   */
  async start(): Promise<void> {
    if (this._running) return
    this._running = true

    try {
      const info = await dockerCmd(['info'])
      if (info.exitCode !== 0) {
        throw new Error('Docker is not available. Ensure Docker is installed and running.')
      }

      const args: string[] = ['run', '-d', '--rm', '--name', this._name, '-w', this.workingDir]

      for (const vol of this._volumes) {
        args.push('-v', vol)
      }
      for (const [key, value] of Object.entries(this._env)) {
        args.push('-e', `${key}=${value}`)
      }
      if (this._memory !== undefined) args.push('--memory', this._memory)
      if (this._cpus !== undefined) args.push('--cpus', String(this._cpus))
      if (this._pidsLimit !== undefined) args.push('--pids-limit', String(this._pidsLimit))
      if (this._network !== undefined) args.push('--network', this._network)
      args.push('--user', this._user)
      if (!this._allowPrivilegeEscalation) {
        args.push('--cap-drop', 'ALL')
        args.push('--security-opt', 'no-new-privileges')
      }

      // docker run requires the image name after all flags. Furthermore, '--' terminates
      // flag parsing so the image is always treated as a positional argument. Without
      // this, an image like '--privileged' would be parsed as a flag, overriding --cap-drop ALL.
      args.push('--', this.image, 'tail', '-f', '/dev/null')

      const result = await dockerCmd(args)
      if (result.exitCode !== 0) {
        throw new Error(`Failed to start Docker container "${this._name}": ${result.stderr}`)
      }
    } catch (err) {
      this._running = false
      throw err
    }
  }

  /** Stop and remove the Docker container. */
  async stop(): Promise<void> {
    if (!this._running) return
    try {
      await dockerCmd(['rm', '-f', this._name])
    } finally {
      this._running = false
    }
  }

  async *executeStreaming(
    command: string,
    options?: ExecuteOptions
  ): AsyncGenerator<StreamChunk | ExecutionResult, void, undefined> {
    if (!this._running) {
      throw new Error(`Docker container "${this._name}" is not running. Call start() before executing commands.`)
    }

    const cwd = options?.cwd ?? this.workingDir
    const execCommand = `cd ${shellQuote(cwd)} && ${command}`

    yield* streamProcess('docker', ['exec', this._name, 'sh', '-c', execCommand], {
      timeout: options?.timeout,
      signal: options?.signal,
    })
  }

  async [Symbol.asyncDispose](): Promise<void> {
    await this.stop()
  }
}
