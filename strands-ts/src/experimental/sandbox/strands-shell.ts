/**
 * Strands Shell sandbox — runs commands and file operations inside Strands Shell.
 *
 * {@link StrandsShellSandbox} backs the {@link Sandbox} abstraction with
 * [Strands Shell](https://github.com/strands-agents/shell): a Bourne-compatible
 * shell that executes entirely in-process, with no `fork`/`exec`/syscalls. The
 * agent only reaches what you declare — bound host paths, allowlisted URLs, and
 * per-URL credentials it never sees.
 *
 * This is an **experimental** feature and may change without notice. It requires
 * the optional `@strands-agents/shell` peer dependency:
 *
 * ```sh
 * npm install @strands-agents/shell
 * ```
 *
 * @example
 * ```typescript
 * import { Agent } from '@strands-agents/sdk'
 * import { StrandsShellSandbox } from '@strands-agents/sdk/experimental/sandbox/strands-shell'
 *
 * const sandbox = new StrandsShellSandbox({
 *   binds: [{ source: '/my/project', destination: '/workspace', mode: 'copy' }],
 *   timeout: 30,
 * })
 * const agent = new Agent({ sandbox })
 * await agent.invoke('List the files in /workspace and summarize them')
 * ```
 */

import { Sandbox } from '../../sandbox/base.js'
import type { ExecuteOptions } from '../../sandbox/base.js'
import { LANGUAGE_PATTERN, shellQuote } from '../../sandbox/constants.js'
import { SandboxPathNotFoundError } from '../../sandbox/errors.js'
import type { ExecutionResult, FileInfo, StreamChunk } from '../../sandbox/types.js'
import type { Tool } from '../../tools/tool.js'
import { makeFileEditor, DEFAULT_FILE_EDITOR_DESCRIPTION } from '../../vended-tools/file-editor/index.js'
import { makeBash, SANDBOX_BASH_DESCRIPTION } from '../../vended-tools/bash/index.js'
import { buildShellEnvPrefix } from '../../sandbox/posix-shell.js'

/**
 * Minimal structural view of the `@strands-agents/shell` runtime API this
 * sandbox depends on. Declared locally so the SDK type-checks without the
 * optional peer dependency installed; the real module is loaded at runtime.
 */
interface NativeShell {
  run(command: string): Promise<{ status: number; stdout: string; stderr: string }>
  readFile(path: string): Promise<Uint8Array>
  writeFile(path: string, content: Uint8Array): Promise<void>
  removeFile(path: string): Promise<void>
  listFiles(path: string): Promise<Array<{ name: string; isDir?: boolean; size?: number }>>
}

interface NativeShellModule {
  Shell: { create(config?: ShellSandboxConfig): Promise<NativeShell> }
}

/** A bind-mount entry mapping a host path into the sandbox VFS. */
export interface ShellBindConfig {
  source: string
  destination: string
  /** `'direct'` passthrough (default) or `'copy'` build-time snapshot. */
  mode?: 'direct' | 'copy'
  /** Reject writes through this mount. Default false. */
  readonly?: boolean
}

/** A per-URL credential injection rule. Exactly one of `token` / `envVar` must be set. */
export interface ShellCredConfig {
  url: string
  token?: string
  envVar?: string
}

/** Resource caps. Every field is optional and independently defaulted by the shell. */
export interface ShellLimits {
  maxOutput?: number
  maxFileSize?: number
  maxFds?: number
  maxBgJobs?: number
  maxPipeline?: number
  maxInput?: number
  maxInodes?: number
  maxDepth?: number
}

/**
 * Configuration for a {@link StrandsShellSandbox}. Mirrors `@strands-agents/shell`'s
 * `ShellConfig`. The agent cannot change this — it is fixed at construction.
 */
export interface ShellSandboxConfig {
  binds?: ShellBindConfig[]
  credentials?: ShellCredConfig[]
  allowedUrls?: string[]
  env?: Record<string, string>
  /** File-creation umask. Default 0o022. */
  umask?: number
  /** Per-command wall-clock timeout in seconds. `undefined` means no timeout. */
  timeout?: number
  limits?: ShellLimits
  /** Path to a TOML config file; merges in first, explicit options win. */
  configFile?: string
}

/**
 * A {@link Sandbox} backed by Strands Shell.
 *
 * File operations use the shell's native VFS API (reporting real `size`
 * metadata); command execution runs through the in-process shell, which keeps
 * session state (env, working directory, functions) across calls. Code
 * execution writes the source to a temporary VFS file and runs the requested
 * interpreter against it (Strands Shell ships `lua`; other interpreters are only
 * available if present in the sandbox).
 *
 * The native shell is created lazily on first use because its constructor is
 * async; every operation awaits {@link getShell}. Tools vended via
 * {@link getTools} describe the sandbox's mounts, timeout, and allowlists so the
 * model knows what it can reach.
 *
 * Note: unlike the base {@link Sandbox} contract, the per-call `timeout` in
 * {@link ExecuteOptions} is ignored. Strands Shell enforces a single wall-clock
 * timeout configured at construction (`config.timeout`); set it there to bound
 * command duration.
 */
export class StrandsShellSandbox extends Sandbox {
  private readonly _config: ShellSandboxConfig
  private _shellPromise: Promise<NativeShell> | undefined

  constructor(config: ShellSandboxConfig = {}) {
    super()
    this._config = config
  }

  /**
   * Resolve the underlying native shell, creating it on first call.
   *
   * @throws If the optional `@strands-agents/shell` package is not installed.
   */
  private getShell(): Promise<NativeShell> {
    if (this._shellPromise === undefined) {
      this._shellPromise = this._createShell()
    }
    return this._shellPromise
  }

  private async _createShell(): Promise<NativeShell> {
    let mod: NativeShellModule
    try {
      mod = (await import('@strands-agents/shell')) as unknown as NativeShellModule
    } catch (err) {
      throw new Error(
        'StrandsShellSandbox requires the "@strands-agents/shell" package. Install it with: npm install @strands-agents/shell',
        { cause: err }
      )
    }
    return mod.Shell.create(this._config)
  }

  // ---- Command execution ----

  async *executeStreaming(
    command: string,
    options?: ExecuteOptions
  ): AsyncGenerator<StreamChunk | ExecutionResult, void, undefined> {
    const shell = await this.getShell()
    const output = await shell.run(wrapCommand(command, options))
    yield* emitOutput(output)
  }

  async *executeCodeStreaming(
    code: string,
    language: string,
    options?: ExecuteOptions
  ): AsyncGenerator<StreamChunk | ExecutionResult, void, undefined> {
    if (!LANGUAGE_PATTERN.test(language)) {
      throw new Error(`language parameter contains invalid characters: ${language}`)
    }
    const shell = await this.getShell()
    // Write the source to a unique temp file in the VFS, then run the interpreter
    // against it. This avoids shell-escaping the code and works without a `base64`
    // command (which Strands Shell does not provide).
    const path = `/tmp/strands_code_${crypto.randomUUID().slice(0, 16)}`
    try {
      await shell.writeFile(path, new TextEncoder().encode(code))
    } catch (err) {
      // A VFS write failure (e.g. inode/size cap) is reported as a failed
      // execution rather than thrown, matching how shell-backed sandboxes
      // surface failures through the stream.
      const message = err instanceof Error ? err.message : String(err)
      yield {
        type: 'executionResult',
        exitCode: 1,
        stdout: '',
        stderr: `failed to stage code for execution: ${message}`,
        outputFiles: [],
      }
      return
    }
    let output
    try {
      output = await shell.run(wrapCommand(`${language} ${path}`, options))
    } finally {
      await shell.removeFile(path).catch(() => {
        /* best-effort cleanup of the temp file */
      })
    }
    yield* emitOutput(output)
  }

  // ---- VFS file operations (native) ----

  async readFile(path: string): Promise<Uint8Array> {
    const shell = await this.getShell()
    return shell.readFile(path)
  }

  async writeFile(path: string, content: Uint8Array): Promise<void> {
    const shell = await this.getShell()
    await shell.writeFile(path, content)
  }

  async removeFile(path: string): Promise<void> {
    const shell = await this.getShell()
    await shell.removeFile(path)
  }

  async listFiles(path: string): Promise<FileInfo[]> {
    const shell = await this.getShell()
    try {
      const entries = await shell.listFiles(path)
      return entries.map((e) => {
        const info: { name: string; isDir?: boolean; size?: number } = { name: e.name }
        if (e.isDir !== undefined) info.isDir = e.isDir
        if (e.size !== undefined) info.size = e.size
        return info
      })
    } catch (err) {
      // Map the shell's missing-path error onto the sandbox contract so the file
      // editor and other callers can distinguish absence from failure.
      if (isNotFound(err)) {
        throw new SandboxPathNotFoundError(path)
      }
      throw err
    }
  }

  // ---- Tools ----

  override getTools(): Tool[] {
    const suffix = this._dynamicInfo()
    return [
      makeFileEditor(this, { description: `${DEFAULT_FILE_EDITOR_DESCRIPTION}${suffix}` }),
      makeBash(this, { description: `${SANDBOX_BASH_DESCRIPTION}${suffix}` }),
    ]
  }

  /** Human-readable description of the sandbox's reachable surface, or `''`. */
  private _dynamicInfo(): string {
    const parts: string[] = []
    const bindDests = (this._config.binds ?? []).map((b) => b.destination).filter(Boolean)
    if (bindDests.length > 0) {
      parts.push(`Host paths are mounted at: ${bindDests.join(', ')}.`)
      parts.push('Writes outside mounted paths are in-memory only and do not reach the host.')
    }
    if (this._config.timeout !== undefined) {
      parts.push(`Commands time out after ${this._config.timeout}s.`)
    }
    const allowedUrls = this._config.allowedUrls ?? []
    if (allowedUrls.length > 0) {
      parts.push(`curl may reach these URL prefixes: ${allowedUrls.join(', ')}.`)
    }
    const credUrls = (this._config.credentials ?? []).map((c) => c.url).filter(Boolean)
    if (credUrls.length > 0) {
      parts.push(
        `Credentials are injected automatically for: ${credUrls.join(', ')} (do not add auth headers or tokens yourself).`
      )
    }
    return parts.length > 0 ? ` ${parts.join(' ')}` : ''
  }
}

/** Wrap a command in a subshell applying `cwd`/`env` without leaking session state. */
function wrapCommand(command: string, options?: ExecuteOptions): string {
  const cwd = options?.cwd
  const env = options?.env
  if (cwd === undefined && (!env || Object.keys(env).length === 0)) {
    return command
  }
  const envPrefix = buildShellEnvPrefix(env)
  const cdPrefix = cwd !== undefined ? `cd ${shellQuote(cwd)} && ` : ''
  return `( ${cdPrefix}${envPrefix}${command} )`
}

/** Emit a native shell result as stdout/stderr chunks followed by the final result. */
function* emitOutput(output: {
  status: number
  stdout: string
  stderr: string
}): Generator<StreamChunk | ExecutionResult, void, undefined> {
  if (output.stdout) {
    yield { type: 'streamChunk', data: output.stdout, streamType: 'stdout' }
  }
  if (output.stderr) {
    yield { type: 'streamChunk', data: output.stderr, streamType: 'stderr' }
  }
  yield {
    type: 'executionResult',
    exitCode: output.status,
    stdout: output.stdout,
    stderr: output.stderr,
    outputFiles: [],
  }
}

/** Whether a shell error denotes a missing path (`code === 'ENOENT'`). */
function isNotFound(err: unknown): boolean {
  return typeof err === 'object' && err !== null && (err as { code?: string }).code === 'ENOENT'
}
