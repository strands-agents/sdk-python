import { readFile, writeFile, unlink, mkdir, readdir, stat } from 'fs/promises'
import { dirname, isAbsolute, join } from 'path'
import { Sandbox } from './base.js'
import type { ExecuteOptions } from './base.js'
import { LANGUAGE_PATTERN } from './constants.js'
import { streamProcess } from './stream-process.js'
import type { ExecutionResult, FileInfo, StreamChunk } from './types.js'
import { buildShellEnvPrefix, shellQuote } from './posix-shell.js'

/**
 * Runs on the host with no isolation. Used as the default when no sandbox is configured.
 */
export class NotASandboxLocalEnvironment extends Sandbox {
  private _resolvePath(path: string): string {
    return isAbsolute(path) ? path : join(process.cwd(), path)
  }

  async *executeStreaming(
    command: string,
    options?: ExecuteOptions
  ): AsyncGenerator<StreamChunk | ExecutionResult, void, undefined> {
    const cwd = options?.cwd ?? process.cwd()
    yield* streamProcess('sh', ['-c', `cd ${shellQuote(cwd)} && ${buildShellEnvPrefix(options?.env)}${command}`], {
      timeout: options?.timeout,
      signal: options?.signal,
    })
  }

  async *executeCodeStreaming(
    code: string,
    language: string,
    options?: ExecuteOptions
  ): AsyncGenerator<StreamChunk | ExecutionResult, void, undefined> {
    if (!LANGUAGE_PATTERN.test(language)) {
      throw new Error(`language parameter contains invalid characters: ${language}`)
    }
    const cwd = options?.cwd ?? process.cwd()
    const encoded = btoa(Array.from(new TextEncoder().encode(code), (b) => String.fromCharCode(b)).join(''))
    const eof = `STRANDS_EOF_${crypto.randomUUID().slice(0, 16)}`
    yield* streamProcess(
      'sh',
      [
        '-c',
        `cd ${shellQuote(cwd)} && ${buildShellEnvPrefix(options?.env)}base64 -d << '${eof}' | ${language}\n${encoded}\n${eof}`,
      ],
      {
        timeout: options?.timeout,
        signal: options?.signal,
        enoentMessage: `Language interpreter not found: ${language}`,
      }
    )
  }

  async readFile(path: string): Promise<Uint8Array> {
    return readFile(this._resolvePath(path))
  }

  async writeFile(path: string, content: Uint8Array): Promise<void> {
    const fullPath = this._resolvePath(path)
    await mkdir(dirname(fullPath), { recursive: true })
    await writeFile(fullPath, content)
  }

  async removeFile(path: string): Promise<void> {
    await unlink(this._resolvePath(path))
  }

  async listFiles(path: string): Promise<FileInfo[]> {
    const fullPath = this._resolvePath(path)
    const entries = await readdir(fullPath, { withFileTypes: true })
    const results: FileInfo[] = []

    for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
      try {
        const entryStat = await stat(join(fullPath, entry.name))
        results.push({ name: entry.name, isDir: entryStat.isDirectory(), size: entryStat.size })
      } catch {
        results.push({ name: entry.name })
      }
    }

    return results
  }
}
