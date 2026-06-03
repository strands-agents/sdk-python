/**
 * Shell sandbox with default implementations for file and code operations.
 *
 * Subclasses only need to implement {@link PosixShellSandbox.executeStreaming} —
 * all other operations are implemented by running shell commands through it.
 * Use this for remote environments where only shell access is available
 * (Docker containers, SSH connections, cloud runtimes).
 */

import { Sandbox } from './base.js'
import type { ExecuteOptions } from './base.js'
import { LANGUAGE_PATTERN } from './constants.js'
import type { ExecutionResult, FileInfo, StreamChunk } from './types.js'

/**
 * Shell-escape a string for safe inclusion in a shell command.
 *
 * Wraps the value in single quotes and escapes any embedded single quotes
 * using the '\'' pattern. Single quotes disable all shell expansion
 * (variables, backticks, globbing), making this safe against injection.
 *
 * @param value - The string to escape.
 * @returns The shell-escaped string wrapped in single quotes.
 */
export function shellQuote(value: string): string {
  return "'" + value.replace(/'/g, "'\\''") + "'"
}

/**
 * Abstract sandbox that provides shell-based defaults for file and code operations.
 * Assumes a POSIX-compatible shell (sh/bash) on the target.
 *
 * Subclasses only need to implement {@link executeStreaming}. The remaining
 * operations — `executeCodeStreaming`, `readFile`, `writeFile`, `removeFile`,
 * and `listFiles` — are implemented via shell commands piped through
 * `executeStreaming`.
 *
 * Subclasses may override any method with a native implementation for
 * better performance or to handle edge cases (e.g., binary-safe file
 * transfer via Docker stdin pipes, or native API calls for cloud backends).
 */
export abstract class PosixShellSandbox extends Sandbox {
  async *executeCodeStreaming(
    code: string,
    language: string,
    options?: ExecuteOptions
  ): AsyncGenerator<StreamChunk | ExecutionResult, void, undefined> {
    if (!LANGUAGE_PATTERN.test(language)) {
      throw new Error(`language parameter contains invalid characters: ${language}`)
    }
    const encoded = btoa(Array.from(new TextEncoder().encode(code), (b) => String.fromCharCode(b)).join(''))
    const eof = `STRANDS_EOF_${crypto.randomUUID().slice(0, 16)}`
    yield* this.executeStreaming(`base64 -d << '${eof}' | ${language}\n${encoded}\n${eof}`, options)
  }

  async readFile(path: string): Promise<Uint8Array> {
    const result = await this.execute(`base64 < ${shellQuote(path)}`)
    if (result.exitCode !== 0) {
      throw new Error(result.stderr || `Failed to read file: ${path}`)
    }
    return Uint8Array.from(atob(result.stdout.replace(/\s/g, '')), (c) => c.charCodeAt(0))
  }

  async writeFile(path: string, content: Uint8Array): Promise<void> {
    const encoded = btoa(Array.from(content, (b) => String.fromCharCode(b)).join(''))
    const quoted = shellQuote(path)
    const eof = `STRANDS_EOF_${crypto.randomUUID().slice(0, 16)}`
    const cmd = `mkdir -p "$(dirname ${quoted})" && base64 -d << '${eof}' > ${quoted}\n${encoded}\n${eof}`
    const result = await this.execute(cmd)
    if (result.exitCode !== 0) {
      throw new Error(result.stderr || `Failed to write file: ${path}`)
    }
  }

  async removeFile(path: string): Promise<void> {
    const result = await this.execute(`rm ${shellQuote(path)}`)
    if (result.exitCode !== 0) {
      throw new Error(result.stderr || `Failed to remove file: ${path}`)
    }
  }

  async listFiles(path: string): Promise<FileInfo[]> {
    const quoted = shellQuote(path)
    const result = await this.execute(`test -d ${quoted} || exit 1; env QUOTING_STYLE=literal ls -1ap ${quoted}`)
    if (result.exitCode !== 0) {
      throw new Error(result.stderr || `Failed to list directory: ${path}`)
    }

    const entries: FileInfo[] = []
    for (const raw of result.stdout.split('\n')) {
      const line = raw.replace(/\r$/, '')
      if (!line || line === './' || line === '../') {
        continue
      }
      const isDir = line.endsWith('/')
      const name = isDir ? line.slice(0, -1) : line
      if (name) {
        entries.push({ name, isDir })
      }
    }
    return entries
  }
}
