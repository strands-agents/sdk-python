import { spawn } from 'node:child_process'
import { PosixShellSandbox } from '@strands-agents/sdk/sandbox'
import type {
  ExecuteOptions,
  ExecutionResult,
  StreamChunk,
} from '@strands-agents/sdk/sandbox'
import type { Tool } from '@strands-agents/sdk'
import { makeBash } from '@strands-agents/sdk/vended-tools/bash'
import { makeFileEditor } from '@strands-agents/sdk/vended-tools/file-editor'

// --8<-- [start:custom_posix]
class FirecrackerSandbox extends PosixShellSandbox {
  constructor(private readonly vmId: string) {
    super()
  }

  async *executeStreaming(
    command: string,
    options?: ExecuteOptions
  ): AsyncGenerator<StreamChunk | ExecutionResult, void, undefined> {
    void options
    const proc = spawn('fc-exec', [this.vmId, 'sh', '-c', command])

    let stdout = ''
    let stderr = ''
    for await (const data of proc.stdout) {
      const text = data.toString()
      stdout += text
      yield { type: 'streamChunk', data: text, streamType: 'stdout' }
    }
    for await (const data of proc.stderr) {
      const text = data.toString()
      stderr += text
      yield { type: 'streamChunk', data: text, streamType: 'stderr' }
    }
    const exitCode: number = await new Promise((resolve) =>
      proc.on('close', (code) => resolve(code ?? 0))
    )
    yield { type: 'executionResult', exitCode, stdout, stderr, outputFiles: [] }
  }
}
// --8<-- [end:custom_posix]

class FirecrackerSandboxWithTools extends FirecrackerSandbox {
  // --8<-- [start:custom_tools]
  override getTools(): Tool[] {
    return [
      makeFileEditor(this, { name: 'sandbox_file_editor' }),
      makeBash(this, { name: 'sandbox_bash' }),
    ]
  }
  // --8<-- [end:custom_tools]
}

void FirecrackerSandboxWithTools
