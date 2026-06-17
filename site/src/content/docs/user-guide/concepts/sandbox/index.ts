import { Agent } from '@strands-agents/sdk'
import { DockerSandbox } from '@strands-agents/sdk/sandbox/docker'
import { SshSandbox } from '@strands-agents/sdk/sandbox/ssh'
import { PosixShellSandbox } from '@strands-agents/sdk/sandbox'
import type { ExecuteOptions, ExecutionResult, StreamChunk } from '@strands-agents/sdk/sandbox'
import type { Tool } from '@strands-agents/sdk'
import { makeBash } from '@strands-agents/sdk/vended-tools/bash'
import { makeFileEditor } from '@strands-agents/sdk/vended-tools/file-editor'
import { spawn } from 'node:child_process'

async function configuring() {
  // --8<-- [start:configuring]
  // Point at a running container; the agent's shell and file tools run inside it
  const sandbox = new DockerSandbox({
    container: 'agent-workspace',
    workingDir: '/workspace',
  })
  const agent = new Agent({ sandbox })

  await agent.invoke('Clone the repo at /workspace and run the test suite')
  // --8<-- [end:configuring]
}

function hostDefault() {
  // --8<-- [start:host_default]
  // No sandbox: command and file tools run on the host with no isolation
  const agent = new Agent()
  console.log(agent.sandbox.constructor.name)
  // Typical output:
  // NotASandboxLocalEnvironment

  // Opt out explicitly to keep that intent stable if the default changes
  const explicitHost = new Agent({ sandbox: false })
  void explicitHost
  // --8<-- [end:host_default]
}

function overrideTool() {
  // --8<-- [start:override_tool]
  const sandbox = new DockerSandbox({ container: 'agent-workspace' })

  // A read-only bash under the same name the sandbox uses
  const lockedBash = makeBash(sandbox, {
    name: 'sandbox_bash',
    description: 'Run read-only shell commands. Do not modify files.',
  })

  // The agent keeps lockedBash; the sandbox's own sandbox_bash is skipped
  const agent = new Agent({ sandbox, tools: [lockedBash] })
  void agent
  // --8<-- [end:override_tool]
}

function docker() {
  // --8<-- [start:docker]
  // workingDir and user are optional; omit them to use the container's defaults
  const sandbox = new DockerSandbox({
    container: 'agent-workspace',
    workingDir: '/workspace',
    user: '1000:1000',
  })
  const agent = new Agent({ sandbox })
  void agent
  // --8<-- [end:docker]
}

function ssh() {
  // --8<-- [start:ssh]
  const sandbox = new SshSandbox({
    host: 'ubuntu@10.0.1.5',
    workingDir: '/home/ubuntu/workspace',
    identityFile: '~/.ssh/agent_key',
  })
  const agent = new Agent({ sandbox })
  void agent
  // --8<-- [end:ssh]
}

async function directUse() {
  // --8<-- [start:direct_use]
  const agent = new Agent({
    sandbox: new DockerSandbox({ container: 'agent-workspace', workingDir: '/workspace' }),
  })

  // Seed an input file, let the agent work, then read the result back
  await agent.sandbox.writeText('/workspace/input.csv', 'id,value\n1,42\n')

  await agent.invoke('Summarize /workspace/input.csv and write the summary to /workspace/out.txt')

  const result = await agent.sandbox.execute('cat /workspace/out.txt')
  console.log(result.exitCode, result.stdout)
  // --8<-- [end:direct_use]
}

async function streaming() {
  // --8<-- [start:streaming]
  const agent = new Agent({
    sandbox: new DockerSandbox({ container: 'agent-workspace', workingDir: '/workspace' }),
  })

  for await (const chunk of agent.sandbox.executeStreaming('npm run build')) {
    if (chunk.type === 'streamChunk') {
      process.stdout.write(chunk.data)
    } else {
      console.log(`\nexit code: ${chunk.exitCode}`)
    }
  }
  // --8<-- [end:streaming]
}

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
    const exitCode: number = await new Promise((resolve) => proc.on('close', (code) => resolve(code ?? 0)))
    yield { type: 'executionResult', exitCode, stdout, stderr, outputFiles: [] }
  }
}
// --8<-- [end:custom_posix]

class FirecrackerSandboxWithTools extends FirecrackerSandbox {
  // --8<-- [start:custom_tools]
  override getTools(): Tool[] {
    return [makeFileEditor(this, { name: 'sandbox_file_editor' }), makeBash(this, { name: 'sandbox_bash' })]
  }
  // --8<-- [end:custom_tools]
}

void configuring
void hostDefault
void overrideTool
void docker
void ssh
void directUse
void streaming
void FirecrackerSandboxWithTools
