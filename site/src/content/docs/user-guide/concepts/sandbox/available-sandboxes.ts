import { Agent } from '@strands-agents/sdk'
import { DockerSandbox } from '@strands-agents/sdk/sandbox/docker'
import { SshSandbox } from '@strands-agents/sdk/sandbox/ssh'

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
    sandbox: new DockerSandbox({
      container: 'agent-workspace',
      workingDir: '/workspace',
    }),
  })

  // Seed an input file, let the agent work, then read the result back
  await agent.sandbox.writeText('/workspace/input.csv', 'id,value\n1,42\n')

  await agent.invoke(
    'Summarize /workspace/input.csv and write the summary to /workspace/out.txt'
  )

  const result = await agent.sandbox.execute('cat /workspace/out.txt')
  console.log(result.exitCode, result.stdout)
  // --8<-- [end:direct_use]
}

async function streaming() {
  // --8<-- [start:streaming]
  const agent = new Agent({
    sandbox: new DockerSandbox({
      container: 'agent-workspace',
      workingDir: '/workspace',
    }),
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

void docker
void ssh
void directUse
void streaming
