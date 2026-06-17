import { Agent } from '@strands-agents/sdk'
import { DockerSandbox } from '@strands-agents/sdk/sandbox/docker'
import { makeBash } from '@strands-agents/sdk/vended-tools/bash'

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

void configuring
void hostDefault
void overrideTool
