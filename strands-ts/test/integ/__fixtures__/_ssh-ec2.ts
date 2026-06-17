/**
 * Sets up a connection to the shared `ssh-ec2` test instance for the SSH
 * sandbox integration test.
 *
 * That instance has no public IP and no inbound security-group rule, so the
 * only way to reach its sshd is an SSM Session Manager port-forward
 * (`aws ssm start-session --document-name AWS-StartPortForwardingSession`).
 * This runs in the global setup (parent process): it resolves the instance and
 * key from SSM, opens the tunnel to a local port, and provides a serializable
 * connection target the test points `SshSandbox` at.
 *
 * Everything self-skips: if the SSM parameters can't be resolved, or the `aws`
 * CLI / `session-manager-plugin` aren't installed, or the tunnel never comes
 * up, the provided context has `shouldSkip` set and the test skips.
 */

import { SSMClient, GetParametersCommand, GetParameterCommand } from '@aws-sdk/client-ssm'
import { spawn, spawnSync, type ChildProcess } from 'node:child_process'
import { mkdtempSync, writeFileSync, rmSync, chmodSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { connect, createServer } from 'node:net'
import { randomBytes } from 'node:crypto'
import type { AddressInfo } from 'node:net'
import type { TestProject } from 'vitest/node'
import type { ProvidedContext } from 'vitest'

const REGION = process.env.AWS_REGION || 'us-east-1'
const SSH_USER = 'ec2-user'
const INSTANCE_ID_PARAM = '/strands/test-infra/ssh-ec2/instance-id'
const KEY_PARAM_NAME_PARAM = '/strands/test-infra/ssh-ec2/private-key-parameter-name'
const TUNNEL_READY_TIMEOUT_MS = 30_000

interface SshEc2Setup {
  context: ProvidedContext['provider-ssh-ec2']
  cleanup: () => void
}

const SKIP: SshEc2Setup = {
  context: { shouldSkip: true, host: undefined, port: undefined, identityFile: undefined, workingDir: undefined },
  cleanup: () => {},
}

/**
 * Resolve the ssh-ec2 instance and open an SSM port-forward to its sshd.
 *
 * @param project - The Vitest test project (used to detect the browser env, where SSH can't run).
 * @returns The connection context to provide, plus a cleanup callback.
 */
export async function getSshEc2TestContext(project: TestProject): Promise<SshEc2Setup> {
  if (project.isBrowserEnabled()) {
    return SKIP
  }

  if (!toolingAvailable()) {
    console.log('⏭️  aws CLI or session-manager-plugin not on PATH - SSH sandbox test will be skipped')
    return SKIP
  }

  const ssm = new SSMClient({ region: REGION })

  const pointers = await ssm
    .send(new GetParametersCommand({ Names: [INSTANCE_ID_PARAM, KEY_PARAM_NAME_PARAM] }))
    .catch((e) => {
      console.warn('Error resolving ssh-ec2 SSM parameters', e)
      return null
    })
  const byName = new Map((pointers?.Parameters ?? []).map((p) => [p.Name, p.Value]))
  const instanceId = byName.get(INSTANCE_ID_PARAM)
  const keyParamName = byName.get(KEY_PARAM_NAME_PARAM)
  if (!instanceId || !keyParamName) {
    console.log('⏭️  ssh-ec2 SSM parameters not available - SSH sandbox test will be skipped')
    return SKIP
  }

  const keyResponse = await ssm
    .send(new GetParameterCommand({ Name: keyParamName, WithDecryption: true }))
    .catch((e) => {
      console.warn('Error reading ssh-ec2 private key', e)
      return null
    })
  const privateKey = keyResponse?.Parameter?.Value
  if (!privateKey) {
    console.log('⏭️  ssh-ec2 private key not readable - SSH sandbox test will be skipped')
    return SKIP
  }

  const keyDir = mkdtempSync(join(tmpdir(), 'strands-ssh-ec2-'))
  const identityFile = join(keyDir, 'key.pem')
  writeFileSync(identityFile, privateKey.endsWith('\n') ? privateKey : `${privateKey}\n`)
  chmodSync(identityFile, 0o600)

  const removeKey = (): void => rmSync(keyDir, { recursive: true, force: true })

  const localPort = await freeLocalPort()
  const tunnelState = openTunnel(instanceId, localPort)

  const ready = await waitForPort(localPort, TUNNEL_READY_TIMEOUT_MS)
  if (!ready) {
    console.log('⏭️  SSM port-forward did not come up - SSH sandbox test will be skipped')
    closeTunnel(tunnelState)
    removeKey()
    return SKIP
  }

  const workingDir = `/home/${SSH_USER}/strands-integ-ssh-${randomBytes(4).toString('hex')}`
  if (!runSsh(localPort, identityFile, `mkdir -p ${workingDir}`)) {
    console.log('⏭️  could not prepare remote working dir - SSH sandbox test will be skipped')
    closeTunnel(tunnelState)
    removeKey()
    return SKIP
  }

  console.log('⏭️  ssh-ec2 instance reachable - SSH sandbox test will run')
  return {
    context: {
      shouldSkip: false,
      host: `${SSH_USER}@127.0.0.1`,
      port: localPort,
      identityFile,
      workingDir,
    },
    cleanup: () => {
      runSsh(localPort, identityFile, `rm -rf ${workingDir}`)
      closeTunnel(tunnelState)
      removeKey()
    },
  }
}

// ---------------------------------------------------------------------------
// Tunnel lifecycle with explicit session termination
// ---------------------------------------------------------------------------

interface TunnelState {
  process: ChildProcess
  sessionId: string | undefined
  exitHandler: () => void
}

function openTunnel(instanceId: string, localPort: number): TunnelState {
  let sessionId: string | undefined
  const proc = spawn(
    'aws',
    [
      'ssm',
      'start-session',
      '--region',
      REGION,
      '--target',
      instanceId,
      '--document-name',
      'AWS-StartPortForwardingSession',
      '--parameters',
      JSON.stringify({ portNumber: ['22'], localPortNumber: [String(localPort)] }),
    ],
    { stdio: ['ignore', 'pipe', 'pipe'] }
  )

  // Capture the session ID from stdout for explicit terminate-session on cleanup.
  proc.stdout?.on('data', (chunk: { toString(): string }) => {
    const match = chunk.toString().match(/SessionId[:\s]+(\S+)/)
    if (match) sessionId = match[1]
  })

  const state: TunnelState = {
    process: proc,
    get sessionId() {
      return sessionId
    },
    exitHandler: () => {},
  }

  // Safety net: if the Node process exits before cleanup is called (e.g. OOM, CI timeout),
  // terminate the SSM session explicitly so it doesn't sit open for 20 minutes.
  state.exitHandler = () => closeTunnel(state)
  process.on('exit', state.exitHandler)

  return state
}

function closeTunnel(state: TunnelState): void {
  process.removeListener('exit', state.exitHandler)

  if (state.sessionId) {
    spawnSync('aws', ['ssm', 'terminate-session', '--region', REGION, '--session-id', state.sessionId], {
      stdio: 'ignore',
    })
  }
  if (!state.process.killed) {
    state.process.kill()
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function toolingAvailable(): boolean {
  const ok = (cmd: string): boolean => {
    const result = spawnSync(cmd, ['--version'], { stdio: 'ignore' })
    return !result.error && result.status === 0
  }
  return ok('aws') && ok('session-manager-plugin')
}

function runSsh(port: number, identityFile: string, remoteCommand: string): boolean {
  const result = spawnSync(
    'ssh',
    [
      '-o',
      'StrictHostKeyChecking=accept-new',
      '-o',
      'BatchMode=yes',
      '-p',
      String(port),
      '-i',
      identityFile,
      '--',
      `${SSH_USER}@127.0.0.1`,
      remoteCommand,
    ],
    { stdio: 'ignore' }
  )
  return !result.error && result.status === 0
}

async function freeLocalPort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = createServer()
    server.on('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const port = (server.address() as AddressInfo).port
      server.close(() => resolve(port))
    })
  })
}

async function waitForPort(port: number, timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const open = await new Promise<boolean>((resolve) => {
      const socket = connect({ host: '127.0.0.1', port })
      socket.setTimeout(1000)
      socket.once('connect', () => {
        socket.destroy()
        resolve(true)
      })
      socket.once('error', () => resolve(false))
      socket.once('timeout', () => {
        socket.destroy()
        resolve(false)
      })
    })
    if (open) return true
    await new Promise((r) => setTimeout(r, 500))
  }
  return false
}
