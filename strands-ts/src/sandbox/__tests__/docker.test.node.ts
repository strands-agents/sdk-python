import { describe, it, expect, vi, beforeEach } from 'vitest'
import { DockerSandbox } from '../docker.js'
import { streamProcess } from '../stream-process.js'
import type { ExecutionResult } from '../types.js'

const OK: ExecutionResult = { type: 'executionResult', exitCode: 0, stdout: '', stderr: '', outputFiles: [] }
const FAIL: ExecutionResult = { type: 'executionResult', exitCode: 1, stdout: '', stderr: 'boom', outputFiles: [] }

vi.mock('../stream-process.js', () => ({
  streamProcess: vi.fn(async function* () {
    yield OK
  }),
}))

const findRunArgs = (): string[] => {
  const call = vi.mocked(streamProcess).mock.calls.find(([, args]) => args[0] === 'run')
  if (!call) throw new Error('docker run was never called')
  return call[1]
}

describe('DockerSandbox', () => {
  beforeEach(() => {
    vi.mocked(streamProcess).mockClear()
  })

  it('throws when a volume mount exposes a sensitive host path', () => {
    expect(() => new DockerSandbox({ image: 'alpine', volumes: ['/etc:/mnt/etc'] })).toThrow(/sensitive host path/)
  })

  it('allows dangerous mounts when allowDangerousMounts is true', () => {
    expect(
      () => new DockerSandbox({ image: 'alpine', volumes: ['/etc:/mnt/etc'], allowDangerousMounts: true })
    ).not.toThrow()
  })

  it('catches path traversal attempts in volume mounts', () => {
    expect(() => new DockerSandbox({ image: 'alpine', volumes: ['/var/../etc:/mnt'] })).toThrow(/sensitive host path/)
  })

  it('does not throw for safe volume mounts', () => {
    expect(() => new DockerSandbox({ image: 'alpine', volumes: ['/opt/myproject:/app'] })).not.toThrow()
  })

  it('runs the container as a non-root user by default', async () => {
    const sandbox = new DockerSandbox({ image: 'alpine' })
    await sandbox.start()

    const args = findRunArgs()
    const userIdx = args.indexOf('--user')
    expect(args[userIdx + 1]).toBe('1000:1000')
  })

  it('drops all capabilities and disables new privileges by default', async () => {
    const sandbox = new DockerSandbox({ image: 'alpine' })
    await sandbox.start()

    const args = findRunArgs()
    expect(args).toEqual(expect.arrayContaining(['--cap-drop', 'ALL', '--security-opt', 'no-new-privileges']))
  })

  it('omits cap-drop and no-new-privileges when allowPrivilegeEscalation is true', async () => {
    const sandbox = new DockerSandbox({ image: 'alpine', allowPrivilegeEscalation: true })
    await sandbox.start()

    const args = findRunArgs()
    expect(args).not.toContain('--cap-drop')
    expect(args).not.toContain('no-new-privileges')
  })

  it('throws a clear error when the Docker daemon is unavailable', async () => {
    vi.mocked(streamProcess).mockImplementationOnce(async function* () {
      yield FAIL
    })
    const sandbox = new DockerSandbox({ image: 'alpine' })
    await expect(sandbox.start()).rejects.toThrow(/Docker is not available/)
  })

  it('throws when executeStreaming is called before start()', async () => {
    const sandbox = new DockerSandbox({ image: 'alpine' })
    await expect(sandbox.execute('echo hi')).rejects.toThrow(/not running.*start\(\)/)
  })

  it('removes the container on stop()', async () => {
    const sandbox = new DockerSandbox({ image: 'alpine', name: 'test-stop' })
    await sandbox.start()
    vi.mocked(streamProcess).mockClear()
    await sandbox.stop()

    expect(streamProcess).toHaveBeenCalledWith('docker', ['rm', '-f', 'test-stop'], {
      enoentMessage: 'docker is not installed or not on PATH',
    })
  })

  it('passes resource limits to docker run', async () => {
    const sandbox = new DockerSandbox({
      image: 'alpine',
      memory: '512m',
      cpus: 1.5,
      pidsLimit: 100,
      network: 'none',
    })
    await sandbox.start()

    const args = findRunArgs()
    expect(args).toEqual(
      expect.arrayContaining(['--memory', '512m', '--cpus', '1.5', '--pids-limit', '100', '--network', 'none'])
    )
  })

  it('executes commands via docker exec with correct args', async () => {
    const sandbox = new DockerSandbox({ image: 'alpine', name: 'test-exec' })
    await sandbox.start()
    vi.mocked(streamProcess).mockClear()
    await sandbox.execute('echo hi')

    expect(streamProcess).toHaveBeenCalledWith('docker', ['exec', 'test-exec', 'sh', '-c', "cd '/tmp' && echo hi"], {
      timeout: undefined,
      signal: undefined,
    })
  })

  it('resets _running on start() failure so retry is possible', async () => {
    vi.mocked(streamProcess).mockImplementationOnce(async function* () {
      yield FAIL
    })
    const sandbox = new DockerSandbox({ image: 'alpine' })
    await expect(sandbox.start()).rejects.toThrow(/Docker is not available/)

    vi.mocked(streamProcess).mockImplementation(async function* () {
      yield OK
    })
    await sandbox.start()
  })
})
