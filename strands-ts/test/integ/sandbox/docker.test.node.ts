import { describe, it, expect, afterEach } from 'vitest'
import { spawnSync } from 'child_process'
import fs from 'fs'
import os from 'os'
import path from 'path'
import { DockerSandbox } from '../../../src/sandbox/docker.js'

function dockerAvailable(): boolean {
  if (process.platform === 'win32') return false
  return spawnSync('docker', ['info'], { stdio: 'pipe' }).status === 0
}

describe.skipIf(!dockerAvailable())('DockerSandbox (integration)', () => {
  let sandbox: DockerSandbox | undefined

  afterEach(async () => {
    if (sandbox) {
      await sandbox.stop()
      sandbox = undefined
    }
  })

  it('start() creates a container, stop() removes it', async () => {
    sandbox = new DockerSandbox({ image: 'alpine:latest', name: 'strands-integ-lifecycle' })
    await sandbox.start()

    const ps = spawnSync('docker', ['ps', '--filter', 'name=strands-integ-lifecycle', '--format', '{{.Names}}'], {
      encoding: 'utf-8',
    })
    expect(ps.stdout.trim()).toBe('strands-integ-lifecycle')

    await sandbox.stop()

    const psAfter = spawnSync(
      'docker',
      ['ps', '-a', '--filter', 'name=strands-integ-lifecycle', '--format', '{{.Names}}'],
      { encoding: 'utf-8' }
    )
    expect(psAfter.stdout.trim()).toBe('')
  })

  it('runs commands and captures stdout, stderr, and exit code', async () => {
    sandbox = new DockerSandbox({ image: 'alpine:latest' })
    await sandbox.start()

    const result = await sandbox.execute('echo hello && echo err >&2')
    expect(result).toStrictEqual({
      type: 'executionResult',
      exitCode: 0,
      stdout: 'hello\n',
      stderr: 'err\n',
      outputFiles: [],
    })

    const failed = await sandbox.execute('exit 42')
    expect(failed.exitCode).toBe(42)
  })

  it('runs python via executeCode', async () => {
    sandbox = new DockerSandbox({ image: 'python:3.12-slim' })
    await sandbox.start()

    const result = await sandbox.executeCode('print(6 * 7)', 'python3')
    expect(result.exitCode).toBe(0)
    expect(result.stdout).toBe('42\n')
  })

  it('round-trips text and binary files', async () => {
    sandbox = new DockerSandbox({ image: 'alpine:latest' })
    await sandbox.start()

    await sandbox.writeText('greeting.txt', 'hello docker')
    expect(await sandbox.readText('greeting.txt')).toBe('hello docker')

    const bytes = new Uint8Array([0, 1, 2, 127, 128, 254, 255])
    await sandbox.writeFile('binary.bin', bytes)
    expect(Array.from(await sandbox.readFile('binary.bin'))).toStrictEqual(Array.from(bytes))
  })

  it('lists and removes files', async () => {
    sandbox = new DockerSandbox({ image: 'alpine:latest' })
    await sandbox.start()

    await sandbox.writeText('a.txt', 'a')
    await sandbox.writeText('b.txt', 'b')

    const names = (await sandbox.listFiles('.')).map((f) => f.name)
    expect(names).toContain('a.txt')
    expect(names).toContain('b.txt')

    await sandbox.removeFile('a.txt')
    await expect(sandbox.readFile('a.txt')).rejects.toThrow()
  })

  it('mounts host volumes when configured', async () => {
    const hostDir = fs.mkdtempSync(path.join(os.tmpdir(), 'strands-integ-vol-'))
    fs.writeFileSync(path.join(hostDir, 'file.txt'), 'from-host\n')

    try {
      sandbox = new DockerSandbox({
        image: 'alpine:latest',
        volumes: [`${hostDir}:/mnt/shared`],
        allowDangerousMounts: true,
      })
      await sandbox.start()

      const result = await sandbox.execute('cat /mnt/shared/file.txt')
      expect(result.stdout.trim()).toBe('from-host')
    } finally {
      fs.rmSync(hostDir, { recursive: true, force: true })
    }
  })

  it('passes environment variables into the container', async () => {
    sandbox = new DockerSandbox({ image: 'alpine:latest', env: { MY_VAR: 'hello-env' } })
    await sandbox.start()

    const result = await sandbox.execute('echo $MY_VAR')
    expect(result.stdout.trim()).toBe('hello-env')
  })

  it('files inside the container do not exist on the host', async () => {
    sandbox = new DockerSandbox({ image: 'alpine:latest' })
    await sandbox.start()

    const marker = `strands-isolation-${Date.now()}`
    await sandbox.writeText(marker, 'sandbox-only')

    const content = await sandbox.readText(marker)
    expect(content).toBe('sandbox-only')

    expect(fs.existsSync(`/tmp/${marker}`)).toBe(false)
  })
})
