import { describe, it, expect } from 'vitest'
import { mkdtempSync, writeFileSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import { createRequire } from 'module'
import { StrandsShellSandbox } from '../../../../src/experimental/sandbox/strands-shell.js'
import { SandboxPathNotFoundError } from '../../../../src/sandbox/errors.js'

// Skips when the optional @strands-agents/shell package is not installed.
function shellAvailable(): boolean {
  try {
    createRequire(import.meta.url).resolve('@strands-agents/shell')
    return true
  } catch {
    return false
  }
}

describe.skipIf(!shellAvailable())('StrandsShellSandbox (integration)', () => {
  it('runs commands and captures stdout, stderr, and exit code', async () => {
    const sandbox = new StrandsShellSandbox({ timeout: 15 })

    const result = await sandbox.execute('echo hello && echo err >&2')
    expect(result.exitCode).toBe(0)
    expect(result.stdout).toBe('hello\n')
    expect(result.stderr).toBe('err\n')

    const failed = await sandbox.execute('exit 42')
    expect(failed.exitCode).toBe(42)
  })

  it('persists session state across calls', async () => {
    const sandbox = new StrandsShellSandbox({ timeout: 15 })
    await sandbox.execute('export GREETING=hi')
    const result = await sandbox.execute('echo $GREETING')
    expect(result.stdout).toBe('hi\n')
  })

  it('scopes cwd and env to a single command', async () => {
    const sandbox = new StrandsShellSandbox({ timeout: 15 })
    const scoped = await sandbox.execute('pwd; echo $SCOPED', { cwd: '/tmp', env: { SCOPED: 'v' } })
    expect(scoped.stdout).toBe('/tmp\nv\n')
    const after = await sandbox.execute('pwd; echo [$SCOPED]')
    expect(after.stdout).not.toContain('/tmp')
    expect(after.stdout).toContain('[]')
  })

  it('runs code through the lua interpreter', async () => {
    const sandbox = new StrandsShellSandbox({ timeout: 15 })
    const result = await sandbox.executeCode('print(6 * 7)', 'lua')
    expect(result.exitCode).toBe(0)
    expect(result.stdout.trim()).toBe('42')
  })

  it('cleans up the temp file after executeCode', async () => {
    const sandbox = new StrandsShellSandbox({ timeout: 15 })
    await sandbox.executeCode('print(1)', 'lua')
    const entries = await sandbox.listFiles('/tmp')
    expect(entries.some((e) => e.name.startsWith('strands_code_'))).toBe(false)
  })

  it('round-trips native file operations with metadata', async () => {
    const sandbox = new StrandsShellSandbox({ timeout: 15 })
    await sandbox.writeFile('/tmp/note.txt', new TextEncoder().encode('content'))
    expect(new TextDecoder().decode(await sandbox.readFile('/tmp/note.txt'))).toBe('content')
    const entries = await sandbox.listFiles('/tmp')
    const note = entries.find((e) => e.name === 'note.txt')!
    expect(note.isDir).toBe(false)
    expect(note.size).toBe(7)
  })

  it('throws SandboxPathNotFoundError for a missing directory', async () => {
    const sandbox = new StrandsShellSandbox({ timeout: 15 })
    await expect(sandbox.listFiles('/does/not/exist')).rejects.toBeInstanceOf(SandboxPathNotFoundError)
  })

  it('exposes host files through a copy bind mount', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'strands-shell-integ-'))
    writeFileSync(join(dir, 'hello.txt'), 'hello from host')
    const sandbox = new StrandsShellSandbox({
      binds: [{ source: dir, destination: '/workspace', mode: 'copy' }],
      timeout: 15,
    })
    const result = await sandbox.execute('cat /workspace/hello.txt')
    expect(result.stdout).toContain('hello from host')
  })

  it('vends bash and fileEditor tools that operate on the sandbox', async () => {
    const sandbox = new StrandsShellSandbox({ timeout: 15 })
    const tools = sandbox.getTools()
    expect(tools.map((t) => t.name).sort()).toStrictEqual(['bash', 'fileEditor'])
  })
})
