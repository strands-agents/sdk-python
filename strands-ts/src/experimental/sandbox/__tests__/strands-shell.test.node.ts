import { describe, it, expect, vi, beforeEach } from 'vitest'
import { StrandsShellSandbox } from '../strands-shell.js'
import { SandboxPathNotFoundError } from '../../../sandbox/errors.js'
import { SANDBOX_BASH_DESCRIPTION } from '../../../vended-tools/bash/types.js'
import { DEFAULT_FILE_EDITOR_DESCRIPTION } from '../../../vended-tools/file-editor/file-editor.js'

// A fake native shell whose `run` echoes the command it received, so tests can
// assert how StrandsShellSandbox wraps cwd/env and maps output. File ops are backed by
// an in-memory map.
function makeFakeShell() {
  const files = new Map<string, Uint8Array>()
  const runs: string[] = []
  let nextResult: { status: number; stdout: string; stderr: string } | undefined
  return {
    files,
    runs,
    setNextResult(r: { status: number; stdout: string; stderr: string }) {
      nextResult = r
    },
    run: vi.fn(async (command: string) => {
      runs.push(command)
      const r = nextResult ?? { status: 0, stdout: `ran: ${command}\n`, stderr: '' }
      nextResult = undefined
      return r
    }),
    readFile: vi.fn(async (path: string) => {
      const data = files.get(path)
      if (!data) {
        const err = new Error('not found') as Error & { code: string }
        err.code = 'ENOENT'
        throw err
      }
      return data
    }),
    writeFile: vi.fn(async (path: string, content: Uint8Array) => {
      files.set(path, content)
    }),
    removeFile: vi.fn(async (path: string) => {
      files.delete(path)
    }),
    listFiles: vi.fn(async (path: string) => {
      if (path === '/missing') {
        const err = new Error('not found') as Error & { code: string }
        err.code = 'ENOENT'
        throw err
      }
      return [{ name: 'a.txt', isDir: false, size: 3 }]
    }),
  }
}

let fakeShell: ReturnType<typeof makeFakeShell>
const createMock = vi.fn(async (_config?: unknown) => fakeShell)

vi.mock('@strands-agents/shell', () => ({
  Shell: {
    create: (config?: unknown) => createMock(config),
  },
}))

describe('StrandsShellSandbox', () => {
  beforeEach(() => {
    fakeShell = makeFakeShell()
    createMock.mockClear()
  })

  describe('execute', () => {
    it('runs a command and maps output', async () => {
      fakeShell.setNextResult({ status: 0, stdout: 'hi\n', stderr: '' })
      const sandbox = new StrandsShellSandbox()
      const result = await sandbox.execute('echo hi')
      expect(result.exitCode).toBe(0)
      expect(result.stdout).toBe('hi\n')
      expect(fakeShell.runs).toStrictEqual(['echo hi'])
    })

    it('maps a non-zero exit code', async () => {
      fakeShell.setNextResult({ status: 2, stdout: '', stderr: 'boom\n' })
      const sandbox = new StrandsShellSandbox()
      const result = await sandbox.execute('false')
      expect(result.exitCode).toBe(2)
      expect(result.stderr).toBe('boom\n')
    })

    it('wraps command in a subshell when cwd or env is set', async () => {
      const sandbox = new StrandsShellSandbox()
      await sandbox.execute('pwd', { cwd: '/workspace', env: { FOO: 'bar' } })
      expect(fakeShell.runs[0]).toBe("( cd '/workspace' && export FOO='bar' && pwd )")
    })

    it('does not wrap the command when no cwd or env is given', async () => {
      const sandbox = new StrandsShellSandbox()
      await sandbox.execute('ls')
      expect(fakeShell.runs[0]).toBe('ls')
    })

    it('creates the native shell only once across calls', async () => {
      const sandbox = new StrandsShellSandbox()
      await sandbox.execute('echo 1')
      await sandbox.execute('echo 2')
      expect(createMock).toHaveBeenCalledTimes(1)
    })

    it('passes the config through to Shell.create', async () => {
      const config = { timeout: 30, binds: [{ source: '/a', destination: '/b' }] }
      const sandbox = new StrandsShellSandbox(config)
      await sandbox.execute('echo hi')
      expect(createMock).toHaveBeenCalledWith(config)
    })

    it('streams stdout/stderr chunks before the result', async () => {
      fakeShell.setNextResult({ status: 0, stdout: 'out', stderr: 'err' })
      const sandbox = new StrandsShellSandbox()
      const chunks = []
      for await (const chunk of sandbox.executeStreaming('cmd')) {
        chunks.push(chunk)
      }
      expect(chunks).toStrictEqual([
        { type: 'streamChunk', data: 'out', streamType: 'stdout' },
        { type: 'streamChunk', data: 'err', streamType: 'stderr' },
        { type: 'executionResult', exitCode: 0, stdout: 'out', stderr: 'err', outputFiles: [] },
      ])
    })
  })

  describe('executeCode', () => {
    it('writes code to a temp file, runs the interpreter, and cleans up', async () => {
      fakeShell.setNextResult({ status: 0, stdout: '42\n', stderr: '' })
      const sandbox = new StrandsShellSandbox()
      const result = await sandbox.executeCode('print(42)', 'lua')
      expect(result.stdout).toBe('42\n')
      // A temp file was written then removed; the run targeted that file.
      expect(fakeShell.writeFile).toHaveBeenCalledTimes(1)
      expect(fakeShell.removeFile).toHaveBeenCalledTimes(1)
      const writtenPath = fakeShell.writeFile.mock.calls[0]![0]
      expect(fakeShell.runs[0]).toBe(`lua ${writtenPath}`)
    })

    it('rejects an invalid interpreter name', async () => {
      const sandbox = new StrandsShellSandbox()
      await expect(sandbox.executeCode('x', 'lua; rm -rf /')).rejects.toThrow('invalid characters')
    })

    it('removes the temp file even when the interpreter run rejects', async () => {
      const sandbox = new StrandsShellSandbox()
      fakeShell.run.mockRejectedValueOnce(new Error('kaboom'))
      await expect(sandbox.executeCode('print(1)', 'lua')).rejects.toThrow('kaboom')
      expect(fakeShell.removeFile).toHaveBeenCalledTimes(1)
    })

    it('reports a failed write to stage code as a result, not a throw', async () => {
      const sandbox = new StrandsShellSandbox()
      fakeShell.writeFile.mockRejectedValueOnce(new Error('disk full'))
      const result = await sandbox.executeCode('print(1)', 'lua')
      expect(result.exitCode).toBe(1)
      expect(result.stderr).toContain('failed to stage code')
      // No interpreter run happened, and no temp file to clean up.
      expect(fakeShell.run).not.toHaveBeenCalled()
    })
  })

  describe('file operations', () => {
    it('round-trips read and write', async () => {
      const sandbox = new StrandsShellSandbox()
      await sandbox.writeFile('/f.txt', new TextEncoder().encode('data'))
      expect(new TextDecoder().decode(await sandbox.readFile('/f.txt'))).toBe('data')
    })

    it('maps a missing directory to SandboxPathNotFoundError', async () => {
      const sandbox = new StrandsShellSandbox()
      await expect(sandbox.listFiles('/missing')).rejects.toBeInstanceOf(SandboxPathNotFoundError)
    })

    it('returns FileInfo metadata from listFiles', async () => {
      const sandbox = new StrandsShellSandbox()
      const entries = await sandbox.listFiles('/tmp')
      expect(entries).toStrictEqual([{ name: 'a.txt', isDir: false, size: 3 }])
    })
  })

  describe('getTools', () => {
    it('vends bash and fileEditor', () => {
      const sandbox = new StrandsShellSandbox()
      const names = sandbox.getTools().map((t) => t.name)
      expect(names).toStrictEqual(['fileEditor', 'bash'])
    })

    it('surfaces mounts, timeout, urls, and credentials in descriptions', () => {
      const sandbox = new StrandsShellSandbox({
        binds: [{ source: '/host', destination: '/workspace', mode: 'copy' }],
        timeout: 15,
        allowedUrls: ['https://api.example.com/'],
        credentials: [{ url: 'https://api.example.com/', token: 'secret' }],
      })
      const bashTool = sandbox.getTools().find((t) => t.name === 'bash')!
      const description = bashTool.toolSpec.description
      expect(description).toContain(SANDBOX_BASH_DESCRIPTION)
      expect(description).toContain('/workspace')
      expect(description).toContain('15s')
      expect(description).toContain('https://api.example.com/')
      expect(description).toContain('Credentials are injected automatically')
      // The secret value must never leak into the description.
      expect(description).not.toContain('secret')
    })

    it('uses the base description for a bare sandbox', () => {
      const sandbox = new StrandsShellSandbox()
      const editorTool = sandbox.getTools().find((t) => t.name === 'fileEditor')!
      expect(editorTool.toolSpec.description).toBe(DEFAULT_FILE_EDITOR_DESCRIPTION)
    })
  })
})
