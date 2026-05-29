import { describe, it, expect, vi, beforeEach } from 'vitest'
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js'
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js'
import { SSEClientTransport } from '@modelcontextprotocol/sdk/client/sse.js'
import { McpClient } from '../mcp.js'

vi.mock('node:fs/promises', () => ({
  readFile: vi.fn(),
}))

vi.mock('node:os', () => ({
  homedir: vi.fn(() => '/home/user'),
}))

vi.mock('node:path', () => ({
  join: (...segments: string[]) => segments.join('/'),
}))

vi.mock('@modelcontextprotocol/sdk/client/stdio.js', () => ({
  StdioClientTransport: vi.fn(function () {}),
  getDefaultEnvironment: vi.fn(() => ({ PATH: '/usr/bin', HOME: '/home/user' })),
}))

vi.mock('@modelcontextprotocol/sdk/client/streamableHttp.js', () => ({
  StreamableHTTPClientTransport: vi.fn(function () {}),
}))

vi.mock('@modelcontextprotocol/sdk/client/sse.js', () => ({
  SSEClientTransport: vi.fn(function () {}),
}))

vi.mock('@modelcontextprotocol/sdk/client/index.js', () => ({
  Client: vi.fn(function (this: Record<string, unknown>) {
    this.connect = vi.fn()
    this.close = vi.fn()
    this.listTools = vi.fn()
    this.callTool = vi.fn()
    this.setRequestHandler = vi.fn()
    this.setNotificationHandler = vi.fn()
    this.getServerCapabilities = vi.fn()
    this.getServerVersion = vi.fn()
    this.getInstructions = vi.fn()
    this.experimental = { tasks: { callToolStream: vi.fn() } }
  }),
}))

describe('McpClient.loadServers', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('transport detection', () => {
    it('creates StdioClientTransport when command is present', async () => {
      const clients = await McpClient.loadServers({
        'my-server': { command: 'node', args: ['server.js'] },
      })

      expect(clients).toHaveLength(1)
      expect(StdioClientTransport).toHaveBeenCalledWith({
        command: 'node',
        args: ['server.js'],
      })
    })

    it('creates McpClient with url when url is present', async () => {
      const clients = await McpClient.loadServers({
        'remote-server': { url: 'https://example.com/mcp' },
      })

      expect(clients).toHaveLength(1)
      expect(StdioClientTransport).not.toHaveBeenCalled()
      expect(SSEClientTransport).not.toHaveBeenCalled()
    })

    it('creates SSEClientTransport when transport is "sse"', async () => {
      const clients = await McpClient.loadServers({
        'sse-server': { url: 'https://example.com/sse', transport: 'sse' },
      })

      expect(clients).toHaveLength(1)
      expect(SSEClientTransport).toHaveBeenCalledWith(new URL('https://example.com/sse'), undefined)
    })

    it('explicit transport overrides auto-detection', async () => {
      const clients = await McpClient.loadServers({
        server: { url: 'https://example.com/mcp', transport: 'sse' },
      })

      expect(clients).toHaveLength(1)
      expect(SSEClientTransport).toHaveBeenCalled()
      expect(StreamableHTTPClientTransport).not.toHaveBeenCalled()
    })

    it('interpolates auth credentials for streamable-http', async () => {
      vi.stubEnv('CLIENT_ID', 'my-id')
      vi.stubEnv('CLIENT_SECRET', 'my-secret')

      const clients = await McpClient.loadServers({
        server: {
          url: 'https://example.com/mcp',
          auth: { clientId: '${CLIENT_ID}', clientSecret: '${CLIENT_SECRET}' },
        },
      })

      expect(clients).toHaveLength(1)
    })

    it('throws when auth credential references missing env var', async () => {
      vi.unstubAllEnvs()

      await expect(
        McpClient.loadServers({
          server: {
            url: 'https://example.com/mcp',
            auth: { clientId: '${MISSING_ID}', clientSecret: 'literal' },
          },
        })
      ).rejects.toThrow('Environment variable "MISSING_ID" is not set')
    })
  })

  describe('env interpolation', () => {
    it('interpolates ${VAR} in env values', async () => {
      vi.stubEnv('MY_SECRET', 'secret123')

      await McpClient.loadServers({
        server: { command: 'node', env: { SECRET: '${MY_SECRET}' } },
      })

      expect(StdioClientTransport).toHaveBeenCalledWith({
        command: 'node',
        env: { PATH: '/usr/bin', HOME: '/home/user', SECRET: 'secret123' },
      })
    })

    it('interpolates ${VAR} in headers', async () => {
      vi.stubEnv('TOKEN', 'abc')

      await McpClient.loadServers({
        server: { url: 'https://example.com/sse', transport: 'sse', headers: { Authorization: 'Bearer ${TOKEN}' } },
      })

      expect(SSEClientTransport).toHaveBeenCalledWith(new URL('https://example.com/sse'), {
        requestInit: { headers: { Authorization: 'Bearer abc' } },
      })
    })

    it('interpolates ${VAR} in url', async () => {
      vi.stubEnv('HOST', 'myhost.com')

      const clients = await McpClient.loadServers({
        server: { url: 'https://${HOST}/mcp', transport: 'sse' },
      })

      expect(clients).toHaveLength(1)
      expect(SSEClientTransport).toHaveBeenCalledWith(new URL('https://myhost.com/mcp'), undefined)
    })

    it('throws when env var is not set', async () => {
      vi.unstubAllEnvs()

      await expect(
        McpClient.loadServers({
          server: { command: 'node', env: { VAL: '${NONEXISTENT_VAR}' } },
        })
      ).rejects.toThrow('Environment variable "NONEXISTENT_VAR" is not set')
    })

    it('skips server with missing env var when continueOnError is true', async () => {
      vi.unstubAllEnvs()

      const clients = await McpClient.loadServers({
        broken: { command: 'node', env: { VAL: '${NONEXISTENT_VAR}' }, continueOnError: true },
        working: { command: 'node' },
      })

      expect(clients).toHaveLength(1)
      expect(clients[0]!.clientName).toBe('working')
    })

    it('interpolates ${VAR} in cwd', async () => {
      vi.stubEnv('PROJECT_DIR', '/home/user/projects')

      await McpClient.loadServers({
        server: { command: 'node', cwd: '${PROJECT_DIR}/my-server' },
      })

      expect(StdioClientTransport).toHaveBeenCalledWith({
        command: 'node',
        cwd: '/home/user/projects/my-server',
      })
    })

    it('merges env with default environment', async () => {
      await McpClient.loadServers({
        server: { command: 'node', env: { CUSTOM: 'value' } },
      })

      expect(StdioClientTransport).toHaveBeenCalledWith({
        command: 'node',
        env: { PATH: '/usr/bin', HOME: '/home/user', CUSTOM: 'value' },
      })
    })
  })

  describe('file config loading', () => {
    it('reads and parses a JSON file', async () => {
      const { readFile } = await import('node:fs/promises')
      vi.mocked(readFile).mockResolvedValue(
        JSON.stringify({
          'my-server': { command: 'node', args: ['server.js'] },
        })
      )

      const clients = await McpClient.loadServers('/path/to/config.json')

      expect(readFile).toHaveBeenCalledWith('/path/to/config.json', 'utf-8')
      expect(clients).toHaveLength(1)
    })

    it('extracts mcpServers key when present', async () => {
      const { readFile } = await import('node:fs/promises')
      vi.mocked(readFile).mockResolvedValue(
        JSON.stringify({
          mcpServers: {
            'server-a': { command: 'node' },
            'server-b': { url: 'https://example.com' },
          },
        })
      )

      const clients = await McpClient.loadServers('/path/to/config.json')

      expect(clients).toHaveLength(2)
    })

    it('uses whole object when mcpServers key is absent', async () => {
      const { readFile } = await import('node:fs/promises')
      vi.mocked(readFile).mockResolvedValue(
        JSON.stringify({
          'server-a': { command: 'node' },
        })
      )

      const clients = await McpClient.loadServers('/path/to/config.json')

      expect(clients).toHaveLength(1)
    })

    it('expands ~ to home directory', async () => {
      const { readFile } = await import('node:fs/promises')
      vi.mocked(readFile).mockResolvedValue(
        JSON.stringify({
          server: { command: 'node' },
        })
      )

      await McpClient.loadServers('~/config/mcp.json')

      expect(readFile).toHaveBeenCalledWith('/home/user/config/mcp.json', 'utf-8')
    })
  })

  describe('defaults and per-server overrides', () => {
    it('per-server continueOnError overrides defaults', async () => {
      const clients = await McpClient.loadServers(
        {
          'strict-server': { command: 'node', continueOnError: false },
          'lenient-server': { command: 'node', continueOnError: true },
        },
        { continueOnError: true }
      )

      expect(clients[0]!.continueOnError).toBe(false)
      expect(clients[1]!.continueOnError).toBe(true)
    })

    it('applies default continueOnError when server does not override', async () => {
      const clients = await McpClient.loadServers({ server: { command: 'node' } }, { continueOnError: true })

      expect(clients[0]!.continueOnError).toBe(true)
    })

    it('uses server name as applicationName when not in defaults', async () => {
      const clients = await McpClient.loadServers({
        'my-named-server': { command: 'node' },
      })

      expect(clients[0]!.clientName).toBe('my-named-server')
    })

    it('uses defaults applicationName over server name', async () => {
      const clients = await McpClient.loadServers({ server: { command: 'node' } }, { applicationName: 'my-app' })

      expect(clients[0]!.clientName).toBe('my-app')
    })
  })

  describe('error cases', () => {
    it('throws when server has neither command nor url', async () => {
      await expect(McpClient.loadServers({ bad: {} })).rejects.toThrow(
        'Server config must include either "command" (stdio) or "url" (http)'
      )
    })

    it('throws when stdio transport specified without command', async () => {
      await expect(McpClient.loadServers({ bad: { transport: 'stdio' } })).rejects.toThrow(
        'Stdio transport requires "command" field'
      )
    })

    it('throws when streamable-http transport specified without url', async () => {
      await expect(McpClient.loadServers({ bad: { transport: 'streamable-http' } })).rejects.toThrow(
        'Streamable HTTP transport requires "url" field'
      )
    })

    it('throws when sse transport specified without url', async () => {
      await expect(McpClient.loadServers({ bad: { transport: 'sse' } })).rejects.toThrow(
        'SSE transport requires "url" field'
      )
    })

    it('throws on invalid file path', async () => {
      const { readFile } = await import('node:fs/promises')
      vi.mocked(readFile).mockRejectedValue(new Error('ENOENT: no such file or directory'))

      await expect(McpClient.loadServers('/nonexistent/path.json')).rejects.toThrow('ENOENT')
    })

    it('throws on malformed JSON', async () => {
      const { readFile } = await import('node:fs/promises')
      vi.mocked(readFile).mockResolvedValue('not json{{{')

      await expect(McpClient.loadServers('/path/to/bad.json')).rejects.toThrow()
    })

    it('throws when auth is used with sse transport', async () => {
      await expect(
        McpClient.loadServers({
          server: {
            url: 'https://example.com',
            transport: 'sse',
            auth: { clientId: 'id', clientSecret: 'secret' },
          },
        })
      ).rejects.toThrow('SSE transport does not support auth')
    })

    it('throws on invalid config shape', async () => {
      const { readFile } = await import('node:fs/promises')
      vi.mocked(readFile).mockResolvedValue(JSON.stringify([1, 2, 3]))

      await expect(McpClient.loadServers('/path/to/bad.json')).rejects.toThrow('MCP config must be a JSON object')
    })

    it('throws when server has both command and url without explicit transport', async () => {
      await expect(McpClient.loadServers({ bad: { command: 'node', url: 'https://example.com' } })).rejects.toThrow(
        'Server config has both "command" and "url"'
      )
    })
  })

  describe('disabled', () => {
    it('skips disabled servers', async () => {
      const clients = await McpClient.loadServers({
        active: { command: 'node' },
        inactive: { command: 'node', disabled: true },
      })

      expect(clients).toHaveLength(1)
      expect(clients[0]!.clientName).toBe('active')
    })
  })

  describe('env interpolation syntax', () => {
    it('supports ${env:VAR} namespaced syntax', async () => {
      vi.stubEnv('MY_TOKEN', 'token123')

      await McpClient.loadServers({
        server: { command: 'node', env: { TOKEN: '${env:MY_TOKEN}' } },
      })

      expect(StdioClientTransport).toHaveBeenCalledWith({
        command: 'node',
        env: { PATH: '/usr/bin', HOME: '/home/user', TOKEN: 'token123' },
      })
    })

    it('interpolates ${VAR} in command and args', async () => {
      vi.stubEnv('MY_CMD', '/usr/local/bin/server')
      vi.stubEnv('MY_ARG', '3000')

      await McpClient.loadServers({
        server: { command: '${MY_CMD}', args: ['--port=${MY_ARG}'] },
      })

      expect(StdioClientTransport).toHaveBeenCalledWith({
        command: '/usr/local/bin/server',
        args: ['--port=3000'],
      })
    })
  })
})
