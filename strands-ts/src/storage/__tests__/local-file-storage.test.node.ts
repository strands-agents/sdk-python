import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { LocalFileStorage } from '../local-file-storage.js'
import { rm, readFile, stat } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { randomUUID } from 'node:crypto'

describe('LocalFileStorage', () => {
  let baseDir: string
  let storage: LocalFileStorage

  beforeEach(() => {
    baseDir = join(tmpdir(), `strands-test-${randomUUID()}`)
    storage = new LocalFileStorage(baseDir)
  })

  afterEach(async () => {
    await rm(baseDir, { recursive: true, force: true })
  })

  describe('write and read', () => {
    it('round-trips bytes', async () => {
      const data = new TextEncoder().encode('hello world')
      await storage.write('test/file.txt', data)
      const result = await storage.read('test/file.txt')
      expect(result).toEqual(data)
    })

    it('creates nested directories', async () => {
      await storage.write('deep/nested/path/file.bin', new Uint8Array([1, 2, 3]))
      const info = await stat(join(baseDir, 'deep/nested/path/file.bin'))
      expect(info.isFile()).toBe(true)
    })

    it('overwrites existing values', async () => {
      await storage.write('key', new TextEncoder().encode('first'))
      await storage.write('key', new TextEncoder().encode('second'))
      const result = await storage.read('key')
      expect(new TextDecoder().decode(result!)).toBe('second')
    })

    it('returns null for missing keys', async () => {
      const result = await storage.read('nonexistent/key')
      expect(result).toBeNull()
    })

    it('writes atomically via tmp file', async () => {
      await storage.write('atomic/test', new Uint8Array([1]))
      const content = await readFile(join(baseDir, 'atomic/test'))
      expect(new Uint8Array(content)).toEqual(new Uint8Array([1]))
      await expect(stat(join(baseDir, 'atomic/test.tmp'))).rejects.toThrow()
    })

    it.skipIf(process.platform === 'win32')('cleans up tmp file on rename failure', async () => {
      const { mkdir, chmod, readdir } = await import('node:fs/promises')
      const dir = join(baseDir, 'readonly')
      await mkdir(dir, { recursive: true })
      const { writeFile } = await import('node:fs/promises')
      await writeFile(join(dir, 'target'), 'original')
      await chmod(dir, 0o555)

      try {
        await expect(storage.write('readonly/target', new Uint8Array([1]))).rejects.toThrow()
        await chmod(dir, 0o755)
        const files = await readdir(dir)
        expect(files.filter((f) => f.includes('.__strands_tmp'))).toHaveLength(0)
      } finally {
        await chmod(dir, 0o755)
      }
    })
  })

  describe('delete', () => {
    it('removes an existing key', async () => {
      await storage.write('deleteme', new Uint8Array([1]))
      await storage.delete('deleteme')
      const result = await storage.read('deleteme')
      expect(result).toBeNull()
    })

    it('is a no-op for missing keys', async () => {
      await expect(storage.delete('nonexistent')).resolves.toBeUndefined()
    })
  })

  describe('list', () => {
    it('lists keys under a prefix', async () => {
      await storage.write('sessions/a/data.json', new Uint8Array([1]))
      await storage.write('sessions/b/data.json', new Uint8Array([2]))
      await storage.write('memory/notes.json', new Uint8Array([3]))

      const keys = await storage.list('sessions/')
      expect(keys).toEqual(['sessions/a/data.json', 'sessions/b/data.json'])
    })

    it('returns all keys for empty prefix', async () => {
      await storage.write('a', new Uint8Array([1]))
      await storage.write('b', new Uint8Array([2]))

      const keys = await storage.list('')
      expect(keys).toEqual(['a', 'b'])
    })

    it('returns empty array when base directory does not exist', async () => {
      const fresh = new LocalFileStorage(join(tmpdir(), `nonexistent-${randomUUID()}`))
      const keys = await fresh.list('')
      expect(keys).toEqual([])
    })

    it('excludes scratch files', async () => {
      await storage.write('real', new Uint8Array([1]))
      const { writeFile, mkdir } = await import('node:fs/promises')
      await mkdir(baseDir, { recursive: true })
      await writeFile(join(baseDir, 'leftover.__strands_tmp'), 'garbage')

      const keys = await storage.list('')
      expect(keys).not.toContain('leftover.__strands_tmp')
    })

    it('does not exclude user .tmp files', async () => {
      await storage.write('notes.tmp', new Uint8Array([1]))
      const keys = await storage.list('')
      expect(keys).toContain('notes.tmp')
    })

    it('returns keys sorted lexicographically', async () => {
      await storage.write('c', new Uint8Array([3]))
      await storage.write('a', new Uint8Array([1]))
      await storage.write('b', new Uint8Array([2]))

      const keys = await storage.list('')
      expect(keys).toEqual(['a', 'b', 'c'])
    })
  })
})
