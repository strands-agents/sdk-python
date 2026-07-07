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

  describe('put and get', () => {
    it('round-trips bytes', async () => {
      const data = new TextEncoder().encode('hello world')
      await storage.put('test/file.txt', data)
      const result = await storage.get('test/file.txt')
      expect(result).toEqual(data)
    })

    it('creates nested directories', async () => {
      await storage.put('deep/nested/path/file.bin', new Uint8Array([1, 2, 3]))
      const info = await stat(join(baseDir, 'deep/nested/path/file.bin'))
      expect(info.isFile()).toBe(true)
    })

    it('overwrites existing values', async () => {
      await storage.put('key', new TextEncoder().encode('first'))
      await storage.put('key', new TextEncoder().encode('second'))
      const result = await storage.get('key')
      expect(new TextDecoder().decode(result!)).toBe('second')
    })

    it('returns null for missing keys', async () => {
      const result = await storage.get('nonexistent/key')
      expect(result).toBeNull()
    })

    it('writes atomically via tmp file', async () => {
      await storage.put('atomic/test', new Uint8Array([1]))
      const content = await readFile(join(baseDir, 'atomic/test'))
      expect(new Uint8Array(content)).toEqual(new Uint8Array([1]))
      await expect(stat(join(baseDir, 'atomic/test.tmp'))).rejects.toThrow()
    })
  })

  describe('delete', () => {
    it('removes an existing key', async () => {
      await storage.put('deleteme', new Uint8Array([1]))
      await storage.delete('deleteme')
      const result = await storage.get('deleteme')
      expect(result).toBeNull()
    })

    it('is a no-op for missing keys', async () => {
      await expect(storage.delete('nonexistent')).resolves.toBeUndefined()
    })
  })

  describe('list', () => {
    it('lists keys under a prefix', async () => {
      await storage.put('sessions/a/data.json', new Uint8Array([1]))
      await storage.put('sessions/b/data.json', new Uint8Array([2]))
      await storage.put('memory/notes.json', new Uint8Array([3]))

      const keys = await storage.list('sessions/')
      expect(keys).toEqual(['sessions/a/data.json', 'sessions/b/data.json'])
    })

    it('returns all keys for empty prefix', async () => {
      await storage.put('a', new Uint8Array([1]))
      await storage.put('b', new Uint8Array([2]))

      const keys = await storage.list('')
      expect(keys).toEqual(['a', 'b'])
    })

    it('returns empty array when base directory does not exist', async () => {
      const fresh = new LocalFileStorage(join(tmpdir(), `nonexistent-${randomUUID()}`))
      const keys = await fresh.list('')
      expect(keys).toEqual([])
    })

    it('excludes scratch files', async () => {
      await storage.put('real', new Uint8Array([1]))
      const { writeFile, mkdir } = await import('node:fs/promises')
      await mkdir(baseDir, { recursive: true })
      await writeFile(join(baseDir, 'leftover.__strands_tmp'), 'garbage')

      const keys = await storage.list('')
      expect(keys).not.toContain('leftover.__strands_tmp')
    })

    it('does not exclude user .tmp files', async () => {
      await storage.put('notes.tmp', new Uint8Array([1]))
      const keys = await storage.list('')
      expect(keys).toContain('notes.tmp')
    })

    it('returns keys sorted lexicographically', async () => {
      await storage.put('c', new Uint8Array([3]))
      await storage.put('a', new Uint8Array([1]))
      await storage.put('b', new Uint8Array([2]))

      const keys = await storage.list('')
      expect(keys).toEqual(['a', 'b', 'c'])
    })
  })
})
