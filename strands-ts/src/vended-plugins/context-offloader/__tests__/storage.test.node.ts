import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { FileStorage } from '../storage.js'
import * as fs from 'node:fs/promises'
import * as path from 'node:path'
import * as os from 'node:os'

describe('FileStorage', () => {
  let tmpDir: string

  beforeEach(async () => {
    tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'context-offloader-test-'))
  })

  afterEach(async () => {
    await fs.rm(tmpDir, { recursive: true, force: true })
  })

  it('stores and retrieves text content', async () => {
    const storage = new FileStorage(tmpDir)
    const content = new TextEncoder().encode('hello world')
    const ref = await storage.store('key1', content, 'text/plain')

    const result = await storage.retrieve(ref)
    expect(new TextDecoder().decode(result.content)).toBe('hello world')
    expect(result.contentType).toBe('text/plain')
  })

  it('stores and retrieves binary content', async () => {
    const storage = new FileStorage(tmpDir)
    const content = new Uint8Array([1, 2, 3, 4, 5])
    const ref = await storage.store('key1', content, 'image/png')

    const result = await storage.retrieve(ref)
    expect(result.content).toEqual(content)
    expect(result.contentType).toBe('image/png')
  })

  it('returns file path as reference preserving configured directory', async () => {
    const storage = new FileStorage(tmpDir)
    const content = new TextEncoder().encode('test')
    const ref = await storage.store('k1', content, 'text/plain')

    expect(ref.startsWith(tmpDir)).toBe(true)
    expect(ref).toMatch(/\.txt$/)
  })

  it('uses correct file extensions', async () => {
    const storage = new FileStorage(tmpDir)
    const content = new TextEncoder().encode('test')

    const txtRef = await storage.store('k1', content, 'text/plain')
    expect(txtRef).toMatch(/\.txt$/)

    const jsonRef = await storage.store('k2', content, 'application/json')
    expect(jsonRef).toMatch(/\.json$/)

    const pngRef = await storage.store('k3', content, 'image/png')
    expect(pngRef).toMatch(/\.png$/)
  })

  it('throws on missing reference', async () => {
    const storage = new FileStorage(tmpDir)
    await expect(storage.retrieve(path.join(tmpDir, 'nonexistent.txt'))).rejects.toThrow('Reference not found')
  })

  it('sanitizes keys for safe filenames', async () => {
    const storage = new FileStorage(tmpDir)
    const content = new TextEncoder().encode('test')
    const ref = await storage.store('../../../etc/passwd', content, 'text/plain')
    expect(ref).not.toContain('..')
  })

  it('prevents path traversal on retrieve', async () => {
    const storage = new FileStorage(tmpDir)
    await expect(storage.retrieve('../../etc/passwd')).rejects.toThrow('Reference not found')
  })

  it('creates artifact directory if it does not exist', async () => {
    const nestedDir = path.join(tmpDir, 'nested', 'dir')
    const storage = new FileStorage(nestedDir)
    const content = new TextEncoder().encode('test')
    await storage.store('key1', content, 'text/plain')

    const stat = await fs.stat(nestedDir)
    expect(stat.isDirectory()).toBe(true)
  })

  it('persists metadata across instances', async () => {
    const storage1 = new FileStorage(tmpDir)
    const content = new TextEncoder().encode('test')
    const ref = await storage1.store('key1', content, 'application/json')

    const storage2 = new FileStorage(tmpDir)
    const result = await storage2.retrieve(ref)
    expect(result.contentType).toBe('application/json')
  })
})
