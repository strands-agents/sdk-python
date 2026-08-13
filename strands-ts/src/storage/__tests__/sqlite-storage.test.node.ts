import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { mkdir, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { randomUUID } from 'node:crypto'
import { setImmediate } from 'node:timers/promises'

import { StorageError } from '../../errors.js'
import { SQLiteStorage } from '../sqlite-storage.js'
import sqlite3 from 'sqlite3'

interface QueryPlanRow {
  detail: string
}

interface SQLiteStorageInternals {
  _databasePromise: Promise<sqlite3.Database>
}

function runStatement(database: sqlite3.Database, sql: string, params: unknown[] = []): Promise<void> {
  return new Promise((resolve, reject) => {
    database.run(sql, params, (error: Error | null) => {
      if (error) reject(error)
      else resolve()
    })
  })
}

function openDatabase(path: string): Promise<sqlite3.Database> {
  return new Promise((resolve, reject) => {
    const database = new sqlite3.Database(path, (error: Error | null) => {
      if (error) reject(error)
      else resolve(database)
    })
  })
}

function closeDatabase(database: sqlite3.Database): Promise<void> {
  return new Promise((resolve, reject) => {
    database.close((error: Error | null) => {
      if (error) reject(error)
      else resolve()
    })
  })
}

function allRows<Row>(database: sqlite3.Database, sql: string): Promise<Row[]> {
  return new Promise((resolve, reject) => {
    database.all<Row>(sql, (error: Error | null, rows: Row[]) => {
      if (error) reject(error)
      else resolve(rows)
    })
  })
}

describe('SQLiteStorage', () => {
  let baseDir: string
  let databasePath: string
  let storage: SQLiteStorage

  beforeEach(() => {
    baseDir = join(tmpdir(), `strands-sqlite-test-${randomUUID()}`)
    databasePath = join(baseDir, 'nested', 'storage.sqlite')
    storage = new SQLiteStorage(databasePath)
  })

  afterEach(async () => {
    await storage.close().catch(() => {})
    await rm(baseDir, { recursive: true, force: true })
  })

  describe('constructor', () => {
    it('rejects an empty database path', () => {
      expect(() => new SQLiteStorage('')).toThrow(
        new TypeError("databasePath must not be empty; use ':memory:' for ephemeral storage")
      )
    })
  })

  describe('write and read', () => {
    it('round-trips raw bytes and creates parent directories', async () => {
      const data = new Uint8Array([0, 1, 127, 128, 255])

      await storage.write('test/value', data)

      expect(await storage.read('test/value')).toEqual(data)
    })

    it('overwrites an existing value', async () => {
      await storage.write('key', new TextEncoder().encode('first'))
      await storage.write('key', new TextEncoder().encode('second'))

      const result = await storage.read('key')
      expect(new TextDecoder().decode(result!)).toBe('second')
    })

    it('returns null for a missing key', async () => {
      expect(await storage.read('missing')).toBeNull()
    })

    it('persists values across storage instances', async () => {
      const data = new TextEncoder().encode('persistent')
      await storage.write('key', data)
      await storage.close()

      storage = new SQLiteStorage(databasePath)

      expect(await storage.read('key')).toEqual(data)
    })
  })

  describe('delete', () => {
    it('removes an existing key', async () => {
      await storage.write('key', new Uint8Array([1]))

      await storage.delete('key')

      expect(await storage.read('key')).toBeNull()
    })

    it('is a no-op for a missing key', async () => {
      await expect(storage.delete('missing')).resolves.toBeUndefined()
    })
  })

  describe('list', () => {
    it('returns matching keys in lexicographic order', async () => {
      await storage.write('sessions/c', new Uint8Array([3]))
      await storage.write('sessions/a', new Uint8Array([1]))
      await storage.write('sessions/b', new Uint8Array([2]))
      await storage.write('memory/a', new Uint8Array([4]))

      expect(await storage.list('sessions/')).toEqual(['sessions/a', 'sessions/b', 'sessions/c'])
    })

    it('returns all keys for an empty prefix', async () => {
      await storage.write('b', new Uint8Array([2]))
      await storage.write('a', new Uint8Array([1]))

      expect(await storage.list('')).toEqual(['a', 'b'])
    })

    it('treats pattern metacharacters as literal prefix content', async () => {
      await storage.write('prefix%/value', new Uint8Array([1]))
      await storage.write('prefix_/value', new Uint8Array([2]))
      await storage.write('prefix*/value', new Uint8Array([3]))
      await storage.write('prefix?/value', new Uint8Array([4]))
      await storage.write('prefix[/value', new Uint8Array([5]))
      await storage.write('prefixA/value', new Uint8Array([6]))

      expect(await storage.list('prefix%/')).toEqual(['prefix%/value'])
      expect(await storage.list('prefix_/')).toEqual(['prefix_/value'])
      expect(await storage.list('prefix*/')).toEqual(['prefix*/value'])
      expect(await storage.list('prefix?/')).toEqual(['prefix?/value'])
      expect(await storage.list('prefix[/')).toEqual(['prefix[/value'])
    })

    it('matches embedded NUL prefixes exactly', async () => {
      await storage.write('a', new Uint8Array([1]))
      await storage.write('a\0b', new Uint8Array([2]))
      await storage.write('a\0b/child', new Uint8Array([3]))
      await storage.write('a\0c', new Uint8Array([4]))

      expect(await storage.list('a\0b')).toEqual(['a\0b', 'a\0b/child'])
    })

    it('preserves distinct unpaired surrogate keys', async () => {
      const firstKey = '\ud800/value'
      const secondKey = '\ud801/value'
      await storage.write(firstKey, new Uint8Array([1]))
      await storage.write(secondKey, new Uint8Array([2]))

      expect(await storage.read(firstKey)).toEqual(new Uint8Array([1]))
      expect(await storage.read(secondKey)).toEqual(new Uint8Array([2]))
      expect(await storage.list('\ud800/')).toEqual([firstKey])
      expect(await storage.list('\ud801/')).toEqual([secondKey])
    })

    it('round-trips prefixes containing the maximum Unicode scalar', async () => {
      const prefix = String.fromCodePoint(0x10ffff)
      await storage.write(prefix, new Uint8Array([1]))
      await storage.write(`${prefix}/child`, new Uint8Array([2]))
      await storage.write('z', new Uint8Array([3]))

      expect(await storage.list(prefix)).toEqual([prefix, `${prefix}/child`])
    })

    it('matches prefixes case-sensitively', async () => {
      await storage.write('Prefix/value', new Uint8Array([1]))
      await storage.write('prefix/value', new Uint8Array([2]))

      expect(await storage.list('prefix/')).toEqual(['prefix/value'])
    })

    it('uses the primary key index for prefix lookups', async () => {
      await storage.write('*sessions/a', new Uint8Array([1]))

      const { _databasePromise } = storage as unknown as SQLiteStorageInternals
      const database = await _databasePromise
      const statements: string[] = []
      const captureStatement = (sql: string): void => {
        statements.push(sql)
      }

      database.on('trace', captureStatement)
      try {
        await storage.list('*sessions/')
        await setImmediate()
      } finally {
        database.off('trace', captureStatement)
      }

      const listStatement = statements.find((sql) => sql.includes('WHERE key >='))
      if (!listStatement) throw new Error('SQLite list query was not traced')

      const queryPlan = await new Promise<QueryPlanRow[]>((resolve, reject) => {
        database.all<QueryPlanRow>(
          `EXPLAIN QUERY PLAN ${listStatement}`,
          (error: Error | null, rows: QueryPlanRow[]) => {
            if (error) reject(error)
            else resolve(rows)
          }
        )
      })
      expect(queryPlan.some((row) => row.detail.includes('SEARCH strands_sdk_storage_v1 USING PRIMARY KEY'))).toBe(true)
    })

    it('returns an empty array when no keys match', async () => {
      await storage.write('other/key', new Uint8Array([1]))

      expect(await storage.list('missing/')).toEqual([])
    })
  })

  describe('key normalization', () => {
    it('normalizes equivalent keys and prefixes', async () => {
      await storage.write('/a//b/', new Uint8Array([1]))

      expect(await storage.read('a/b')).toEqual(new Uint8Array([1]))
      expect(await storage.list('/a//')).toEqual(['a/b'])
    })

    it('rejects invalid keys and prefixes', async () => {
      await expect(storage.write('', new Uint8Array([1]))).rejects.toThrow(StorageError)
      await expect(storage.read('a/../b')).rejects.toThrow(StorageError)
      await expect(storage.delete('../a')).rejects.toThrow(StorageError)
      await expect(storage.list('../')).rejects.toThrow(StorageError)
    })
  })

  describe('namespace', () => {
    it('scopes keys, normalizes trailing slashes, and preserves lifecycle control', async () => {
      const namespaced = storage.namespace('sessions/')
      const nested = namespaced.namespace('abc/')

      await nested.write('snapshot', new Uint8Array([1]))

      expect(await storage.list('')).toEqual(['sessions/abc/snapshot'])
      expect(await namespaced.list('')).toEqual(['abc/snapshot'])
      expect(await nested.list('')).toEqual(['snapshot'])

      await nested.close()
      await expect(storage.read('sessions/abc/snapshot')).rejects.toThrow(StorageError)
    })
  })

  describe('errors', () => {
    it('enforces BLOB values and rejects malformed persisted rows', async () => {
      await storage.write('key', new Uint8Array([1]))

      const { _databasePromise } = storage as unknown as SQLiteStorageInternals
      const database = await _databasePromise
      await expect(runStatement(database, 'UPDATE strands_sdk_storage_v1 SET value = ?', ['corrupt'])).rejects.toThrow()

      await runStatement(database, 'PRAGMA ignore_check_constraints = ON')
      for (const malformedValue of ['corrupt', 8]) {
        await runStatement(database, 'UPDATE strands_sdk_storage_v1 SET value = ?', [malformedValue])
        await expect(storage.read('key')).rejects.toMatchObject({
          name: 'StorageError',
          cause: expect.objectContaining({
            message: expect.stringContaining('non-BLOB'),
          }),
        })
      }
    })

    it('rejects malformed encoded keys', async () => {
      await storage.write('valid', new Uint8Array([1]))

      const { _databasePromise } = storage as unknown as SQLiteStorageInternals
      const database = await _databasePromise
      await runStatement(database, 'PRAGMA ignore_check_constraints = ON')
      for (const malformedKey of ['', '002f', '0061002f002f0062', '002e002e']) {
        await runStatement(database, 'INSERT INTO strands_sdk_storage_v1 (key, value) VALUES (?, ?)', [
          malformedKey,
          new Uint8Array([1]),
        ])
        await expect(storage.list('')).rejects.toThrow(StorageError)
        await runStatement(database, 'DELETE FROM strands_sdk_storage_v1 WHERE key = ?', [malformedKey])
      }
    })

    it('rejects an incompatible existing SDK table before writing', async () => {
      await mkdir(dirname(databasePath), { recursive: true })
      const database = await openDatabase(databasePath)
      await runStatement(database, 'CREATE TABLE strands_sdk_storage_v1 (key TEXT PRIMARY KEY, customer_value TEXT)')
      await runStatement(database, "INSERT INTO strands_sdk_storage_v1 VALUES ('customer-row', 'untouched')")
      await closeDatabase(database)

      await expect(storage.write('key', new Uint8Array([1]))).rejects.toMatchObject({
        name: 'StorageError',
        cause: expect.objectContaining({
          message: expect.stringContaining('incompatible schema'),
        }),
      })

      const inspectionDatabase = await openDatabase(databasePath)
      expect(
        await allRows<{ key: string; customer_value: string }>(
          inspectionDatabase,
          'SELECT key, customer_value FROM strands_sdk_storage_v1'
        )
      ).toEqual([{ key: 'customer-row', customer_value: 'untouched' }])
      await closeDatabase(inspectionDatabase)
    })

    it('rejects matching columns with incompatible constraints', async () => {
      await mkdir(dirname(databasePath), { recursive: true })
      const database = await openDatabase(databasePath)
      await runStatement(
        database,
        `CREATE TABLE strands_sdk_storage_v1 (
          key TEXT PRIMARY KEY NOT NULL CHECK(length(key) > 0),
          value BLOB NOT NULL CHECK(typeof(value) = 'blob') CHECK(length(value) < 2)
        ) WITHOUT ROWID`
      )
      await closeDatabase(database)

      await expect(storage.write('key', new Uint8Array([1, 2]))).rejects.toMatchObject({
        name: 'StorageError',
        cause: expect.objectContaining({
          message: expect.stringContaining('incompatible schema'),
        }),
      })
    })

    it('wraps database errors and retries transient initialization failures', async () => {
      const directoryPath = join(baseDir, 'database-directory')
      await mkdir(directoryPath, { recursive: true })
      storage = new SQLiteStorage(directoryPath)

      await expect(storage.write('key', new Uint8Array([1]))).rejects.toMatchObject({
        name: 'StorageError',
        message: expect.stringContaining(directoryPath),
        cause: expect.any(Error),
      })

      await rm(directoryPath, { recursive: true })
      await storage.write('key', new Uint8Array([1]))
      expect(await storage.read('key')).toEqual(new Uint8Array([1]))
    })

    it('drains operations that started before close', async () => {
      const writes = Array.from({ length: 64 }, (_, index) =>
        storage.write(`shutdown/${index}`, new Uint8Array([index]))
      )

      const firstClose = storage.close()
      const secondClose = storage.close()

      expect(secondClose).toBe(firstClose)
      await expect(Promise.all([...writes, firstClose, secondClose])).resolves.toHaveLength(66)

      storage = new SQLiteStorage(databasePath)
      expect(await storage.list('shutdown/')).toHaveLength(64)
    })

    it('rejects operations after close', async () => {
      await storage.close()

      await expect(storage.read('key')).rejects.toThrow(StorageError)
      await expect(storage.close()).resolves.toBeUndefined()
    })
  })
})
