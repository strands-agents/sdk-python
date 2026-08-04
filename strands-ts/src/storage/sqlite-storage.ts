import type sqlite3 from 'sqlite3'
import type { Storage } from './storage.js'

import { StorageError } from '../errors.js'
import { namespace, normalizeKey, normalizePrefix } from './storage.js'

const DEFAULT_DATABASE_PATH = './.strands-storage.sqlite'
const BUSY_TIMEOUT_MS = 5000
const TABLE_NAME = 'strands_sdk_storage_v1'
const TABLE_DEFINITION_SQL = `
  ${TABLE_NAME} (
    key TEXT PRIMARY KEY NOT NULL CHECK(length(key) > 0),
    value BLOB NOT NULL CHECK(typeof(value) = 'blob')
  ) WITHOUT ROWID
`
const CREATE_TABLE_SQL = `CREATE TABLE IF NOT EXISTS ${TABLE_DEFINITION_SQL}`
const EXPECTED_TABLE_SQL = normalizeSQL(`CREATE TABLE ${TABLE_DEFINITION_SQL}`)

interface ValueRow {
  value: unknown
  valueType: string
}

interface KeyRow {
  key: string
}

interface TableDefinitionRow {
  sql: string | null
}

/**
 * A namespaced SQLite storage view that retains control of the shared database lifecycle.
 */
export interface SQLiteStorageView extends Storage {
  /**
   * Returns a nested view that shares the same underlying database.
   *
   * @param prefix - Prefix to prepend to all keys
   * @returns A closeable SQLite storage view scoped to `prefix`
   */
  namespace(prefix: string): SQLiteStorageView
  /**
   * Closes the shared underlying database connection.
   *
   * @returns A promise that resolves after pending operations drain and the connection closes
   */
  close(): Promise<void>
}

// sqlite3's TEXT binding replaces unpaired UTF-16 surrogates. Fixed-width hex
// keeps every JavaScript code unit distinct while preserving prefix and sort order.
function encodeKey(key: string): string {
  let encoded = ''
  for (let index = 0; index < key.length; index++) {
    encoded += key.charCodeAt(index).toString(16).padStart(4, '0')
  }
  return encoded
}

function decodeKey(encoded: string): string {
  if (!encoded || encoded.length % 4 !== 0 || !/^[0-9a-f]+$/.test(encoded)) {
    throw new Error('Invalid encoded SQLite storage key')
  }

  let key = ''
  for (let index = 0; index < encoded.length; index += 4) {
    key += String.fromCharCode(Number.parseInt(encoded.slice(index, index + 4), 16))
  }
  try {
    if (normalizeKey(key) !== key) {
      throw new Error('Invalid encoded SQLite storage key')
    }
  } catch {
    throw new Error('Invalid encoded SQLite storage key')
  }
  return key
}

function encodedPrefixUpperBound(prefix: string): string | undefined {
  if (!prefix) return undefined
  const lastIndex = prefix.length - 1
  return `${prefix.slice(0, lastIndex)}${String.fromCharCode(prefix.charCodeAt(lastIndex) + 1)}`
}

function normalizeSQL(sql: string): string {
  return sql.replace(/\s+/g, ' ').trim()
}

function toStorageError(message: string, error: unknown): StorageError {
  return error instanceof StorageError ? error : new StorageError(message, { cause: error })
}

function isMissingSQLiteDependency(error: unknown): boolean {
  if (!(error instanceof Error) || !('code' in error)) return false
  if (error.code !== 'ERR_MODULE_NOT_FOUND' && error.code !== 'MODULE_NOT_FOUND') return false
  return /(?:package|module) ['"]sqlite3['"]/.test(error.message)
}

/**
 * SQLite-backed {@link Storage} implementation.
 *
 * Stores normalized keys in a lossless internal encoding and raw byte values in
 * a single SQLite database. The optional `sqlite3` peer dependency is loaded
 * lazily on first use, so importing or constructing `SQLiteStorage` does not
 * require the driver. Install it with `npm install sqlite3@^6.0.1` before
 * performing storage operations.
 *
 * This backend is Node.js-only and requires Node.js 20.17 or later with a
 * compatible `sqlite3` 6.x release (`^6.0.1`). The default database path is
 * `./.strands-storage.sqlite`, kept separate from {@link LocalFileStorage}'s
 * default keyspace.
 *
 * @example
 * ```typescript
 * import { SQLiteStorage } from '@strands-agents/sdk/storage'
 *
 * const storage = new SQLiteStorage('./.strands/agent.sqlite')
 * try {
 *   await storage.write('sessions/abc/snapshot.json', bytes)
 * } finally {
 *   await storage.close()
 * }
 * ```
 */
export class SQLiteStorage implements Storage {
  private readonly _databasePath: string
  private _databasePromise: Promise<sqlite3.Database> | undefined
  private _closePromise: Promise<void> | undefined
  private readonly _operations = new Set<Promise<unknown>>()
  private _closed = false

  /**
   * @param databasePath - SQLite database path. Defaults to `./.strands-storage.sqlite`.
   * @throws TypeError if `databasePath` is empty
   */
  constructor(databasePath: string = DEFAULT_DATABASE_PATH) {
    if (databasePath.length === 0) {
      throw new TypeError("databasePath must not be empty; use ':memory:' for ephemeral storage")
    }
    this._databasePath = databasePath
  }

  /**
   * Stores `data` under `key`, overwriting any existing value.
   *
   * @param key - Opaque, `/`-separated key identifying the value
   * @param data - Raw bytes to persist
   * @throws {@link StorageError} if the key is invalid or the write fails
   */
  async write(key: string, data: Uint8Array): Promise<void> {
    const normalized = normalizeKey(key)
    try {
      await this._withDatabase(async (database) => {
        const { Buffer } = await import('node:buffer')
        await this._run(
          database,
          `
          INSERT INTO ${TABLE_NAME} (key, value)
          VALUES (?, ?)
          ON CONFLICT(key) DO UPDATE SET value = excluded.value
        `,
          [encodeKey(normalized), Buffer.from(data)]
        )
      })
    } catch (error: unknown) {
      throw toStorageError(`Failed to write '${normalized}' to SQLite database '${this._databasePath}'`, error)
    }
  }

  /**
   * Retrieves the bytes previously stored under `key`.
   *
   * @param key - The key to read
   * @returns The stored bytes, or `null` if no value exists for `key`
   * @throws {@link StorageError} if the key is invalid or the read fails
   */
  async read(key: string): Promise<Uint8Array | null> {
    const normalized = normalizeKey(key)
    try {
      return await this._withDatabase(async (database) => {
        const row = await this._get<ValueRow>(
          database,
          `SELECT value, typeof(value) AS valueType FROM ${TABLE_NAME} WHERE key = ?`,
          [encodeKey(normalized)]
        )
        if (!row) return null
        if (row.valueType !== 'blob' || !(row.value instanceof Uint8Array)) {
          throw new Error(`Invalid non-BLOB value stored for key '${normalized}'`)
        }
        return new Uint8Array(row.value)
      })
    } catch (error: unknown) {
      throw toStorageError(`Failed to read '${normalized}' from SQLite database '${this._databasePath}'`, error)
    }
  }

  /**
   * Deletes the value stored under `key`. A no-op if the key does not exist.
   *
   * @param key - The key to delete
   * @throws {@link StorageError} if the key is invalid or the delete fails
   */
  async delete(key: string): Promise<void> {
    const normalized = normalizeKey(key)
    try {
      await this._withDatabase(async (database) => {
        await this._run(database, `DELETE FROM ${TABLE_NAME} WHERE key = ?`, [encodeKey(normalized)])
      })
    } catch (error: unknown) {
      throw toStorageError(`Failed to delete '${normalized}' from SQLite database '${this._databasePath}'`, error)
    }
  }

  /**
   * Lists the keys whose names begin with `prefix`, sorted lexicographically.
   *
   * @param prefix - Key prefix to match. An empty string matches all keys.
   * @returns The matching keys, sorted ascending
   * @throws {@link StorageError} if the prefix is invalid or the listing fails
   */
  async list(prefix: string): Promise<string[]> {
    const normalized = normalizePrefix(prefix)
    try {
      return await this._withDatabase(async (database) => {
        const encodedPrefix = encodeKey(normalized)
        const upperBound = encodedPrefixUpperBound(encodedPrefix)
        const [sql, params] =
          upperBound === undefined
            ? [
                `
          SELECT key
          FROM ${TABLE_NAME}
          WHERE key >= ?
          ORDER BY key ASC
        `,
                [encodedPrefix],
              ]
            : [
                `
          SELECT key
          FROM ${TABLE_NAME}
          WHERE key >= ? AND key < ?
          ORDER BY key ASC
        `,
                [encodedPrefix, upperBound],
              ]
        const rows = await this._all<KeyRow>(database, sql, params)
        return rows
          .map((row) => decodeKey(row.key))
          .filter((key) => key.startsWith(normalized))
          .sort()
      })
    } catch (error: unknown) {
      throw toStorageError(`Failed to list SQLite database '${this._databasePath}' under '${normalized}'`, error)
    }
  }

  /**
   * Returns a prefixed view that shares this instance's database and lifecycle.
   * Closing any view closes the database for the original instance and all views.
   *
   * @param prefix - Prefix to prepend to all keys
   * @returns A closeable SQLite storage view scoped to `prefix`
   */
  namespace(prefix: string): SQLiteStorageView {
    return this._namespace(this, prefix)
  }

  /**
   * Closes the underlying database connection. Calling `close` more than once is a no-op.
   *
   * @throws {@link StorageError} if closing the database fails
   */
  close(): Promise<void> {
    this._closePromise ??= this._close()
    return this._closePromise
  }

  private async _close(): Promise<void> {
    this._closed = true
    await Promise.allSettled(this._operations)
    const databasePromise = this._databasePromise
    if (!databasePromise) return

    let database: sqlite3.Database
    try {
      database = await databasePromise
    } catch {
      return
    }

    try {
      await this._closeDatabase(database)
    } catch (error: unknown) {
      throw new StorageError(`Failed to close SQLite database '${this._databasePath}'`, { cause: error })
    }
  }

  private _withDatabase<Result>(operation: (database: sqlite3.Database) => Promise<Result>): Promise<Result> {
    if (this._closed) {
      return Promise.reject(new StorageError(`SQLite database '${this._databasePath}' is closed`))
    }

    const operationPromise = this._getDatabase().then(operation)
    this._operations.add(operationPromise)
    void operationPromise.then(
      () => this._operations.delete(operationPromise),
      () => this._operations.delete(operationPromise)
    )
    return operationPromise
  }

  private async _getDatabase(): Promise<sqlite3.Database> {
    if (this._closed) {
      throw new StorageError(`SQLite database '${this._databasePath}' is closed`)
    }
    if (this._databasePromise) return this._databasePromise

    const opening = this._openDatabase()
    this._databasePromise = opening
    void opening.catch(() => {
      if (this._databasePromise === opening) {
        this._databasePromise = undefined
      }
    })
    return opening
  }

  private async _openDatabase(): Promise<sqlite3.Database> {
    if (this._databasePath !== ':memory:') {
      const { mkdir } = await import('node:fs/promises')
      const { dirname } = await import('node:path')
      await mkdir(dirname(this._databasePath), { recursive: true })
    }

    const sqlite = await this._loadSQLite()
    const database = await new Promise<sqlite3.Database>((resolve, reject) => {
      const opened = new sqlite.Database(this._databasePath, (error: Error | null) => {
        if (error) {
          reject(error)
          return
        }
        resolve(opened)
      })
    })

    database.configure('busyTimeout', BUSY_TIMEOUT_MS)
    try {
      await this._exec(database, CREATE_TABLE_SQL)
      await this._validateSchema(database)
      return database
    } catch (error: unknown) {
      await this._closeDatabase(database).catch(() => {})
      throw error
    }
  }

  private async _loadSQLite(): Promise<typeof sqlite3> {
    try {
      const { default: sqlite } = await import('sqlite3')
      return sqlite
    } catch (error: unknown) {
      if (!isMissingSQLiteDependency(error)) {
        throw new StorageError(
          "SQLiteStorage could not load the optional peer dependency 'sqlite3'. Reinstall a compatible 6.x release with 'npm install sqlite3@^6.0.1' and verify it supports this Node.js runtime.",
          { cause: error }
        )
      }
      throw new StorageError(
        "SQLiteStorage requires the optional peer dependency 'sqlite3' ^6.0.1. Install it with 'npm install sqlite3@^6.0.1'.",
        { cause: error }
      )
    }
  }

  private async _validateSchema(database: sqlite3.Database): Promise<void> {
    const definition = await this._get<TableDefinitionRow>(
      database,
      "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
      [TABLE_NAME]
    )
    if (
      definition?.sql === null ||
      definition?.sql === undefined ||
      normalizeSQL(definition.sql) !== EXPECTED_TABLE_SQL
    ) {
      throw new Error(`SQLite table '${TABLE_NAME}' has an incompatible schema`)
    }
  }

  private _namespace(storage: Storage, prefix: string): SQLiteStorageView {
    const view = namespace(storage, prefix)
    return {
      ...view,
      namespace: (nestedPrefix: string): SQLiteStorageView => this._namespace(view, nestedPrefix),
      close: (): Promise<void> => this.close(),
    }
  }

  private _run(database: sqlite3.Database, sql: string, params: unknown[]): Promise<void> {
    return new Promise((resolve, reject) => {
      database.run(sql, params, (error: Error | null) => {
        if (error) reject(error)
        else resolve()
      })
    })
  }

  private _get<Row>(database: sqlite3.Database, sql: string, params: unknown[]): Promise<Row | undefined> {
    return new Promise((resolve, reject) => {
      database.get<Row>(sql, params, (error: Error | null, row: Row | undefined) => {
        if (error) reject(error)
        else resolve(row)
      })
    })
  }

  private _all<Row>(database: sqlite3.Database, sql: string, params: unknown[]): Promise<Row[]> {
    return new Promise((resolve, reject) => {
      database.all<Row>(sql, params, (error: Error | null, rows: Row[]) => {
        if (error) reject(error)
        else resolve(rows)
      })
    })
  }

  private _exec(database: sqlite3.Database, sql: string): Promise<void> {
    return new Promise((resolve, reject) => {
      database.exec(sql, (error: Error | null) => {
        if (error) reject(error)
        else resolve()
      })
    })
  }

  private _closeDatabase(database: sqlite3.Database): Promise<void> {
    return new Promise((resolve, reject) => {
      database.close((error: Error | null) => {
        if (error) reject(error)
        else resolve()
      })
    })
  }
}
