import { existsSync, readFileSync } from 'node:fs'
import { spawn } from 'node:child_process'

const DATA_STORE_PATH = '.astro/data-store.json'
const TIMEOUT_MS = 60_000
const SYNCED_MARKER = 'Synced content'

function isDataStorePopulated(): boolean {
  if (!existsSync(DATA_STORE_PATH)) return false
  try {
    return readFileSync(DATA_STORE_PATH, 'utf-8').trim().length > 100
  } catch {
    return false
  }
}

export async function setup() {
  if (isDataStorePopulated()) return

  console.log('[global-setup] Data store not populated — starting astro dev temporarily...')

  const server = spawn('npx', ['astro', 'dev'], {
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: true,
  })

  try {
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('Timed out waiting for "Synced content"')), TIMEOUT_MS)

      const onData = (chunk: Buffer) => {
        process.stdout.write(chunk)
        if (chunk.toString().includes(SYNCED_MARKER)) {
          clearTimeout(timer)
          resolve()
        }
      }

      server.stdout.on('data', onData)
      server.stderr.on('data', onData)
      server.on('error', (err) => {
        clearTimeout(timer)
        reject(err)
      })
      server.on('close', (code) => {
        if (code !== null) {
          clearTimeout(timer)
          reject(new Error(`Astro dev server exited with code ${code}`))
        }
      })
    })
  } finally {
    if (server.pid != null) {
      process.kill(-server.pid, 'SIGTERM')
    }
  }

  console.log('[global-setup] Data store ready.')
}
