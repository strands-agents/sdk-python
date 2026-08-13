import { execFileSync } from 'node:child_process'
import { readdirSync, realpathSync, statSync } from 'node:fs'
import { join } from 'node:path'

const DATA_STORE_PATH = '.astro/data-store.json'
const TIMEOUT_MS = 120_000

// sync writes the store under cacheDir (getDataStoreFile in astro/dist/content/paths.js), so pointing
// cacheDir at .astro targets the path the tests read. Child process: getViteConfig aliases in-process
// 'astro' imports to a types-only stub.
const SYNC_SCRIPT = "const { sync } = await import('astro'); await sync({ cacheDir: './.astro' })"

// Everything the content-layer snapshot is derived from.
const CONTENT_SOURCES = ['src/content', 'src/content.config.ts']

// Follows symlinks (docs `_generated` dirs point into .build/api-docs); `visited` breaks cycles.
// Directory mtimes are included so a deleted file, which bumps only its parent dir, registers.
function newestMtime(path: string, visited: Set<string>): number {
  let stat
  try {
    stat = statSync(path)
  } catch {
    return 0
  }
  if (!stat.isDirectory()) return stat.mtimeMs
  const resolved = realpathSync(path)
  if (visited.has(resolved)) return 0
  visited.add(resolved)
  let newest = stat.mtimeMs
  for (const entry of readdirSync(path)) {
    newest = Math.max(newest, newestMtime(join(path, entry), visited))
  }
  return newest
}

// getCollection() reads the snapshot in DATA_STORE_PATH, not the content
// files, so a stale snapshot silently tests against outdated entries.
function isDataStoreFresh(): boolean {
  let storeStat
  try {
    storeStat = statSync(DATA_STORE_PATH)
  } catch {
    return false
  }
  // A store this small cannot hold real collections; treat it as absent.
  if (storeStat.size <= 100) return false
  return CONTENT_SOURCES.every((source) => newestMtime(source, new Set()) <= storeStat.mtimeMs)
}

export function setup() {
  if (isDataStoreFresh()) return

  console.log('[global-setup] Data store missing or stale — running astro sync...')

  execFileSync('node', ['--input-type=module', '-e', SYNC_SCRIPT], { stdio: 'inherit', timeout: TIMEOUT_MS })

  console.log('[global-setup] Data store ready.')
}
