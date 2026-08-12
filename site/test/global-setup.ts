import { execFileSync } from 'node:child_process'
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync } from 'node:fs'
import { dirname, join } from 'node:path'

const DATA_STORE_PATH = '.astro/data-store.json'
// `astro sync` writes the store under the cache dir (Astro's default is
// node_modules/.astro), while the test environment reads the dev-mode path
// above; see getDataStoreFile in astro/dist/content/paths.js.
const SYNC_DATA_STORE_PATH = 'node_modules/.astro/data-store.json'
const TIMEOUT_MS = 120_000

// Everything the content-layer snapshot is derived from: the collection
// sources and the schemas that shape them.
const CONTENT_SOURCES = ['src/content', 'src/content.config.ts']

/**
 * Newest mtime under a path, following symlinks (docs `_generated` dirs point
 * into .build/api-docs). Directory mtimes are included so a deleted file,
 * which bumps only its parent directory, still counts as a change. Broken
 * symlinks and unreadable paths report 0: nothing there to be newer than the
 * snapshot.
 */
function newestMtime(path: string): number {
  let stat
  try {
    stat = statSync(path)
  } catch {
    return 0
  }
  if (!stat.isDirectory()) return stat.mtimeMs
  let newest = stat.mtimeMs
  for (const entry of readdirSync(path)) {
    newest = Math.max(newest, newestMtime(join(path, entry)))
  }
  return newest
}

/**
 * getCollection() reads the content-layer snapshot in DATA_STORE_PATH, not
 * the content files themselves, so a snapshot older than the content would
 * make every collection test silently assert against outdated entries.
 */
function isDataStoreFresh(): boolean {
  if (!existsSync(DATA_STORE_PATH)) return false
  try {
    if (readFileSync(DATA_STORE_PATH, 'utf-8').trim().length <= 100) return false
    const storeMtime = statSync(DATA_STORE_PATH).mtimeMs
    return CONTENT_SOURCES.every((source) => newestMtime(source) <= storeMtime)
  } catch {
    return false
  }
}

export async function setup() {
  if (isDataStoreFresh()) return

  console.log('[global-setup] Data store missing or stale — running astro sync...')

  // `astro sync` runs to completion and exits, so unlike a watch-mode dev
  // server there is no process left to clean up (an unkilled dev server would
  // keep rewriting the store from a stale worktree state). The stale dev-path
  // store is removed first so a sync failure fails the tests loudly instead
  // of leaving them reading the outdated snapshot.
  rmSync(DATA_STORE_PATH, { force: true })
  execFileSync('npx', ['astro', 'sync'], { stdio: 'inherit', timeout: TIMEOUT_MS })
  mkdirSync(dirname(DATA_STORE_PATH), { recursive: true })
  copyFileSync(SYNC_DATA_STORE_PATH, DATA_STORE_PATH)

  console.log('[global-setup] Data store ready.')
}
