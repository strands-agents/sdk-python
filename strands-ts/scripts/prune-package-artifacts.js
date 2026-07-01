import { readdir, rm } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { join } from 'node:path'

const distDir = new URL('../dist/', import.meta.url)
const testFilePattern = /\.test(?:\.[^/.]+)*\.(?:js|d\.ts)(?:\.map)?$/

async function pruneGeneratedTestArtifacts(directory) {
  let entries

  try {
    entries = await readdir(directory, { withFileTypes: true })
  } catch (error) {
    if (error?.code === 'ENOENT') {
      return
    }

    throw error
  }

  await Promise.all(
    entries.map(async (entry) => {
      const entryPath = join(fileURLToPath(directory), entry.name)

      if (entry.isDirectory()) {
        if (entry.name === '__tests__' || entry.name === '__fixtures__') {
          await rm(entryPath, { recursive: true, force: true })
          return
        }

        await pruneGeneratedTestArtifacts(new URL(`${entry.name}/`, directory))
        return
      }

      if (entry.isFile() && testFilePattern.test(entry.name)) {
        await rm(entryPath, { force: true })
      }
    }),
  )
}

await pruneGeneratedTestArtifacts(distDir)
