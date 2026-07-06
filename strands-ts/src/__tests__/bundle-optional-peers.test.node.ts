import { describe, it, expect } from 'vitest'
import { build } from 'esbuild'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

// strands-ts/src — the directory that holds the package entry `index.ts`.
const srcDir = resolve(dirname(fileURLToPath(import.meta.url)), '..')

/**
 * Bundle a source entry with esbuild and return the set of resolved input
 * modules (the static import graph). Dynamically imported modules are still
 * resolved and appear here, which is exactly what we want: a bundler eagerly
 * resolves string-literal `import()` calls while building the graph.
 */
async function bundleGraphInputs(entryContents: string): Promise<string[]> {
  const result = await build({
    stdin: { contents: entryContents, resolveDir: srcDir, loader: 'ts', sourcefile: 'entry.ts' },
    bundle: true,
    platform: 'node',
    format: 'esm',
    metafile: true,
    write: false,
    logLevel: 'silent',
  })
  return Object.keys(result.metafile.inputs)
}

describe('optional peer dependencies stay out of the core import graph', () => {
  // Regression test for #3016: importing only `Agent` must not drag the optional
  // `@aws-sdk/client-s3` peer (used solely by the opt-in context-offloader
  // `S3Storage` backend) into a bundler's static graph, which broke esbuild
  // builds for apps that never use context offloading.
  it('importing Agent does not pull @aws-sdk/client-s3 into the bundle graph', async () => {
    const inputs = await bundleGraphInputs("import { Agent } from './index.ts'\nconsole.log(Agent)\n")
    const s3Inputs = inputs.filter((path) => path.includes('@aws-sdk/client-s3'))
    expect(s3Inputs).toEqual([])
  }, 30_000)
})
