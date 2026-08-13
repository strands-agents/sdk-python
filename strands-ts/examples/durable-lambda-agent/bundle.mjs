import { rmSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { build } from 'esbuild'

const exampleDirectory = path.dirname(fileURLToPath(import.meta.url))
const outputDirectory = path.join(exampleDirectory, 'dist')

rmSync(outputDirectory, { recursive: true, force: true })

await build({
  entryPoints: [path.join(exampleDirectory, 'src/handler.ts')],
  outfile: path.join(outputDirectory, 'handler.js'),
  bundle: true,
  platform: 'node',
  target: 'node22',
  format: 'cjs',
  sourcemap: true,
  logLevel: 'info',
})
