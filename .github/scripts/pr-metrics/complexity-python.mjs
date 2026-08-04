#!/usr/bin/env node
// Report cognitive complexity for Python functions, via complexipy.
//
// complexipy's JSON output omits line numbers, so we read its SARIF output
// instead: that carries the `startLine`/`endLine` range each function spans,
// which is what diff scoping needs.
//
// Usage: complexity-python.mjs <sarif-file>
// Writes normalized `{file, name, complexity, startLine, endLine}` records to stdout.

import fs from 'node:fs'
import path from 'node:path'

// "Function 'x' has a cognitive complexity of 37, which exceeds ..."
const COMPLEXITY_RE = /cognitive complexity of (\d+)/

export function parseSarif(sarif, repoRoot) {
  const results = sarif?.runs?.[0]?.results ?? []
  const functions = []

  for (const result of results) {
    const match = COMPLEXITY_RE.exec(result?.message?.text ?? '')
    if (!match) continue

    const location = result?.locations?.[0]?.physicalLocation
    const region = location?.region
    const uri = location?.artifactLocation?.uri
    if (!region?.startLine || !uri) continue

    functions.push({
      file: toRepoRelative(uri, repoRoot),
      name: result?.locations?.[0]?.logicalLocations?.[0]?.name ?? '<anonymous>',
      complexity: Number(match[1]),
      startLine: region.startLine,
      endLine: region.endLine ?? region.startLine,
    })
  }
  return functions
}

function toRepoRelative(uri, repoRoot) {
  const filePath = uri.startsWith('file://') ? uri.slice('file://'.length) : uri
  if (!path.isAbsolute(filePath)) return filePath.split(path.sep).join('/')
  // Resolve symlinks on both sides so a symlinked checkout root (/tmp ->
  // /private/tmp on macOS) does not yield an escaping ../.. path.
  const relative = path.relative(realpath(repoRoot), realpath(filePath))
  return relative.split(path.sep).join('/')
}

function realpath(p) {
  try {
    return fs.realpathSync(p)
  } catch {
    return p
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const [sarifPath] = process.argv.slice(2)
  if (!sarifPath) {
    console.error('usage: complexity-python.mjs <sarif-file>')
    process.exit(2)
  }
  const repoRoot = process.env.COMPLEXITY_REPO_ROOT ?? process.cwd()
  const sarif = JSON.parse(fs.readFileSync(sarifPath, 'utf8'))
  process.stdout.write(JSON.stringify(parseSarif(sarif, repoRoot)))
}
