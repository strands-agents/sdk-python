#!/usr/bin/env node
// Decide the size/* and complexity/* labels for a PR.
//
// Pure data in, pure data out: takes a unified diff plus per-function
// complexity records and emits the label decision as JSON. The same code runs
// locally (`hatch run complexity` / `npm run complexity`) and in CI, so the
// number a contributor sees matches the label the bot applies.

import fs from 'node:fs'
import { countsTowardSize, isAnalyzable, sizeLabel, complexityLabel } from './classify.mjs'
import { parseChangedLines, rangeTouched } from './diff.mjs'

/**
 * @param {object} input
 * @param {string} input.diff - `git diff -U0` between merge base and head
 * @param {Array<{path: string, additions: number, deletions: number}>} input.files
 * @param {Array<{file, name, complexity, startLine, endLine}>} input.functions
 */
export function buildReport({ diff, files, functions }) {
  const countedLines = files
    .filter((f) => countsTowardSize(f.path))
    .reduce((sum, f) => sum + f.additions + f.deletions, 0)
  const totalLines = files.reduce((sum, f) => sum + f.additions + f.deletions, 0)

  const changedLines = parseChangedLines(diff)
  const touched = functions
    .filter((fn) => isAnalyzable(fn.file))
    .filter((fn) => rangeTouched(changedLines.get(fn.file), fn.startLine, fn.endLine))
    .sort((a, b) => b.complexity - a.complexity)

  // No analyzable source touched (docs-only, tests-only) means no complexity
  // signal exists — distinct from "we measured it and it was simple".
  const maxComplexity = touched.length > 0 ? touched[0].complexity : null

  return {
    size: {
      label: sizeLabel(countedLines),
      countedLines,
      totalLines,
      excludedLines: totalLines - countedLines,
    },
    complexity: {
      label: maxComplexity === null ? null : complexityLabel(maxComplexity),
      maxComplexity,
      offenders: touched.slice(0, 10).map((fn) => ({
        file: fn.file,
        name: fn.name,
        complexity: fn.complexity,
        startLine: fn.startLine,
      })),
    },
  }
}

/** Human-readable summary for local runs and the CI job log. */
export function formatReport(report) {
  const lines = []
  const { size, complexity } = report
  lines.push(`size:       ${size.label}  (${size.countedLines} lines counted, ${size.excludedLines} excluded)`)
  if (complexity.label === null) {
    lines.push('complexity: n/a  (no SDK source functions touched)')
  } else {
    lines.push(`complexity: ${complexity.label}  (max ${complexity.maxComplexity})`)
    for (const fn of complexity.offenders) {
      lines.push(`  ${String(fn.complexity).padStart(4)}  ${fn.file}:${fn.startLine}  ${fn.name}`)
    }
  }
  return lines.join('\n')
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const read = (p) => JSON.parse(fs.readFileSync(p, 'utf8'))
  const { DIFF_FILE, FILES_FILE, FUNCTIONS_FILE, REPORT_FILE } = process.env
  if (!DIFF_FILE || !FILES_FILE || !FUNCTIONS_FILE) {
    console.error('set DIFF_FILE, FILES_FILE and FUNCTIONS_FILE')
    process.exit(2)
  }
  const report = buildReport({
    diff: fs.readFileSync(DIFF_FILE, 'utf8'),
    files: read(FILES_FILE),
    functions: read(FUNCTIONS_FILE),
  })
  if (REPORT_FILE) fs.writeFileSync(REPORT_FILE, JSON.stringify(report))
  console.log(formatReport(report))
}
