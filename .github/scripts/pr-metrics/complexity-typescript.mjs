#!/usr/bin/env node
// Report cognitive complexity for TypeScript functions, via eslint-plugin-sonarjs.
//
// sonarjs anchors each finding at the function's *name*, not its body, so
// `line === endLine` for every result. Diff scoping needs the full span, so we
// resolve each anchor to the smallest enclosing function via the TypeScript AST.
//
// Usage: complexity-typescript.mjs <eslint-json-file>
// Writes normalized `{file, name, complexity, startLine, endLine}` records to stdout.

import fs from 'node:fs'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const RULE_ID = 'sonarjs/cognitive-complexity'
// "Refactor this function to reduce its Cognitive Complexity from 31 to the 15 allowed."
const COMPLEXITY_RE = /Complexity from (\d+) to/

/**
 * Load the TypeScript compiler from the pinned tools install.
 *
 * Resolved at call time rather than by a static import because the analyzers
 * live outside this script's own module tree (see PR_METRICS_TOOLS_DIR).
 */
export async function loadTypescript(toolsDir) {
  const specifier = toolsDir
    ? pathToFileURL(path.join(toolsDir, 'node_modules', 'typescript', 'lib', 'typescript.js')).href
    : 'typescript'
  const mod = await import(specifier)
  return mod.default ?? mod
}

function isFunctionNode(ts, node) {
  return (
    ts.isFunctionDeclaration(node) ||
    ts.isFunctionExpression(node) ||
    ts.isArrowFunction(node) ||
    ts.isMethodDeclaration(node) ||
    ts.isConstructorDeclaration(node) ||
    ts.isGetAccessor(node) ||
    ts.isSetAccessor(node)
  )
}

/** All function line ranges in a source file, 1-indexed and inclusive. */
export function functionRanges(ts, sourceText, fileName) {
  const source = ts.createSourceFile(fileName, sourceText, ts.ScriptTarget.Latest, true)
  const ranges = []
  const visit = (node) => {
    if (isFunctionNode(ts, node)) {
      ranges.push({
        startLine: source.getLineAndCharacterOfPosition(node.getStart(source)).line + 1,
        endLine: source.getLineAndCharacterOfPosition(node.getEnd()).line + 1,
      })
    }
    ts.forEachChild(node, visit)
  }
  visit(source)
  return ranges
}

function toRepoRelative(filePath, repoRoot) {
  if (!path.isAbsolute(filePath)) return filePath.split(path.sep).join('/')
  // Resolve symlinks on both sides so a symlinked checkout root does not yield
  // an escaping ../.. path.
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

/** Smallest range containing `line`; sonarjs anchors nested arrows inside methods. */
function smallestEnclosing(ranges, line) {
  let best = null
  for (const range of ranges) {
    if (range.startLine > line || range.endLine < line) continue
    if (best === null || range.endLine - range.startLine < best.endLine - best.startLine) {
      best = range
    }
  }
  return best
}

/**
 * ESLint reports the offending token's position but not the function's name, so
 * recover it from the source line for a readable report.
 */
function nameAt(sourceText, line, column) {
  const text = sourceText.split('\n')[line - 1]
  if (text === undefined) return '<anonymous>'
  const identifier = /([A-Za-z_$][\w$]*)/.exec(text.slice(column - 1))
  return identifier ? identifier[1] : '<anonymous>'
}

export function parseEslintReport(ts, report, repoRoot, readFile = (p) => fs.readFileSync(p, 'utf8')) {
  const functions = []

  for (const fileReport of report) {
    const findings = (fileReport.messages ?? []).filter((m) => m.ruleId === RULE_ID)
    if (findings.length === 0) continue

    let ranges = []
    let sourceText = ''
    try {
      sourceText = readFile(fileReport.filePath)
      ranges = functionRanges(ts, sourceText, fileReport.filePath)
    } catch {
      // Unreadable file: fall back to the anchor line below.
    }

    const relative = toRepoRelative(fileReport.filePath, repoRoot)

    for (const finding of findings) {
      const match = COMPLEXITY_RE.exec(finding.message ?? '')
      if (!match) continue
      const enclosing = smallestEnclosing(ranges, finding.line)
      functions.push({
        file: relative,
        name: nameAt(sourceText, finding.line, finding.column),
        complexity: Number(match[1]),
        startLine: enclosing?.startLine ?? finding.line,
        endLine: enclosing?.endLine ?? finding.line,
      })
    }
  }
  return functions
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const [reportPath] = process.argv.slice(2)
  if (!reportPath) {
    console.error('usage: complexity-typescript.mjs <eslint-json-file>')
    process.exit(2)
  }
  const repoRoot = process.env.COMPLEXITY_REPO_ROOT ?? process.cwd()
  const ts = await loadTypescript(process.env.PR_METRICS_TOOLS_DIR)
  const report = JSON.parse(fs.readFileSync(reportPath, 'utf8'))
  process.stdout.write(JSON.stringify(parseEslintReport(ts, report, repoRoot)))
}
