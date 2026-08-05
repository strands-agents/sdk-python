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
import { toRepoRelative } from './classify.mjs'

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

/**
 * The name a function is known by.
 *
 * Anonymous function expressions and arrows carry no name of their own, so fall
 * back to whatever binds them — `const handler = () => {}` reads as `handler`,
 * and `{ onEvent() {} }` as `onEvent`. Methods are qualified with their class so
 * two same-named methods are distinguishable in a report.
 */
function functionName(ts, node) {
  if (node.name) {
    const own = node.name.getText()
    if (ts.isMethodDeclaration(node) || ts.isGetAccessor(node) || ts.isSetAccessor(node)) {
      const className = node.parent?.name?.getText()
      return className ? `${className}::${own}` : own
    }
    return own
  }
  if (ts.isConstructorDeclaration(node)) {
    const className = node.parent?.name?.getText()
    return className ? `${className}::constructor` : 'constructor'
  }
  const parent = node.parent
  if (parent) {
    // const f = () => {} / f: () => {} / f = function () {}
    if (ts.isVariableDeclaration(parent) || ts.isPropertyDeclaration(parent) || ts.isPropertyAssignment(parent)) {
      return parent.name?.getText() ?? '<anonymous>'
    }
    if (ts.isBinaryExpression(parent) && parent.operatorToken?.kind === ts.SyntaxKind.EqualsToken) {
      return parent.left?.getText() ?? '<anonymous>'
    }
    // A callback has no name of its own; the call it is passed to is the useful
    // label, so `items.map(...)` reads as `map callback`.
    if (ts.isCallExpression(parent)) {
      const callee = parent.expression
      const calleeName = ts.isPropertyAccessExpression(callee) ? callee.name.getText() : callee.getText?.()
      if (calleeName) return `${calleeName} callback`
    }
  }
  return '<anonymous>'
}

/**
 * All function ranges in a source file, 1-indexed and inclusive, with names.
 *
 * The AST is the authority on names: sonarjs reports only a token position, and
 * recovering a name from the source text at that position misreads arrow
 * parameters and `async function*` as the function's name.
 */
export function functionRanges(ts, sourceText, fileName) {
  const source = ts.createSourceFile(fileName, sourceText, ts.ScriptTarget.Latest, true)
  const ranges = []
  const visit = (node) => {
    if (isFunctionNode(ts, node)) {
      ranges.push({
        startLine: source.getLineAndCharacterOfPosition(node.getStart(source)).line + 1,
        endLine: source.getLineAndCharacterOfPosition(node.getEnd()).line + 1,
        name: functionName(ts, node),
      })
    }
    ts.forEachChild(node, visit)
  }
  visit(source)
  return ranges
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

export function parseEslintReport(ts, report, repoRoot) {
  const functions = []

  for (const fileReport of report) {
    const findings = (fileReport.messages ?? []).filter((m) => m.ruleId === RULE_ID)
    if (findings.length === 0) continue

    let ranges = []
    try {
      ranges = functionRanges(ts, fs.readFileSync(fileReport.filePath, 'utf8'), fileReport.filePath)
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
        name: enclosing?.name ?? '<anonymous>',
        complexity: Number(match[1]),
        startLine: enclosing?.startLine ?? finding.line,
        endLine: enclosing?.endLine ?? finding.line,
      })
    }
  }
  return functions
}
