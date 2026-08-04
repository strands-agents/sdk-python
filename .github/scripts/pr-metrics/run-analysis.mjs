#!/usr/bin/env node
// Analyze a working tree for cognitive complexity and report the labels a PR
// would receive.
//
// This is the entry point behind `npm run complexity` and `hatch run
// complexity`, and the same script CI invokes, so a contributor sees locally
// exactly what the labeler will apply.
//
// Analysis is read-only: the SDK sources are parsed, never imported or
// executed, which is what makes it safe to run against an untrusted PR.
//
// Usage: run-analysis.mjs [--base <ref>] [--json <out>] [--root <dir>]

import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { isAnalyzable } from './classify.mjs'
import { buildReport, formatReport } from './report.mjs'
import { parseSarif } from './complexity-python.mjs'
import { loadTypescript, parseEslintReport } from './complexity-typescript.mjs'

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url))

/**
 * Where the pinned analyzers live. In CI this points at the base revision's
 * install so a PR cannot substitute its own analyzer; locally it defaults to
 * the checked-in tools directory.
 */
const TOOLS_DIR = process.env.PR_METRICS_TOOLS_DIR
  ? path.resolve(process.env.PR_METRICS_TOOLS_DIR)
  : path.join(SCRIPT_DIR, 'tools')

function parseArgs(argv) {
  const args = { base: 'origin/main', root: process.cwd(), json: null }
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === '--base') args.base = argv[++i]
    else if (argv[i] === '--json') args.json = argv[++i]
    else if (argv[i] === '--root') args.root = path.resolve(argv[++i])
  }
  return args
}

function git(root, ...args) {
  return execFileSync('git', args, { cwd: root, encoding: 'utf8', maxBuffer: 256 * 1024 * 1024 })
}

/** Merge base, so a stale branch is not blamed for changes that landed on main. */
function mergeBase(root, base) {
  try {
    return git(root, 'merge-base', base, 'HEAD').trim()
  } catch {
    return base
  }
}

function changedFiles(root, from) {
  const numstat = git(root, 'diff', '--numstat', from, '--')
  const files = []
  for (const line of numstat.trim().split('\n')) {
    if (!line) continue
    const [additions, deletions, filePath] = line.split('\t')
    if (!filePath) continue
    files.push({
      path: filePath,
      // "-" marks a binary file, which has no reviewable line count.
      additions: additions === '-' ? 0 : Number(additions),
      deletions: deletions === '-' ? 0 : Number(deletions),
    })
  }
  return files
}

function run(cmd, args, options = {}) {
  try {
    execFileSync(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'], ...options })
    return true
  } catch (error) {
    // complexipy and eslint exit non-zero when findings exist; the report file
    // is still written, so only a missing report is a real failure.
    return error.status !== undefined
  }
}

/**
 * ESLint flat config for complexity reporting.
 *
 * Generated rather than checked in because the plugin paths must resolve to the
 * pinned tools install, which may sit outside the analyzed tree. The threshold
 * is 0 so sonarjs reports a score for every function; the label thresholds live
 * in classify.mjs.
 */
function complexityConfig() {
  const toUrl = (specifier) =>
    pathToFileURL(path.join(TOOLS_DIR, 'node_modules', specifier)).href
  return `
import sonarjs from ${JSON.stringify(toUrl('eslint-plugin-sonarjs/cjs/plugin.js'))}
import tsparser from ${JSON.stringify(toUrl('@typescript-eslint/parser/dist/index.js'))}

export default [
  {
    files: ['**/*.ts', '**/*.tsx', '**/*.mts', '**/*.cts'],
    languageOptions: {
      parser: tsparser,
      parserOptions: { ecmaVersion: 2022, sourceType: 'module' },
    },
    plugins: { sonarjs },
    rules: { 'sonarjs/cognitive-complexity': ['warn', 0] },
  },
]
`
}

function analyzePython(root, tmp, pythonFiles) {
  if (pythonFiles.length === 0) return []
  const sarif = path.join(tmp, 'python.sarif')
  const ok = run(
    'complexipy',
    [
      '--quiet',
      // Report every function, not just those over a threshold; the labeler
      // decides the bucket. SARIF is the only format carrying line ranges.
      '--ignore-complexity',
      '--max-complexity-allowed',
      '0',
      '--output-format',
      'sarif',
      '--output',
      sarif,
      ...pythonFiles,
    ],
    { cwd: root },
  )
  if (!ok || !fs.existsSync(sarif)) {
    console.error('warning: complexipy did not produce a report; skipping Python complexity')
    console.error('         run: pip install -r .github/scripts/pr-metrics/tools/requirements.txt')
    return []
  }
  return parseSarif(JSON.parse(fs.readFileSync(sarif, 'utf8')), root)
}

async function analyzeTypescript(root, tmp, tsFiles) {
  if (tsFiles.length === 0) return []
  const report = path.join(tmp, 'typescript.json')
  const eslint = path.join(TOOLS_DIR, 'node_modules', '.bin', 'eslint')
  if (!fs.existsSync(eslint)) {
    console.error(
      `warning: complexity analyzers not installed in ${TOOLS_DIR}; skipping TypeScript complexity\n` +
        '         run: npm ci --ignore-scripts --prefix .github/scripts/pr-metrics/tools',
    )
    return []
  }
  // ESLint treats the config file's directory as the base path and silently
  // ignores anything outside it, so the config must be written into the tree
  // being analyzed rather than a temp directory.
  const configPath = path.join(root, `.eslint.complexity.${process.pid}.mjs`)
  fs.writeFileSync(configPath, complexityConfig())
  let ok
  try {
    ok = run(
      eslint,
      ['--no-config-lookup', '-c', configPath, '--format', 'json', '--output-file', report, ...tsFiles],
      { cwd: root },
    )
  } finally {
    fs.rmSync(configPath, { force: true })
  }
  if (!ok || !fs.existsSync(report)) {
    console.error('warning: eslint did not produce a report; skipping TypeScript complexity')
    return []
  }
  const ts = await loadTypescript(TOOLS_DIR)
  return parseEslintReport(ts, JSON.parse(fs.readFileSync(report, 'utf8')), root)
}

async function main() {
  const args = parseArgs(process.argv.slice(2))
  const from = mergeBase(args.root, args.base)
  const files = changedFiles(args.root, from)
  const diff = git(args.root, 'diff', '-U0', from, '--')

  // Analyze only files the PR touches that still exist in the head revision.
  // isAnalyzable is the same predicate the report uses, so a file is never
  // analyzed only to be discarded later.
  const present = files
    .map((f) => f.path)
    .filter(isAnalyzable)
    .filter((p) => fs.existsSync(path.join(args.root, p)))
  const pythonFiles = present.filter((p) => p.endsWith('.py'))
  const tsFiles = present.filter((p) => !p.endsWith('.py'))

  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'pr-complexity-'))
  try {
    const functions = [
      ...analyzePython(args.root, tmp, pythonFiles),
      ...(await analyzeTypescript(args.root, tmp, tsFiles)),
    ]
    const report = buildReport({ diff, files, functions })
    if (args.json) fs.writeFileSync(args.json, JSON.stringify(report))
    console.log(formatReport(report))
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  await main()
}
