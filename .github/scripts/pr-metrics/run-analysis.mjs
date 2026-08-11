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
  // quotePath=false keeps non-ASCII paths literal instead of octal-escaped and
  // quoted, so they match the paths the analyzers report.
  return execFileSync('git', ['-c', 'core.quotePath=false', ...args], {
    cwd: root,
    encoding: 'utf8',
    maxBuffer: 256 * 1024 * 1024,
  })
}

/** Merge base, so a stale branch is not blamed for changes that landed on main. */
function mergeBase(root, base) {
  try {
    return git(root, 'merge-base', base, 'HEAD').trim()
  } catch {
    // No merge base (shallow clone, unrelated histories, or a ref this clone
    // lacks). Fall back to the ref itself, but only if git can resolve it —
    // otherwise every later git call fails with an opaque error.
    try {
      git(root, 'rev-parse', '--verify', `${base}^{commit}`)
      return base
    } catch {
      throw new Error(
        `cannot resolve a diff base from '${base}': it is neither mergeable with HEAD nor a known commit. ` +
          'Pass --base <ref> with a ref this clone contains.'
      )
    }
  }
}

/**
 * Files the range changes, with their churn.
 *
 * `-z` is what makes renames usable. In the default text format a rename is
 * reported as the single combined path `dir/{old.py => new.py}`, which matches
 * none of the classification patterns, so a renamed test would count toward the
 * size bucket and a renamed source file would never be analyzed. With `-z`, git
 * emits the old and new paths as separate NUL-delimited fields, and the churn
 * reflects only the lines that actually changed rather than the whole file.
 * `-z` also stops git quoting paths with spaces or non-ASCII bytes.
 */
export function changedFiles(root, from) {
  return parseNumstatZ(git(root, 'diff', '--numstat', '-z', from, '--'))
}

/**
 * Parse `git diff --numstat -z` output.
 *
 * Records are NUL-delimited. A normal change is "adds\tdels\tpath"; a rename or
 * copy is "adds\tdels\t" followed by the old path and the new path as two
 * further NUL-delimited fields.
 */
export function parseNumstatZ(numstat) {
  const files = []
  const fields = numstat.split('\0')

  for (let i = 0; i < fields.length; i += 1) {
    const record = fields[i]
    if (!record) continue

    // Only the first two tabs are delimiters, since only the counts are
    // guaranteed tab-free. Splitting on every tab would read a path that itself
    // begins with a tab as a rename record — whose third field is also empty —
    // and the skip below would then swallow the next two files.
    const firstTab = record.indexOf('\t')
    const secondTab = firstTab < 0 ? -1 : record.indexOf('\t', firstTab + 1)
    if (secondTab < 0) continue

    const additions = record.slice(0, firstTab)
    const deletions = record.slice(firstTab + 1, secondTab)
    let filePath = record.slice(secondTab + 1)

    if (filePath === '') {
      // Rename or copy: skip the old path, keep the new one.
      i += 2
      filePath = fields[i]
    }
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

/**
 * Run an analyzer, reporting whether its output can be trusted.
 *
 * `okStatuses` lists the exit codes that still mean "ran and produced a report",
 * since both analyzers use a non-zero exit to report findings. Anything else — a
 * missing binary (no status at all), a signal, or a fatal configuration error —
 * means the report is absent or partial and must not be read as "no findings".
 */
function run(cmd, args, okStatuses, options = {}) {
  try {
    execFileSync(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'], ...options })
    return true
  } catch (error) {
    if (!Number.isInteger(error.status)) return false
    return okStatuses.includes(error.status)
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
  const toUrl = (specifier) => pathToFileURL(path.join(TOOLS_DIR, 'node_modules', specifier)).href
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
    // complexipy exits 1 when any function exceeds the (zero) threshold, which
    // is every run; the report is still written.
    [1],
    { cwd: root }
  )
  if (!ok || !fs.existsSync(sarif)) {
    console.error('warning: complexipy did not produce a report; skipping Python complexity')
    console.error('         run: pip install -r .github/scripts/pr-metrics/tools/requirements.txt')
    return []
  }
  return parseSarif(JSON.parse(fs.readFileSync(sarif, 'utf8')), root)
}

/**
 * Score the merge-base versions of the changed Python files.
 *
 * Python complexity counts only functions whose score increased over the base
 * (complexipy's snapshot-watermark semantics): editing inside an already
 * complex function without making it worse should not inherit the function's
 * whole score. Returns a map of `file::name` to base complexity. A file with
 * no base version (new or renamed) yields no entries, so its functions count
 * in full as new code. TypeScript has no analyzer support for this and keeps
 * absolute scoring.
 */
function baselinePython(root, tmp, from, pythonFiles) {
  const baseline = new Map()
  if (pythonFiles.length === 0) return baseline

  const baseDir = path.join(tmp, 'base')
  const extracted = []
  for (const file of pythonFiles) {
    let content
    try {
      content = git(root, 'show', `${from}:${file}`)
    } catch {
      continue
    }
    const dest = path.join(baseDir, file)
    fs.mkdirSync(path.dirname(dest), { recursive: true })
    fs.writeFileSync(dest, content)
    extracted.push(dest)
  }
  if (extracted.length === 0) return baseline

  const sarif = path.join(tmp, 'python-base.sarif')
  const ok = run(
    'complexipy',
    [
      '--quiet',
      '--ignore-complexity',
      '--max-complexity-allowed',
      '0',
      '--output-format',
      'sarif',
      '--output',
      sarif,
      ...extracted,
    ],
    [1],
    { cwd: baseDir }
  )
  if (!ok || !fs.existsSync(sarif)) {
    // Without a baseline every touched function counts in full, which is the
    // pre-existing behavior rather than a wrong label; warn and continue.
    console.error('warning: complexipy failed on the base revision; python complexity falls back to absolute scoring')
    return baseline
  }
  for (const fn of parseSarif(JSON.parse(fs.readFileSync(sarif, 'utf8')), baseDir)) {
    const key = `${fn.file}::${fn.name}`
    // Names are already qualified (`Class::method`) and nested functions fold
    // into their parent's score, so within one file a key collision requires a
    // module-level redefinition. If one occurs, keeping the highest base score
    // can mask a real increase in the lower-scored twin; that under-counts,
    // which is the safe direction for an advisory label.
    baseline.set(key, Math.max(baseline.get(key) ?? 0, fn.complexity))
  }
  return baseline
}

async function analyzeTypescript(root, tmp, tsFiles) {
  if (tsFiles.length === 0) return []
  const report = path.join(tmp, 'typescript.json')
  const eslint = path.join(TOOLS_DIR, 'node_modules', '.bin', 'eslint')
  if (!fs.existsSync(eslint)) {
    console.error(
      `warning: complexity analyzers not installed in ${TOOLS_DIR}; skipping TypeScript complexity\n` +
        '         run: npm ci --ignore-scripts --prefix .github/scripts/pr-metrics/tools'
    )
    return []
  }
  // With an explicit -c, ESLint's base path is the cwd, not the config's
  // directory — so the config lives in a temp dir and `cwd: root` is what keeps
  // the analyzed files inside the base path.
  const configPath = path.join(tmp, 'eslint.complexity.config.mjs')
  fs.writeFileSync(configPath, complexityConfig())
  const ok = run(
    eslint,
    ['--no-config-lookup', '-c', configPath, '--format', 'json', '--output-file', report, ...tsFiles],
    // The complexity rule is a warning, so a clean run exits 0 and exit 1 means
    // some other rule errored; either way the report is written. Exit 2 is a
    // fatal config or parse failure and must not be read as "no findings".
    [1],
    { cwd: root }
  )
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
  // Rename detection stays on to match changedFiles(): a renamed file then
  // reports only its genuinely changed lines, instead of every line looking
  // added and every pre-existing function counting as touched.
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
    const baseline = baselinePython(args.root, tmp, from, pythonFiles)
    const pythonFunctions = analyzePython(args.root, tmp, pythonFiles).map((fn) => ({
      ...fn,
      baseComplexity: baseline.get(`${fn.file}::${fn.name}`) ?? null,
    }))
    const functions = [...pythonFunctions, ...(await analyzeTypescript(args.root, tmp, tsFiles))]
    const report = buildReport({ diff, files, functions })
    if (args.json) fs.writeFileSync(args.json, JSON.stringify(report))
    console.log(formatReport(report))
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
}

// pathToFileURL percent-encodes, which a bare string concatenation does not, so
// comparing raw paths would silently no-op for a checkout path containing a
// space or non-ASCII character. It also rejects a missing argv[1], which is the
// case when this module is imported from `node -e`, a worker, or the REPL.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main()
}
