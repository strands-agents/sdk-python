// Tests for the PR metrics labeler.
//
// Uses node:test so they run with no install step: `node --test .github/scripts/pr-metrics/`

import assert from 'node:assert/strict'
import test from 'node:test'
import { classify, FileKind, countsTowardSize, isAnalyzable, sizeLabel, complexityLabel } from './classify.mjs'
import { parseChangedLines, rangeTouched } from './diff.mjs'
import { buildReport } from './report.mjs'
import { escapeHtml, labelsFromMetrics } from './apply-labels.mjs'
import { parseSarif } from './complexity-python.mjs'

test('classify separates tests from source', () => {
  assert.equal(classify('strands-py/src/strands/agent.py'), FileKind.SOURCE)
  assert.equal(classify('strands-py/tests/strands/test_agent.py'), FileKind.TEST)
  assert.equal(classify('strands-py/tests_integ/test_model.py'), FileKind.TEST)
  assert.equal(classify('strands-ts/src/agent/__tests__/agent.test.ts'), FileKind.TEST)
  assert.equal(classify('strands-ts/src/mcp/client.test.ts'), FileKind.TEST)
  assert.equal(classify('strands-py/src/strands/conftest.py'), FileKind.TEST)
  assert.equal(classify('site/src/content/docs/index.mdx'), FileKind.DOCS)
  assert.equal(classify('package-lock.json'), FileKind.GENERATED)
  assert.equal(classify('strands-py/uv.lock'), FileKind.GENERATED)
})

test('size counts source and docs but not tests or lockfiles', () => {
  assert.equal(countsTowardSize('strands-ts/src/agent/agent.ts'), true)
  assert.equal(countsTowardSize('site/src/content/docs/index.mdx'), true)
  assert.equal(countsTowardSize('strands-py/tests/test_agent.py'), false)
  assert.equal(countsTowardSize('package-lock.json'), false)
})

test('only SDK source is analyzed for complexity', () => {
  assert.equal(isAnalyzable('strands-py/src/strands/agent.py'), true)
  assert.equal(isAnalyzable('strands-ts/src/agent/agent.ts'), true)
  // Docs tooling and tests are out of scope even though they are real code.
  assert.equal(isAnalyzable('site/src/scripts/build.ts'), false)
  assert.equal(isAnalyzable('strands-py/tests/test_agent.py'), false)
  assert.equal(isAnalyzable('strands-ts/src/agent/__tests__/agent.test.ts'), false)
})

test('label thresholds sit on documented boundaries', () => {
  assert.equal(sizeLabel(20), 'size/xs')
  assert.equal(sizeLabel(21), 'size/s')
  assert.equal(sizeLabel(1000), 'size/l')
  assert.equal(sizeLabel(1001), 'size/xl')
  assert.equal(complexityLabel(10), 'complexity/low')
  assert.equal(complexityLabel(11), 'complexity/medium')
  assert.equal(complexityLabel(25), 'complexity/medium')
  assert.equal(complexityLabel(26), 'complexity/high')
})

test('parseChangedLines tracks post-image line numbers', () => {
  const diff = [
    'diff --git a/a.py b/a.py',
    '--- a/a.py',
    '+++ b/a.py',
    '@@ -10,0 +11,2 @@',
    '+one',
    '+two',
    '@@ -30,1 +32,1 @@',
    '-old',
    '+new',
  ].join('\n')
  const changed = parseChangedLines(diff)
  assert.deepEqual(
    [...changed.get('a.py')].sort((x, y) => x - y),
    [11, 12, 32]
  )
})

test('parseChangedLines ignores pure deletions and deleted files', () => {
  const diff = [
    'diff --git a/gone.py b/gone.py',
    '--- a/gone.py',
    '+++ /dev/null',
    '@@ -1,3 +0,0 @@',
    '-a',
    '-b',
    '-c',
  ].join('\n')
  assert.equal(parseChangedLines(diff).size, 0)
})

test('rangeTouched only matches overlapping ranges', () => {
  const lines = new Set([50])
  assert.equal(rangeTouched(lines, 40, 60), true)
  assert.equal(rangeTouched(lines, 50, 50), true)
  assert.equal(rangeTouched(lines, 51, 90), false)
  assert.equal(rangeTouched(undefined, 1, 10), false)
})

// The core guarantee: a pre-existing hotspot elsewhere in a touched file must
// not drive the label, or every PR touching that file is permanently 'high'.
test('complexity ignores untouched functions in a touched file', () => {
  const diff = [
    'diff --git a/strands-py/src/strands/m.py b/strands-py/src/strands/m.py',
    '--- a/strands-py/src/strands/m.py',
    '+++ b/strands-py/src/strands/m.py',
    '@@ -20,0 +21,1 @@',
    '+    pass',
  ].join('\n')
  const report = buildReport({
    diff,
    files: [{ path: 'strands-py/src/strands/m.py', additions: 1, deletions: 0 }],
    functions: [
      { file: 'strands-py/src/strands/m.py', name: 'simple', complexity: 4, startLine: 10, endLine: 30 },
      { file: 'strands-py/src/strands/m.py', name: 'monster', complexity: 90, startLine: 100, endLine: 400 },
    ],
  })
  assert.equal(report.complexity.maxComplexity, 4)
  assert.equal(report.complexity.label, 'complexity/low')
})

test('size excludes test churn from the bucket', () => {
  const report = buildReport({
    diff: '',
    files: [
      { path: 'strands-py/src/strands/agent.py', additions: 10, deletions: 5 },
      { path: 'strands-py/tests/test_agent.py', additions: 800, deletions: 100 },
      { path: 'package-lock.json', additions: 5000, deletions: 4000 },
    ],
    functions: [],
  })
  assert.equal(report.size.countedLines, 15)
  assert.equal(report.size.label, 'size/xs')
  assert.equal(report.size.totalLines, 9915)
})

test('a docs-only PR gets no complexity label', () => {
  const report = buildReport({
    diff: '',
    files: [{ path: 'site/src/content/docs/index.mdx', additions: 40, deletions: 2 }],
    functions: [],
  })
  assert.equal(report.complexity.label, null)
  assert.equal(report.size.label, 'size/s')
})

// The artifact comes from a job that ran untrusted code, so labels must be
// derived from its integers, never from any string it supplies.
test('labelsFromMetrics ignores artifact-supplied label strings', () => {
  const labels = labelsFromMetrics({
    size: { countedLines: 5, label: 'approved' },
    complexity: { maxComplexity: 3, label: 'lgtm; ship it' },
  })
  assert.deepEqual(labels, ['size/xs', 'complexity/low'])
})

test('labelsFromMetrics rejects malformed and out-of-range numbers', () => {
  assert.deepEqual(labelsFromMetrics({ size: { countedLines: -1 }, complexity: { maxComplexity: null } }), [])
  assert.deepEqual(labelsFromMetrics({ size: { countedLines: 1e12 }, complexity: { maxComplexity: null } }), [])
  assert.deepEqual(labelsFromMetrics({ size: { countedLines: 'ten' }, complexity: { maxComplexity: null } }), [])
  assert.deepEqual(labelsFromMetrics({ size: { countedLines: 1.5 }, complexity: { maxComplexity: null } }), [])
  // A valid size still applies when complexity is unusable.
  assert.deepEqual(labelsFromMetrics({ size: { countedLines: 10 }, complexity: { maxComplexity: 'x' } }), ['size/xs'])
})

// `git diff --numstat` reports a rename as the combined path `dir/{a => b}` in
// its text format, which matches none of the patterns below — a renamed test
// would count toward size and renamed source would go unanalyzed. changedFiles()
// uses -z, where the new path arrives as its own field; this pins the contract
// that classification only ever sees a plain path.
test('renamed paths resolve to plain paths, not brace notation', () => {
  assert.equal(classify('strands-ts/src/agent/__tests__/b.test.ts'), FileKind.TEST)
  assert.equal(countsTowardSize('strands-ts/src/agent/__tests__/b.test.ts'), false)
  assert.equal(isAnalyzable('strands-py/src/strands/agent2.py'), true)
  // The brace form must NOT be mistaken for analyzable source.
  assert.equal(isAnalyzable('strands-py/src/strands/{agent.py => agent2.py}'), false)
})

test('parseChangedLines is not fooled by an added line whose content starts with ++', () => {
  const diff = [
    'diff --git a/strands-py/src/strands/m.py b/strands-py/src/strands/m.py',
    '--- a/strands-py/src/strands/m.py',
    '+++ b/strands-py/src/strands/m.py',
    '@@ -5,0 +6,3 @@',
    '+# a comment',
    '++ b/nowhere.py',
    '+more code',
  ].join('\n')
  const changed = parseChangedLines(diff)
  assert.deepEqual([...changed.keys()], ['strands-py/src/strands/m.py'])
  assert.deepEqual(
    [...changed.get('strands-py/src/strands/m.py')].sort((x, y) => x - y),
    [6, 7, 8]
  )
})

test('parseChangedLines handles multiple files and a no-count hunk header', () => {
  const diff = [
    'diff --git a/a.py b/a.py',
    '--- a/a.py',
    '+++ b/a.py',
    '@@ -1 +1 @@',
    '-old',
    '+new',
    'diff --git a/b.py b/b.py',
    '--- a/b.py',
    '+++ b/b.py',
    '@@ -10,0 +11,1 @@',
    '+added',
  ].join('\n')
  const changed = parseChangedLines(diff)
  assert.deepEqual([...changed.get('a.py')], [1])
  assert.deepEqual([...changed.get('b.py')], [11])
})

test('parseChangedLines ignores the no-newline marker', () => {
  const diff = [
    'diff --git a/a.py b/a.py',
    '--- a/a.py',
    '+++ b/a.py',
    '@@ -1 +1,2 @@',
    ' first',
    '+second',
    '\\ No newline at end of file',
  ].join('\n')
  assert.deepEqual([...parseChangedLines(diff).get('a.py')], [2])
})

test('parseChangedLines keeps paths containing spaces intact', () => {
  const diff = [
    'diff --git a/dir/we ird.py b/dir/we ird.py',
    '--- a/dir/we ird.py',
    // git appends a tab after a path containing whitespace.
    '+++ b/dir/we ird.py\t',
    '@@ -0,0 +1 @@',
    '+x = 1',
  ].join('\n')
  assert.deepEqual([...parseChangedLines(diff).get('dir/we ird.py')], [1])
})

// The artifact is produced by a job that ran untrusted code. Number(null) is 0,
// so coercing before validating would turn a missing metric into size/xs.
test('labelsFromMetrics rejects falsy non-numbers instead of reading them as 0', () => {
  assert.deepEqual(labelsFromMetrics({ size: { countedLines: null }, complexity: { maxComplexity: false } }), [])
  assert.deepEqual(labelsFromMetrics({ size: { countedLines: [] }, complexity: { maxComplexity: '' } }), [])
  assert.deepEqual(labelsFromMetrics({ size: { countedLines: '0' }, complexity: { maxComplexity: null } }), [])
  // A real zero is still a valid measurement.
  assert.deepEqual(labelsFromMetrics({ size: { countedLines: 0 }, complexity: { maxComplexity: 0 } }), [
    'size/xs',
    'complexity/low',
  ])
})

// core.summary interpolates into HTML without escaping, and a PR controls its
// own file paths, so an unescaped path could close the surrounding tag and
// inject markup into the trusted labeling job's summary.
test('escapeHtml neutralizes markup in artifact-supplied paths', () => {
  assert.equal(escapeHtml('</code></pre><h1>Approved</h1>'), '&lt;/code&gt;&lt;/pre&gt;&lt;h1&gt;Approved&lt;/h1&gt;')
  assert.equal(escapeHtml('<img src=x onerror="alert(1)">'), '&lt;img src=x onerror=&quot;alert(1)&quot;&gt;')
  // Ampersands escape first so existing entities are not double-decoded.
  assert.equal(escapeHtml('a & b'), 'a &amp; b')
  assert.equal(escapeHtml('ordinary/path.ts'), 'ordinary/path.ts')
})

test('parseSarif reads complexity and line ranges from complexipy output', () => {
  const sarif = {
    runs: [
      {
        results: [
          {
            message: {
              text: "Function 'f' has a cognitive complexity of 37, which exceeds the maximum of 15.",
            },
            locations: [
              {
                physicalLocation: {
                  artifactLocation: { uri: '/repo/strands-py/src/strands/m.py' },
                  region: { startLine: 447, endLine: 654 },
                },
                logicalLocations: [{ name: 'f', kind: 'function' }],
              },
            ],
          },
        ],
      },
    ],
  }
  assert.deepEqual(parseSarif(sarif, '/repo'), [
    { file: 'strands-py/src/strands/m.py', name: 'f', complexity: 37, startLine: 447, endLine: 654 },
  ])
})
