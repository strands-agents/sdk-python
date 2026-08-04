// Tests for the PR metrics labeler.
//
// Uses node:test so they run with no install step: `node --test .github/scripts/pr-metrics/`

import assert from 'node:assert/strict'
import test from 'node:test'
import { classify, FileKind, countsTowardSize, isAnalyzable, sizeLabel, complexityLabel } from './classify.mjs'
import { parseChangedLines, rangeTouched } from './diff.mjs'
import { buildReport } from './report.mjs'
import { labelsFromMetrics } from './apply-labels.mjs'
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
  assert.deepEqual([...changed.get('a.py')].sort((x, y) => x - y), [11, 12, 32])
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
  assert.deepEqual(labelsFromMetrics({ size: { countedLines: 10 }, complexity: { maxComplexity: 'x' } }), [
    'size/xs',
  ])
})

test('parseSarif reads complexity and line ranges from complexipy output', () => {
  const sarif = {
    runs: [
      {
        results: [
          {
            message: { text: "Function 'f' has a cognitive complexity of 37, which exceeds the maximum allowed complexity of 15." },
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
