// Apply the size/* and complexity/* labels from an analysis artifact.
//
// Runs in the trusted half of the labeler with `pull-requests: write`, so the
// artifact is treated as hostile input throughout: it was produced by a job
// that ran contributor-controlled code.
//
//   - Label names are derived here from clamped integers, never read as strings
//     from the artifact, so an arbitrary label cannot be injected.
//   - The PR number is verified against the triggering run's head SHA, so one
//     PR cannot relabel another.
//   - Only labels in the managed namespace are added or removed, so labels set
//     by maintainers or other automation are left alone.

import fs from 'node:fs'
import path from 'node:path'
import {
  ALL_MANAGED_LABELS,
  SIZE_OVERFLOW_LABEL,
  COMPLEXITY_OVERFLOW_LABEL,
  sizeLabel,
  complexityLabel,
} from './classify.mjs'

// Guards against an artifact claiming an absurd count to force a bucket.
const MAX_REASONABLE_LINES = 10_000_000
const MAX_REASONABLE_COMPLEXITY = 100_000

function readNonNegativeInt(value, max) {
  const n = Number(value)
  if (!Number.isFinite(n) || !Number.isInteger(n) || n < 0 || n > max) return null
  return n
}

/**
 * Recompute labels from the artifact's numbers. Deliberately ignores the
 * artifact's own `label` strings so only names from classify.mjs can be applied.
 */
export function labelsFromMetrics(metrics) {
  const labels = []

  const countedLines = readNonNegativeInt(metrics?.size?.countedLines, MAX_REASONABLE_LINES)
  if (countedLines !== null) labels.push(sizeLabel(countedLines))

  // null max complexity is legitimate: docs-only and test-only PRs touch no
  // analyzable function, so they get no complexity label at all.
  const rawComplexity = metrics?.complexity?.maxComplexity
  if (rawComplexity !== null && rawComplexity !== undefined) {
    const maxComplexity = readNonNegativeInt(rawComplexity, MAX_REASONABLE_COMPLEXITY)
    if (maxComplexity !== null) labels.push(complexityLabel(maxComplexity))
  }

  return labels
}

/** Resolve the PR this artifact belongs to, verifying it against the run's head SHA. */
async function resolvePrNumber({ github, context, core, claimed }) {
  const run = context.payload.workflow_run
  const candidates = (run.pull_requests ?? []).map((pr) => pr.number)

  if (candidates.length === 0) {
    // Forks do not populate `pull_requests`, so search by head SHA instead.
    const { data } = await github.rest.repos.listPullRequestsAssociatedWithCommit({
      owner: context.repo.owner,
      repo: context.repo.repo,
      commit_sha: run.head_sha,
    })
    candidates.push(...data.map((pr) => pr.number))
  }

  if (candidates.length === 0) {
    core.info(`no pull request found for head sha ${run.head_sha}`)
    return null
  }
  // The artifact's claim is only honored if the API independently agrees the PR
  // is associated with this run's head commit.
  if (claimed !== null && candidates.includes(claimed)) return claimed
  if (claimed !== null) {
    core.warning(`artifact claimed PR #${claimed}, which is not associated with ${run.head_sha}; ignoring it`)
  }
  return candidates[0]
}

export async function applyLabels({ github, context, core, workspace }) {
  const metricsDir = path.join(workspace, 'metrics')
  const metrics = JSON.parse(fs.readFileSync(path.join(metricsDir, 'pr-metrics.json'), 'utf8'))

  const claimedRaw = fs.readFileSync(path.join(metricsDir, 'pr-number.txt'), 'utf8').trim()
  const claimed = readNonNegativeInt(claimedRaw, Number.MAX_SAFE_INTEGER)

  const prNumber = await resolvePrNumber({ github, context, core, claimed })
  if (prNumber === null) return

  const desired = labelsFromMetrics(metrics)
  if (desired.length === 0) {
    core.warning('artifact contained no usable metrics; leaving labels unchanged')
    return
  }

  const identity = { owner: context.repo.owner, repo: context.repo.repo, issue_number: prNumber }
  const { data: current } = await github.rest.issues.listLabelsOnIssue(identity)
  const currentManaged = current.map((l) => l.name).filter((name) => ALL_MANAGED_LABELS.includes(name))

  // Remove only stale labels in our namespace, so re-runs are quiet and other
  // automation's labels survive.
  for (const name of currentManaged) {
    if (!desired.includes(name)) {
      await github.rest.issues.removeLabel({ ...identity, name })
      core.info(`removed ${name}`)
    }
  }
  const missing = desired.filter((name) => !currentManaged.includes(name))
  if (missing.length > 0) {
    await github.rest.issues.addLabels({ ...identity, labels: missing })
    core.info(`added ${missing.join(', ')}`)
  }

  await summarize({ core, metrics, desired })
}

async function summarize({ core, metrics, desired }) {
  const { size, complexity } = metrics
  core.summary.addHeading('PR metrics', 3).addTable([
    [
      { data: 'Metric', header: true },
      { data: 'Label', header: true },
      { data: 'Detail', header: true },
    ],
    ['Size', desired.find((l) => l.startsWith('size/')) ?? 'n/a', `${size?.countedLines ?? '?'} lines counted, ${size?.excludedLines ?? '?'} excluded (tests and generated files)`],
    [
      'Complexity',
      desired.find((l) => l.startsWith('complexity/')) ?? 'n/a',
      complexity?.maxComplexity == null
        ? 'no SDK source functions touched'
        : `max cognitive complexity ${complexity.maxComplexity} among functions this PR touches`,
    ],
  ])

  const offenders = Array.isArray(complexity?.offenders) ? complexity.offenders : []
  if (desired.includes(COMPLEXITY_OVERFLOW_LABEL) && offenders.length > 0) {
    core.summary.addRaw('\nMost complex functions this PR touches:\n')
    // Rendered as text, not markdown, so artifact-supplied names cannot inject
    // markup into the summary.
    core.summary.addCodeBlock(
      offenders
        .slice(0, 10)
        .map((fn) => `${String(fn.complexity).padStart(4)}  ${fn.file}:${fn.startLine}  ${fn.name}`)
        .join('\n'),
    )
  }
  if (desired.includes(SIZE_OVERFLOW_LABEL)) {
    core.summary.addRaw('\nThis PR is large enough that reviewers may ask for it to be split.\n')
  }
  await core.summary.write()
}
