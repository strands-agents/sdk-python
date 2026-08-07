import * as fs from 'fs';
import * as path from 'path';
import {
  DEPLOY_APPROVAL_ENVIRONMENT,
  DEPLOY_ENVIRONMENT,
  DEPLOY_WORKFLOW_BRANCH,
  DEPLOY_WORKFLOW_PATH,
  DIFF_ENVIRONMENTS,
} from '../lib/constructs/github-oidc';
import { DIFF_ROLE_NAME } from '../lib/constructs/github-diff-role';

/**
 * The deploy workflow and the stack are two halves of one mechanism: the roles
 * trust one workflow file on one branch, running in specific GitHub
 * environments, and the workflow has to supply every environment variable
 * internal mode requires. Neither half fails visibly when the other drifts — the
 * deploy just breaks, or worse, deploys a role with an emptied list, or hands a
 * pull request's code credentials it should not have. These assertions are what
 * notice.
 */
const workflowPath = path.join(__dirname, '..', '..', DEPLOY_WORKFLOW_PATH);
const workflow = fs.readFileSync(workflowPath, 'utf-8');

/**
 * The YAML block of one job, from `  <name>:` to the next key at the same
 * indent. Enough to assert what a single job does and does not have, without
 * adding a YAML parser to this package's dependencies.
 */
function jobBlock(name: string): string {
  const start = workflow.indexOf(`\n  ${name}:\n`);
  expect(start).toBeGreaterThan(-1);
  const rest = workflow.slice(start + 1);
  const next = rest.slice(1).search(/\n {2}[a-z-]+:\n/);
  return next === -1 ? rest : rest.slice(0, next + 1);
}

/** The `for name in … ; do` list of a job's preflight step. */
function preflightList(block: string): string {
  const start = block.indexOf('for name in');
  expect(start).toBeGreaterThan(-1);
  return block.slice(start, block.indexOf('; do', start));
}

const INTERNAL_ENV_LISTS = [
  'STRANDS_TEST_INFRA_PRIVATE_REPOS',
  'STRANDS_TEST_INFRA_BUCKET_NAMES',
  'STRANDS_TEST_INFRA_PERSISTENT_BUCKET_NAMES',
  'STRANDS_TEST_INFRA_SECRET_NAMES',
  'STRANDS_TEST_INFRA_RUNNER_ROLES',
];

test('the workflow the CI roles trust exists at that path', () => {
  expect(fs.existsSync(workflowPath)).toBe(true);
});

test('the workflow deploys internal mode from the trusted branch', () => {
  expect(workflow).toContain("STRANDS_TEST_INFRA_INTERNAL: 'true'");
  expect(workflow).toContain(`refs/heads/${DEPLOY_WORKFLOW_BRANCH}`);
  expect(workflow).toContain(`branches: [${DEPLOY_WORKFLOW_BRANCH}]`);
});

// `job_workflow_ref` carries the branch the workflow file lives on, and
// pull_request_target is the only pull-request trigger that runs the base
// branch's copy. On `pull_request` the file would come from the pull request
// itself — and a fork PR would get no secrets, so the diff would be wrong even
// if it ran.
test('pull requests trigger through pull_request_target, not pull_request', () => {
  expect(workflow).toContain('pull_request_target:');
  expect(workflow).not.toMatch(/^ {2}pull_request:/m);
});

test.each([
  ['deploy', ['STRANDS_TEST_INFRA_DEPLOY_ROLE', 'STRANDS_TEST_INFRA_DEPLOYMENT_ACCOUNT']],
  ['diff', ['STRANDS_TEST_INFRA_DEPLOYMENT_ACCOUNT']],
])('the %s job supplies and preflights every value internal mode needs', (job, extras) => {
  const block = jobBlock(job);
  const preflight = preflightList(block);

  for (const name of extras) {
    expect(preflight).toContain(name);
  }

  // Each of these is read by IntegTestRole in internal mode, and an unset or
  // empty value narrows the deployed role (or misreports the diff) instead of
  // failing — so the job both supplies it and refuses to start without it.
  for (const name of INTERNAL_ENV_LISTS) {
    expect(block).toContain(`${name}: \${{ secrets.${name} }}`);
    expect(preflight).toContain(name);
  }
});

// --- The environment gate ---

// The deploy role's trust policy names exactly these two subjects, so this
// expression is the whole authorization decision: unprotected environment for a
// merge, required-reviewer environment for a pull request.
test('the deploy job runs in the two environments the deploy role trusts', () => {
  const block = jobBlock('deploy');

  expect(block).toContain(`'${DEPLOY_APPROVAL_ENVIRONMENT}'`);
  expect(block).toContain(`'${DEPLOY_ENVIRONMENT}'`);
  expect(block).toMatch(
    new RegExp(
      `environment:\\s*\\n\\s*name: \\$\\{\\{ github.event_name == 'pull_request_target' && '${DEPLOY_APPROVAL_ENVIRONMENT}' \\|\\| '${DEPLOY_ENVIRONMENT}' \\}\\}`,
    ),
  );
});

test('the diff job runs in the environments the diff role trusts', () => {
  const block = jobBlock('diff');

  expect(block).toContain('name: ${{ needs.authorization-check.outputs.approval-env }}');
  // The approval-env output is one of these two, which is what the diff role
  // trusts; the gate that produces it must stay in the workflow.
  expect(workflow).toContain('strands-agents/devtools/authorization-check@main');
  for (const environment of DIFF_ENVIRONMENTS) {
    expect(['auto-approve', 'manual-approval']).toContain(environment);
  }
});

// The isolation between the two roles is the environment name and nothing else.
// If a diff job could run in a deploy environment, unreviewed pull-request code
// would hold an OIDC token that satisfies the deploy role's trust policy.
test('the diff and deploy environments are disjoint', () => {
  const deployEnvironments = [DEPLOY_ENVIRONMENT, DEPLOY_APPROVAL_ENVIRONMENT];
  expect(DIFF_ENVIRONMENTS.filter((name) => deployEnvironments.includes(name))).toEqual([]);
});

// --- The diff job's containment ---

// The default diff method creates a change set with the *deploy* role, which the
// diff role cannot assume; --method=template uses the read-only lookup role.
// Without this flag the diff job fails, and the fix would look like "give the
// diff role more permissions".
test('the diff job diffs through the lookup role', () => {
  expect(jobBlock('diff')).toContain('--method=template');
});

test('the diff job never sees the deploy role', () => {
  const block = jobBlock('diff');

  expect(block).not.toContain('STRANDS_TEST_INFRA_DEPLOY_ROLE');
  expect(block).toContain(
    `role-to-assume: arn:aws:iam::\${{ secrets.STRANDS_TEST_INFRA_DEPLOYMENT_ACCOUNT }}:role/${DIFF_ROLE_NAME}`,
  );
});

// Anything printed by a job that holds these lists ends up in a world-readable
// log, job summary or artifact. Masking covers the log; the redactor covers the
// files, and it is inlined in the workflow rather than read from the (untrusted)
// pull-request checkout.
test('the diff job masks and redacts the secret-derived names', () => {
  const block = jobBlock('diff');

  expect(block).toContain('::add-mask::');
  expect(block).toContain('cat > "$RUNNER_TEMP/redact.py"');
  expect(block).toContain('python3 "$RUNNER_TEMP/redact.py"');
  for (const name of [...INTERNAL_ENV_LISTS, 'STRANDS_TEST_INFRA_DEPLOYMENT_ACCOUNT']) {
    // Every list the job holds has to reach the redactor, or its entries survive
    // into the published diff.
    expect(block.slice(block.indexOf('redact.py'))).toContain(`'${name}'`);
  }
});

// A job that runs pull-request code must not also hold a token that can write to
// the pull request, so the comment is posted from a job with no checkout.
test('the diff is published from a job that runs no pull-request code', () => {
  const block = jobBlock('publish-diff');

  expect(block).toContain('pull-requests: write');
  expect(block).not.toContain('actions/checkout');
  expect(block).not.toContain('npm ci');
  expect(jobBlock('diff')).not.toContain('pull-requests: write');
});
