import * as fs from 'fs';
import * as path from 'path';
import {
  DEPLOY_ENVIRONMENTS,
  DIFF_ENVIRONMENTS,
  DIFF_ROLE_NAME,
  WORKFLOW_PATH,
} from '../lib/constructs/github-ci-roles';

// The workflow and the roles are two halves of one mechanism: the roles trust one
// workflow file on main running in named environments, and the workflow has to
// supply every value internal mode needs. Neither half fails visibly when the
// other drifts — the deploy just breaks, or deploys a role with an emptied list,
// or hands a pull request credentials it should not have.
const workflow = fs.readFileSync(path.join(__dirname, '..', '..', WORKFLOW_PATH), 'utf-8');

/** One job's YAML block, from `  <name>:` to the next key at that indent. */
function job(name: string): string {
  const start = workflow.indexOf(`\n  ${name}:\n`);
  expect(start).toBeGreaterThan(-1);
  const rest = workflow.slice(start + 1);
  const next = rest.slice(1).search(/\n {2}[a-z-]+:\n/);
  return next === -1 ? rest : rest.slice(0, next + 1);
}

/** The inlined redactor of one job, between the heredoc markers. */
function redactor(name: string): string {
  const block = job(name);
  const open = block.indexOf(`cat > "$RUNNER_TEMP/redact.py" <<'PY'`);
  expect(open).toBeGreaterThan(-1);
  const from = block.indexOf('\n', open) + 1;
  return block.slice(from, block.indexOf('\n          PY', from));
}

const SECRET_LISTS = [
  'STRANDS_TEST_INFRA_PRIVATE_REPOS',
  'STRANDS_TEST_INFRA_BUCKET_NAMES',
  'STRANDS_TEST_INFRA_PERSISTENT_BUCKET_NAMES',
  'STRANDS_TEST_INFRA_SECRET_NAMES',
  'STRANDS_TEST_INFRA_RUNNER_ROLES',
  'STRANDS_TEST_INFRA_DEPLOYMENT_ACCOUNT',
];

/** The words of the workflow-level `REDACT` folded scalar. */
function redactList(): string[] {
  const start = workflow.indexOf('REDACT: >-');
  expect(start).toBeGreaterThan(-1);
  const body = workflow.slice(workflow.indexOf('\n', start) + 1);
  // The folded scalar runs until a line indented less than its content.
  const end = body.search(/\n(?! {4}\S)/);
  return body.slice(0, end).trim().split(/\s+/);
}

// One list drives both the preflight and the redactor, which is the point — but it
// also means dropping a name here silently stops masking and redacting that
// secret's values on a public repo, with nothing else failing. A reviewer deleted
// one and all 41 tests still passed; this closes that hole.
test('REDACT names every secret-derived value, and nothing else', () => {
  expect(redactList()).toEqual(SECRET_LISTS);
});

test('the workflow the roles trust exists, and deploys internal mode from main', () => {
  expect(workflow).toContain("STRANDS_TEST_INFRA_INTERNAL: 'true'");
  expect(workflow).toContain('branches: [main]');
});

// job_workflow_ref carries the branch the file lives on, and pull_request_target
// is the only PR trigger that runs the base branch's copy. On `pull_request` the
// file would come from the PR, and a fork would get no secrets anyway.
test('pull requests trigger through pull_request_target', () => {
  expect(workflow).toContain('pull_request_target:');
  expect(workflow).not.toMatch(/^ {2}pull_request:/m);
});

// Every one of these is read by IntegTestRole in internal mode, and an empty value
// narrows the deployed role (or misreports the diff) instead of failing.
test.each(['diff', 'deploy'])('the %s job supplies and preflights every value', (name) => {
  const block = job(name);
  for (const secret of SECRET_LISTS) {
    expect(workflow).toContain(`REDACT: >-`);
    expect(workflow).toContain(secret);
    expect(block).toContain(`${secret}: \${{ secrets.${secret} }}`);
  }
  expect(block).toContain('REQUIRED: ${{ env.REDACT }}');
  expect(block).toMatch(/for name in \$REQUIRED; do \[ -n "\$\{!name:-\}" \] \|\| missing\+=/);
});

// The deploy job's fail-closed guard: a workflow_dispatch from another branch is
// refused loudly here rather than dying on an opaque STS error.
test('the deploy job refuses to run off main unless it is a pull request', () => {
  expect(job('deploy')).toContain(
    '[ "$GITHUB_EVENT_NAME" != \'pull_request_target\' ] && [ "$GITHUB_REF" != \'refs/heads/main\' ]',
  );
});

test('only the deploy job requires the deploy-role secret', () => {
  expect(job('deploy')).toContain('REQUIRED: ${{ env.REDACT }} STRANDS_TEST_INFRA_DEPLOY_ROLE');
  expect(job('diff')).not.toContain('STRANDS_TEST_INFRA_DEPLOY_ROLE');
});

// The deploy role's trust names exactly these two subjects, so this expression is
// the whole authorization decision: unprotected for a merge, required reviewers
// for a pull request.
test('the deploy job runs in the environments the deploy role trusts', () => {
  const [unprotected, approval] = DEPLOY_ENVIRONMENTS;
  expect(job('deploy')).toContain(
    `name: \${{ github.event_name == 'pull_request_target' && '${approval}' || '${unprotected}' }}`,
  );
});

test('the diff job runs in the authorization-check environments', () => {
  expect(job('diff')).toContain('name: ${{ needs.authorization-check.outputs.approval-env }}');
  expect(workflow).toContain('strands-agents/devtools/authorization-check@main');
  expect(DIFF_ENVIRONMENTS).toEqual(['auto-approve', 'manual-approval']);
});

// The default diff method creates a change set with the deploy role, which the
// diff role cannot assume. Without this flag the diff job fails, and the tempting
// fix would be to widen the role.
test('the diff job diffs through the lookup role and holds only the diff role', () => {
  const block = job('diff');
  expect(block).toContain('--method=template');
  expect(block).toContain(
    `role-to-assume: arn:aws:iam::\${{ secrets.STRANDS_TEST_INFRA_DEPLOYMENT_ACCOUNT }}:role/${DIFF_ROLE_NAME}`,
  );
});

// Anything either job prints lands in a world-readable log or job summary, and
// ::add-mask:: covers neither a summary (a file) nor the individual entries of a
// comma-separated secret.
test.each(['diff', 'deploy'])('the %s job masks and redacts before printing', (name) => {
  const block = job(name);
  expect(block).toContain('::add-mask::');
  expect(block).toContain('python3 "$RUNNER_TEMP/redact.py"');
  expect(block).not.toMatch(/cat "\$RUNNER_TEMP\/raw\.txt"/);
  // Newline-terminated, or `read` drops each list's last entry and a
  // single-value secret is never masked at all.
  expect(block).toContain(`printf '%s\\n' "\${!name:-}"`);
});

// Inlined per job because a composite action or tracked script would be the pull
// request's own code on pull_request_target. Drift between the copies is the risk
// that buys: a reviewer broke one copy's sort order and nothing failed.
test('the two redactor copies are identical, and neutralize commands and fences', () => {
  expect(redactor('deploy')).toBe(redactor('diff'));
  expect(redactor('diff')).toContain('key=len, reverse=True');
  expect(redactor('diff')).toContain("re.sub('`{3,}'");
  expect(redactor('diff')).toContain("re.sub('^([ \\t]*)::'");
});

// cdk deploy streams for minutes, so the redactor has to work as a filter. 2>&1
// included: CDK writes progress and errors to stderr.
test('the deploy job pipes cdk deploy, stderr included, through the redactor', () => {
  expect(job('deploy')).toMatch(
    /npx cdk deploy [^\n]*2>&1 \|\n\s*python3 "\$RUNNER_TEMP\/redact\.py"/,
  );
});

// Every deploy touches the one stack and CloudFormation rejects a concurrent
// UPDATE, but the workflow-level group is per pull request.
test('all deploys serialize on one concurrency group', () => {
  expect(job('deploy')).toMatch(/concurrency:\s*\n\s*group: test-infra-deploy-stack/);
});
