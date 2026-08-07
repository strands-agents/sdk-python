import * as fs from 'fs';
import * as path from 'path';
import {
  DEPLOY_ROLE_BRANCH,
  DEPLOY_ROLE_WORKFLOW_PATH,
} from '../lib/constructs/github-deploy-role';

/**
 * The deploy workflow and the stack are two halves of one mechanism: the role
 * trusts one workflow file on one branch, and the workflow has to supply every
 * environment variable internal mode requires. Neither half fails visibly when
 * the other drifts — the deploy just breaks, or worse, deploys a role with an
 * emptied list. These assertions are what notice.
 */
const workflowPath = path.join(__dirname, '..', '..', DEPLOY_ROLE_WORKFLOW_PATH);

test('the workflow the deploy role trusts exists at that path', () => {
  expect(fs.existsSync(workflowPath)).toBe(true);
});

test('the workflow deploys internal mode from the trusted branch only', () => {
  const workflow = fs.readFileSync(workflowPath, 'utf-8');

  expect(workflow).toContain("STRANDS_TEST_INFRA_INTERNAL: 'true'");
  expect(workflow).toContain(`refs/heads/${DEPLOY_ROLE_BRANCH}`);
  expect(workflow).toContain(`branches: [${DEPLOY_ROLE_BRANCH}]`);
});

test('the workflow passes every environment variable internal mode requires', () => {
  const workflow = fs.readFileSync(workflowPath, 'utf-8');
  // The `for name in … ; do` list of the preflight step.
  const preflight = workflow.slice(workflow.indexOf('for name in'), workflow.indexOf('; do'));
  expect(preflight).toContain('STRANDS_TEST_INFRA_DEPLOY_ROLE');

  // Each of these is read by IntegTestRole in internal mode, and an unset or
  // empty value narrows the deployed role instead of failing the deploy — so the
  // workflow both supplies it and refuses to deploy when it is missing.
  for (const name of [
    'STRANDS_TEST_INFRA_PRIVATE_REPOS',
    'STRANDS_TEST_INFRA_BUCKET_NAMES',
    'STRANDS_TEST_INFRA_PERSISTENT_BUCKET_NAMES',
    'STRANDS_TEST_INFRA_SECRET_NAMES',
    'STRANDS_TEST_INFRA_RUNNER_ROLES',
  ]) {
    expect(workflow).toContain(`${name}: \${{ secrets.${name} }}`);
    expect(preflight).toContain(name);
  }
});
