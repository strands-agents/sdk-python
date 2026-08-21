import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as iam from 'aws-cdk-lib/aws-iam';

const REPOSITORY = 'strands-agents/harness-sdk';
const BRANCH = 'main';

/** Renaming this file breaks both roles' trust; a unit test guards it. */
export const WORKFLOW_PATH = '.github/workflows/test-infra-deploy.yml';

/** Deploy environments: unprotected post-merge, required reviewers for a PR. */
export const DEPLOY_ENVIRONMENTS = ['test-infra-deploy', 'test-infra-deploy-approval'];

/** Diff environments: what `devtools/authorization-check` emits. */
export const DIFF_ENVIRONMENTS = ['auto-approve', 'manual-approval'];

export const DEPLOY_ROLE_NAME = 'StrandsTestInfraDeployRole';
export const DIFF_ROLE_NAME = 'StrandsTestInfraDiffRole';

/**
 * OIDC principal for the deploy workflow, restricted to jobs declaring one of
 * `environments`.
 *
 * The subject is the environment, not the ref: `pull_request_target` reports the
 * default branch in `GITHUB_REF`, so a pull-request job with no `environment:`
 * mints the same subject a push to main does. `job_workflow_ref` is what pins
 * the branch and the workflow file. So the two roles' environment sets must stay
 * disjoint (asserted in a test) — otherwise a token from the diff job, which
 * runs unreviewed pull-request code, would satisfy the deploy role's trust.
 */
function principal(account: string, environments: string[]): iam.FederatedPrincipal {
  return new iam.FederatedPrincipal(
    `arn:aws:iam::${account}:oidc-provider/token.actions.githubusercontent.com`,
    {
      StringEquals: {
        'token.actions.githubusercontent.com:aud': 'sts.amazonaws.com',
        'token.actions.githubusercontent.com:sub': environments.map(
          (name) => `repo:${REPOSITORY}:environment:${name}`,
        ),
        'token.actions.githubusercontent.com:job_workflow_ref': `${REPOSITORY}/${WORKFLOW_PATH}@refs/heads/${BRANCH}`,
      },
    },
    'sts:AssumeRoleWithWebIdentity',
  );
}

function bootstrapRole(stack: cdk.Stack, name: string): string {
  return `arn:aws:iam::${stack.account}:role/cdk-hnb659fds-${name}-role-${stack.account}-${stack.region}`;
}

function assumeOnly(role: iam.Role, bootstrapRoles: string[]): void {
  const stack = cdk.Stack.of(role);
  role.addToPolicy(
    new iam.PolicyStatement({
      actions: ['sts:AssumeRole'],
      resources: bootstrapRoles.map((name) => bootstrapRole(stack, name)),
    }),
  );
}

/**
 * The two identities the deploy workflow uses. Internal mode only — they trust
 * the GitHub OIDC provider, which exists only in the team's test account. Both
 * are created by the stack they deploy, so repairing a broken trust policy means
 * a human running `cdk deploy` with account credentials.
 *
 * - Deploy: may assume the CDK bootstrap roles, which is all `cdk deploy` needs.
 *   Narrow on paper, powerful in effect (the bootstrap deploy role can pass an
 *   AdministratorAccess execution role), so the trust policy is what contains it.
 * - Diff: may assume the bootstrap `lookup` role only (`ReadOnlyAccess`). This is
 *   the role a pull request's own code holds, before anyone has approved it.
 */
export class GitHubCiRoles extends Construct {
  constructor(scope: Construct, id: string) {
    super(scope, id);

    const stack = cdk.Stack.of(this);

    const deploy = new iam.Role(this, 'DeployRole', {
      roleName: DEPLOY_ROLE_NAME,
      maxSessionDuration: cdk.Duration.hours(1),
      assumedBy: principal(stack.account, DEPLOY_ENVIRONMENTS),
    });
    // deploy (CloudFormation), file/image-publishing (assets — none today, but
    // adding one should not need an IAM change), lookup (context queries).
    assumeOnly(deploy, ['deploy', 'file-publishing', 'image-publishing', 'lookup']);

    const diff = new iam.Role(this, 'DiffRole', {
      roleName: DIFF_ROLE_NAME,
      maxSessionDuration: cdk.Duration.hours(1),
      assumedBy: principal(stack.account, DIFF_ENVIRONMENTS),
    });
    // Only lookup: `cdk diff --method=template` needs nothing else, and `deploy`
    // is what would make this role dangerous in a pull request's hands.
    assumeOnly(diff, ['lookup']);
  }
}
