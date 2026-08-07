import * as iam from 'aws-cdk-lib/aws-iam';

/** Repository whose workflow runs may assume the test-infra CI roles. */
export const GITHUB_REPOSITORY = 'strands-agents/harness-sdk';

/**
 * Branch whose copy of the deploy workflow may assume them.
 *
 * `pull_request_target` runs the *base branch* copy of a workflow, so a PR run
 * reports this branch in `job_workflow_ref` as well. That is what makes a
 * PR-triggered diff — and an approval-gated PR deploy — possible without ever
 * trusting a workflow file a pull request can edit.
 */
export const DEPLOY_WORKFLOW_BRANCH = 'main';

/**
 * Repo-relative path of the only workflow allowed to assume these roles.
 * Renaming the file without updating this breaks CI, which a unit test guards.
 */
export const DEPLOY_WORKFLOW_PATH = '.github/workflows/test-infra-deploy.yml';

/**
 * GitHub environment the post-merge deploy job runs in. Must exist with **no**
 * protection rules: a merge to `main` deploys without waiting for anyone.
 */
export const DEPLOY_ENVIRONMENT = 'test-infra-deploy';

/**
 * GitHub environment a pull-request deploy job runs in. Must exist **with
 * required reviewers** — that approval is the only thing standing between a
 * pull request's TypeScript and credentials that can change the account.
 */
export const DEPLOY_APPROVAL_ENVIRONMENT = 'test-infra-deploy-approval';

/**
 * Environments the read-only diff job runs in: the two values
 * `strands-agents/devtools/authorization-check` emits. A PR author with write
 * access lands in `auto-approve` and the diff runs immediately; anyone else
 * lands in `manual-approval` and it waits for a reviewer.
 *
 * These must stay **disjoint** from {@link DEPLOY_ENVIRONMENT} and
 * {@link DEPLOY_APPROVAL_ENVIRONMENT}. The environment name is the only claim
 * separating the diff job's OIDC token from the deploy job's, and the diff job
 * executes unreviewed pull-request code — an overlap would let that code assume
 * the deploy role. A unit test asserts the disjointness.
 */
export const DIFF_ENVIRONMENTS = ['auto-approve', 'manual-approval'];

/**
 * OIDC principal for the deploy workflow, narrowed to jobs running in one of
 * `environments`.
 *
 * Two conditions carry the whole trust decision:
 *
 * - `job_workflow_ref` pins the entry workflow to
 *   {@link DEPLOY_WORKFLOW_PATH} on {@link DEPLOY_WORKFLOW_BRANCH}. No other
 *   workflow can assume these roles, and — because a workflow file only reaches
 *   `main` through review — the steps that run are always reviewed steps. This
 *   is also what pins the ref: a `workflow_dispatch` from another branch reports
 *   that branch here and is refused.
 * - `sub` pins the GitHub environment. GitHub mints
 *   `repo:<owner>/<repo>:environment:<name>` only for a job that declares that
 *   environment, so a role trusted for a protected environment is unreachable
 *   until its required reviewers approve.
 *
 * The subject deliberately does **not** pin `ref:refs/heads/main` any more.
 * `pull_request_target` sets `GITHUB_REF` to the default branch, so a PR job
 * with no environment gets the *same* subject a push-to-main job does — the ref
 * subject cannot tell reviewed code from a pull request's.
 */
export function deployWorkflowPrincipal(
  account: string,
  environments: string[],
): iam.FederatedPrincipal {
  return new iam.FederatedPrincipal(
    `arn:aws:iam::${account}:oidc-provider/token.actions.githubusercontent.com`,
    {
      StringEquals: {
        'token.actions.githubusercontent.com:aud': 'sts.amazonaws.com',
        // A list is an OR over exact matches: no wildcards, so an environment
        // that is not named here cannot assume the role.
        'token.actions.githubusercontent.com:sub': environments.map(
          (name) => `repo:${GITHUB_REPOSITORY}:environment:${name}`,
        ),
        'token.actions.githubusercontent.com:job_workflow_ref': `${GITHUB_REPOSITORY}/${DEPLOY_WORKFLOW_PATH}@refs/heads/${DEPLOY_WORKFLOW_BRANCH}`,
      },
    },
    'sts:AssumeRoleWithWebIdentity',
  );
}

/** ARN of a CDK bootstrap role in this account/region. */
export function bootstrapRoleArn(stack: { account: string; region: string }, name: string): string {
  // Not stack.formatArn: callers pass the resolved account/region and the
  // qualifier is the bootstrap default.
  return `arn:aws:iam::${stack.account}:role/cdk-hnb659fds-${name}-role-${stack.account}-${stack.region}`;
}
