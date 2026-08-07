import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as iam from 'aws-cdk-lib/aws-iam';
import {
  DIFF_ENVIRONMENTS,
  GITHUB_REPOSITORY,
  bootstrapRoleArn,
  deployWorkflowPrincipal,
} from './github-oidc';

/** Fixed role name: the diff job builds this ARN from the account secret. */
export const DIFF_ROLE_NAME = 'StrandsTestInfraDiffRole';

/**
 * Read-only identity the `Test Infra: Deploy` workflow assumes to show a pull
 * request what it would change in the account — `cdk diff --method=template`,
 * which compares templates using the CDK bootstrap *lookup* role rather than
 * creating a change set with the deploy role.
 *
 * Why a second role instead of reusing {@link GitHubDeployRole}: the diff runs
 * `test-infra/`'s TypeScript *from the pull request*, before anyone has approved
 * it. Whatever credentials that job holds, the pull request's code holds. So this
 * role can only reach the lookup role — `ReadOnlyAccess` in the bootstrap stack,
 * scoped to the team's test account — and cannot reach the bootstrap deploy role
 * or the CloudFormation execution role behind it.
 *
 * It is assumable only from the deploy workflow's diff job, whose GitHub
 * environments ({@link DIFF_ENVIRONMENTS}) are disjoint from the deploy job's.
 * That disjointness is the isolation: an OIDC token minted in the diff job does
 * not satisfy the deploy role's trust policy.
 *
 * What this does *not* contain, and should not be mistaken for containing: a
 * pull request whose diff job runs can read anything `ReadOnlyAccess` covers in
 * the test account, and the job is handed the `STRANDS_TEST_INFRA_*` lists it
 * needs to synthesize a comparable template. For a PR from someone without write
 * access that job waits for an environment approval first — approving it is
 * saying the code is safe to run, not just safe to read.
 */
export class GitHubDiffRole extends Construct {
  public readonly role: iam.Role;

  constructor(scope: Construct, id: string) {
    super(scope, id);

    const stack = cdk.Stack.of(this);

    this.role = new iam.Role(this, 'Role', {
      roleName: DIFF_ROLE_NAME,
      description: `Diffs ${stack.stackName} for pull requests on ${GITHUB_REPOSITORY} (read-only)`,
      maxSessionDuration: cdk.Duration.hours(1),
      assumedBy: deployWorkflowPrincipal(stack.account, DIFF_ENVIRONMENTS),
    });

    this.role.addToPolicy(
      new iam.PolicyStatement({
        actions: ['sts:AssumeRole'],
        // Only `lookup`. `deploy` is what makes the deploy role powerful, and a
        // change set — the diff method that needs it — is not available here by
        // design: the diff job passes `--method=template`.
        resources: [bootstrapRoleArn(stack, 'lookup')],
      }),
    );
  }
}
