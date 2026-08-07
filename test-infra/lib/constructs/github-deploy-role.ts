import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as iam from 'aws-cdk-lib/aws-iam';

/**
 * Fixed role name so the ARN is stable: the deploy workflow reads it from a
 * repository secret, which would otherwise have to be re-pointed after every
 * replacement of this role.
 */
export const DEPLOY_ROLE_NAME = 'StrandsTestInfraDeployRole';

/** Repository whose workflow runs may assume the deploy role. */
export const DEPLOY_ROLE_REPOSITORY = 'strands-agents/harness-sdk';

/** Branch whose workflow runs may assume the deploy role. */
export const DEPLOY_ROLE_BRANCH = 'main';

/**
 * Repo-relative path of the only workflow allowed to assume the deploy role.
 * Renaming the file without updating this breaks deploys, which a unit test
 * guards against.
 */
export const DEPLOY_ROLE_WORKFLOW_PATH = '.github/workflows/test-infra-deploy.yml';

/**
 * Identity the `Test Infra: Deploy` workflow assumes to deploy this stack, so a
 * merged change to `test-infra/` reaches the account without an out-of-band
 * `cdk deploy`.
 *
 * The role's only permission is to assume the CDK bootstrap roles, which is all
 * `cdk deploy` needs. That is a narrow grant on paper but a powerful one in
 * effect — the bootstrap deploy role can pass the CloudFormation execution role,
 * which `cdk bootstrap` gives AdministratorAccess by default. So the trust
 * policy, not the permission policy, is what actually contains this role: only
 * the deploy workflow, running on `main` of this repository, can assume it.
 *
 * Three consequences worth knowing before changing either side:
 *
 * - A job that declares `environment:` gets the subject
 *   `repo:<owner>/<repo>:environment:<name>` instead and can no longer assume
 *   this role.
 * - Renaming or moving the deploy workflow changes `job_workflow_ref` and breaks
 *   deploys until {@link DEPLOY_ROLE_WORKFLOW_PATH} is updated with it.
 * - The role is created by the stack it deploys, so the first deployment (and
 *   any repair after a broken trust policy) has to be run by a human with
 *   account credentials.
 */
export class GitHubDeployRole extends Construct {
  public readonly role: iam.Role;

  constructor(scope: Construct, id: string) {
    super(scope, id);

    const stack = cdk.Stack.of(this);

    this.role = new iam.Role(this, 'Role', {
      roleName: DEPLOY_ROLE_NAME,
      description: `Deploys ${stack.stackName} from GitHub Actions on ${DEPLOY_ROLE_REPOSITORY}@${DEPLOY_ROLE_BRANCH}`,
      maxSessionDuration: cdk.Duration.hours(1),
      assumedBy: new iam.FederatedPrincipal(
        `arn:aws:iam::${stack.account}:oidc-provider/token.actions.githubusercontent.com`,
        {
          StringEquals: {
            'token.actions.githubusercontent.com:aud': 'sts.amazonaws.com',
            // StringEquals, not StringLike: exactly one branch of one repo, so a
            // workflow on any other ref (or a fork) cannot assume the role.
            'token.actions.githubusercontent.com:sub': `repo:${DEPLOY_ROLE_REPOSITORY}:ref:refs/heads/${DEPLOY_ROLE_BRANCH}`,
            // The subject alone would trust any workflow on that branch that
            // requests an id-token; this narrows it to the deploy workflow.
            'token.actions.githubusercontent.com:job_workflow_ref': `${DEPLOY_ROLE_REPOSITORY}/${DEPLOY_ROLE_WORKFLOW_PATH}@refs/heads/${DEPLOY_ROLE_BRANCH}`,
          },
        },
        'sts:AssumeRoleWithWebIdentity',
      ),
    });

    // The bootstrap roles `cdk deploy` assumes: deploy (CloudFormation),
    // file-publishing and image-publishing (asset upload — no assets today, but
    // adding one should not need an IAM change), lookup (context queries and the
    // bootstrap version parameter). They all trust this account, so permission
    // to assume them is the only grant this role needs.
    const qualifier = cdk.DefaultStackSynthesizer.DEFAULT_QUALIFIER;
    this.role.addToPolicy(
      new iam.PolicyStatement({
        actions: ['sts:AssumeRole'],
        resources: ['deploy', 'file-publishing', 'image-publishing', 'lookup'].map((name) =>
          stack.formatArn({
            service: 'iam',
            region: '',
            resource: 'role',
            resourceName: `cdk-${qualifier}-${name}-role-${stack.account}-${stack.region}`,
          }),
        ),
      }),
    );
  }
}
