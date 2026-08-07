import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as iam from 'aws-cdk-lib/aws-iam';
import {
  DEPLOY_APPROVAL_ENVIRONMENT,
  DEPLOY_ENVIRONMENT,
  GITHUB_REPOSITORY,
  bootstrapRoleArn,
  deployWorkflowPrincipal,
} from './github-oidc';

/**
 * Fixed role name so the ARN is stable: the deploy workflow reads it from a
 * repository secret, which would otherwise have to be re-pointed after every
 * replacement of this role.
 */
export const DEPLOY_ROLE_NAME = 'StrandsTestInfraDeployRole';

/** The bootstrap roles `cdk deploy` assumes. */
const BOOTSTRAP_ROLES = ['deploy', 'file-publishing', 'image-publishing', 'lookup'];

/**
 * Identity the `Test Infra: Deploy` workflow assumes to deploy this stack, so a
 * merged change to `test-infra/` reaches the account without an out-of-band
 * `cdk deploy` — and so a maintainer can deploy a pull request's version of the
 * stack, behind an approval, before merging it.
 *
 * The role's only permission is to assume the CDK bootstrap roles, which is all
 * `cdk deploy` needs. That is a narrow grant on paper but a powerful one in
 * effect — the bootstrap deploy role can pass the CloudFormation execution role,
 * which `cdk bootstrap` gives AdministratorAccess by default. So the trust
 * policy, not the permission policy, is what actually contains this role.
 *
 * It is assumable only from the deploy workflow on `main`
 * ({@link deployWorkflowPrincipal}) and only from a job running in one of two
 * GitHub environments:
 *
 * - {@link DEPLOY_ENVIRONMENT} — the post-merge path. Unprotected, so a merge
 *   deploys unattended.
 * - {@link DEPLOY_APPROVAL_ENVIRONMENT} — the pull-request path. Its required
 *   reviewers are the gate, and approving it means accepting that the pull
 *   request's own TypeScript will run with these credentials.
 *
 * Three consequences worth knowing before changing either side:
 *
 * - Which environment a job declares *is* its authorization. Never point the
 *   PR path at an unprotected environment, and never reuse either name for a
 *   job that runs unreviewed code (that is what
 *   {@link GitHubDiffRole} and its own environments are for).
 * - Renaming or moving the deploy workflow changes `job_workflow_ref` and breaks
 *   deploys until `DEPLOY_WORKFLOW_PATH` is updated with it.
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
      description: `Deploys ${stack.stackName} from the test-infra deploy workflow on ${GITHUB_REPOSITORY}`,
      maxSessionDuration: cdk.Duration.hours(1),
      assumedBy: deployWorkflowPrincipal(stack.account, [
        DEPLOY_ENVIRONMENT,
        DEPLOY_APPROVAL_ENVIRONMENT,
      ]),
    });

    // deploy (CloudFormation), file-publishing and image-publishing (asset
    // upload — no assets today, but adding one should not need an IAM change),
    // lookup (context queries and the bootstrap version parameter). They all
    // trust this account, so permission to assume them is the only grant this
    // role needs.
    this.role.addToPolicy(
      new iam.PolicyStatement({
        actions: ['sts:AssumeRole'],
        resources: BOOTSTRAP_ROLES.map((name) => bootstrapRoleArn(stack, name)),
      }),
    );
  }
}
