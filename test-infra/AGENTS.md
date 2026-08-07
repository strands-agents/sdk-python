# Test Infrastructure — Agent Guidance

**You almost certainly do not need to deploy or modify this stack.**

This CDK stack provisions AWS resources (Bedrock knowledge bases, EC2 instances) that a small subset of integration tests depend on. The vast majority of tests — including most integration tests — work without it.

## When to deploy this stack

Only if you are:
- Modifying the test infrastructure CDK code itself
- Iterating on the specific tests that resolve SSM parameters from this stack (KB ingestion, SSH sandbox)

## When NOT to deploy this stack

- Running unit tests (`npm test` in any package)
- Running most integration tests (they don't need provisioned infra)
- Reviewing or modifying SDK code, docs, tools, or CLI
- Opening a PR — CI runs infrastructure-dependent tests automatically against pre-provisioned resources

## Never set `STRANDS_TEST_INFRA_INTERNAL=true`

This flag attaches a broad internal IAM policy and GitHub OIDC trust that only makes sense in the Strands team's own test account. Setting it in any other account creates a role with permissions to resources that don't exist.

## If you do need to deploy

See `README.md` in this directory for setup instructions. The default `npx cdk deploy` with your AWS credentials configured is all most cases need.

## Changes to this directory deploy themselves

Once a change under `test-infra/` merges to `main`, `.github/workflows/test-infra-deploy.yml` deploys the stack to the team's test account (internal mode). So a permission you add to `integ-test-role.ts` reaches the live role without anyone deploying by hand — and a mistake reaches it just as fast. A pull request additionally gets a read-only `cdk diff` posted to it, and can be deployed before merge by approving the `test-infra-deploy-approval` environment. Three constraints that workflow depends on:

- **The GitHub environment a job declares *is* its authorization.** The deploy role trusts the subjects `…:environment:test-infra-deploy` (post-merge, unprotected) and `…:environment:test-infra-deploy-approval` (pull request, required reviewers) — and nothing else. Never point the pull-request path at an unprotected environment, and never reuse either name for a job that runs unreviewed code. The read-only diff job has its own role and its own environments (`auto-approve` / `manual-approval`) precisely so that a pull request's code cannot hold a token the deploy role accepts.
- **`job_workflow_ref` pins this one workflow file on `main`.** Renaming or moving it breaks both roles until `DEPLOY_WORKFLOW_PATH` is updated with it (a unit test fails if they disagree). It is also why the PR trigger is `pull_request_target`: that runs `main`'s copy of the workflow, so a pull request cannot rewrite the steps that handle it.
- `StrandsTestInfraDeployRole` and `StrandsTestInfraDiffRole` are created by the stack they deploy, so never rename or narrow them without a plan for the manual deploy that repairs them.

Approving a pull-request deploy authorizes that pull request's TypeScript to run with credentials that can change the account. The diff tells you what it *says* it will change; only the code tells you what it will do.

And the inverse of the rule above: never deploy to the team account **without** `STRANDS_TEST_INFRA_INTERNAL=true`. Community mode omits the deploy role and the integ role's OIDC trust from the template, so CloudFormation deletes both — taking CI's integration tests and its ability to deploy this stack with them.

## Convention: always set removal policy DESTROY

All resources in this stack must specify `removalPolicy: cdk.RemovalPolicy.DESTROY` (or `applyRemovalPolicy(DESTROY)` for L1 constructs). This is test infrastructure — it must tear down cleanly on `cdk destroy` with no orphaned resources or naming collisions on redeploy. Never use RETAIN here.
