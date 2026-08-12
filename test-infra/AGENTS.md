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

A change under `test-infra/` that merges to `main` is deployed to the team's test account by `.github/workflows/test-infra-deploy.yml`, so a permission you add to `integ-test-role.ts` reaches the live role without anyone deploying by hand — and a mistake reaches it just as fast. A pull request gets a read-only `cdk diff` in its job summary, and can be deployed before merge by approving the `test-infra-deploy-approval` environment, which also runs that pull request's TypeScript with deploy credentials.

Three things that workflow depends on, all in `lib/constructs/github-ci-roles.ts`:

- **The GitHub environment a job declares *is* its authorization.** The deploy role trusts `…:environment:test-infra-deploy` (post-merge, unprotected) and `…:environment:test-infra-deploy-approval` (required reviewers) and nothing else. Never point the pull-request path at an unprotected environment, and never reuse either name for a job running unreviewed code — that is what the diff role and its own environments are for.
- **`job_workflow_ref` pins that one workflow file on `main`.** Renaming it breaks both roles (a unit test fails if they disagree), and it is why the PR trigger is `pull_request_target`: that runs `main`'s copy, so a pull request cannot rewrite the steps handling it.
- **Both roles are created by the stack they deploy**, so never rename or narrow them without a plan for the manual deploy that repairs them.

And the inverse of the rule above: never deploy to the team account **without** `STRANDS_TEST_INFRA_INTERNAL=true`. Community mode omits both CI roles and the integ role's OIDC trust, so CloudFormation deletes them — taking CI's integration tests and its ability to deploy this stack with them.

## Convention: always set removal policy DESTROY

All resources in this stack must specify `removalPolicy: cdk.RemovalPolicy.DESTROY` (or `applyRemovalPolicy(DESTROY)` for L1 constructs). This is test infrastructure — it must tear down cleanly on `cdk destroy` with no orphaned resources or naming collisions on redeploy. Never use RETAIN here.
