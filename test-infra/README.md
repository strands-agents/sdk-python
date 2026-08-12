# Strands Test Infrastructure

## Who this is for (and who it's not for)

This CDK stack provisions shared AWS resources (Bedrock knowledge bases, EC2 instances, etc.) that a **small subset** of Strands integration tests run against — specifically, tests that exercise infrastructure-dependent features like RAG knowledge bases and SSH sandboxes. The vast majority of integration tests do not require these resources.

**Most contributors do not need to deploy this.** If you just want to run integration tests:

- **Open a PR** — the Strands CI GitHub Action runs all integration tests (including the infrastructure-dependent ones) automatically against the team's pre-provisioned resources. No AWS account required on your end.
- **Run locally without this stack** — most integration tests work with just AWS credentials and model access. Only the tests that explicitly resolve SSM parameters from this stack need it deployed.

Deploy this stack only if you:
- Are developing or modifying the test infrastructure itself
- Are working on the specific features (KB ingestion, SSH sandbox) that depend on these resources and want to iterate locally
- Are setting up a new AWS account to run the full test suite independently

## Features

The stack provisions independently-toggleable features:

| Feature | What it deploys | SSM parameters |
|---|---|---|
| `bedrock-knowledge-base` | Bedrock KB + S3 Vectors index + S3 and CUSTOM data sources + source bucket | `/strands/test-infra/bedrock-knowledge-base/{knowledge-base-id, s3-data-source-id, custom-data-source-id, s3-source-bucket-name}` |
| `ssh-ec2` | t4g.nano EC2 instance (private, SSM-only access) + VPC + interface endpoints + ED25519 key pair | `/strands/test-infra/ssh-ec2/{instance-id, private-key-parameter-name}` |

By default all features deploy. Use `-c testFeatures=bedrock-knowledge-base` to deploy a subset.

## Quick start

### Prerequisites

- Node.js 20+
- AWS CLI configured with credentials for the target account
- CDK bootstrapped in the target account/region (`npx cdk bootstrap`)

### Install

```sh
cd test-infra
npm install
```

### Deploy

```sh
npx cdk deploy
```

This deploys all features with a test role your account can assume. Account and region are inferred from your AWS CLI profile.

### Deploy a single feature

```sh
npx cdk deploy -c testFeatures=bedrock-knowledge-base
```

### Deploy with explicit account/region

```sh
export STRANDS_TEST_INFRA_DEPLOYMENT_ACCOUNT=123456789012
export STRANDS_TEST_INFRA_DEPLOYMENT_REGION=us-east-1
npx cdk deploy
```

### Run tests against the deployed stack

Integration tests import `ssmParameterPath` from `lib/constants.ts` and resolve resource IDs at runtime:

```ts
import { ssmParameterPath } from 'test-infra/lib/constants';

const kbId = await ssm.getParameter({ Name: ssmParameterPath('bedrock-knowledge-base', 'knowledge-base-id') });
```

No hardcoded IDs — tests work against any deployed instance of this stack.

### Type-check and unit tests

```sh
npm run build   # type-check only (noEmit)
npm test        # CDK assertions covering all features, toggling, IAM
```

### Destroy

```sh
npx cdk destroy
```

All resources use `DESTROY` removal policies — `cdk destroy` cleans up everything.

## Automated deployment (team account)

`.github/workflows/test-infra-deploy.yml` keeps the deployed stack in sync with this directory. Without it a permission added to `integ-test-role.ts` has no effect until someone remembers to run `cdk deploy`, and the integ test that needs it fails with an `AccessDenied` that looks like a missing grant rather than a missing deployment. It always deploys with `STRANDS_TEST_INFRA_INTERNAL=true`, which is why it must stay pointed at the team account.

| Trigger | What runs | Identity | Waits for a human? |
|---|---|---|---|
| Push to `main`, or **Run workflow** from `main` | `cdk diff` → `cdk deploy` | `StrandsTestInfraDeployRole` in `test-infra-deploy` | No |
| Pull request | Type-check, unit tests, read-only `cdk diff` in the job summary | `StrandsTestInfraDiffRole` in `auto-approve` / `manual-approval` | Only if the author lacks write access |
| Pull request, `deploy` job | `cdk diff` → `cdk deploy` of **the PR's** code | `StrandsTestInfraDeployRole` in `test-infra-deploy-approval` | **Always** |

**Approving a pull-request deploy runs that pull request's TypeScript with credentials that can change the account** — `cdk deploy` executes `bin/test-infra.ts` and everything it imports, from the PR, fork included. The approval is the only gate; read the code, not just the diff. The account then holds an unmerged stack until `main` deploys again.

### Reading the diff

`test-infra/` computes IAM from repository secrets, so the TypeScript diff does not tell you what the live role ends up with. The `diff` job synthesizes the PR's code with the live values and writes the result to its job summary, on the same run page as the approval button. It uses `--method=template` (compare templates via the bootstrap lookup role) rather than the default change set, which needs the deploy role — so it will not show resource replacements a change set would catch.

Secret-derived names are redacted to `***` everywhere the workflow prints, including the deploy job: this repository is public, and `::add-mask::` covers neither a job summary (a file) nor the individual entries of a comma-separated secret. Read the live template in CloudFormation if you need a resolved name.

Note that `pull_request_target` runs `main`'s copy of the workflow, so a PR that edits the workflow is handled by the *old* one; the change takes effect on merge.

### Credentials

Both roles are defined in `lib/constructs/github-ci-roles.ts`, internal mode only, and are created by the stack they deploy — so repairing a broken trust policy means a human running `cdk deploy`.

- **`StrandsTestInfraDeployRole`** may assume the CDK bootstrap roles, which is all `cdk deploy` needs. Narrow on paper, powerful in effect: the bootstrap deploy role can pass CloudFormation's execution role, which `cdk bootstrap` grants AdministratorAccess.
- **`StrandsTestInfraDiffRole`** may assume the bootstrap **lookup** role only (`ReadOnlyAccess`). This is the role a pull request's own code holds.

Both pin `job_workflow_ref` (this workflow file, on `main` — so no other workflow can assume them, and a `workflow_dispatch` from another branch is refused) and a `sub` list of `repo:strands-agents/harness-sdk:environment:<name>`. GitHub mints that subject only for a job declaring that environment, so required reviewers are enforced by IAM as well as by GitHub. The subject deliberately does **not** pin `ref:refs/heads/main`: `pull_request_target` reports the default branch in `GITHUB_REF`, so a ref subject cannot distinguish a reviewed push from a pull request. Which means the two roles' environment sets must stay disjoint, or unreviewed code would hold a token the deploy role accepts. A unit test asserts it.

### One-time setup

1. **Environments:** `test-infra-deploy` with **no** protection rules (a merge must deploy unattended) and `test-infra-deploy-approval` with **required reviewers**. Neither may be reused by another workflow — their names are what the deploy role trusts. `auto-approve` / `manual-approval` already exist for the rest of the repo.
2. **Secrets:** `STRANDS_TEST_INFRA_DEPLOY_ROLE` (the deploy role ARN), `STRANDS_TEST_INFRA_DEPLOYMENT_ACCOUNT` (the team account id, which pins the target — the CDK CLI refuses a mismatch), and the lists `STRANDS_TEST_INFRA_PRIVATE_REPOS`, `_BUCKET_NAMES`, `_PERSISTENT_BUCKET_NAMES`, `_SECRET_NAMES`, `_RUNNER_ROLES`. Every list is load-bearing: an empty value does not fail the deploy, it deploys a role with those entries **removed**, so the workflow refuses to start if any is unset. The diff role needs no secret — its ARN is derived from the account id.
3. **The first deploy is manual**, since the stack defines the roles that deploy it:

```sh
cd test-infra
npx cdk bootstrap   # once per account/region

STRANDS_TEST_INFRA_INTERNAL=true \
STRANDS_TEST_INFRA_PRIVATE_REPOS=... \
STRANDS_TEST_INFRA_BUCKET_NAMES=... \
STRANDS_TEST_INFRA_PERSISTENT_BUCKET_NAMES=... \
STRANDS_TEST_INFRA_SECRET_NAMES=... \
STRANDS_TEST_INFRA_RUNNER_ROLES=... \
  npx cdk deploy    # with account credentials
```

Then set `STRANDS_TEST_INFRA_DEPLOY_ROLE` to the created ARN.

## Configuration reference

| Input | Channel | Purpose | Default |
|---|---|---|---|
| `testFeatures` | CDK context (`-c testFeatures=a,b`) | Which features to provision | `all` |
| `STRANDS_TEST_INFRA_INTERNAL` | Env var (`=true`) | Attach internal legacy policy + GitHub OIDC trust | `false` |
| `STRANDS_TEST_INFRA_DEPLOYMENT_ACCOUNT` | Env var | Target AWS account | Inferred from CLI profile |
| `STRANDS_TEST_INFRA_DEPLOYMENT_REGION` | Env var | Target AWS region | Inferred from CLI profile |

## Internal mode

> **Do not set `STRANDS_TEST_INFRA_INTERNAL=true` unless you are deploying to the Strands team's own test account.** This mode attaches a broad legacy policy (model invocation, KB management, secrets access, AOSS, CloudWatch) and configures the role trust for Strands GitHub Actions OIDC. It is not meaningful outside the internal account and will create a role with permissions to resources that don't exist in your account.

```sh
STRANDS_TEST_INFRA_INTERNAL=true npx cdk deploy
```

## Architecture

```
bin/test-infra.ts                          Entry point, env resolution, feature validation
lib/
  constants.ts                             TestFeature type, VALID_TEST_FEATURES, ssmParameterPath()
  stacks/test-infra-stack.ts               Thin orchestrator: feature gating + role
  constructs/
    test-feature-construct.ts              Base class: ssmPath() + grantSsmParameterRead()
    integ-test-role.ts                     Shared test role (OIDC trust in internal mode)
    github-deploy-role.ts                  Role CI deploys this stack with (internal mode only)
    bedrock-knowledge-base-test-resources  KB + S3 Vectors + data sources + SSM + grants
    ssh-ec2-test-resources.ts              EC2 + VPC + endpoints + key pair + SSM + grants
test/
  test-infra.test.ts                       CDK assertions: features, toggling, IAM
  deploy-workflow.test.ts                  Pins the deploy workflow to what the stack expects
```

## Adding a new feature

1. Add the feature name to `TestFeature` type and `VALID_TEST_FEATURES` in `lib/constants.ts`
2. Create a construct in `lib/constructs/` extending `TestFeatureConstruct`
3. Set `readonly featureName = 'your-feature' as const`
4. Publish SSM params via `this.ssmPath('param-name')`
5. Implement `grantUsage(role)` and call `this.grantSsmParameterRead(role)` at the end
6. Wire it into `lib/stacks/test-infra-stack.ts` with `if (enabled('your-feature'))`
7. Add tests
