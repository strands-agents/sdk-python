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

`.github/workflows/test-infra-deploy.yml` keeps the deployed stack in sync with this directory. Without it the stack drifts silently: a permission added to `integ-test-role.ts` has no effect until someone remembers to run `cdk deploy`, and the integ test that needs it fails with an `AccessDenied` that looks like a missing grant rather than a missing deployment.

It always deploys with `STRANDS_TEST_INFRA_INTERNAL=true` — that is the whole point of it, and why it must stay pointed at the team account.

| Trigger | What runs | Identity | Waits for a human? |
|---|---|---|---|
| Push to `main` under `test-infra/**`, or **Run workflow** from `main` | `cdk diff` then `cdk deploy` | `StrandsTestInfraDeployRole` in the `test-infra-deploy` environment | No |
| Pull request | Type-check, unit tests, and a read-only `cdk diff` posted to the PR | `StrandsTestInfraDiffRole` in `auto-approve` / `manual-approval` | Only if the author lacks write access |
| Pull request, `deploy` job | `cdk diff` then `cdk deploy` of **the PR's** code | `StrandsTestInfraDeployRole` in the `test-infra-deploy-approval` environment | **Always** |

### Reviewing a pull request's diff

`test-infra/` computes IAM from repository secrets, so reading the TypeScript diff does not tell you what the live role ends up with. The `diff` job synthesizes the PR's code with the same `STRANDS_TEST_INFRA_*` values the real deploy uses and posts the result as a comment (updated in place on each push), a job summary, and the `test-infra-cdk-diff` artifact.

Because this repository is public and all three are world-readable, secret-derived names — the account id, private repos, bucket names, secret names, runner roles — are replaced with `***` before anything is written. The structure of the change stays visible; the literals do not. The unredacted diff is in the deploy job's own `cdk diff`, which only runs once someone approves.

Two caveats on the read-only diff. It uses `--method=template` (compare templates through the bootstrap lookup role) rather than the default change set, which needs the deploy role — so it will not show resource replacements a change set would catch. And `pull_request_target` runs `main`'s copy of the workflow, so a PR that edits the workflow itself is diffed by the *old* workflow; that change only takes effect once merged.

### Deploying a pull request before merge

Approve the run's `test-infra-deploy-approval` deployment. Two things to know before you do:

- **It runs the pull request's code with credentials that can change the account.** `cdk deploy` executes `bin/test-infra.ts` and everything it imports, from the PR — including a fork's. The environment approval is the only gate. Read the code, not just the diff.
- **The account then holds an unmerged stack** until `main` deploys again. The job says so in its summary. Merging the PR (or re-running the workflow on `main`) restores it.

### Credentials

Two roles, both internal mode only, both defined in `lib/constructs/` and trusted through GitHub OIDC:

- **`StrandsTestInfraDeployRole`** (`github-deploy-role.ts`) — its only permission is to assume the CDK bootstrap roles, so `cdk deploy` works and nothing else does. That is narrow on paper but powerful in effect: the bootstrap deploy role can pass CloudFormation's execution role, which `cdk bootstrap` grants AdministratorAccess. So the trust policy is what really contains it.
- **`StrandsTestInfraDiffRole`** (`github-diff-role.ts`) — can assume the bootstrap **lookup** role only (`ReadOnlyAccess`). This is the role a pull request's own code holds, so it deliberately cannot reach the deploy role or the execution role behind it.

Both pin the same two claims (`github-oidc.ts`):

- **`job_workflow_ref`** — this workflow file, on `main`. No other workflow can assume either role, and a `workflow_dispatch` from another branch is refused (the workflow also fails its first step with a clear message rather than an opaque STS error).
- **`sub`** — the GitHub environment, e.g. `repo:strands-agents/harness-sdk:environment:test-infra-deploy`. GitHub mints that subject only for a job declaring that environment, so a protected environment's required reviewers are enforced by IAM as well as by GitHub.

The subject deliberately does **not** pin `ref:refs/heads/main`. `pull_request_target` reports the default branch in `GITHUB_REF`, so a ref subject cannot distinguish a reviewed push from a pull request — the environment can. Which means the diff job's environments must stay disjoint from the deploy job's, or unreviewed code would hold a token the deploy role accepts. A unit test asserts that.

### Repository secrets

| Secret | Purpose |
|---|---|
| `STRANDS_TEST_INFRA_DEPLOY_ROLE` | ARN of `StrandsTestInfraDeployRole` — `arn:aws:iam::<account>:role/StrandsTestInfraDeployRole` |
| `STRANDS_TEST_INFRA_DEPLOYMENT_ACCOUNT` | The team test account id. Pins the deployment: the CDK CLI refuses to deploy if the assumed role belongs to a different account, so a mis-set role ARN cannot apply internal mode elsewhere. The diff role's ARN is built from this plus its fixed name, so it needs no secret of its own |
| `STRANDS_TEST_INFRA_PRIVATE_REPOS` | Comma-separated private repos trusted to assume the integ test role |
| `STRANDS_TEST_INFRA_BUCKET_NAMES` | Comma-separated bucket name patterns the integ test role may manage |
| `STRANDS_TEST_INFRA_PERSISTENT_BUCKET_NAMES` | Comma-separated persistent bucket name patterns (no `DeleteBucket`) |
| `STRANDS_TEST_INFRA_SECRET_NAMES` | Comma-separated Secrets Manager secret names the integ test role may read |
| `STRANDS_TEST_INFRA_RUNNER_ROLES` | Comma-separated role names also trusted to assume the integ test role |

The deploy job fails on its first step if **any** of these is missing, and the diff job if any but the deploy-role ARN is (it never receives that one). Every list is load-bearing: an empty value does not fail the deploy, it deploys a role with those entries removed. That includes `STRANDS_TEST_INFRA_RUNNER_ROLES`, which the stack treats as optional — set it to the live list even though a local community deploy can leave it unset.

### GitHub environments

Three must exist, or the roles are unassumable:

| Environment | Protection | Used by |
|---|---|---|
| `test-infra-deploy` | **None** — a merge must deploy unattended | The post-merge / `workflow_dispatch` deploy |
| `test-infra-deploy-approval` | **Required reviewers** | A pull-request deploy. This approval is the entire gate |
| `auto-approve`, `manual-approval` | As the rest of the repo has them (none / required reviewers) | The read-only diff job, via `strands-agents/devtools/authorization-check` |

Neither `test-infra-*` environment may be reused by another workflow: their names are what the deploy role trusts.

### One-time setup

The stack defines the roles that deploy it, so the first deployment is manual:

```sh
cd test-infra
npx cdk bootstrap   # once per account/region

# Internal mode needs every list below; it throws rather than deploy a role with
# one of them empty. Use the same values as the repository secrets.
STRANDS_TEST_INFRA_INTERNAL=true \
STRANDS_TEST_INFRA_PRIVATE_REPOS=... \
STRANDS_TEST_INFRA_BUCKET_NAMES=... \
STRANDS_TEST_INFRA_PERSISTENT_BUCKET_NAMES=... \
STRANDS_TEST_INFRA_SECRET_NAMES=... \
STRANDS_TEST_INFRA_RUNNER_ROLES=... \
  npx cdk deploy    # with account credentials
```

Then set `STRANDS_TEST_INFRA_DEPLOY_ROLE` to the ARN of the created deploy role and create the environments above. The same manual path is the repair route if a change ever breaks a trust policy.

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
