# Durable Lambda Agent

This deployable example combines the experimental Strands `Checkpoint` API,
`SessionManager`, and
[AWS Lambda durable execution](https://docs.aws.amazon.com/lambda/latest/dg/durable-functions.html).
The agent pauses at `afterModel` and `afterTools`, persists each `CheckpointData`
value in the Lambda durable journal, and saves conversation snapshots to S3.

## How persistence works

The handler creates the agent with `checkpointing: true` and invokes it until it
returns a terminal result. Each `stopReason: 'checkpoint'` result is serialized with
`Checkpoint.toJSON()`, persisted by `DurableContext.step()`, validated with
`Checkpoint.fromJSON()`, and passed into the next invocation through
`checkpointResume`.

`CheckpointData` contains only the boundary position, cycle index, and schema
version. `SessionManager` saves messages and other session snapshot fields to S3
after each agent invocation. The S3 storage calls are themselves durable steps, so
a Lambda replay restores the original session read and does not repeat writes.
Model and tool middleware steps restore completed calls into the fresh agent.

In the TypeScript SDK, resuming `afterModel` intentionally invokes the model again
because the pending assistant tool-use message is not part of that checkpoint. The
middleware records it as a separate durable model call; subsequent Lambda replays
restore both calls instead of contacting the model again.

## Prerequisites

- Node.js 22+
- AWS SAM CLI 1.143+
- AWS CLI v2
- AWS credentials with permission to deploy the template
- Bedrock model access in the deployment region

Use least-privilege credentials in a development or sandbox account. Review the IAM
policies and S3 bucket configuration in `template.yml` before deployment. Bedrock
and Lambda durable actions use wildcard resources so the example works with model
IDs and inference profiles across accounts; session access is scoped to the created
bucket and object prefix.

## Install and validate

From the monorepo root:

```bash
npm ci
npm --prefix strands-ts/examples/durable-lambda-agent ci
npm --prefix strands-ts/examples/durable-lambda-agent run build
npm --prefix strands-ts/examples/durable-lambda-agent run bundle
npm --prefix strands-ts/examples/durable-lambda-agent run validate:template
```

## Deploy

Deployment creates or updates a CloudFormation stack, IAM role, log group, durable
Lambda function, and encrypted versioned S3 session bucket. SAM displays the change
set for confirmation before applying it. The bucket has retention policies so stack
deletion does not remove conversation snapshots.

```bash
cd strands-ts/examples/durable-lambda-agent
AWS_REGION=us-west-2 npm run deploy
```

Optional environment variables:

```bash
STACK_NAME=my-durable-agent \
FUNCTION_NAME=my-durable-agent \
BEDROCK_MODEL_ID=global.anthropic.claude-sonnet-4-6 \
AWS_REGION=us-west-2 \
npm run deploy
```

## Invoke

Each invocation gets a unique durable execution and session ID by default.

```bash
npm run invoke          # normal execution
npm run invoke:restart  # wait-driven replay
npm run invoke:crash    # fail after the first tool, then resume with the second tool
```

Set `SESSION_ID` to continue a conversation across separate durable executions.
Session IDs may contain lowercase letters, numbers, hyphens, and underscores.

```bash
SESSION_ID=seattle_trip PROMPT='Add a hotel to the plan.' npm run invoke
```

The invocation command follows CloudWatch logs until interrupted with `Ctrl-C`.
For replay and retry scenarios, each tool and session-storage write should execute
once across the full durable execution.

## Remove the example stack

Stack deletion removes the function and IAM role and is destructive. The retained
S3 bucket and its versioned session data remain. Verify the account, region, and
stack name before running deletion manually:

```bash
aws sts get-caller-identity
aws cloudformation delete-stack \
  --stack-name "${STACK_NAME:-strands-durable-agent}" \
  --region "${AWS_REGION:-us-west-2}"
```

## Scope

This example supports synchronous tools and non-streaming durable replay. Stream
events are emitted during initial execution but are not re-emitted from completed
steps on replay. Session snapshots and durable step payloads must remain within the
respective S3 and Lambda service limits. MCP sessions and Strands interrupts require
separate lifecycle handling and are not covered here.
