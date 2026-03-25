# Deploy Module

Deploy Strands agents to AWS from Python. One method call packages your agent's code, provisions cloud infrastructure, and returns a live endpoint.

```python
from strands import Agent

agent = Agent(model="us.anthropic.claude-sonnet-4-20250514", system_prompt="You are a helpful assistant.")
result = agent.deploy(target="agentcore", name="my-agent")
print(result.agent_runtime_arn)
```

No Terraform. No CloudFormation. No Docker. No YAML.

## Why Use This

### The Problem

Deploying an AI agent to AWS today requires leaving your Python environment and context-switching into infrastructure work:

1. Write a handler/entrypoint that wraps your agent
2. Package your code with the right dependencies for Linux
3. Create IAM roles with the correct trust policies and permissions
4. Provision compute (Lambda, ECS, AgentCore) via console, CLI, or IaC
5. Configure networking, endpoints, and authentication
6. Track which resources you created so you can update or tear them down

For a developer iterating on agent logic, this overhead is the difference between shipping in minutes and shipping in hours.

### The Solution

`.deploy()` handles all of that behind a single method call. It follows the same pattern as [Prefect's `flow.deploy()`](https://docs.prefect.io/latest/deploy/run-flows-in-local-processes/) and [Modal's `@app.function`](https://modal.com/) -- infrastructure from code.

### Use Cases

**Rapid prototyping.** You're experimenting with a new agent -- different system prompts, tools, model configurations. You want to test it as a real HTTP endpoint that teammates or a frontend can call, without setting up infrastructure first.

```python
agent = Agent(model="us.anthropic.claude-sonnet-4-20250514", system_prompt="You are a code reviewer.")
result = agent.deploy(name="code-reviewer-v3")
# Share the ARN with your team -- they can invoke it immediately
```

**Demo and stakeholder review.** You need to show a working agent to stakeholders. Instead of running it on your laptop during the meeting, deploy it to AgentCore and hand them an endpoint they can test independently.

**CI/CD integration.** Add `.deploy()` to your pipeline so every merge to main automatically deploys the latest version of your agent. The state file tracks the existing deployment and updates it in place.

```python
# deploy_agent.py -- called by CI
from strands import Agent
from my_project.tools import search, summarize

agent = Agent(
    model="us.anthropic.claude-sonnet-4-20250514",
    system_prompt="You are a research assistant.",
    tools=[search, summarize],
)
agent.deploy(name="research-assistant", region="us-west-2")
```

**Multi-agent systems.** Deploy multiple agents independently and compose them. Each agent gets its own runtime, IAM role, and endpoint.

```python
planner = Agent(system_prompt="You break tasks into steps.")
executor = Agent(system_prompt="You execute individual steps.")

planner.deploy(name="planner-agent")
executor.deploy(name="executor-agent")
```

## Getting Started

### Prerequisites

- **Python 3.10+**
- **AWS credentials** configured (via `aws configure`, environment variables, or IAM role)
- **Permissions** to create IAM roles, S3 buckets, and AgentCore runtimes (see [Permissions](#permissions) below)
- **boto3** with AgentCore support (already a Strands dependency -- run `pip install --upgrade boto3 botocore` if you get service errors)

### Basic Deployment

```python
from strands import Agent

agent = Agent(
    model="us.anthropic.claude-sonnet-4-20250514",
    system_prompt="You are a helpful assistant.",
)

result = agent.deploy(
    target="agentcore",  # Deploy to Bedrock AgentCore
    name="my-agent",     # Name for the cloud resource
)

print(result.agent_runtime_arn)       # arn:aws:bedrock-agentcore:us-west-2:123456789:runtime/...
print(result.agent_runtime_endpoint_arn)  # arn:aws:bedrock-agentcore:us-west-2:123456789:runtime/.../endpoint/...
print(result.role_arn)                # arn:aws:iam::123456789:role/strands-my-agent-agentcore-role
```

### Updating an Existing Deployment

Call `.deploy()` again with the same name. The module reads `.strands/state.json` and updates the existing runtime instead of creating a new one:

```python
# First deploy -- creates everything
agent.deploy(name="my-agent")

# Change the agent...
agent = Agent(system_prompt="You are now a pirate assistant. Arrr!")

# Second deploy -- updates the existing runtime
result = agent.deploy(name="my-agent")
print(result.created)  # False -- updated, not created
```

### Tearing Down

Use the `AgentCoreTarget.destroy()` method to clean up all resources:

```python
from strands.deploy._agentcore import AgentCoreTarget
from strands.deploy._state import StateManager

target = AgentCoreTarget()
target.destroy("my-agent", StateManager())
```

This deletes the AgentCore runtime, IAM role, and removes the state entry.

## API Reference

### `Agent.deploy()`

```python
def deploy(
    self,
    target: Literal["agentcore"] = "agentcore",
    *,
    name: str | None = None,
    auth: Literal["public", "iam"] = "public",
    region: str | None = None,
    description: str | None = None,
    environment_variables: dict[str, str] | None = None,
) -> DeployResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target` | `"agentcore"` | `"agentcore"` | Deployment target. Currently only AgentCore is supported. |
| `name` | `str \| None` | Sanitized `agent.name` | Name for the deployed resource. Used in IAM role names, S3 keys, and runtime names. |
| `auth` | `"public" \| "iam"` | `"public"` | Authentication mode for the endpoint. |
| `region` | `str \| None` | Auto-detected | AWS region. Resolved from: explicit value > model config > boto3 session > `AWS_REGION` env var > `us-east-1`. |
| `description` | `str \| None` | `None` | Description attached to the AgentCore runtime. |
| `environment_variables` | `dict[str, str] \| None` | `None` | Environment variables set in the runtime. Useful for API keys or feature flags. |

### `DeployResult`

Returned by `agent.deploy()`:

| Field | Type | Description |
|-------|------|-------------|
| `target` | `str` | The deployment target (`"agentcore"`). |
| `name` | `str` | The deployment name. |
| `region` | `str` | The AWS region. |
| `created` | `bool` | `True` if a new deployment was created, `False` if an existing one was updated. |
| `agent_runtime_id` | `str \| None` | The AgentCore runtime ID. |
| `agent_runtime_arn` | `str \| None` | The full ARN of the runtime. |
| `agent_runtime_endpoint_arn` | `str \| None` | The ARN of the runtime endpoint. |
| `role_arn` | `str \| None` | The IAM execution role ARN. |

### `DeployConfig`

If you prefer the functional API over the method:

```python
from strands.deploy import DeployConfig, deploy

config = DeployConfig(
    target="agentcore",
    name="my-agent",
    region="eu-west-1",
    environment_variables={"LOG_LEVEL": "DEBUG"},
)
result = deploy(agent, config)
```

## How It Works

### Deployment Flow

When you call `agent.deploy(target="agentcore", name="my-agent")`, the module executes these steps:

1. **Validate** -- checks AWS credentials and boto3 compatibility
2. **Resolve region** -- from the explicit parameter, agent's model config, boto3 session, or `AWS_REGION`
3. **Create IAM role** (if new) -- `strands-{name}-agentcore-role` with trust policy for `bedrock-agentcore.amazonaws.com` and permissions for Bedrock model invocation, CloudWatch, and X-Ray
4. **Generate entrypoint** -- a Python file that wraps your agent in a `BedrockAgentCoreApp` with an HTTP `/invocations` endpoint
5. **Package code** -- zips your working directory (excluding `.git/`, `.venv/`, `__pycache__/`, `.strands/`) plus the generated entrypoint
6. **Upload to S3** -- creates a bucket `strands-deploy-{account_id}-{region}` and uploads the zip
7. **Create/update runtime** -- calls `bedrock-agentcore-control` API with the S3 code artifact
8. **Wait for READY** -- polls `get_agent_runtime()` until status is `READY` (up to 5 minutes)
9. **Create endpoint** (if new) -- creates a callable endpoint for the runtime
10. **Save state** -- writes all ARNs to `.strands/state.json` for future updates

### State File

Deployment state is tracked in `.strands/state.json` in your working directory:

```json
{
  "version": "1",
  "deployments": {
    "my-agent": {
      "target": "agentcore",
      "region": "us-west-2",
      "agent_runtime_id": "rt-abc123",
      "agent_runtime_arn": "arn:aws:bedrock-agentcore:us-west-2:123456789:runtime/rt-abc123",
      "agent_runtime_endpoint_arn": "arn:aws:bedrock-agentcore:...",
      "role_arn": "arn:aws:iam::123456789:role/strands-my-agent-agentcore-role",
      "s3_bucket": "strands-deploy-123456789-us-west-2",
      "s3_key": "my-agent/20260325T103000.zip",
      "last_deployed": "2026-03-25T10:30:00+00:00",
      "created_at": "2026-03-25T10:30:00+00:00"
    }
  }
}
```

Add `.strands/` to your `.gitignore` -- it contains account-specific ARNs that shouldn't be committed.

### What Gets Packaged

The zip archive contains:
- All files in your working directory (excluding `.git/`, `.venv/`, `venv/`, `.env/`, `__pycache__/`, `.strands/`, `node_modules/`, `*.pyc`, `*.egg-info/`, build artifacts)
- `_strands_entrypoint.py` -- auto-generated `BedrockAgentCoreApp` wrapper that reconstructs your agent

### Architecture

The module follows the **strategy pattern** so new deployment targets can be added without modifying existing code:

```
Agent.deploy()
    |
    v
deploy() function (src/strands/deploy/__init__.py)
    |
    v
DeployTarget (abstract base)
    |
    +-- AgentCoreTarget  (bedrock-agentcore-control)
    +-- [LambdaTarget]   (future)
    +-- [FargateTarget]  (future)
```

## Permissions

The AWS principal running `.deploy()` needs these permissions:

### IAM
- `iam:CreateRole`, `iam:GetRole` -- for the execution role
- `iam:PutRolePolicy`, `iam:DeleteRolePolicy` -- for the inline execution policy
- `iam:DeleteRole` -- for `destroy()`

### S3
- `s3:CreateBucket`, `s3:HeadBucket` -- for the deployment artifact bucket
- `s3:PutObject` -- for uploading the code zip
- `s3:PutPublicAccessBlock` -- for securing the bucket

### Bedrock AgentCore
- `bedrock-agentcore:CreateAgentRuntime`, `bedrock-agentcore:UpdateAgentRuntime`, `bedrock-agentcore:GetAgentRuntime`, `bedrock-agentcore:DeleteAgentRuntime`
- `bedrock-agentcore:CreateAgentRuntimeEndpoint`

### STS
- `sts:GetCallerIdentity` -- to resolve the account ID for resource naming

## Current Limitations

- **AgentCore only.** Lambda, Fargate, and other compute targets are planned but not yet implemented.
- **Basic agent reconstruction.** The generated entrypoint reconstructs the agent from its `model`, `system_prompt`, and `name`. Custom `@tool` functions defined in your code are included in the zip but must be importable by the entrypoint to be used. Complex tool configurations (MCP servers, tool providers) are not yet serialized.
- **No streaming endpoint.** The generated entrypoint uses synchronous invocation. WebSocket/streaming support is planned.
- **Single endpoint per deployment.** Each `.deploy()` call manages one runtime and one endpoint.

## Extending

To add a new deployment target, implement the `DeployTarget` interface:

```python
from strands.deploy._base import DeployTarget

class LambdaTarget(DeployTarget):
    def validate(self, config):
        ...

    def deploy(self, agent, config, state_manager):
        ...

    def destroy(self, name, state_manager, region=None):
        ...
```

Then register it in `src/strands/deploy/__init__.py` in the `targets` dict.
