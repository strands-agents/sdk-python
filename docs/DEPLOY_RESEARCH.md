# Deploy Research: Starter Toolkit vs Raw Boto3

## Discovery

Two existing AWS packages already solve the deployment problem we're building:

| Package | PyPI Name | Purpose |
|---------|-----------|---------|
| **bedrock-agentcore-sdk-python** | `bedrock-agentcore` | Runtime wrapper only. `BedrockAgentCoreApp` + decorators (`@app.entrypoint`, `@app.websocket`). No deployment logic. |
| **bedrock-agentcore-starter-toolkit** | `bedrock-agentcore-starter-toolkit` | Full deployment lifecycle. CLI (`agentcore deploy`) AND a programmatic `Runtime` class with `configure()`, `launch()`, `invoke()`, `destroy()`, `status()`. |

## The Key Finding

**The starter toolkit already has a Python API that does everything our `_agentcore.py` does — and more.**

The `Runtime` class at `bedrock_agentcore_starter_toolkit.notebook.runtime.bedrock_agentcore` provides:

```python
from bedrock_agentcore_starter_toolkit import Runtime

runtime = Runtime()
runtime.configure(
    entrypoint="agent.py",
    agent_name="my-agent",
    deployment_type="direct_code_deploy",
    runtime_type="PYTHON_3_12",
    region="us-west-2",
)
result = runtime.launch()  # Deploys to AWS
print(result.agent_arn)

runtime.invoke({"prompt": "Hello"})
runtime.status()
runtime.destroy()
```

This `Runtime.launch()` internally handles:
- IAM role creation (with proper trust policies, service-linked roles)
- S3 bucket creation and code upload
- `bedrock-agentcore-control` API calls (`create_agent_runtime`, `update_agent_runtime`)
- Polling for READY status
- CodeBuild for container deployments
- ECR management
- VPC configuration
- Memory service provisioning (STM/LTM)
- Observability setup

## Comparison: Our Implementation vs Starter Toolkit

| Capability | Our `_agentcore.py` | Starter Toolkit `Runtime` |
|------------|---------------------|---------------------------|
| IAM role creation | Basic (inline policy) | Production-grade (proper naming, service-linked roles) |
| S3 upload | Manual bucket + put_object | Managed with auto-create, proper naming |
| Runtime creation | Raw boto3 calls | `BedrockAgentCoreClient` wrapper with retries |
| Deployment modes | `direct_code_deploy` only | `direct_code_deploy` + `container` (CodeBuild/ECR) |
| VPC support | None | Full (subnets, security groups, validation) |
| Memory integration | None | STM, LTM, STM+LTM modes |
| Observability | None | OpenTelemetry, X-Ray integration |
| Auth configuration | None | JWT authorizer, OAuth2, request headers |
| Protocol support | HTTP only | HTTP, MCP, A2A, AG-UI |
| Lifecycle config | None | idle timeout, max lifetime |
| Dependency packaging | Zip CWD only | `uv` for dependency resolution + zip |
| Config persistence | `.strands/state.json` (custom) | `.bedrock_agentcore.yaml` (standard) |
| Destroy/cleanup | Basic (delete runtime + role) | Comprehensive (ECR, CodeBuild, memory, policies) |
| Error handling | Basic | Production-grade with rich console output |
| Multi-agent | No | Yes (named agents in config) |

## The Two Approaches

### Approach A: Keep Current (Raw Boto3)

What we built: 7 files in `src/strands/deploy/`, ~500 lines, raw boto3 calls.

**Pros:**
- Zero additional dependencies beyond boto3 (already in strands-agents)
- Full control over every API call
- Lighter install footprint

**Cons:**
- Reimplements what the starter toolkit already does (and does better)
- Missing: VPC, container deploys, memory, observability, auth, lifecycle config
- We'd need to maintain parity with AgentCore API changes ourselves
- Our IAM policies, S3 bucket naming, and error handling are less battle-tested
- Duplicated effort — AWS already ships and maintains this

### Approach B: Wrap the Starter Toolkit (Recommended)

Replace `_agentcore.py` internals to delegate to `Runtime` from `bedrock-agentcore-starter-toolkit`.

```python
# What agent.deploy() would do internally:
from bedrock_agentcore_starter_toolkit import Runtime

runtime = Runtime()
runtime.configure(
    entrypoint=generated_entrypoint_path,
    agent_name=config.name,
    deployment_type="direct_code_deploy",
    runtime_type=python_runtime,
    region=region,
    non_interactive=True,
)
result = runtime.launch()
```

**Pros:**
- Dramatically simpler implementation (~100 lines instead of ~500)
- Inherits all production features: VPC, memory, auth, container deploys, observability
- Maintained by AWS — stays current with AgentCore API changes
- Battle-tested by real users via the `agentcore` CLI
- Our `.deploy()` becomes a thin, Pythonic facade over proven infrastructure
- Users who want advanced config can drop down to `Runtime` directly

**Cons:**
- Adds `bedrock-agentcore-starter-toolkit` as a dependency (it's heavy: ~25 deps including rich, typer, jinja2, httpx)
- Should be an optional extra: `pip install strands-agents[deploy]`
- Config stored in `.bedrock_agentcore.yaml` (toolkit's format) rather than our `.strands/state.json`
- Less control over internal details

### Approach C: Hybrid

Keep our `DeployTarget` abstraction and `DeployConfig`/`DeployResult` types, but use `Runtime` internally for the AgentCore target. Keep `.strands/state.json` as our state layer that wraps the toolkit's config.

```python
class AgentCoreTarget(DeployTarget):
    def deploy(self, agent, config, state_manager):
        # 1. Generate entrypoint file (we keep this)
        entrypoint_path = self._write_entrypoint(agent)

        # 2. Delegate to starter toolkit (new)
        runtime = Runtime()
        runtime.configure(
            entrypoint=str(entrypoint_path),
            agent_name=f"strands-{config.name}",
            deployment_type="direct_code_deploy",
            runtime_type=get_python_runtime(),
            region=region,
            non_interactive=True,
            **self._map_config(config),  # auth, env vars, etc.
        )
        launch_result = runtime.launch()

        # 3. Map result back to our types (we keep this)
        state_manager.save(config.name, {...})
        return DeployResult(...)
```

**This gives us:**
- Clean Strands-native DX (`agent.deploy()`)
- All the toolkit's production capabilities
- Our state management layer on top
- The `DeployTarget` strategy pattern for future Lambda/Fargate targets
- Optional dependency via `[deploy]` extra

## Recommendation

**Approach C (Hybrid)** is the strongest path. It keeps our clean API surface and extensible architecture while leveraging AWS's production-grade deployment machinery under the hood.

The key changes needed:
1. Add `bedrock-agentcore-starter-toolkit>=0.3.0` to a `[deploy]` optional extra in `pyproject.toml`
2. Rewrite `_agentcore.py` to delegate to `Runtime` (~100 lines instead of ~250)
3. Keep `_packaging.py` for entrypoint generation (still needed)
4. Keep `_state.py` as our state layer
5. Remove our manual IAM role creation, S3 bucket management, and runtime polling — the toolkit handles all of that
6. Drop `_constants.py` IAM policies (no longer needed)

## What We Keep

- `Agent.deploy()` method and its signature
- `DeployConfig`, `DeployResult` dataclasses
- `DeployTarget` strategy pattern (for future Lambda target)
- `StateManager` and `.strands/state.json`
- Entrypoint generation (`_packaging.py`)
- Exception hierarchy

## What We Replace

- All raw boto3 calls in `_agentcore.py` (IAM, S3, bedrock-agentcore-control)
- IAM policy constants
- Runtime polling loop
- S3 bucket creation/management

## Source Repos Examined

- `/Users/afarn/workplace/dev/junk/bedrock-agentcore-starter-toolkit` — v0.3.3, provides `Runtime` class and `agentcore` CLI
- `/Users/afarn/workplace/dev/junk/bedrock-agentcore-sdk-python` — v1.4.7, provides `BedrockAgentCoreApp` runtime wrapper (no deployment logic)
