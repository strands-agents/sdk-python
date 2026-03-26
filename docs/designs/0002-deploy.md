# Strands High Level Features - Deploy

## Where we are

The Strands Agents SDK makes it easy to build an agent in a few lines of code. Once an agent works locally, the next question is always the same: *how do I run this in production?*

Today, deploying a Strands agent to the cloud requires the developer to:

1. Write a service wrapper (Flask, FastAPI, or a framework-specific entrypoint).
2. Package the agent code, its dependencies, and the wrapper into a deployable artifact.
3. Provision cloud infrastructure: IAM roles, compute runtimes, networking, and endpoints.
4. Track what was deployed so subsequent updates go to the same resource.
5. Repeat all of the above each time the agent changes.

Each of these steps involves decisions that have nothing to do with the agent itself. The developer must become an infrastructure engineer before their agent can serve its first request. This friction slows adoption, increases the barrier to entry, and pulls focus away from the agent's actual purpose.

## The gap between "it works locally" and "it works in production"

AWS Bedrock AgentCore provides managed infrastructure for running agents: auto-scaling runtimes, built-in IAM, and a service endpoint. The `bedrock-agentcore-starter-toolkit` Python package wraps AgentCore's APIs and handles the heavy lifting of IAM role creation, code upload, runtime provisioning, and endpoint polling.

Even with the toolkit, a developer still needs to:

- **Learn AgentCore's mental model.** AgentCore introduces its own concepts that developers need to learn.
- **Write a `BedrockAgentCoreApp` entrypoint from scratch.** The toolkit expects a specific entrypoint format: a `BedrockAgentCoreApp` instance with an `@app.entrypoint` decorator. The developer must manually reconstruct their Strands agent inside this file. No bridge exists between the Strands `Agent` class and AgentCore's expected format.
- **Track deployed resources manually.** Each deployment creates IAM roles, runtime IDs, and endpoint ARNs. The developer must track these across create-vs-update cycles to avoid orphaned resources or duplicates. No built-in state management exists.

This is too many concepts, too many files, and too many failure modes for someone who just wants to ship an agent.

## Introducing `deploy` - one call from local to cloud

The end goal is the simplest possible deployment experience — a method on the agent itself:

```python
agent = Agent(name="ac_agent", plugins=[plugins] tools=[my_tool])
agent.deploy()
```

This is the API we want to graduate to. `agent.deploy()` communicates exactly the right intent: *take this agent and put it in the cloud.* It requires no new imports, no separate modules, and no mental context switch.

However, deployment touches cloud infrastructure, source capture, state management, and dependency resolution. The API will evolve as we add targets and learn from real usage. Shipping `agent.deploy()` on day one would lock us into a stable contract before the feature is ready.

For the initial launch, we expose deployment as a standalone function under `strands.experimental`:

```python
from strands import Agent
from strands.experimental.deploy import deploy

agent = Agent(name="ac_agent", tools=[my_tool])
result = deploy(agent)
```

This gives us room to iterate on the function signature, config shape, and result fields while signaling that the API may change. Once the feature stabilizes — validated across project structures, with multiple targets and real-world usage — we promote it to `agent.deploy()` on the Agent class.

Behind that call, the module:

1. Captures the caller's source and strips the `deploy()` call.
2. Appends a `BedrockAgentCoreApp` wrapper to create a deployable entrypoint.
3. Resolves the AWS region from the agent's model config, the boto3 session, or the environment.
4. Merges SDK-required packages (`bedrock-agentcore`, `strands-agents`) with the user's `requirements.txt`.
5. Delegates provisioning to the `bedrock-agentcore-starter-toolkit` `Runtime` class.
6. Saves deployment state to `.strands_deploy/state.json` so subsequent calls update rather than recreate.
7. Cleans up generated artifacts (`_strands_entrypoint.py`, `dependencies.hash`, `dependencies.zip`).

### Why experimental first?

Placing `deploy` under `strands.experimental` rather than directly on the Agent class serves three purposes:

- **API freedom.** The function signature, config shape, and result fields can change without breaking stable SDK consumers.
- **Dependency isolation.** Deployment pulls in `bedrock-agentcore-starter-toolkit` and `boto3`. These should not be required to import `Agent`. The experimental function makes the dependency opt-in:

```python
pip install 'strands-agents[deploy]'
```

- **Graduation criteria.** Moving to `agent.deploy()` becomes a deliberate milestone — one that requires validated source capture, multiple targets, and confidence in the API surface.

## Architecture

### Strategy pattern for deployment targets

A core tenet of the Strands SDK is platform independence. Strands agents already use any model provider — Bedrock, OpenAI, Gemini — and the same principle extends to deployment. A developer should deploy the same agent to AgentCore today and to Lambda, ECS, or a non-AWS platform tomorrow without rewriting deployment code. The deploy module cannot be built around a single target. It must support multiple backends behind a stable interface.

The module uses the Strategy pattern to achieve this:

```
deploy()  -->  DeployTarget (ABC)
                    |
                    +-- AgentCoreTarget
                    +-- (future: LambdaTarget, ECSTarget, ...)
```

```python
class DeployTarget(ABC):
    @abstractmethod
    def validate(self, config: DeployConfig) -> None: ...

    @abstractmethod
    def deploy(self, agent: Agent, config: DeployConfig,
               state_manager: StateManager) -> DeployResult: ...

    @abstractmethod
    def destroy(self, name: str, state_manager: StateManager,
                region: str | None = None) -> None: ...
```

Each target owns its full lifecycle: validation, provisioning, updating, and teardown. The `deploy()` function selects the target, constructs a `DeployConfig`, and delegates.

### AgentCore target

`AgentCoreTarget` is the first concrete target. Rather than calling AWS APIs directly, it delegates to the `bedrock-agentcore-starter-toolkit` `Runtime` class:

```
AgentCoreTarget.deploy()
    |
    +-- generate_agentcore_entrypoint(agent)  # _packaging.py
    +-- Runtime().configure(...)               # starter toolkit
    +-- Runtime().launch()                     # starter toolkit
    +-- StateManager.save(...)                 # _state.py
```

This delegation is deliberate. The starter toolkit handles IAM role creation, S3 upload, runtime polling, and endpoint management. Reimplementing that logic would be fragile and would drift as AgentCore evolves. By delegating, the deploy module stays thin and benefits from toolkit improvements automatically.

### Entrypoint generation via source capture

A Strands agent carries tools, plugins, hooks, and custom model configurations — arbitrary Python objects that cannot be reliably serialized. The deploy module sidesteps this by copying the caller's actual source file and transforming it into a valid AgentCore entrypoint.

The process works in three steps:

1. **Find the caller.** `_find_caller_info()` walks the call stack to locate the source file that called `deploy()`, and identifies the variable name holding the agent object.
2. **Strip the deploy call.** An AST transformer (`_DeployStripper`) removes the `deploy()` import and call, any code after it, and any `if __name__ == '__main__'` block. It also converts relative imports to absolute so the code works as a standalone script.
3. **Append the AgentCore wrapper.** A small template is appended that wraps the agent in a `BedrockAgentCoreApp` entrypoint.

Given this caller source:

```python
from strands import Agent
from strands.experimental import deploy
from my_tools import calculator, search

agent = Agent(
    name="ac_agent",
    system_prompt="You are a helpful assistant.",
    tools=[calculator, search],
)
result = deploy(agent)
```

The generated entrypoint becomes:

```python
"""Auto-generated Strands Agent entrypoint for AgentCore."""
import sys
import os

_here = os.path.dirname(__file__)
sys.path.insert(0, _here)

from strands import Agent
from my_tools import calculator, search

agent = Agent(
    system_prompt='You are a helpful assistant.',
    tools=[calculator, search],
    name="ac_agent"
)

# --- AgentCore wrapper (auto-generated by strands deploy) ---
from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload):
    prompt = payload.get("prompt", "Hello!")
    result = agent(prompt)
    return {"result": str(result), "stop_reason": result.stop_reason}

if __name__ == "__main__":
    app.run()
```

This approach preserves the full agent definition — tools, plugins, hooks, and all constructor parameters. The entire working directory is packaged into the deployment zip alongside the entrypoint, so local imports (like `my_tools`) resolve correctly.

If the caller's source file cannot be found (e.g., when called from a REPL or `<stdin>`), the module raises a `DeployPackagingException`.

### State management

Deployments are stateful. Creating an AgentCore runtime produces ARNs and IDs that must persist so the next `deploy()` call updates rather than duplicates.

State lives in `.strands_deploy/state.json` in the working directory:

```json
{
  "version": "1",
  "deployments": {
    "my-agent": {
      "target": "agentcore",
      "region": "us-west-2",
      "agent_runtime_id": "rt-abc123",
      "agent_runtime_arn": "arn:aws:bedrock-agentcore:...",
      "last_deployed": "2026-03-25T10:30:00+00:00",
      "created_at": "2026-03-25T10:00:00+00:00"
    }
  }
}
```

The `StateManager` uses atomic writes (temporary file + `os.replace`) to prevent corruption. Multiple named deployments can coexist in a single state file.

### Exception hierarchy

All deployment errors inherit from `DeployException`:

```
DeployException
    +-- DeployTargetException   # Target-specific failures (AWS API errors)
    +-- DeployPackagingException # Code packaging failures
    +-- DeployStateException     # State file read/write failures
```

Callers can catch the base class broadly or handle specific failure modes.

## Module structure

```
src/strands/experimental/deploy/
    __init__.py        # deploy(), DeployConfig, DeployResult
    _base.py           # DeployTarget ABC
    _agentcore.py      # AgentCoreTarget implementation
    _constants.py      # Python version mapping, packaging excludes
    _packaging.py      # Entrypoint generation, code zipping
    _state.py          # StateManager, DeployState TypedDict
    _exceptions.py     # Exception hierarchy
```

All internal modules are prefixed with `_` to signal they are private. The public API is the `deploy()` function and the dataclasses exported from `__init__.py`.

## Configuration

### `DeployConfig`

```python
@dataclass
class DeployConfig:
    target: Literal["agentcore"]
    name: str
    region: str | None = None
    description: str | None = None
    environment_variables: dict[str, str] = field(default_factory=dict)
```

### `DeployResult`

```python
@dataclass
class DeployResult:
    target: str
    name: str
    region: str
    created: bool = True
    agent_runtime_id: str | None = None
    agent_runtime_arn: str | None = None
    agent_runtime_endpoint_arn: str | None = None
    role_arn: str | None = None
```

### Region resolution

Region is resolved in priority order:
1. Explicit `region` parameter
2. Agent model's config (`model.config["region"]`)
3. boto3 session default region
4. `AWS_REGION` environment variable
5. Fallback to `us-east-1`

### Name sanitization

If no explicit name is provided, the agent's name is sanitized: lowercased, non-alphanumeric characters replaced with underscores, stripped, and truncated to 40 characters.

## Experimental status

The module lives under `strands.experimental` for three reasons:

1. **Source capture has edge cases.** The AST-based approach works well for straightforward scripts, but complex project layouts — editable installs, custom `PYTHONPATH`, deeply nested packages — may require additional `sys.path` handling that has not yet been validated.
2. **Single target.** Only AgentCore is implemented. The target abstraction needs validation against a second backend.
3. **API surface.** The function signature, config shape, and result fields may change as we learn from real usage.

Graduating to `agent.deploy()` on the Agent class requires validating source capture across diverse project structures, adding a second deployment target, and gaining confidence in the API surface through real-world usage.

## FAQ

### Why not just use the AgentCore CLI directly?

Programmatic deployment from Python offers advantages the CLI cannot:

- **No learning curve.** Developers already know Python. `deploy(agent)` is self-explanatory — no CLI flags, config files, or authentication flows to learn.
- **Rapid experimentation and sharing.** Tweak a prompt, swap a tool, redeploy — without leaving the script. The feedback loop between "change the agent" and "test it live" shrinks to seconds. Share a live endpoint with teammates by sharing an ARN.
- **Pipeline integration.** Deployment becomes a line of Python in a CI job, an eval harness, or a test suite:

```python
agent = Agent(name="my_agent", tools=[my_tool])
results = run_evals(agent)
if results.passed:
    deploy(agent)
```

- **No boilerplate.** The CLI still requires a `BedrockAgentCoreApp` entrypoint, managed requirements, and explicit configuration. `deploy()` handles all of that.
- **Familiar paradigm.** Python developers expect this pattern from Airflow, Prefect, and MLflow — infrastructure managed programmatically alongside the code it runs.

### Does this compromise Strands' platform agnosticism?

No. The `DeployTarget` abstraction keeps deployment platform-agnostic. AgentCore is the first target, but the interface supports any backend — Lambda, ECS, Kubernetes, or non-AWS platforms. Each target is a separate implementation behind the same `deploy()` function, and users can always specify `target=` explicitly.

Strands favors making deployment easy, not a particular cloud provider.

### Why not ship `agent.deploy()` on the Agent class from day one?

The end goal is `agent.deploy()`. But deployment touches source capture, cloud infrastructure, state management, and dependency resolution — the API will evolve as we add targets and learn from usage.

Deployment starts as an experimental function: `from strands.experimental.deploy import deploy` followed by `deploy(agent)`. This lets us iterate without locking the stable Agent class into a premature contract. Once validated across project structures and with multiple targets, we promote it to `agent.deploy()`.

### What happens if I call `deploy()` twice?

The second call updates the existing deployment rather than creating a duplicate. State is tracked in `.strands_deploy/state.json`, which maps deployment names to their cloud resource IDs. If state exists for the given name, the deploy module passes the existing resource identifiers to the toolkit so it performs an update instead of a create.

### What gets packaged in the deployment?

The entire working directory (minus excluded paths like `.git`, `__pycache__`, `.venv`, etc.) is zipped and uploaded. The caller's source file is transformed into an AgentCore entrypoint, with the `deploy()` call stripped and a wrapper appended. This means all local Python files — tool definitions, utility modules, config files — are available to the deployed agent.

## Follow-up items

- **Complex project layouts.** Validate and improve source capture for editable installs, custom `PYTHONPATH`, and deeply nested package hierarchies where `sys.path` manipulation may not suffice.
- **Automatic dependency detection.** Extract dependencies from `pyproject.toml` rather than requiring a separate `requirements.txt`.
- **`destroy()` as public API.** Expose teardown through the top-level module, not just the target class.
- **CLI integration.** Add a `strands deploy` CLI command for deployment outside Python scripts.
- **Multi-target validation.** Implement a second target (Lambda, ECS, or similar) to stress-test the `DeployTarget` abstraction.
