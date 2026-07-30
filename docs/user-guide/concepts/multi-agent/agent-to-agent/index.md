Strands Agents supports the [Agent-to-Agent (A2A) protocol](https://a2aproject.github.io/A2A/latest/), enabling seamless communication between AI agents across different platforms and implementations.

## What is Agent-to-Agent (A2A)?

The Agent-to-Agent protocol is an open standard that defines how AI agents can discover, communicate, and collaborate with each other.

### Use Cases

A2A protocol support enables several powerful use cases:

-   **Multi-Agent Workflows**: Chain multiple specialized agents together
-   **Agent Marketplaces**: Discover and use agents from different providers
-   **Cross-Platform Integration**: Connect Strands agents with other A2A-compatible systems
-   **Distributed AI Systems**: Build scalable, distributed agent architectures

Learn more about the A2A protocol:

-   [A2A GitHub Organization](https://github.com/a2aproject/A2A)
-   [A2A Python SDK](https://github.com/a2aproject/a2a-python)
-   [A2A Documentation](https://a2aproject.github.io/A2A/latest/)

Complete Examples Available

Check out the [Native A2A Support samples](https://github.com/strands-agents/samples/tree/main/python/03-integrate/protocols/a2a-native) for complete, ready-to-run client, server and tool implementations.

## Installation

To use A2A functionality with Strands, install the package with the A2A dependencies:

(( tab "Python" ))
```bash
pip install 'strands-agents[a2a]'
```

This installs the core Strands SDK along with the necessary A2A protocol dependencies.
(( /tab "Python" ))

(( tab "TypeScript" ))
```bash
npm install @strands-agents/sdk @a2a-js/sdk express
```

`@a2a-js/sdk` and `express` are optional peer dependencies of `@strands-agents/sdk` and must be installed explicitly.
(( /tab "TypeScript" ))

## Consuming Remote Agents

The `A2AAgent` class provides the simplest way to consume remote A2A agents. It wraps the A2A protocol communication and presents a familiar interface—you can invoke it just like a regular Strands `Agent`.

Without `A2AAgent`, you need to manually resolve agent cards, configure HTTP clients, build protocol messages, and parse responses. The `A2AAgent` class handles all of this automatically.

### Basic Usage

(( tab "Python" ))
```python
from strands.agent.a2a_agent import A2AAgent

# Create an A2AAgent pointing to a remote A2A server
a2a_agent = A2AAgent(endpoint="http://localhost:9000")

# Invoke it just like a regular Agent
result = a2a_agent("Show me 10 ^ 6")
print(result.message)
# {'role': 'assistant', 'content': [{'text': '10^6 = 1,000,000'}]}
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { A2AAgent } from '@strands-agents/sdk/a2a'

// Create an A2AAgent pointing to a remote A2A server
const a2aAgent = new A2AAgent({ url: 'http://localhost:9000' })

// Invoke it just like a regular Agent
const result = await a2aAgent.invoke('Show me 10 ^ 6')
console.log(result.lastMessage.content)
```
(( /tab "TypeScript" ))

The `A2AAgent` returns an `AgentResult` just like a local `Agent`, making it easy to integrate remote agents into your existing code.

### Configuration Options

(( tab "Python" ))
The `A2AAgent` constructor accepts these parameters.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `endpoint` | `str` | Required | Base URL of the remote A2A agent |
| `name` | `str` | None | Agent name (auto-populated from agent card if not provided) |
| `description` | `str` | None | Agent description (auto-populated from agent card if not provided) |
| `timeout` | `int` | 300 | Timeout for HTTP operations in seconds |
| `a2a_client_factory` | `ClientFactory` | None | Optional pre-configured A2A client factory |
(( /tab "Python" ))

(( tab "TypeScript" ))
The `A2AAgent` constructor accepts a config object with these properties.

| Property | Type | Default | Description |
| --- | --- | --- | --- |
| `url` | `string` | Required | Base URL of the remote A2A agent |
| `agentCardPath` | `string` | `/.well-known/agent-card.json` | Path to the agent card endpoint |
| `id` | `string` | The `url` value | Unique identifier for the agent instance |
| `name` | `string` | From agent card | Agent name (auto-populated from agent card if not provided) |
| `description` | `string` | From agent card | Agent description (auto-populated from agent card if not provided) |

The agent card is fetched lazily on the first `invoke()` or `stream()` call.
(( /tab "TypeScript" ))

### Asynchronous Invocation

(( tab "Python" ))
For async workflows, use `invoke_async`:

```python
import asyncio
from strands.agent.a2a_agent import A2AAgent

async def main():
    a2a_agent = A2AAgent(endpoint="http://localhost:9000")
    result = await a2a_agent.invoke_async("Calculate the square root of 144")
    print(result.message)

asyncio.run(main())
```
(( /tab "Python" ))

(( tab "TypeScript" ))
In TypeScript, `invoke` is always async:

```typescript
import { A2AAgent } from '@strands-agents/sdk/a2a'

const a2aAgent = new A2AAgent({ url: 'http://localhost:9000' })
const result = await a2aAgent.invoke('Calculate the square root of 144')
console.log(result.lastMessage.content)
```
(( /tab "TypeScript" ))

### Streaming Responses

(( tab "Python" ))
For real-time streaming of responses, use `stream_async`:

```python
import asyncio
from strands.agent.a2a_agent import A2AAgent

async def main():
    a2a_agent = A2AAgent(endpoint="http://localhost:9000")

    async for event in a2a_agent.stream_async("Explain quantum computing"):
        if "data" in event:
            print(event["data"], end="", flush=True)

asyncio.run(main())
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
const remoteAgent = new A2AAgent({ url: 'http://localhost:9000' })

// stream() yields A2AStreamUpdateEvent for each protocol event,
// then an AgentResultEvent with the final result
const stream = remoteAgent.stream('Explain quantum computing')
let next = await stream.next()
while (!next.done) {
  console.log(next.value)
  next = await stream.next()
}
// Final result
console.log(next.value)
```

`A2AAgent.stream()` uses `sendMessageStream` from the A2A SDK. It yields `A2AStreamUpdateEvent` for each protocol event (messages, task status updates, artifact updates) followed by an `AgentResultEvent` with the final result.
(( /tab "TypeScript" ))

### Fetching the Agent Card

(( tab "Python" ))
You can retrieve the remote agent’s metadata using `get_agent_card`:

```python
import asyncio
from strands.agent.a2a_agent import A2AAgent

async def main():
    a2a_agent = A2AAgent(endpoint="http://localhost:9000")
    card = await a2a_agent.get_agent_card()
    print(f"Agent: {card.name}")
    print(f"Description: {card.description}")
    print(f"Skills: {card.skills}")

asyncio.run(main())
```
(( /tab "Python" ))

(( tab "TypeScript" ))
The agent card is fetched and cached internally on the first `invoke()` or `stream()` call. There is no separate public method to retrieve it.
(( /tab "TypeScript" ))

## A2AAgent in Multi-Agent Patterns

The `A2AAgent` class integrates with Strands multi-agent patterns that support it. Currently, you can use remote A2A agents in [Graph](/docs/user-guide/concepts/multi-agent/graph/index.md) workflows (Python only) and as [tools in an orchestrator agent](#as-a-tool).

### As a Tool

You can wrap an `A2AAgent` as a tool in an orchestrator agent’s toolkit:

(( tab "Python" ))
```python
from strands import Agent, tool
from strands.agent.a2a_agent import A2AAgent

calculator_agent = A2AAgent(
    endpoint="http://calculator-service:9000",
    name="calculator"
)

@tool
def calculate(expression: str) -> str:
    """Perform a mathematical calculation."""
    result = calculator_agent(expression)
    return str(result.message["content"][0]["text"])

orchestrator = Agent(
    system_prompt="You are a helpful assistant. Use the calculate tool for math.",
    tools=[calculate]
)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
const calculatorAgent = new A2AAgent({
  url: 'http://calculator-service:9000',
})

const calculate = tool({
  name: 'calculate',
  description: 'Perform a mathematical calculation.',
  inputSchema: z.object({
    expression: z.string().describe('The math expression to evaluate'),
  }),
  callback: async (input) => {
    const calcResult = await calculatorAgent.invoke(input.expression)
    return String(calcResult.lastMessage.content[0])
  },
})

const orchestrator = new Agent({
  systemPrompt: 'You are a helpful assistant. Use the calculate tool for math.',
  tools: [calculate],
})
```
(( /tab "TypeScript" ))

### In Graph Workflows

The `A2AAgent` works as a node in [Graph](/docs/user-guide/concepts/multi-agent/graph/index.md) workflows. See [Remote Agents with A2AAgent](/docs/user-guide/concepts/multi-agent/graph/index.md#remote-agents-with-a2aagent) for detailed examples of mixing local and remote agents in graph-based pipelines.

### In Swarm Patterns

Not yet supported

`A2AAgent` is not currently supported in Swarm patterns in either SDK. Swarm coordination relies on tool-based handoffs that require capabilities not yet available in the A2A protocol. Use [Graph](/docs/user-guide/concepts/multi-agent/graph/index.md) workflows for multi-agent patterns with remote A2A agents.

## Creating an A2A Server

### Basic Server Setup

Create a Strands agent and expose it as an A2A server:

(( tab "Python" ))
```python
import logging
from strands_tools.calculator import calculator
from strands import Agent
from strands.multiagent.a2a import A2AServer

logging.basicConfig(level=logging.INFO)

# Build a fresh agent for each A2A context so callers stay isolated
def create_agent(context_id: str) -> Agent:
    return Agent(
        name="Calculator Agent",
        description="A calculator agent that can perform basic arithmetic operations.",
        tools=[calculator],
        callback_handler=None
    )

# Create A2A server with a per context agent factory (streaming enabled by default)
a2a_server = A2AServer(agent_factory=create_agent)

# Start the server
a2a_server.serve()
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { Agent } from '@strands-agents/sdk'
import { A2AExpressServer } from '@strands-agents/sdk/a2a/express'

// Build a fresh agent for each A2A context so callers stay isolated
const server = new A2AExpressServer({
  agentFactory: (contextId) =>
    new Agent({
      systemPrompt: 'You are a calculator agent that can perform basic arithmetic.',
    }),
  name: 'Calculator Agent',
  description: 'A calculator agent that can perform basic arithmetic operations.',
})

await server.serve()
```
(( /tab "TypeScript" ))

The server serves the agent card at `/.well-known/agent-card.json` and handles JSON-RPC requests at the root path. Streaming is supported by default.

### Conversation Isolation

The A2A protocol identifies each conversation with a `context_id`. The server isolates conversation state per `context_id` so that callers in different contexts never read or influence each other’s history. Two modes control how that isolation works.

The recommended mode is `agent_factory` - you provide a callable that takes a `context_id` and returns a dedicated agent for each conversation, reusing it for later requests in that context. This enables different conversations to run concurrently because each owns an independent agent.

The factory is also where you wire per-conversation concerns such as a `session_manager` to persist that conversation’s history.

(( tab "Python" ))
```python
from strands import Agent
from strands.multiagent.a2a import A2AServer

def create_agent(context_id: str) -> Agent:
    return Agent(
        name="Calculator Agent",
        description="A calculator agent.",
        callback_handler=None
    )

a2a_server = A2AServer(agent_factory=create_agent)
a2a_server.serve()
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { Agent, SessionManager, FileStorage } from '@strands-agents/sdk'
import { A2AExpressServer } from '@strands-agents/sdk/a2a/express'

// The factory runs once per contextId and returns a dedicated agent, so each conversation
// is isolated. Wire an optional sessionManager here to persist that conversation's history,
// scoped to the contextId.
const storage = new FileStorage('./sessions')

const server = new A2AExpressServer({
  agentFactory: (contextId) =>
    new Agent({
      name: 'Calculator Agent',
      description: 'A calculator agent.',
      sessionManager: new SessionManager({
        sessionId: contextId,
        storage: { snapshot: storage },
      }),
    }),
  name: 'Calculator Agent',
  maxContexts: 1000,
})

await server.serve()
```
(( /tab "TypeScript" ))

The server retains at most `max_contexts` contexts at once (default 1000). When that cap is exceeded, the server evicts the least recently used context, and a later request that reuses the evicted `context_id` starts a fresh conversation. Tune this cap to bound memory in long running servers.

`context_id` is not an authentication boundary

Contexts are keyed on the client supplied `context_id`. A caller who knows another caller’s `context_id` can attach to that conversation. Multi-tenant deployments must enforce authenticated identity at the transport or gateway layer.

Passing a single `agent` is deprecated

In single-agent mode the server reuses one agent across every context, swapping each context’s saved state on and off the shared instance under a lock, which serializes all requests. A single `agent` with a configured `session_manager` is rejected, because the session manager would persist every context into one interleaved session. Prefer `agent_factory` for any deployment that serves more than one conversation.

### Answering Interrupts

Python only

This feature is currently available in the Python SDK only.

When a tool or hook on the served agent raises an [interrupt](/docs/user-guide/concepts/interrupts/index.md), the task moves to the A2A `input_required` state and waits. The client answers it, and the task resumes the paused tool exactly where it stopped.

Answering an interrupt is the one flow on this page that needs the raw [`a2a-sdk`](https://github.com/a2aproject/a2a-python) client rather than `A2AAgent`. `A2AAgent` speaks in text, so it raises `ValueError` if you pass it interrupt responses, and it drops the `DataPart` carrying the interrupt ids when it reads the reply. The examples below build A2A messages directly.

Each interrupt has a server-generated id, and an answer is bound to the id of the interrupt that raised it. The server advertises the pending interrupts on the `input_required` status message as a `DataPart`, alongside the human-readable `TextPart`:

```json
{
  "kind": "data",
  "data": {
    "interrupts": [
      {
        "interruptId": "v1:tool_call:tu-1:a71adb48-e65d-55fa-b155-0359c9cd3b66",
        "name": "approve_campaign",
        "reason": {"name": "spring-launch"}
      }
    ]
  }
}
```

To answer, send a new message on the same `taskId` containing a `DataPart` that echoes the `interruptId` back with the response:

```json
{
  "kind": "data",
  "data": {
    "interruptResponse": {
      "interruptId": "v1:tool_call:tu-1:a71adb48-e65d-55fa-b155-0359c9cd3b66",
      "response": {"approved": true}
    }
  }
}
```

The `response` becomes the return value of the `interrupt()` call that paused the tool. It is any JSON value except `null`, which the server refuses — a null answer would leave the interrupt unsatisfied and re-raise it. `false` and `0` are fine. Answer several interrupts in one message by sending one `DataPart` for each.

Reading the ids off the status message and answering them:

```python
from a2a.types import DataPart, Part

# The task parked in input_required; read the interrupts it is waiting on.
pending = next(
    part.root.data["interrupts"]
    for part in task.status.message.parts
    if isinstance(part.root, DataPart) and "interrupts" in part.root.data
)

# Answer each one on the same taskId.
answers = [
    Part(root=DataPart(data={
        "interruptResponse": {"interruptId": item["interruptId"], "response": {"approved": True}}
    }))
    for item in pending
]
```

The server rejects an answer it cannot bind, before the agent runs, so a refused answer leaves the interrupt pending and the task still answerable. An answer is rejected when:

-   its `interruptId` does not match an interrupt the task is waiting on
-   no interrupt is pending
-   the same `interruptId` is answered twice in one message
-   the payload is missing `interruptId`, or its `response` is missing or `null`
-   it is sent alongside other content parts

A task with a pending interrupt also rejects an ordinary conversational message — answer the interrupt, or cancel the task.

A `DataPart` without an `interruptResponse` key is unaffected and continues to reach the agent as structured data.

### Server Configuration Options

(( tab "Python" ))
The `A2AServer` constructor accepts several configuration options:

-   `agent_factory`: Callable that takes a `context_id` and returns a fresh agent per context (recommended)
-   `agent`: A single Strands agent reused across contexts (deprecated; prefer `agent_factory`)
-   `max_contexts`: Maximum number of per context agents to retain concurrently (default: 1000, must be at least 1)
-   `host`: Hostname or IP address to bind to (default: “127.0.0.1”)
-   `port`: Port to bind to (default: 9000)
-   `version`: Version of the agent (default: “0.0.1”)
-   `skills`: Custom list of agent skills (default: auto-generated from tools)
-   `http_url`: Public HTTP URL where this agent will be accessible (optional, enables path-based mounting)
-   `serve_at_root`: Forces server to serve at root path regardless of http\_url path (default: False)
-   `task_store`: Custom task storage implementation (defaults to InMemoryTaskStore)
-   `queue_manager`: Custom message queue management (optional)
-   `push_config_store`: Custom push notification configuration storage (optional)
-   `push_sender`: Custom push notification sender implementation (optional)

Provide exactly one of `agent_factory` or `agent`, recommend `agent_factory`.
(( /tab "Python" ))

(( tab "TypeScript" ))
The TypeScript SDK provides two server classes:

-   **`A2AServer`**: Base class that manages the agent card and request handler. Use this when integrating with your own HTTP framework.
-   **`A2AExpressServer`**: Express based server with `serve()` and `createMiddleware()` methods.

The `A2AExpressServer` constructor accepts a config object:

-   `agentFactory`: Callable that takes a `contextId` and returns a fresh agent per context (recommended)
-   `agent`: A single Strands Agent reused across contexts (deprecated; prefer `agentFactory`)
-   `maxContexts`: Maximum number of per context agents to retain concurrently (default: 1000, must be at least 1)
-   `name` (required): Human-readable name for the agent
-   `description`: Description of the agent’s purpose
-   `host`: Host to bind the server to (default: `'127.0.0.1'`)
-   `port`: Port to listen on (default: `9000`)
-   `version`: Version string for the agent card (default: `'0.0.1'`)
-   `httpUrl`: Public URL override for the agent card
-   `skills`: Skills to advertise in the agent card
-   `taskStore`: Task store for persisting task state (defaults to InMemoryTaskStore)
-   `userBuilder`: User builder for authentication (default: no authentication)

Provide exactly one of `agentFactory` or `agent`, recommend `agentFactory`.

```typescript
const server = new A2AExpressServer({
  agentFactory: (contextId) =>
    new Agent({
      systemPrompt: 'You are a helpful agent.',
    }),
  name: 'My Agent',
  description: 'A helpful agent',
  // Retain at most 1000 per context agents; evict least recently used
  maxContexts: 1000,
  host: '0.0.0.0',
  port: 8080,
  version: '1.0.0',
  httpUrl: 'https://my-agent.example.com', // Public URL override
  skills: [
    { id: 'math', name: 'Math', description: 'Performs calculations', tags: [] },
  ],
})

await server.serve()
```
(( /tab "TypeScript" ))

### Advanced Server Customization

(( tab "Python" ))
The `A2AServer` provides access to the underlying FastAPI or Starlette application objects allowing you to further customize server behavior.

```python
from contextlib import asynccontextmanager
from strands import Agent
from strands.multiagent.a2a import A2AServer
import uvicorn

# Create your agent factory and A2A server
def create_agent(context_id: str) -> Agent:
    return Agent(name="My Agent", description="A customizable agent", callback_handler=None)

a2a_server = A2AServer(agent_factory=create_agent)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan with proper error handling."""
    # Startup tasks
    yield  # Application runs here
    # Shutdown tasks

# Access the underlying FastAPI app
# Allows passing keyword arguments to FastAPI constructor for further customization
fastapi_app = a2a_server.to_fastapi_app(app_kwargs={"lifespan": lifespan})
# Add custom middleware, routes, or configuration
fastapi_app.add_middleware(...)

# Or access the Starlette app
# Allows passing keyword arguments to FastAPI constructor for further customization
starlette_app = a2a_server.to_starlette_app(app_kwargs={"lifespan": lifespan})
# Customize as needed

# You can then serve the customized app directly
uvicorn.run(fastapi_app, host="127.0.0.1", port=9000)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
The `A2AExpressServer` exposes a `createMiddleware()` method that returns an Express Router, which you can mount in your own Express app:

```typescript
const express = (await import('express')).default

const server = new A2AExpressServer({
  agentFactory: (contextId) =>
    new Agent({ systemPrompt: 'You are a customizable agent.' }),
  name: 'My Agent',
  description: 'A customizable agent',
})

// Get the A2A middleware as an Express Router
const a2aRouter = server.createMiddleware()

// Create your own Express app with custom routes/middleware
const app = express()
app.get('/health', (_req, res) => {
  res.json({ status: 'ok' })
})
app.use(a2aRouter)

app.listen(9000, '127.0.0.1', () => {
  console.log('Server listening on http://127.0.0.1:9000')
})
```

You can also use an `AbortSignal` for graceful shutdown:

```typescript
const server = new A2AExpressServer({
  agentFactory: (contextId) => new Agent({ systemPrompt: 'You are a helpful agent.' }),
  name: 'My Agent',
})

const controller = new AbortController()
await server.serve({ signal: controller.signal })

// Later, to stop the server:
controller.abort()
```
(( /tab "TypeScript" ))

#### Configurable Request Handler Components

(( tab "Python" ))
The `A2AServer` supports configurable request handler components for advanced customization:

```python
from strands import Agent
from strands.multiagent.a2a import A2AServer
from a2a.server.tasks import TaskStore, PushNotificationConfigStore, PushNotificationSender
from a2a.server.events import QueueManager

# Custom task storage implementation
class CustomTaskStore(TaskStore):
    # Implementation details...
    pass

# Custom queue manager
class CustomQueueManager(QueueManager):
    # Implementation details...
    pass

# Create an agent factory with custom components
def create_agent(context_id: str) -> Agent:
    return Agent(name="My Agent", description="A customizable agent", callback_handler=None)

a2a_server = A2AServer(
    agent_factory=create_agent,
    task_store=CustomTaskStore(),
    queue_manager=CustomQueueManager(),
    push_config_store=MyPushConfigStore(),
    push_sender=MyPushSender()
)
```

**Interface Requirements:**

Custom implementations must follow these interfaces:

-   `task_store`: Must implement `TaskStore` interface from `a2a.server.tasks`
-   `queue_manager`: Must implement `QueueManager` interface from `a2a.server.events`
-   `push_config_store`: Must implement `PushNotificationConfigStore` interface from `a2a.server.tasks`
-   `push_sender`: Must implement `PushNotificationSender` interface from `a2a.server.tasks`
(( /tab "Python" ))

(( tab "TypeScript" ))
The TypeScript `A2AExpressServer` supports a custom `taskStore` for persisting task state:

```typescript
import { Agent } from '@strands-agents/sdk'
import { A2AExpressServer } from '@strands-agents/sdk/a2a/express'

const server = new A2AExpressServer({
  agentFactory: (contextId) => new Agent({ systemPrompt: 'You are a helpful agent.' }),
  name: 'My Agent',
  taskStore: myCustomTaskStore, // Must implement TaskStore from @a2a-js/sdk/server
})
```
(( /tab "TypeScript" ))

#### Path-Based Mounting for Containerized Deployments

(( tab "Python" ))
The `A2AServer` supports automatic path-based mounting for deployment scenarios involving load balancers or reverse proxies. This allows you to deploy agents behind load balancers with different path prefixes.

```python
from strands import Agent
from strands.multiagent.a2a import A2AServer

# Create an agent factory
def create_agent(context_id: str) -> Agent:
    return Agent(
        name="Calculator Agent",
        description="A calculator agent",
        callback_handler=None
    )

# Deploy with path-based mounting
# The agent will be accessible at http://my-alb.amazonaws.com/calculator/
a2a_server = A2AServer(
    agent_factory=create_agent,
    http_url="http://my-alb.amazonaws.com/calculator"
)

# For load balancers that strip path prefixes, use serve_at_root=True
a2a_server_with_root = A2AServer(
    agent_factory=create_agent,
    http_url="http://my-alb.amazonaws.com/calculator",
    serve_at_root=True  # Serves at root even though URL has /calculator path
)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
Use the `httpUrl` option to set the public URL for the agent card. For custom path mounting, use `createMiddleware()` and mount the router at any path in your Express app:

```typescript
import { Agent } from '@strands-agents/sdk'
import { A2AExpressServer } from '@strands-agents/sdk/a2a/express'

const server = new A2AExpressServer({
  agentFactory: (contextId) => new Agent({ systemPrompt: 'A calculator agent.' }),
  name: 'Calculator Agent',
  httpUrl: 'http://my-alb.amazonaws.com/calculator',
})

const express = (await import('express')).default
const app = express()
app.use('/calculator', server.createMiddleware())
app.listen(9000)
```
(( /tab "TypeScript" ))

## Strands A2A Tool

### Installation

To use the A2A client tool, install strands-agents-tools with the A2A extra:

```bash
pip install 'strands-agents-tools[a2a_client]'
```

Strands provides this tool for discovering and interacting with A2A agents without manually writing client code:

```python
import asyncio
import logging
from strands import Agent
from strands_tools.a2a_client import A2AClientToolProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create A2A client tool provider with known agent URLs
# Assuming you have an A2A server running on 127.0.0.1:9000
# known_agent_urls is optional
provider = A2AClientToolProvider(known_agent_urls=["http://127.0.0.1:9000"])

# Create agent with A2A client tools
agent = Agent(tools=provider.tools)

# The agent can now discover and interact with A2A servers
# Standard usage
response = agent("pick an agent and make a sample call")
logger.info(response)

# Alternative Async usage
# async def main():
#     response = await agent.invoke_async("pick an agent and make a sample call")
#     logger.info(response)
# asyncio.run(main())
```

The A2A client tool provides three main capabilities:

-   **Agent Discovery**: Automatically discover available A2A agents and their capabilities
-   **Protocol Communication**: Send messages to A2A agents using the standardized protocol
-   **Natural Language Interface**: Interact with remote agents using natural language commands

## Troubleshooting

If you encounter bugs or need to request features for A2A support:

1.  Check the [A2A documentation](https://a2aproject.github.io/A2A/latest/) for protocol-specific issues
2.  Report Strands-specific issues on [GitHub](https://github.com/strands-agents/harness-sdk/issues/new/choose)
3.  Include relevant error messages and code samples in your reports

## Related pages

- [Agent Workflows: Building Multi-Agent Systems with Strands Agents SDK](/docs/user-guide/concepts/multi-agent/workflow/index.md) (1 shared tag)
- [Graph Multi-Agent Pattern](/docs/user-guide/concepts/multi-agent/graph/index.md) (1 shared tag)
- [Multi-agent Patterns](/docs/user-guide/concepts/multi-agent/multi-agent-patterns/index.md) (1 shared tag)
- [Swarm Multi-Agent Pattern](/docs/user-guide/concepts/multi-agent/swarm/index.md) (1 shared tag)
- [Agents as Tools with Strands Agents SDK](/docs/user-guide/concepts/multi-agent/agents-as-tools/index.md) (1 shared tag)


## Implementation

### Python

- [harness-sdk/strands-py/src/strands/agent/a2a_agent.py](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/agent/a2a_agent.py)
- [harness-sdk/strands-py/src/strands/multiagent/a2a/server.py](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/a2a/server.py)
- [harness-sdk/strands-py/src/strands/multiagent/a2a/executor.py](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/a2a/executor.py)

### TypeScript

- [harness-sdk/strands-ts/src/a2a/a2a-agent.ts](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/a2a/a2a-agent.ts)
- [harness-sdk/strands-ts/src/a2a/server.ts](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/a2a/server.ts)
- [harness-sdk/strands-ts/src/a2a/executor.ts](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/a2a/executor.ts)
