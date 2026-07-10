# Harness Python v1.25.0

Released 2026-02-05
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.25.0 · Package: https://pypi.org/project/strands-agents/1.25.0/

## Features
- add A2AAgent class (https://github.com/strands-agents/sdk-python/pull/1441)
- make structured output prompt message configurable (#1288) (https://github.com/strands-agents/sdk-python/pull/1627)
- Add AgentBase support for A2AAgent compatibility (https://github.com/strands-agents/sdk-python/pull/1615)

## Fixes
- preserve nullable semantics for required Union\[T, None\] params (https://github.com/strands-agents/sdk-python/pull/1584)
- LedgerProvider handles parallel tool calls (https://github.com/strands-agents/sdk-python/pull/1559)
- Handles Bedrock-style context overflow errors for OpenAI-compatible endpoints (https://github.com/strands-agents/sdk-python/pull/1529)
- Update retry\_strategy=None to turn off retries (https://github.com/strands-agents/sdk-python/pull/1630)
- update agent card URL when host/port overridden in A2AServer.ser… (https://github.com/strands-agents/sdk-python/pull/1626)

## Other
- Increase pytest timeout to 45 seconds (https://github.com/strands-agents/sdk-python/pull/1586)
- Publish integ tests results to cloudwatch (https://github.com/strands-agents/sdk-python/pull/1587)
- Feature: Allow s3Location as Document, Image, and Video location source (https://github.com/strands-agents/sdk-python/pull/1572)
- Clone main metrics upload script for integ tests (https://github.com/strands-agents/sdk-python/pull/1600)
- Skip location for non bedrock model providers (https://github.com/strands-agents/sdk-python/pull/1602)
- Add conditional execution for finalize step (https://github.com/strands-agents/sdk-python/pull/1605)
- interrupts - graph - multiagent nodes (https://github.com/strands-agents/sdk-python/pull/1606)
- fix various test warnings (https://github.com/strands-agents/sdk-python/pull/1613)
- Fix bedrock file warnings (https://github.com/strands-agents/sdk-python/pull/1603)
- increase test timeout (https://github.com/strands-agents/sdk-python/pull/1623)
- Fix openai test (https://github.com/strands-agents/sdk-python/pull/1624)
- bump actions/setup-python from 4 to 6 (https://github.com/strands-agents/sdk-python/pull/1548)
- bump aws-actions/configure-aws-credentials from 4 to 5 (https://github.com/strands-agents/sdk-python/pull/1547)
- bump actions/download-artifact from 4 to 7 (https://github.com/strands-agents/sdk-python/pull/1609)
- bump actions/upload-artifact from 4 to 6 (https://github.com/strands-agents/sdk-python/pull/1608)
- remove broken MCP transport timeout test (https://github.com/strands-agents/sdk-python/pull/1635)

## Major Features

### A2AAgent: First-Class Client for Remote A2A Agents - [PR#1441](https://github.com/strands-agents/sdk-python/pull/1441)

The new `A2AAgent` class makes it simple to connect to and invoke remote agents that implement the [Agent-to-Agent (A2A) protocol](https://github.com/google/a2a). `A2AAgent` implements the `AgentBase` protocol, so it can be called synchronously, asynchronously, or used in streaming mode just like a local `Agent`. It automatically discovers the remote agent's card to populate its name and description, and manages HTTP client lifecycle for you.

```python
from strands.agent.a2a_agent import A2AAgent

# Connect to a remote A2A agent
a2a_agent = A2AAgent(endpoint="http://localhost:9000")

# Invoke it like any other agent
result = a2a_agent("Show me 10 ^ 6")
print(result.message)

# Or stream events asynchronously
async for event in a2a_agent.stream_async("Summarize this report"):
    if event.get("type") == "a2a_stream":
        print(f"A2A event: {event['event']}")
    elif "result" in event:
        print(f"Final: {event['result'].message}")
```

### A2AAgent Support in Graph Workflows - [PR#1615](https://github.com/strands-agents/sdk-python/pull/1615)

Graph nodes now accept any `AgentBase` implementation as an executor, not just the concrete `Agent` class. This means `A2AAgent` instances and other custom `AgentBase` implementations can participate in graph-based multi-agent workflows alongside local agents. The `Agent` class also now explicitly extends `AgentBase` for compile-time protocol verification.

```python
from strands import Agent
from strands.agent.a2a_agent import A2AAgent
from strands.multiagent.graph import GraphBuilder

local_agent = Agent(name="summarizer", system_prompt="Summarize input concisely.")
remote_agent = A2AAgent(endpoint="http://remote-agent:9000")

builder = GraphBuilder()
builder.add_node(remote_agent, "research")
builder.add_node(local_agent, "summarize")
builder.add_edge("research", "summarize")
graph = builder.build()

result = graph("Analyze recent AI trends")
```

### Interrupt Support for MultiAgent Graph Nodes - [PR#1606](https://github.com/strands-agents/sdk-python/pull/1606)

Interrupts now propagate correctly through nested multi-agent graph nodes. When an agent inside a nested `Graph`, `Swarm`, or any custom `MultiAgentBase` node raises an interrupt, the outer graph pauses execution and surfaces the interrupt to the caller. After the caller provides a response, the outer graph resumes execution from where it left off. This builds on the interrupt support for single-agent graph nodes added in v1.24.0.

```python
from strands import Agent, tool
from strands.interrupt import Interrupt
from strands.multiagent import GraphBuilder, Status
from strands.types.tools import ToolContext

@tool(context=True)
def approval_tool(tool_context: ToolContext) -> str:
    return tool_context.interrupt("approval", reason="Needs human approval")

agent = Agent(name="reviewer", tools=[approval_tool])

inner_graph = GraphBuilder()
inner_graph.add_node(agent, "reviewer")
outer_graph = GraphBuilder()
outer_graph.add_node(inner_graph.build(), "review_pipeline")

graph = outer_graph.build()
result = graph("Review this document")

while result.status == Status.INTERRUPTED:
    responses = [
        {"interruptResponse": {"interruptId": i.id, "response": "Approved"}}
        for i in result.interrupts
    ]
    result = graph(responses)
```

### S3 Location Support for Documents, Images, and Videos - [PR#1572](https://github.com/strands-agents/sdk-python/pull/1572)

Media content types now support S3 locations as a source, allowing you to reference documents, images, and videos stored in Amazon S3 directly without base64 encoding. The new `S3Location` type includes a required `uri` field and an optional `bucketOwner` field for cross-account access. On Bedrock, S3 locations are passed through to the API natively.

```python
from strands import Agent

agent = Agent()

response = agent([{
    "role": "user",
    "content": [
        {"text": "Summarize this document:"},
        {
            "document": {
                "format": "pdf",
                "name": "report",
                "source": {
                    "location": {
                        "type": "s3",
                        "uri": "s3://my-bucket/documents/report.pdf",
                        "bucketOwner": "123456789012"  # optional, for cross-account
                    }
                }
            }
        }
    ]
}])
```

### Configurable Structured Output Prompt - [PR#1627](https://github.com/strands-agents/sdk-python/pull/1627)

The prompt message the agent uses to request structured output formatting is now configurable via the `structured_output_prompt` parameter. Previously, the hardcoded message `"You must format the previous response as structured output."` could trigger Bedrock Guardrails prompt-attack filters. You can now customize this message at the agent level or per invocation to work around guardrail rules or to better suit your use case.

```python
from strands import Agent
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    age: int

# Custom prompt avoids triggering Bedrock Guardrails prompt attack filter
agent = Agent(
    structured_output_model=UserInfo,
    structured_output_prompt="Please use the output tool now."
)

# Or override per invocation
result = agent(
    "Extract user info from: John is 30 years old",
    structured_output_prompt="Format the response using the output tool."
)
```

---

## Major Bug Fixes

* **Nullable Semantics Preserved for Required `Union[T, None]` Tool Parameters** - [PR#1584](https://github.com/strands-agents/sdk-python/pull/1584)
  When a `@tool` parameter was typed as `T | None` without a default value, the schema cleaning logic was stripping `null` from the `anyOf` array, making the field required with no way to express null. This caused LLMs to fall back to passing `"null"` as a string. The `anyOf` simplification is now skipped for fields in the `required` array, preserving proper nullable semantics.

* **LedgerProvider Now Handles Parallel Tool Calls Correctly** - [PR#1559](https://github.com/strands-agents/sdk-python/pull/1559)
  When agents proposed multiple tool calls in a single model response, the ledger provider only updated the last tool in the batch (`ledger['tool_calls'][-1]`), leaving earlier pending tools without proper status updates. The provider now correctly tracks and updates each individual tool call.

* **Context Overflow Detection for OpenAI-Compatible Bedrock Endpoints** - [PR#1529](https://github.com/strands-agents/sdk-python/pull/1529)
  OpenAI-compatible endpoints that wrap Bedrock models (e.g., Databricks Model Serving) return Bedrock-style error messages like `"Input is too long for requested model"` as `APIError` instead of `BadRequestError` with code `context_length_exceeded`. These errors are now detected and converted to `ContextWindowOverflowException`, allowing conversation managers like `SummarizingConversationManager` to trigger properly.

* **`retry_strategy=None` Now Disables Retries** - [PR#1630](https://github.com/strands-agents/sdk-python/pull/1630)
  Passing `retry_strategy=None` to the Agent constructor previously applied the default retry strategy instead of disabling retries. Now `None` explicitly disables all SDK retries (equivalent to `ModelRetryStrategy(max_attempts=1)`), while omitting the parameter entirely still applies the default strategy. **Note:** This is a minor breaking change in behavior for code that explicitly passed `None`.

* **A2A Server Agent Card URL Updated on Host/Port Override** - [PR#1626](https://github.com/strands-agents/sdk-python/pull/1626)
  When overriding `host` and `port` in `A2AServer.serve()`, the agent card at `/.well-known/agent-card.json` still returned the original URL from constructor defaults, causing A2A clients to connect to the wrong address. The server now updates the agent card URL to match the overridden host/port, unless an explicit `http_url` was provided in the constructor.

---

## Minor Changes

* Increase pytest timeout to 45 seconds - [PR#1586](https://github.com/strands-agents/sdk-python/pull/1586)
* Publish integ test results to CloudWatch - [PR#1587](https://github.com/strands-agents/sdk-python/pull/1587)
* Clone main metrics upload script for integ tests - [PR#1600](https://github.com/strands-agents/sdk-python/pull/1600)
* Skip S3 location for non-Bedrock model providers - [PR#1602](https://github.com/strands-agents/sdk-python/pull/1602)
* Add conditional execution for finalize step - [PR#1605](https://github.com/strands-agents/sdk-python/pull/1605)
* Fix various test warnings - [PR#1613](https://github.com/strands-agents/sdk-python/pull/1613)
* Fix Bedrock file warnings - [PR#1603](https://github.com/strands-agents/sdk-python/pull/1603)
* Increase test timeout - [PR#1623](https://github.com/strands-agents/sdk-python/pull/1623)
* Fix OpenAI test - [PR#1624](https://github.com/strands-agents/sdk-python/pull/1624)
* Remove broken MCP transport timeout test - [PR#1635](https://github.com/strands-agents/sdk-python/pull/1635)
* Bump `actions/setup-python` from 4 to 6 - [PR#1548](https://github.com/strands-agents/sdk-python/pull/1548)
* Bump `aws-actions/configure-aws-credentials` from 4 to 5 - [PR#1547](https://github.com/strands-agents/sdk-python/pull/1547)
* Bump `actions/download-artifact` from 4 to 7 - [PR#1609](https://github.com/strands-agents/sdk-python/pull/1609)
* Bump `actions/upload-artifact` from 4 to 6 - [PR#1608](https://github.com/strands-agents/sdk-python/pull/1608)

---
