## The Concept: Agents as Tools

“Agents as Tools” is an architectural pattern in AI systems where specialized AI agents are wrapped as callable functions (tools) that can be used by other agents. This creates a hierarchical structure where:

1.  **A primary “orchestrator” agent** handles user interaction and determines which specialized agent to call
2.  **Specialized “tool agents”** perform domain-specific tasks when called by the orchestrator

This approach mimics human team dynamics, where a manager coordinates specialists, each bringing unique expertise to solve complex problems. Rather than a single agent trying to handle everything, tasks are delegated to the most appropriate specialized agent.

## Key Benefits and Core Principles

The “Agents as Tools” pattern offers several advantages:

-   **Separation of concerns**: Each agent has a focused area of responsibility, making the system easier to understand and maintain
-   **Hierarchical delegation**: The orchestrator decides which specialist to invoke, creating a clear chain of command
-   **Modular architecture**: Specialists can be added, removed, or modified independently without affecting the entire system
-   **Improved performance**: Each agent can have tailored system prompts and tools optimized for its specific task

## Strands Agents SDK Best Practices for Agent Tools

When implementing the “Agents as Tools” pattern with Strands Agents SDK:

1.  **Clear tool documentation**: Write descriptive names and descriptions that explain the agent’s expertise
2.  **Focused system prompts**: Keep each specialized agent tightly focused on its domain
3.  **Proper response handling**: Use consistent patterns to extract and format responses
4.  **Tool selection guidance**: Give the orchestrator clear criteria for when to use each specialized agent

## Implementing Agents as Tools with Strands Agents SDK

Strands Agents SDK provides three ways to implement the “Agents as Tools” pattern: passing agents directly in the `tools` array for the simplest setup, `.as_tool()``.asTool()` when you need to customize tool name, description, or context behavior, and the `@tool` decorator or `tool()` function for full control over how the agent is invoked.

```mermaid
flowchart TD
    User([User]) <--> Orchestrator["Orchestrator Agent"]
    Orchestrator --> RA["Research Assistant"]
    Orchestrator --> PA["Product Recommendation Assistant"]
    Orchestrator --> TA["Trip Planning Assistant"]

    RA --> Orchestrator
    PA --> Orchestrator
    TA --> Orchestrator
```

### Passing Agents Directly

The simplest way to use an agent as a tool is to pass it directly in the `tools` array. The SDK automatically converts it into a tool that accepts an `input` string parameter and returns the agent’s text response.

(( tab "Python" ))
```python
from strands import Agent
from strands_tools import retrieve, http_request

# Create specialized agents
research_agent = Agent(
    system_prompt="""You are a specialized research assistant. Focus only on providing
    factual, well-sourced information in response to research questions.
    Always cite your sources when possible.""",
    tools=[retrieve, http_request],
)

product_agent = Agent(
    system_prompt="""You are a specialized product recommendation assistant.
    Provide personalized product suggestions based on user preferences.""",
    tools=[retrieve, http_request],
)

travel_agent = Agent(
    system_prompt="""You are a specialized travel planning assistant.
    Create detailed travel itineraries based on user preferences.""",
    tools=[retrieve, http_request],
)

# Create the orchestrator — agents are automatically converted to tools
orchestrator = Agent(
    system_prompt="""You are an assistant that routes queries to specialized agents:
    - For research questions and factual information → Use the research_agent tool
    - For product recommendations and shopping advice → Use the product_agent tool
    - For travel planning and itineraries → Use the travel_agent tool
    - For simple questions not requiring specialized knowledge → Answer directly

    Always select the most appropriate tool based on the user's query.""",
    tools=[research_agent, product_agent, travel_agent],
)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
// Create specialized agents
const researchAgent = new Agent({
  name: 'research_agent',
  description:
    'Provides factual, well-sourced information in response to research questions.',
  systemPrompt: `You are a specialized research assistant. Focus only on providing
factual, well-sourced information in response to research questions.
Always cite your sources when possible.`,
  printer: false,
})

const productAgent = new Agent({
  name: 'product_agent',
  description: 'Provides personalized product suggestions based on user preferences.',
  systemPrompt: `You are a specialized product recommendation assistant.
Provide personalized product suggestions based on user preferences.`,
  printer: false,
})

const travelAgent = new Agent({
  name: 'travel_agent',
  description: 'Creates detailed travel itineraries based on user preferences.',
  systemPrompt: `You are a specialized travel planning assistant.
Create detailed travel itineraries based on user preferences.`,
  printer: false,
})

// Create the orchestrator — agents are automatically converted to tools
const orchestrator = new Agent({
  systemPrompt: `You are an assistant that routes queries to specialized agents:
- For research questions and factual information → Use the research_agent tool
- For product recommendations and shopping advice → Use the product_agent tool
- For travel planning and itineraries → Use the travel_agent tool
- For simple questions not requiring specialized knowledge → Answer directly

Always select the most appropriate tool based on the user's query.`,
  tools: [researchAgent, productAgent, travelAgent],
})
```
(( /tab "TypeScript" ))

### Customizing Agent Tools

When you need to customize the tool name, description, or context behavior, use `.as_tool()``.asTool()` explicitly:

(( tab "Python" ))
```python
orchestrator = Agent(
    system_prompt="You are an assistant that routes queries to specialized agents.",
    tools=[
        research_agent.as_tool(
            name="research_assistant",
            description="Process and respond to research-related queries requiring factual information.",
        ),
    ],
)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
const orchestrator = new Agent({
  systemPrompt: 'You are an assistant that routes queries to specialized agents.',
  tools: [
    researchAgent.asTool({
      name: 'research_assistant',
      description:
        'Process and respond to research-related queries requiring factual information.',
    }),
  ],
})
```
(( /tab "TypeScript" ))

#### Context Management

By default, both direct passing and `.as_tool()``.asTool()` reset the agent’s conversation context between invocations, ensuring every call starts from a clean baseline. To preserve the agent’s conversation history across invocations:

(( tab "Python" ))
```python
# Agent will remember prior interactions within the same orchestrator session
orchestrator = Agent(
    system_prompt="You are an assistant that routes queries to specialized agents.",
    tools=[research_agent.as_tool(preserve_context=True)],
)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
// Agent will remember prior interactions within the same orchestrator session
const orchestrator = new Agent({
  systemPrompt: 'You are an assistant that routes queries to specialized agents.',
  tools: [researchAgent.asTool({ preserveContext: true })],
})
```
(( /tab "TypeScript" ))

### Creating Custom Agent Tools

For more control over how the agent is invoked — such as custom pre/post-processing, error handling, or passing multiple parameters — you can create a custom tool that wraps an agent:

(( tab "Python" ))
```python
from strands import Agent, tool
from strands_tools import retrieve, http_request

RESEARCH_ASSISTANT_PROMPT = """
You are a specialized research assistant. Focus only on providing
factual, well-sourced information in response to research questions.
Always cite your sources when possible.
"""

@tool
def research_assistant(query: str) -> str:
    """
    Process and respond to research-related queries.

    Args:
        query: A research question requiring factual information

    Returns:
        A detailed research answer with citations
    """
    try:
        research_agent = Agent(
            system_prompt=RESEARCH_ASSISTANT_PROMPT,
            tools=[retrieve, http_request]
        )

        response = research_agent(query)
        return str(response)
    except Exception as e:
        return f"Error in research assistant: {str(e)}"
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
const researchAssistant = tool({
  name: 'research_assistant',
  description:
    'Process and respond to research-related queries requiring factual information.',
  inputSchema: z.object({
    query: z.string().describe('A research question requiring factual information'),
  }),
  callback: async (input) => {
    const researchAgent = new Agent({
      systemPrompt: `You are a specialized research assistant. Focus only on providing
factual, well-sourced information in response to research questions.
Always cite your sources when possible.`,
    })

    const response = await researchAgent.invoke(input.query)
    return response.lastMessage.content
      .map((block) => ('text' in block ? block.text : ''))
      .join('')
  },
})
```
(( /tab "TypeScript" ))

You can create multiple specialized agents following the same pattern:

(( tab "Python" ))
```python
@tool
def product_recommendation_assistant(query: str) -> str:
    """
    Handle product recommendation queries by suggesting appropriate products.

    Args:
        query: A product inquiry with user preferences

    Returns:
        Personalized product recommendations with reasoning
    """
    try:
        product_agent = Agent(
            system_prompt="""You are a specialized product recommendation assistant.
            Provide personalized product suggestions based on user preferences.""",
            tools=[retrieve, http_request, dialog],
        )
        # Implementation with response handling
        # ...
        return processed_response
    except Exception as e:
        return f"Error in product recommendation: {str(e)}"

@tool
def trip_planning_assistant(query: str) -> str:
    """
    Create travel itineraries and provide travel advice.

    Args:
        query: A travel planning request with destination and preferences

    Returns:
        A detailed travel itinerary or travel advice
    """
    try:
        travel_agent = Agent(
            system_prompt="""You are a specialized travel planning assistant.
            Create detailed travel itineraries based on user preferences.""",
            tools=[retrieve, http_request],
        )
        # Implementation with response handling
        # ...
        return processed_response
    except Exception as e:
        return f"Error in trip planning: {str(e)}"
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
const productRecommendationAssistant = tool({
  name: 'product_recommendation_assistant',
  description:
    'Handle product recommendation queries by suggesting appropriate products.',
  inputSchema: z.object({
    query: z.string().describe('A product inquiry with user preferences'),
  }),
  callback: async (input) => {
    const productAgent = new Agent({
      systemPrompt: `You are a specialized product recommendation assistant.
Provide personalized product suggestions based on user preferences.`,
    })

    const response = await productAgent.invoke(input.query)
    return response.lastMessage.content
      .map((block) => ('text' in block ? block.text : ''))
      .join('')
  },
})

const tripPlanningAssistant = tool({
  name: 'trip_planning_assistant',
  description: 'Create travel itineraries and provide travel advice.',
  inputSchema: z.object({
    query: z
      .string()
      .describe('A travel planning request with destination and preferences'),
  }),
  callback: async (input) => {
    const travelAgent = new Agent({
      systemPrompt: `You are a specialized travel planning assistant.
Create detailed travel itineraries based on user preferences.`,
    })

    const response = await travelAgent.invoke(input.query)
    return response.lastMessage.content
      .map((block) => ('text' in block ? block.text : ''))
      .join('')
  },
})
```
(( /tab "TypeScript" ))

#### Creating the Orchestrator Agent

Create an orchestrator agent that has access to all specialized agents as tools:

(( tab "Python" ))
```python
from strands import Agent
from .specialized_agents import research_assistant, product_recommendation_assistant, trip_planning_assistant

MAIN_SYSTEM_PROMPT = """
You are an assistant that routes queries to specialized agents:
- For research questions and factual information → Use the research_assistant tool
- For product recommendations and shopping advice → Use the product_recommendation_assistant tool
- For travel planning and itineraries → Use the trip_planning_assistant tool
- For simple questions not requiring specialized knowledge → Answer directly

Always select the most appropriate tool based on the user's query.
"""

orchestrator = Agent(
    system_prompt=MAIN_SYSTEM_PROMPT,
    callback_handler=None,
    tools=[research_assistant, product_recommendation_assistant, trip_planning_assistant]
)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
const orchestrator = new Agent({
    systemPrompt: `You are an assistant that routes queries to specialized agents:
- For research questions and factual information → Use the research_assistant tool
- For recommendations and advice → Use the product_recommendation_assistant tool
- For travel planning and itineraries → Use the trip_planning_assistant tool
- For simple questions not requiring specialized knowledge → Answer directly

Always select the most appropriate tool based on the user's query.`,
    tools: [researchAssistant, productRecommendationAssistant, tripPlanningAssistant],
  })
```
(( /tab "TypeScript" ))

Here’s how this multi-agent system might handle a complex user query:

(( tab "Python" ))
```python
# Example: E-commerce Customer Service System
customer_query = "I'm looking for hiking boots for a trip to Patagonia next month"

# The orchestrator automatically determines that this requires multiple specialized agents
response = orchestrator(customer_query)

# Behind the scenes, the orchestrator will:
# 1. First call the trip_planning_assistant to understand travel requirements for Patagonia
#    - Weather conditions in the region next month
#    - Typical terrain and hiking conditions
# 2. Then call product_recommendation_assistant with this context to suggest appropriate boots
#    - Waterproof options for potential rain
#    - Proper ankle support for uneven terrain
#    - Brands known for durability in harsh conditions
# 3. Combine these specialized responses into a cohesive answer that addresses both the
#    travel planning and product recommendation aspects of the query
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
const response = await orchestrator.invoke(
  "I'm looking for hiking boots for a trip to Patagonia next month"
)
```
(( /tab "TypeScript" ))

This example demonstrates how Strands Agents SDK enables specialized experts to collaborate on complex queries requiring multiple domains of knowledge. The orchestrator intelligently routes different aspects of the query to the appropriate specialized agents, then synthesizes their responses into a comprehensive answer.

## Delegation

Sometimes, you do not want a sub-agent’s response to be re-processed by the orchestrator agent, since it would incur unnecessary token usage and potentially corrupt the sub-agent’s response. For example:

-   A coding agent returning generated source code that shouldn’t be rephrased
-   A customer service agent whose compliance-reviewed language must reach the user unchanged
-   A retrieval agent returning structured data the caller consumes directly

In that case, you can mark a tool agent as a delegate by passing `delegate: true` to `.as_tool()``.asTool()`. When you do so, the orchestrator agent returns the delegate’s response directly to the user, skipping the extra orchestrator model round-trip.

(( tab "Python" ))
```python
from pydantic import BaseModel
from strands import Agent


class BillingResponse(BaseModel):
    summary: str
    refund_amount: float


billing_agent = Agent(
    name="billing_expert",
    description="Answers billing questions: charges, refunds, and invoices.",
    system_prompt="You handle billing questions with precision.",
    structured_output_model=BillingResponse,
)

orchestrator = Agent(
    system_prompt="""Route billing questions to billing_expert.
Answer general questions yourself.""",
    tools=[billing_agent.as_tool(delegate=True)],
)

result = orchestrator("Why was I charged twice?")
# result contains the billing agent's structured JSON response, unchanged
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
const BillingResponseSchema = z.object({
    summary: z.string(),
    refundAmount: z.number(),
  })

  const billingAgent = new Agent({
    name: 'billing_expert',
    description: 'Answers billing questions: charges, refunds, and invoices.',
    systemPrompt: 'You handle billing questions with precision.',
    structuredOutputSchema: BillingResponseSchema,
    printer: false,
  })

  const orchestrator = new Agent({
    systemPrompt: `Route billing questions to billing_expert.
Answer general questions yourself.`,
    tools: [billingAgent.asTool({ delegate: true })],
  })

  const result = await orchestrator.invoke('Why was I charged twice?')
  // result contains the billing agent's structured JSON response, unchanged
```
(( /tab "TypeScript" ))

On success, the agent loop stops with `stop_reason == 'end_turn'``stopReason === 'endTurn'` and the `AgentResult` contains the delegate’s content without additional model processing. Streaming events from the delegate surface natively in the parent’s stream, so callers see the specialist’s tokens as they arrive.

Limitations to be aware of:

-   **Single delegated tool per turn**: An agent can have both delegated and non-delegated tools. However, if the agent calls a delegated tool during a turn, that tool must be called alone. The SDK cancels the tool batch if other tools are requested alongside the delegated tool.
-   **Incompatible with stateful models**: Delegation exits the agent loop early, which would leave an unclosed function call on the server for models that manage conversation state server-side. Combining delegated tools with stateful models is not supported.

## Remote Agents with A2A

You can also use remote agents as tools through the [Agent-to-Agent (A2A) protocol](/docs/user-guide/concepts/multi-agent/agent-to-agent/index.md). The `A2AAgent` class lets you wrap a remote A2A-compatible agent as a tool in your orchestrator, following the same pattern described above but communicating over the network. See [A2AAgent as a Tool](/docs/user-guide/concepts/multi-agent/agent-to-agent/index.md#as-a-tool) for details.

## Complete Working Example

For complete implementations of this pattern, see the following examples:

(( tab "Python" ))
The [Teacher’s Assistant](/docs/examples/python/multi_agent_example/multi_agent_example/index.md) example demonstrates an orchestrator agent that routes student queries to specialized agents for math, English, language translation, computer science, and general knowledge.
(( /tab "Python" ))

(( tab "TypeScript" ))
The [Agents as Tools](https://github.com/strands-agents/harness-sdk/tree/main/strands-ts/examples/agents-as-tools) example demonstrates an orchestrator agent that routes student queries to specialized tool agents for math, English, computer science, and general knowledge.
(( /tab "TypeScript" ))

## Related pages

- [Agent Workflows: Building Multi-Agent Systems with Strands Agents SDK](/docs/user-guide/concepts/multi-agent/workflow/index.md) (1 shared tag)
- [Agent-to-Agent (A2A) Protocol](/docs/user-guide/concepts/multi-agent/agent-to-agent/index.md) (1 shared tag)
- [Graph Multi-Agent Pattern](/docs/user-guide/concepts/multi-agent/graph/index.md) (1 shared tag)
- [Multi-agent Patterns](/docs/user-guide/concepts/multi-agent/multi-agent-patterns/index.md) (1 shared tag)
- [Swarm Multi-Agent Pattern](/docs/user-guide/concepts/multi-agent/swarm/index.md) (1 shared tag)
- [Community Built Tools](/docs/user-guide/concepts/tools/community-tools-package/index.md) (1 shared tag)
- [Creating Custom Tools](/docs/user-guide/concepts/tools/custom-tools/index.md) (1 shared tag)
- [Vended Tools](/docs/user-guide/concepts/tools/vended-tools/index.md) (1 shared tag)
- [Creating a Custom Model Provider](/docs/user-guide/concepts/model-providers/custom_model_provider/index.md) (1 shared tag)
- [Build with AI](/docs/user-guide/build-with-ai/index.md) (1 shared tag)


## Implementation

### Python

- [harness-sdk/strands-py/src/strands/agent/_agent_as_tool.py](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/agent/_agent_as_tool.py)
- [harness-sdk/strands-py/src/strands/agent/_agent_delegation.py](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/agent/_agent_delegation.py)

### TypeScript

- [harness-sdk/strands-ts/src/agent/agent-as-tool.ts](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/agent/agent-as-tool.ts)
- [harness-sdk/strands-ts/src/agent/agent-delegation.ts](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/agent/agent-delegation.ts)
