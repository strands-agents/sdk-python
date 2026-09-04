This quickstart guide shows you how to create your first basic Strands agent with TypeScript, add built-in and custom tools to your agent, use different model providers, emit debug logs, and run the agent locally.

After completing this guide you can integrate your agent with a web server or browser, evaluate and improve your agent, along with deploying to production and running at scale.

## Install the SDK

First, ensure that you have Node.js 22+ and npm installed. See the [npm documentation](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm) for installation instructions.

Create a new directory for your project and initialize it:

```bash
mkdir my-agent
cd my-agent
npm init -y
npm pkg set type=module
```

Learn more about the [npm init command](https://docs.npmjs.com/cli/v8/commands/npm-init) and its options.

Next, install the `@strands-agents/sdk` package:

```bash
npm install @strands-agents/sdk
```

The Strands Agents SDK includes optional vended tools that are built-in and production-ready for your agents to use. These tools can be imported directly as follows:

```typescript
import { bash } from '@strands-agents/sdk/vended-tools/bash'
```

## Configuring Credentials

Strands supports many different model providers. By default, agents use the Amazon Bedrock model provider with Claude Sonnet 4.6.

To use the examples in this guide, you’ll need to configure your environment with AWS credentials that have permissions to invoke Claude Sonnet 4.6. You can set up your credentials in several ways:

1.  **Environment variables**: Set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optionally `AWS_SESSION_TOKEN`
2.  **AWS credentials file**: Configure credentials using `aws configure` CLI command
3.  **IAM roles**: If running on AWS services like EC2, ECS, or Lambda, use IAM roles
4.  **Bedrock API keys**: Set the `AWS_BEARER_TOKEN_BEDROCK` environment variable

Make sure your AWS credentials have the necessary permissions to access Amazon Bedrock and invoke Claude Sonnet 4.6.

## Project Setup

Now we’ll continue building out the nodejs project by adding TypeScript to the project where our agent will reside. We’ll use this directory structure:

```plaintext
my-agent/
├── src/
│   └── agent.ts
├── package.json
└── README.md
```

Create the directory: `mkdir src`

Install the dev dependencies:

```bash
npm install --save-dev @types/node typescript
```

And finally our `src/agent.ts` file where the goodies are:

```typescript
// Define a custom tool as a TypeScript function
import { Agent, tool } from '@strands-agents/sdk'
import z from 'zod'

const letterCounter = tool({
  name: 'letter_counter',
  description:
    'Count occurrences of a specific letter in a word. Performs case-insensitive matching.',
  // Zod schema for letter counter input validation
  inputSchema: z
    .object({
      word: z.string().describe('The input word to search in'),
      letter: z.string().describe('The specific letter to count'),
    })
    .refine((data) => data.letter.length === 1, {
      message: "The 'letter' parameter must be a single character",
    }),
  callback: (input) => {
    const { word, letter } = input

    // Convert both to lowercase for case-insensitive comparison
    const lowerWord = word.toLowerCase()
    const lowerLetter = letter.toLowerCase()

    // Count occurrences
    let count = 0
    for (const char of lowerWord) {
      if (char === lowerLetter) {
        count++
      }
    }

    // Return result as string (following the pattern of other tools in this project)
    return `The letter '${letter}' appears ${count} time(s) in '${word}'`
  },
})

// Create an agent with tools with our custom letterCounter tool
const agent = new Agent({
  tools: [letterCounter],
})

// Ask the agent a question that uses the available tools
const message = `Tell me how many letter R's are in the word "strawberry" 🍓`
const result = await agent.invoke(message)
console.log(result.lastMessage)
```

This basic quickstart agent can now count letters in words. The agent automatically determines when to use tools based on the input query and context.

Note

The `tool()` function also accepts plain JSON Schema objects instead of Zod. See [Creating Custom Tools](/docs/user-guide/concepts/tools/custom-tools/index.md) for details.

```mermaid
flowchart LR
    A[Input & Context] --> Loop

    subgraph Loop[" "]
        direction TB
        B["Reasoning (LLM)"] --> C["Tool Selection"]
        C --> D["Tool Execution"]
        D --> B
    end

    Loop --> E[Response]
```

More details can be found in the [Agent Loop](/docs/user-guide/concepts/agents/agent-loop/index.md) documentation.

## Running Agents

Our agent is just TypeScript, so we can run it using Node.js, Bun, Deno, or any TypeScript runtime!

To test our agent, we’ll use [`tsx`](https://tsx.is/) to run the file on Node.js:

```bash
npx tsx src/agent.ts
```

And that’s it! We now have a running agent with powerful tools and abilities in just a few lines of code 🥳.

## Understanding What Agents Did

After running an agent, you can understand what happened during execution by examining the agent’s messages, traces, and metrics. Every agent invocation returns an `AgentResult` object that contains the data the agent used along with comprehensive observability data including [local execution traces](/docs/user-guide/observability-evaluation/metrics/index.md#local-execution-traces) and [metrics](/docs/user-guide/observability-evaluation/metrics/index.md).

```typescript
// Access the agent's message array
const result = await agent.invoke('What is the square root of 144?')
console.log(agent.messages)
```

## Console Output

Agents display their reasoning and responses in real-time to the console by default. You can disable this output by setting `printer: false` when creating your agent:

```typescript
const quietAgent = new Agent({
  tools: [letterCounter],
  printer: false, // Disable console output
})
```

## Model Providers

### Identifying a configured model

Strands defaults to the Bedrock model provider using Claude Sonnet 4.6. The model your agent is using can be retrieved by accessing `model.config`:

```typescript
// Check the model configuration
const myAgent = new Agent()
console.log(myAgent['model'].getConfig().modelId)
// Output: { modelId: 'global.anthropic.claude-sonnet-4-6' }
```

You can specify a different model by creating a model provider instance with specific configurations

### Amazon Bedrock (Default)

For more control over model configuration, you can create a model provider instance:

```typescript
import { BedrockModel } from '@strands-agents/sdk'

// Create a BedrockModel with custom configuration
const bedrockModel = new BedrockModel({
  modelId: 'global.anthropic.claude-opus-4-6-v1',
  region: 'us-west-2',
  temperature: 0.3,
})

const bedrockAgent = new Agent({ model: bedrockModel })
```

For the Amazon Bedrock model provider, AWS credentials are typically defined in `AWS_` prefixed environment variables or configured with the `aws configure` CLI command.

You will also need to enable model access in Amazon Bedrock for the models that you choose to use with your agents, following the [AWS documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access-modify.html) to enable access.

More details in the [Amazon Bedrock Model Provider](/docs/user-guide/concepts/model-providers/amazon-bedrock/index.md) documentation.

### Additional Model Providers

Strands Agents supports several other model providers beyond Amazon Bedrock:

-   **[OpenAI](/docs/user-guide/concepts/model-providers/openai/index.md)** - Access to OpenAI or OpenAI-compatible models
-   **[Google](/docs/user-guide/concepts/model-providers/google/index.md)** - Access to Google’s Gemini models

## Capturing Streamed Data & Events

Strands provides two main approaches to capture streaming events from an agent: async iterators and callback functions.

### Async Iterators

For asynchronous applications (like web servers or APIs), Strands provides an async iterator approach using `stream()`. This is particularly useful with async frameworks like Express, Fastify, or NestJS.

```typescript
// Async function that iterates over streamed agent events
async function processStreamingResponse() {
  const prompt = 'What is 25 * 48 and explain the calculation'

  // Stream the response as it's generated from the agent:
  for await (const event of agent.stream(prompt)) {
    console.log('Event:', event.type)
  }
}

// Run the streaming example
await processStreamingResponse()
```

The async iterator yields the same event types as the callback handler callbacks, including text generation events, tool events, and lifecycle events. This approach is ideal for integrating Strands agents with async web frameworks.

See the [Async Iterators](/docs/user-guide/concepts/streaming/async-iterators/index.md) documentation for full details.

## Next Steps

Ready to learn more? Check out these resources:

-   [Examples](https://github.com/strands-agents/harness-sdk/tree/main/strands-ts/examples) - Examples for many use cases
-   [TypeScript SDK Repository](https://github.com/strands-agents/harness-sdk/tree/main/strands-ts) - Explore the TypeScript SDK source code and contribute
-   [Agent Loop](/docs/user-guide/concepts/agents/agent-loop/index.md) - Learn how Strands agents work under the hood
-   [State](/docs/user-guide/concepts/agents/state/index.md) - Understand how agents maintain context and state across a conversation
-   [Operating Agents in Production](/docs/user-guide/deploy/operating-agents-in-production/index.md) - Taking agents from development to production, operating them responsibly at scale

## Related pages

- [Get started](/docs/user-guide/quickstart/overview/index.md) (1 shared tag)
- [Python Quickstart](/docs/user-guide/quickstart/python/index.md) (1 shared tag)
- [Strands Evaluation Quickstart](/docs/user-guide/evals-sdk/quickstart/index.md) (1 shared tag)
- [Strands Shell Quickstart](/docs/user-guide/shell/quickstart/index.md) (1 shared tag)
- [Red Teaming Quickstart](/docs/user-guide/evals-sdk/red-teaming/quickstart/index.md) (1 shared tag)
- [Voice & Realtime Quickstart](/docs/user-guide/concepts/bidirectional-streaming/quickstart/index.md) (1 shared tag)
