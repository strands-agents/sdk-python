## What are Model Providers?

A model provider is a service or platform that hosts and serves large language models through an API. The Strands Agents SDK abstracts away the complexity of working with different providers, offering a unified interface that makes it easy to switch between models or use multiple providers in the same application.

## Supported Providers

The following table shows all model providers supported by Strands Agents SDK and their availability in Python and TypeScript:

| Provider | Python Supported | TypeScript Supported |
| --- | --- | --- |
| [Amazon Bedrock](/docs/user-guide/concepts/model-providers/amazon-bedrock/index.md) | ✅ | ✅ |
| [Amazon Nova](/docs/user-guide/concepts/model-providers/amazon-nova/index.md) | ✅ | ❌ |
| [Anthropic](/docs/user-guide/concepts/model-providers/anthropic/index.md) | ✅ | ✅ |
| [Custom Providers](/docs/user-guide/concepts/model-providers/custom_model_provider/index.md) | ✅ | ✅ |
| [Google](/docs/user-guide/concepts/model-providers/google/index.md) | ✅ | ✅ |
| [LiteLLM](/docs/user-guide/concepts/model-providers/litellm/index.md) | ✅ | ❌ |
| [llama.cpp](/docs/user-guide/concepts/model-providers/llamacpp/index.md) | ✅ | ❌ |
| [LlamaAPI](/docs/user-guide/concepts/model-providers/llamaapi/index.md) | ✅ | ❌ |
| [MistralAI](/docs/user-guide/concepts/model-providers/mistral/index.md) | ✅ | ❌ |
| [Ollama](/docs/user-guide/concepts/model-providers/ollama/index.md) | ✅ | ❌ |
| [OpenAI](/docs/user-guide/concepts/model-providers/openai/index.md) | ✅ | ✅ |
| [OpenAI Responses API](/docs/user-guide/concepts/model-providers/openai-responses/index.md) | ✅ | ✅ |
| [SageMaker](/docs/user-guide/concepts/model-providers/sagemaker/index.md) | ✅ | ❌ |
| [Vercel](/docs/user-guide/concepts/model-providers/vercel/index.md) | ❌ | ✅ |
| [Writer](/docs/user-guide/concepts/model-providers/writer/index.md) | ✅ | ❌ |

### Community providers

The following providers are built and maintained by the Strands community. Browse the [integrations page](/integrations/index.md) to explore additional community packages.

| Provider | Python Supported | TypeScript Supported |
| --- | --- | --- |
| [CLOVA Studio](/docs/integrations/model-providers/clova-studio/index.md) | ✅ | ❌ |
| [Cohere](/docs/integrations/model-providers/cohere/index.md) | ✅ | ❌ |
| [Crusoe](/docs/integrations/model-providers/crusoe/index.md) | ✅ | ❌ |
| [Fireworks AI](/docs/integrations/model-providers/fireworksai/index.md) | ✅ | ❌ |
| [MLX](/docs/integrations/model-providers/mlx/index.md) | ✅ | ❌ |
| [Nebius Token Factory](/docs/integrations/model-providers/nebius-token-factory/index.md) | ✅ | ❌ |
| [NVIDIA NIM](/docs/integrations/model-providers/nvidia-nim/index.md) | ✅ | ❌ |
| [OVHcloud AI Endpoints](/docs/integrations/model-providers/ovhcloud-ai-endpoints/index.md) | ✅ | ❌ |
| [SGLang](/docs/integrations/model-providers/sglang/index.md) | ✅ | ❌ |
| [vLLM](/docs/integrations/model-providers/vllm/index.md) | ✅ | ❌ |
| [xAI](/docs/integrations/model-providers/xai/index.md) | ✅ | ❌ |

## Getting Started

### Installation

Most providers are available as optional dependencies. Install the provider you need:

(( tab "Python" ))
```bash
# Install with specific provider
pip install 'strands-agents[bedrock]'
pip install 'strands-agents[openai]'
pip install 'strands-agents[anthropic]'

# Or install with all providers
pip install 'strands-agents[all]'
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```bash
# Core SDK includes BedrockModel by default
npm install @strands-agents/sdk

# To use OpenAI, install the openai package
npm install openai
```

> **Note:** All model providers except Bedrock are listed as optional dependencies in the SDK. This means npm will attempt to install them automatically, but won’t fail if they’re unavailable. You can explicitly install them when needed.
(( /tab "TypeScript" ))

### Basic Usage

Each provider follows a similar pattern for initialization and usage. Models are interchangeable - you can easily switch between providers by changing the model instance:

(( tab "Python" ))
```python
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.models.openai import OpenAIModel

# Use Bedrock
bedrock_model = BedrockModel()
agent = Agent(model=bedrock_model)
response = agent("What can you help me with?")

# Alternatively, use OpenAI by just switching model provider
openai_model = OpenAIModel(
    client_args={"api_key": "<KEY>"},
    model_id="gpt-4o"
)
agent = Agent(model=openai_model)
response = agent("What can you help me with?")
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { Agent } from '@strands-agents/sdk'
import { BedrockModel } from '@strands-agents/sdk/models/bedrock'
import { OpenAIModel } from '@strands-agents/sdk/models/openai'

// Use Bedrock
const bedrockModel = new BedrockModel()
let agent = new Agent({ model: bedrockModel })
let response = await agent.invoke('What can you help me with?')

// Alternatively, use OpenAI by just switching model provider
const openaiModel = new OpenAIModel({
  api: 'chat',
  apiKey: process.env.OPENAI_API_KEY,
  modelId: 'gpt-5.4',
})
agent = new Agent({ model: openaiModel })
response = await agent.invoke('What can you help me with?')
```
(( /tab "TypeScript" ))

## Next Steps

### Explore Model Providers

-   **[Amazon Bedrock](/docs/user-guide/concepts/model-providers/amazon-bedrock/index.md)** - Default provider with wide model selection, enterprise features, and full Python/TypeScript support
-   **[OpenAI](/docs/user-guide/concepts/model-providers/openai/index.md)** - GPT models with streaming support
-   **[Google](/docs/user-guide/concepts/model-providers/google/index.md)** - Google’s Gemini models with tool calling support
-   **[Custom Providers](/docs/user-guide/concepts/model-providers/custom_model_provider/index.md)** - Build your own model integration
-   **[Anthropic](/docs/user-guide/concepts/model-providers/anthropic/index.md)** - Direct Claude API access