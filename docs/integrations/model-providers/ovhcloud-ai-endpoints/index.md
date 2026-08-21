[OVHcloud AI Endpoints](https://www.ovhcloud.com/en/public-cloud/ai-endpoints/) offers access to a wide range of language models from a European cloud provider, with data sovereignty and GDPR compliance.

OpenAI compatibility

This integration works through the SDK’s built-in [OpenAI provider](/docs/user-guide/concepts/model-providers/openai/index.md) pointed at OVHcloud’s OpenAI-compatible endpoint; there is no separate OVHcloud integration. Compatible endpoints can have quirks that deviate from the exact OpenAI API spec, so some features may behave differently than they do against OpenAI itself.

## Installation

The OpenAI provider is an optional dependency. To install, run:

(( tab "Python" ))
```bash
pip install 'strands-agents[openai]'
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```bash
npm install @strands-agents/sdk openai
```
(( /tab "TypeScript" ))

## Usage

Generate an API key in the [OVHcloud Manager](https://ovh.com/manager) under **Public Cloud** > **AI & Machine Learning** > **AI Endpoints** > **API keys**, export it as `OVHCLOUD_API_KEY`, and point the provider at the AI Endpoints endpoint:

(( tab "Python" ))
```python
import os

from strands import Agent
from strands.models.openai import OpenAIModel

model = OpenAIModel(
    client_args={
        # An empty key selects the rate-limited free tier
        "api_key": os.environ.get("OVHCLOUD_API_KEY", ""),
        "base_url": "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
    },
    model_id="Meta-Llama-3_3-70B-Instruct",
)

agent = Agent(model=model)
response = agent("Explain tool calling in one sentence.")
print(response)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { Agent } from '@strands-agents/sdk'
import { OpenAIModel } from '@strands-agents/sdk/models/openai'

const model = new OpenAIModel({
  api: 'chat',
  // A non-empty API key is required; the keyless free tier is not supported
  apiKey: process.env.OVHCLOUD_API_KEY,
  clientConfig: {
    baseURL: 'https://oai.endpoints.kepler.ai.cloud.ovh.net/v1',
  },
  modelId: 'Meta-Llama-3_3-70B-Instruct',
})

const agent = new Agent({ model })
const response = await agent.invoke('Explain tool calling in one sentence.')
console.log(response)
```
(( /tab "TypeScript" ))

## Configuration

Two client settings connect the provider to OVHcloud AI Endpoints:

-   **API key**: from the [OVHcloud Manager](https://ovh.com/manager). Without a key, requests use a rate-limited free tier; the Python provider accepts an empty key string, while the TypeScript provider requires a non-empty key.
-   **Base URL**: `https://oai.endpoints.kepler.ai.cloud.ovh.net/v1`

Model IDs come from the [AI Endpoints catalog](https://www.ovhcloud.com/en/public-cloud/ai-endpoints/catalog/), for example `Meta-Llama-3_3-70B-Instruct`. For model parameters and other provider options, see the [OpenAI provider](/docs/user-guide/concepts/model-providers/openai/index.md) guide.

## References

-   [OVHcloud AI Endpoints](https://www.ovhcloud.com/en/public-cloud/ai-endpoints/)
-   [OVHcloud AI Endpoints catalog](https://www.ovhcloud.com/en/public-cloud/ai-endpoints/catalog/)
-   [OpenAI provider](/docs/user-guide/concepts/model-providers/openai/index.md)