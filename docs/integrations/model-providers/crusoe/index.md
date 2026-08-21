[Crusoe](https://www.crusoe.ai/cloud/managed-inference) provides managed inference for leading open-weight language models on renewable-powered GPU infrastructure.

OpenAI compatibility

This integration works through the SDK’s built-in [OpenAI provider](/docs/user-guide/concepts/model-providers/openai/index.md) pointed at Crusoe’s OpenAI-compatible endpoint; there is no separate Crusoe integration. Compatible endpoints can have quirks that deviate from the exact OpenAI API spec, so some features may behave differently than they do against OpenAI itself.

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

Generate an API key in the **Security** tab of the [Crusoe Cloud Console](https://console.crusoecloud.com/), export it as `CRUSOE_API_KEY`, and point the provider at Crusoe’s endpoint:

(( tab "Python" ))
```python
import os

from strands import Agent
from strands.models.openai import OpenAIModel

model = OpenAIModel(
    client_args={
        "api_key": os.environ["CRUSOE_API_KEY"],
        "base_url": "https://api.inference.crusoecloud.com/v1/",
    },
    model_id="zai/GLM-5.2",
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
  apiKey: process.env.CRUSOE_API_KEY,
  clientConfig: {
    baseURL: 'https://api.inference.crusoecloud.com/v1/',
  },
  modelId: 'zai/GLM-5.2',
})

const agent = new Agent({ model })
const response = await agent.invoke('Explain tool calling in one sentence.')
console.log(response)
```
(( /tab "TypeScript" ))

## Configuration

Two client settings connect the provider to Crusoe Managed Inference:

-   **API key**: from the **Security** tab of the [Crusoe Cloud Console](https://console.crusoecloud.com/)
-   **Base URL**: `https://api.inference.crusoecloud.com/v1/`

Model IDs come from the [Crusoe Managed Inference overview](https://docs.crusoecloud.com/managed-inference/overview), for example `zai/GLM-5.2` or `nvidia/Nemotron-3-Super-120B-A12B`. For model parameters and other provider options, see the [OpenAI provider](/docs/user-guide/concepts/model-providers/openai/index.md) guide.

## References

-   [Crusoe Managed Inference documentation](https://docs.crusoecloud.com/managed-inference/overview)
-   [Crusoe Cloud Console](https://console.crusoecloud.com/)
-   [OpenAI provider](/docs/user-guide/concepts/model-providers/openai/index.md)