[Nebius Token Factory](https://tokenfactory.nebius.com) provides fast inference for open-source language models.

OpenAI compatibility

This integration works through the SDK’s built-in [OpenAI provider](/docs/user-guide/concepts/model-providers/openai/index.md) pointed at Nebius Token Factory’s OpenAI-compatible endpoint; there is no separate integration. Compatible endpoints can have quirks that deviate from the exact OpenAI API spec, so some features may behave differently than they do against OpenAI itself.

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

Create an API key in the [Nebius Token Factory Console](https://tokenfactory.nebius.com/), export it as `NEBIUS_API_KEY`, and point the provider at the Token Factory endpoint:

(( tab "Python" ))
```python
import os

from strands import Agent
from strands.models.openai import OpenAIModel

model = OpenAIModel(
    client_args={
        "api_key": os.environ["NEBIUS_API_KEY"],
        "base_url": "https://api.tokenfactory.nebius.com/v1/",
    },
    model_id="deepseek-ai/DeepSeek-R1-0528",
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
  apiKey: process.env.NEBIUS_API_KEY,
  clientConfig: {
    baseURL: 'https://api.tokenfactory.nebius.com/v1/',
  },
  modelId: 'deepseek-ai/DeepSeek-R1-0528',
})

const agent = new Agent({ model })
const response = await agent.invoke('Explain tool calling in one sentence.')
console.log(response)
```
(( /tab "TypeScript" ))

## Configuration

Two client settings connect the provider to Nebius Token Factory:

-   **API key**: from the [Token Factory Console](https://tokenfactory.nebius.com/)
-   **Base URL**: `https://api.tokenfactory.nebius.com/v1/`

Model IDs come from the [Token Factory model catalog](https://docs.tokenfactory.nebius.com/ai-models-inference/overview), for example `deepseek-ai/DeepSeek-R1-0528` or `meta-llama/Meta-Llama-3.1-70B-Instruct`. For model parameters and other provider options, see the [OpenAI provider](/docs/user-guide/concepts/model-providers/openai/index.md) guide.

## References

-   [Nebius Token Factory documentation](https://docs.tokenfactory.nebius.com/)
-   [Nebius Token Factory API reference](https://docs.tokenfactory.nebius.com/api-reference)
-   [OpenAI provider](/docs/user-guide/concepts/model-providers/openai/index.md)