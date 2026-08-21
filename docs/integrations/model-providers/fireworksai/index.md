[Fireworks AI](https://fireworks.ai) provides fast hosted inference for open-source language models.

OpenAI compatibility

This integration works through the SDK’s built-in [OpenAI provider](/docs/user-guide/concepts/model-providers/openai/index.md) pointed at Fireworks AI’s OpenAI-compatible endpoint; there is no separate integration. Compatible endpoints can have quirks that deviate from the exact OpenAI API spec, so some features may behave differently than they do against OpenAI itself.

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

Create an [API key](https://app.fireworks.ai/settings/users/api-keys), export it as `FIREWORKS_API_KEY`, and point the provider at Fireworks AI’s endpoint:

(( tab "Python" ))
```python
import os

from strands import Agent
from strands.models.openai import OpenAIModel

model = OpenAIModel(
    client_args={
        "api_key": os.environ["FIREWORKS_API_KEY"],
        "base_url": "https://api.fireworks.ai/inference/v1",
    },
    model_id="accounts/fireworks/models/deepseek-v3p1-terminus",
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
  apiKey: process.env.FIREWORKS_API_KEY,
  clientConfig: {
    baseURL: 'https://api.fireworks.ai/inference/v1',
  },
  modelId: 'accounts/fireworks/models/deepseek-v3p1-terminus',
})

const agent = new Agent({ model })
const response = await agent.invoke('Explain tool calling in one sentence.')
console.log(response)
```
(( /tab "TypeScript" ))

## Configuration

Two client settings connect the provider to Fireworks AI:

-   **API key**: from your [account settings](https://app.fireworks.ai/settings/users/api-keys)
-   **Base URL**: `https://api.fireworks.ai/inference/v1`

Model IDs come from the [Fireworks AI model library](https://fireworks.ai/models) and carry the `accounts/fireworks/models/` prefix. For model parameters and other provider options, see the [OpenAI provider](/docs/user-guide/concepts/model-providers/openai/index.md) guide.

## References

-   [Fireworks AI OpenAI compatibility guide](https://fireworks.ai/docs/tools-sdks/openai-compatibility#openai-compatibility)
-   [Fireworks AI model library](https://fireworks.ai/models)
-   [OpenAI provider](/docs/user-guide/concepts/model-providers/openai/index.md)