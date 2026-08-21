[OpenRouter](https://openrouter.ai/) routes requests to hundreds of models from many labs behind a single API and API key.

OpenAI compatibility

This integration works through the SDK’s built-in [OpenAI provider](/docs/user-guide/concepts/model-providers/openai/index.md) pointed at OpenRouter’s OpenAI-compatible endpoint; there is no separate OpenRouter integration. Compatible endpoints can have quirks that deviate from the exact OpenAI API spec, so some features may behave differently than they do against OpenAI itself.

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

Create an [API key](https://openrouter.ai/settings/keys), export it as `OPENROUTER_API_KEY`, and point the provider at OpenRouter’s endpoint:

(( tab "Python" ))
```python
import os

from strands import Agent
from strands.models.openai import OpenAIModel

model = OpenAIModel(
    client_args={
        "api_key": os.environ["OPENROUTER_API_KEY"],
        "base_url": "https://openrouter.ai/api/v1",
    },
    model_id="openai/gpt-5.4",
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
  apiKey: process.env.OPENROUTER_API_KEY,
  clientConfig: {
    baseURL: 'https://openrouter.ai/api/v1',
  },
  modelId: 'openai/gpt-5.4',
})

const agent = new Agent({ model })
const response = await agent.invoke('Explain tool calling in one sentence.')
console.log(response)
```
(( /tab "TypeScript" ))

## Configuration

Two client settings connect the provider to OpenRouter:

-   **API key**: from your [OpenRouter settings](https://openrouter.ai/settings/keys)
-   **Base URL**: `https://openrouter.ai/api/v1`

Model IDs come from the [OpenRouter catalog](https://openrouter.ai/models) and are prefixed with the lab name, for example `openai/gpt-5.4` or `anthropic/claude-sonnet-4-5`. For model parameters and other provider options, see the [OpenAI provider](/docs/user-guide/concepts/model-providers/openai/index.md) guide.

## References

-   [OpenRouter quickstart](https://openrouter.ai/docs/quickstart)
-   [OpenRouter model catalog](https://openrouter.ai/models)
-   [OpenAI provider](/docs/user-guide/concepts/model-providers/openai/index.md)