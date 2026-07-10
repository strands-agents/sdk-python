[Google Gemini](https://ai.google.dev/api) is Google’s family of multimodal large language models designed for advanced reasoning, code generation, and creative tasks. The Strands Agents SDK implements a Google/Gemini provider, allowing you to run agents against the Gemini models available through Google’s AI API.

## Installation

Gemini is configured as an optional dependency in Strands Agents.

To install it, run:

(( tab "Python" ))
```bash
pip install 'strands-agents[gemini]' strands-agents-tools
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```bash
npm install @strands-agents/sdk @google/genai
```
(( /tab "TypeScript" ))

## Usage

After installing dependencies, you can import and initialize the Strands Agents’ Gemini provider as follows:

(( tab "Python" ))
```python
from strands import Agent
from strands.models.gemini import GeminiModel
from strands_tools import calculator

model = GeminiModel(
    client_args={
        "api_key": "<KEY>",
    },
    # **model_config
    model_id="gemini-2.5-flash",
    params={
        # some sample model parameters
        "temperature": 0.7,
        "max_output_tokens": 2048,
        "top_p": 0.9,
        "top_k": 40
    }
)

agent = Agent(model=model, tools=[calculator])
response = agent("What is 2+2")
print(response)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { Agent } from '@strands-agents/sdk'
import { GoogleModel } from '@strands-agents/sdk/models/google'

const model = new GoogleModel({
  apiKey: '<KEY>',
  modelId: 'gemini-2.5-flash',
  params: {
    temperature: 0.7,
    maxOutputTokens: 2048,
    topP: 0.9,
    topK: 40,
  },
})

const agent = new Agent({ model })
const response = await agent.invoke('What is 2+2')
console.log(response)
```
(( /tab "TypeScript" ))

## Configuration

### Client Configuration

(( tab "Python" ))
The `client_args` configure the underlying Google GenAI client. For a complete list of available arguments, please refer to the [Google GenAI documentation](https://googleapis.github.io/python-genai/).
(( /tab "Python" ))

(( tab "TypeScript" ))
The `clientConfig` configures the underlying Google GenAI client. You can also pass a pre-configured `client` instance directly. For a complete list of available options, please refer to the [@google/genai documentation](https://github.com/googleapis/js-genai).
(( /tab "TypeScript" ))

### Model Configuration

(( tab "Python" ))
The `model_config` configures the underlying model selected for inference. The supported configurations are:

| Parameter | Description | Example | Options |
| --- | --- | --- | --- |
| `model_id` | ID of a Gemini model to use | `"gemini-2.5-flash"` | [Available models](#available-models) |
| `params` | Model specific parameters | `{"temperature": 0.7, "maxOutputTokens": 2048}` | [Parameter reference](#model-parameters) |
(( /tab "Python" ))

(( tab "TypeScript" ))
| Parameter | Description | Example | Options |
| --- | --- | --- | --- |
| `modelId` | ID of a Gemini model to use | `'gemini-2.5-flash'` | [Available models](#available-models) |
| `params` | Model specific parameters | `{ temperature: 0.7, maxOutputTokens: 2048 }` | [Parameter reference](#model-parameters) |
(( /tab "TypeScript" ))

### Model Parameters

For a complete list of supported parameters, see the [Gemini API documentation](https://ai.google.dev/api/generate-content#generationconfig).

(( tab "Python" ))
| Parameter | Description | Type |
| --- | --- | --- |
| `temperature` | Controls randomness in responses | `float` |
| `max_output_tokens` | Maximum tokens to generate | `int` |
| `top_p` | Nucleus sampling parameter | `float` |
| `top_k` | Top-k sampling parameter | `int` |
| `candidate_count` | Number of response candidates | `int` |
| `stop_sequences` | Custom stopping sequences | `list[str]` |

**Example:**

```python
params = {
    "temperature": 0.8,
    "max_output_tokens": 4096,
    "top_p": 0.95,
    "top_k": 40,
    "candidate_count": 1,
    "stop_sequences": ['STOP!']
}
```
(( /tab "Python" ))

(( tab "TypeScript" ))
| Parameter | Description | Type |
| --- | --- | --- |
| `temperature` | Controls randomness in responses | `number` |
| `maxOutputTokens` | Maximum tokens to generate | `number` |
| `topP` | Nucleus sampling parameter | `number` |
| `topK` | Top-k sampling parameter | `number` |
| `candidateCount` | Number of response candidates | `number` |
| `stopSequences` | Custom stopping sequences | `string[]` |

**Example:**

```typescript
const params = {
  temperature: 0.8,
  maxOutputTokens: 4096,
  topP: 0.95,
  topK: 40,
  candidateCount: 1,
  stopSequences: ['STOP!'],
}
```
(( /tab "TypeScript" ))

### Available Models

For a complete list of supported models, see the [Gemini API documentation](https://ai.google.dev/gemini-api/docs/models).

**Popular Models:**

-   `gemini-2.5-pro` - Most advanced model for complex reasoning and thinking
-   `gemini-2.5-flash` - Best balance of performance and cost
-   `gemini-2.5-flash-lite` - Most cost-efficient option
-   `gemini-2.0-flash` - Next-gen features with improved speed
-   `gemini-2.0-flash-lite` - Cost-optimized version of 2.0

### Built-in Tools

(( tab "Python" ))
Google’s built-in tools (Google Search, Code Execution, URL Context) can be passed via the `gemini_tools` config option. These are appended alongside any function tools registered on the agent.

```python
from google import genai
from strands import Agent
from strands.models.gemini import GeminiModel

model = GeminiModel(
    client_args={"api_key": "<KEY>"},
    model_id="gemini-2.5-flash",
    gemini_tools=[
        genai.types.Tool(google_search=genai.types.GoogleSearch()),
    ],
)

agent = Agent(model=model)
response = agent("What are the latest AI news today?")
print(response)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
Google’s built-in tools (Google Search, Code Execution, URL Context) can be passed via the `builtInTools` config option. These are appended alongside any function tools registered on the agent.

```typescript
const model = new GoogleModel({
  apiKey: '<KEY>',
  modelId: 'gemini-2.5-flash',
  builtInTools: [{ googleSearch: {} }],
})

const agent = new Agent({ model })
const response = await agent.invoke('What are the latest AI news today?')
console.log(response)
```
(( /tab "TypeScript" ))

For available built-in tools, see the [Gemini tools documentation](https://ai.google.dev/gemini-api/docs/tools).

## Troubleshooting

### Module Not Found

(( tab "Python" ))
If you encounter the error `ModuleNotFoundError: No module named 'google.genai'`, this means the `google-genai` dependency hasn’t been properly installed in your environment. To fix this, run `pip install 'strands-agents[gemini]'`.
(( /tab "Python" ))

(( tab "TypeScript" ))
If you encounter import errors for `@google/genai`, ensure the package is installed: `npm install @google/genai`.
(( /tab "TypeScript" ))

### API Key Issues

Make sure your Google AI API key is properly set via `client_args``apiKey`, or as the `GOOGLE_API_KEY` / `GEMINI_API_KEY` environment variable. You can obtain an API key from the [Google AI Studio](https://aistudio.google.com/app/apikey).

### Rate Limiting and Safety Issues

The Gemini provider handles several types of errors automatically:

-   **Safety/Content Policy**: When content is blocked due to safety concerns, the model will return a safety message
-   **Rate Limiting**: When quota limits are exceeded, a `ModelThrottledException` is raised
-   **Server Errors**: Temporary server issues are handled with appropriate error messages

(( tab "Python" ))
```python
from strands.types.exceptions import ModelThrottledException

try:
    response = agent("Your query here")
except ModelThrottledException as e:
    print(f"Rate limit exceeded: {e}")
    # Implement backoff strategy
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
try {
  const response = await agent.invoke('Your query here')
} catch (error) {
  console.error('Error:', error)
  // Implement backoff strategy
}
```
(( /tab "TypeScript" ))

## Advanced Features

### Structured Output

Gemini models support structured output through their native JSON schema capabilities. Pass a schema to the agent, and Strands converts it to Gemini’s JSON schema format and validates the response.

(( tab "Python" ))
Define a Pydantic model and pass it to [`agent.structured_output()`](/docs/api/python/strands.agent.agent#Agent.structured_output):

```python
from pydantic import BaseModel, Field
from strands import Agent
from strands.models.gemini import GeminiModel

class MovieReview(BaseModel):
    """Analyze a movie review."""
    title: str = Field(description="Movie title")
    rating: int = Field(description="Rating from 1-10", ge=1, le=10)
    genre: str = Field(description="Primary genre")
    sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")
    summary: str = Field(description="Brief summary of the review")

model = GeminiModel(
    client_args={"api_key": "<KEY>"},
    model_id="gemini-2.5-flash",
    params={
        "temperature": 0.3,
        "max_output_tokens": 1024,
        "top_p": 0.85
    }
)

agent = Agent(model=model)

result = agent.structured_output(
    MovieReview,
    """
    Just watched "The Matrix" - what an incredible sci-fi masterpiece!
    The groundbreaking visual effects and philosophical themes make this
    a must-watch. Keanu Reeves delivers a solid performance. 9/10!
    """
)

print(f"Movie: {result.title}")
print(f"Rating: {result.rating}/10")
print(f"Genre: {result.genre}")
print(f"Sentiment: {result.sentiment}")
```
(( /tab "Python" ))

(( tab "TypeScript" ))
Define a Zod schema and pass it as `structuredOutputSchema`. Validated output is on `result.structuredOutput`:

```typescript
import { Agent } from '@strands-agents/sdk'
import { GoogleModel } from '@strands-agents/sdk/models/google'
import { z } from 'zod'

const MovieReview = z.object({
  title: z.string().describe('Movie title'),
  rating: z.number().min(1).max(10).describe('Rating from 1-10'),
  genre: z.string().describe('Primary genre'),
  sentiment: z.enum(['positive', 'negative', 'neutral']).describe('Overall sentiment'),
  summary: z.string().describe('Brief summary of the review'),
})

const model = new GoogleModel({
  apiKey: '<KEY>',
  modelId: 'gemini-2.5-flash',
})

const agent = new Agent({ model, structuredOutputSchema: MovieReview })

const result = await agent.invoke(
  `Just watched "The Matrix" - what an incredible sci-fi masterpiece!
   The groundbreaking visual effects and philosophical themes make this
   a must-watch. Keanu Reeves delivers a solid performance. 9/10!`
)

const review = result.structuredOutput as z.infer<typeof MovieReview>
console.log(`Movie: ${review.title}`)
console.log(`Rating: ${review.rating}/10`)
console.log(`Genre: ${review.genre}`)
console.log(`Sentiment: ${review.sentiment}`)
```
(( /tab "TypeScript" ))

For schema patterns, error handling, and per-invocation overrides, see [Structured Output](/docs/user-guide/concepts/agents/structured-output/index.md).

### Custom client

Users can pass their own custom Gemini client to the GeminiModel for Strands Agents to use directly. Users are responsible for handling the lifecycle (e.g., closing) of the client.

(( tab "Python" ))
```python
from google import genai
from strands import Agent
from strands.models.gemini import GeminiModel
from strands_tools import calculator

client = genai.Client(api_key="<KEY>")

model = GeminiModel(
    client=client,
    # **model_config
    model_id="gemini-2.5-flash",
    params={
        # some sample model parameters
        "temperature": 0.7,
        "max_output_tokens": 2048,
        "top_p": 0.9,
        "top_k": 40
    }
)

agent = Agent(model=model, tools=[calculator])
response = agent("What is 2+2")
print(response)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { GoogleGenAI } from '@google/genai'
import { Agent } from '@strands-agents/sdk'
import { GoogleModel } from '@strands-agents/sdk/models/google'

const client = new GoogleGenAI({ apiKey: '<KEY>' })

const model = new GoogleModel({
  client,
  modelId: 'gemini-2.5-flash',
  params: {
    temperature: 0.7,
    maxOutputTokens: 2048,
    topP: 0.9,
    topK: 40,
  },
})

const agent = new Agent({ model })
const response = await agent.invoke('What is 2+2')
console.log(response)
```
(( /tab "TypeScript" ))

### Multimodal Capabilities

Gemini models support text, image, document, and video inputs, making them ideal for multimodal applications.

#### Image Input

(( tab "Python" ))
```python
from strands import Agent
from strands.models.gemini import GeminiModel

model = GeminiModel(
    client_args={"api_key": "<KEY>"},
    model_id="gemini-2.5-flash",
    params={
        "temperature": 0.5,
        "max_output_tokens": 2048,
        "top_p": 0.9
    }
)

agent = Agent(model=model)

# Process image with text
response = agent([
    {
        "role": "user",
        "content": [
            {"text": "What do you see in this image?"},
            {"image": {"format": "png", "source": {"bytes": image_bytes}}}
        ]
    }
])
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { Agent, ImageBlock, TextBlock } from '@strands-agents/sdk'
import { GoogleModel } from '@strands-agents/sdk/models/google'

const model = new GoogleModel({
  apiKey: '<KEY>',
  modelId: 'gemini-2.5-flash',
})

const agent = new Agent({ model })

// Process image with text
const result = await agent.invoke([
  new TextBlock('What do you see in this image?'),
  new ImageBlock({
    format: 'png',
    source: { bytes: imageBytes },
  }),
])
```
(( /tab "TypeScript" ))

#### Document Input

(( tab "Python" ))
```python
response = agent([
    {
        "role": "user",
        "content": [
            {"text": "Summarize this document"},
            {"document": {"format": "pdf", "source": {"bytes": document_bytes}}}
        ]
    }
])
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { DocumentBlock, TextBlock } from '@strands-agents/sdk'

const result = await agent.invoke([
  new TextBlock('Summarize this document'),
  new DocumentBlock({
    name: 'my-document',
    format: 'pdf',
    source: { bytes: pdfBytes },
  }),
])
```
(( /tab "TypeScript" ))

#### Video Input

(( tab "Python" ))
```python
response = agent([
    {
        "role": "user",
        "content": [
            {"text": "Describe what happens in this video"},
            {"video": {"format": "mp4", "source": {"bytes": video_bytes}}}
        ]
    }
])
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { VideoBlock, TextBlock } from '@strands-agents/sdk'

const result = await agent.invoke([
  new TextBlock('Describe what happens in this video'),
  new VideoBlock({
    format: 'mp4',
    source: { bytes: videoBytes },
  }),
])
```
(( /tab "TypeScript" ))

**Supported formats:**

-   **Images**: PNG, JPEG, GIF, WebP (automatically detected via MIME type)
-   **Documents**: PDF and other binary formats (automatically detected via MIME type)
-   **Video**: MP4 and other video formats (automatically detected via MIME type)

### Token Counting

Token counting is used by context management strategies to estimate input tokens before each model call.

(( tab "Python" ))
The Google provider can use the native `models.count_tokens()` API for message content. However, the Gemini API does not support counting system instructions or tool specifications natively. These are estimated separately using a character-based heuristic (characters ÷ 4 for text, characters ÷ 2 for JSON).

You can enable native token counting with:

```python
model = GoogleModel(
    model_id="gemini-2.5-flash",
    use_native_token_count=True,
)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
The Google provider can use the native `models.countTokens()` API for message content. However, the Gemini API does not support counting system instructions or tool specifications natively. These are estimated separately using a character-based heuristic (characters ÷ 4 for text, characters ÷ 2 for JSON).

You can enable native token counting with:

```typescript
const model = new GoogleModel({
  modelId: 'gemini-2.5-flash',
  useNativeTokenCount: true,
})
```

When disabled (or if the API call fails), falls back to estimation using the character-based heuristic.
(( /tab "TypeScript" ))

## References

-   [Python API](/docs/api/python/strands.models.model)
-   [Google Gemini](https://ai.google.dev/api)
-   [Google GenAI SDK documentation](https://googleapis.github.io/python-genai/)
-   [Google AI Studio](https://aistudio.google.com/)
-   [@google/genai TypeScript SDK](https://github.com/googleapis/js-genai)

## Related pages

- [Vercel](/docs/user-guide/concepts/model-providers/vercel/index.md) (1 shared tag)
- [Multimodal Correctness Evaluator](/docs/user-guide/evals-sdk/evaluators/multimodal_correctness_evaluator/index.md) (1 shared tag)
- [Multimodal Faithfulness Evaluator](/docs/user-guide/evals-sdk/evaluators/multimodal_faithfulness_evaluator/index.md) (1 shared tag)
- [Multimodal Instruction Following Evaluator](/docs/user-guide/evals-sdk/evaluators/multimodal_instruction_following_evaluator/index.md) (1 shared tag)
- [Multimodal Output Evaluator](/docs/user-guide/evals-sdk/evaluators/multimodal_output_evaluator/index.md) (1 shared tag)
- [Multimodal Overall Quality Evaluator](/docs/user-guide/evals-sdk/evaluators/multimodal_overall_quality_evaluator/index.md) (1 shared tag)
- [OpenAI](/docs/user-guide/concepts/model-providers/openai/index.md) (1 shared tag)
- [Writer](/docs/user-guide/concepts/model-providers/writer/index.md) (1 shared tag)
- [Amazon Nova](/docs/user-guide/concepts/model-providers/amazon-nova/index.md) (1 shared tag)
- [Amazon Bedrock](/docs/user-guide/concepts/model-providers/amazon-bedrock/index.md) (1 shared tag)


## Implementation

### Python

- [harness-sdk/strands-py/src/strands/models/gemini.py](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/gemini.py)

### TypeScript

- [harness-sdk/strands-ts/src/models/google/model.ts](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/models/google/model.ts)
