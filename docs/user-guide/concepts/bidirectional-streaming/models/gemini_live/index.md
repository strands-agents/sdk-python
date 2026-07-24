The [Gemini Live API](https://ai.google.dev/gemini-api/docs/live) lets developers create natural conversations by enabling a two-way WebSocket connection with the Gemini models. The Live API processes data streams in real time. Users can interrupt the AI’s responses with new input, similar to a real conversation. Key features include:

-   **Multimodal Streaming**: The API supports streaming of text, audio, and video data.
-   **Bidirectional Interaction**: The user and the model can provide input and output at the same time.
-   **Interruptibility**: Users can interrupt the model’s response, and the model adjusts its response.
-   **Tool Use and Function Calling**: The API can use external tools to perform actions and get context while maintaining a real-time connection.
-   **Session Management**: Supports managing long conversations through sessions, providing context and continuity.
-   **Secure Authentication**: Uses tokens for secure client-side authentication.

## Installation

Gemini Live is configured as an optional dependency in Strands Agents.

To install it, run:

```bash
pip install 'strands-agents[bidi-gemini]'
```

Or to install all bidirectional streaming providers at once:

```bash
pip install 'strands-agents[bidi-all]'
```

## Usage

After installing `strands-agents[bidi-gemini]`, you can import and initialize the Strands Agents’ Gemini Live provider as follows:

```python
import asyncio

from strands.experimental.bidi import BidiAgent
from strands.experimental.bidi.io import BidiAudioIO, BidiTextIO
from strands.experimental.bidi.models import BidiGeminiLiveModel
from strands_tools import stop

from strands_tools import calculator


async def main() -> None:
    model = BidiGeminiLiveModel(
        model_id="gemini-2.5-flash-native-audio-preview-09-2025",
        provider_config={
            "audio": {
                "voice": "Kore",
            },
        },
        client_config={"api_key": "<GOOGLE_AI_API_KEY>"},
    )
    # stop tool allows user to verbally stop agent execution.
    agent = BidiAgent(model=model, tools=[calculator, stop])

    audio_io = BidiAudioIO()
    text_io = BidiTextIO()
    await agent.run(inputs=[audio_io.input()], outputs=[audio_io.output(), text_io.output()])


if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration

### Client Configs

For details on the supported client configs, see [here](https://googleapis.github.io/python-genai/genai.html#genai.client.Client).

### Provider Configs

| Parameter | Description | Example | Options |
| --- | --- | --- | --- |
| `audio` | `AudioConfig` instance. | `{"voice": "Kore"}` | [reference](/docs/api/python/strands.experimental.bidi.types.model#AudioConfig) |
| `inference` | Dict of inference fields specified in the Gemini `LiveConnectConfig`. | `{"temperature": 0.7}` | [reference](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfig) |

For the list of supported voices and languages, see [here](https://docs.cloud.google.com/text-to-speech/docs/list-voices-and-types).

## Session Management

Currently, `BidiGeminiLiveModel` does not produce a message history and so has limited compatability with the Strands [session manager](/docs/user-guide/concepts/bidirectional-streaming/session-management/index.md). However, the provider does utilize Gemini’s [Session Resumption](https://ai.google.dev/gemini-api/docs/live-session) as part of the [connection restart](/docs/user-guide/concepts/bidirectional-streaming/agent/index.md#connection-restart) workflow. This allows Gemini Live connections to persist up to 24 hours. After this time limit, a new `BidiGeminiLiveModel` instance must be created to continue conversations.

## Troubleshooting

### Module Not Found

If you encounter the error `ModuleNotFoundError: No module named 'google.genai'`, this means the `google-genai` dependency hasn’t been properly installed in your environment. To fix this, run `pip install 'strands-agents[bidi-gemini]'`.

### API Key Issues

Make sure your Google AI API key is properly set in `client_config` or as the `GOOGLE_API_KEY` environment variable. You can obtain an API key from the [Google AI Studio](https://aistudio.google.com/app/apikey).

## References

-   [Gemini Live API](https://ai.google.dev/gemini-api/docs/live)
-   [Gemini API Reference](https://googleapis.github.io/python-genai/genai.html#)
-   [Python API Reference](/docs/api/python/strands.experimental.bidi.models.gemini_live#BidiGeminiLiveModel)

## Related pages

- [BidiAgent](/docs/user-guide/concepts/bidirectional-streaming/agent/index.md) (1 shared tag)
- [Events](/docs/user-guide/concepts/bidirectional-streaming/events/index.md) (1 shared tag)
- [I/O Channels](/docs/user-guide/concepts/bidirectional-streaming/io/index.md) (1 shared tag)
- [Interruptions](/docs/user-guide/concepts/bidirectional-streaming/interruption/index.md) (1 shared tag)
- [OpenAI Realtime](/docs/user-guide/concepts/bidirectional-streaming/models/openai_realtime/index.md) (1 shared tag)
- [Bidirectional Streaming Hooks](/docs/user-guide/concepts/bidirectional-streaming/hooks/index.md) (1 shared tag)
- [Voice & Realtime Quickstart](/docs/user-guide/concepts/bidirectional-streaming/quickstart/index.md) (1 shared tag)
- [Nova Sonic](/docs/user-guide/concepts/bidirectional-streaming/models/nova_sonic/index.md) (1 shared tag)
- [Bidirectional Streaming Session Management](/docs/user-guide/concepts/bidirectional-streaming/session-management/index.md) (1 shared tag)


## Implementation

### Python

- [harness-sdk/strands-py/src/strands/experimental/bidi/models/gemini_live.py](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/gemini_live.py)
