# OpenAI Realtime API Bidirectional Model Provider

This module provides integration with OpenAI's Realtime API for bidirectional streaming conversations with real-time audio and text interaction.

## Features

- **Real-time Audio Streaming**: Native support for speech-to-speech conversations
- **Text Interaction**: Full text input/output capabilities
- **Function Calling**: Execute custom tools during conversations
- **Voice Activity Detection**: Automatic turn detection and interruption handling
- **Multiple Voices**: Choose from various OpenAI voices (alloy, echo, fable, onyx, nova, shimmer)
- **Official SDK**: Uses the official OpenAI Python SDK for robust integration

## Installation

```bash
# Install with OpenAI support
pip install strands-agents[openai,bidirectional-streaming]

# Or install dependencies manually
pip install openai>=1.107.0 pyaudio>=0.2.13
```

## Authentication

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Basic Usage

### Text Conversation

```python
import asyncio
from strands.experimental.bidirectional_streaming.agent.agent import BidirectionalAgent
from strands.experimental.bidirectional_streaming.models.openai_realtime import OpenAIRealtimeBidirectionalModel

async def text_chat():
    # Initialize model
    model = OpenAIRealtimeBidirectionalModel(
        model_id="gpt-realtime",
        api_key="your-api-key",
        params={
            "output_modalities": ["text"],
            "instructions": "You are a helpful assistant."
        }
    )
    
    # Create agent
    agent = BidirectionalAgent(model=model)
    
    # Start conversation
    await agent.start()
    
    # Send message
    await agent.send("Hello! How are you?")
    
    # Receive response
    async for event in agent.receive():
        if "textOutput" in event:
            print(f"Assistant: {event['textOutput']['text']}")
        elif "BidirectionalConnectionEnd" in event:
            break
    
    await agent.end()

asyncio.run(text_chat())
```

### Audio Conversation

```python
import asyncio
import pyaudio
from strands.experimental.bidirectional_streaming.agent.agent import BidirectionalAgent
from strands.experimental.bidirectional_streaming.models.openai_realtime import OpenAIRealtimeBidirectionalModel

async def voice_chat():
    # Initialize model with audio
    model = OpenAIRealtimeBidirectionalModel(
        model_id="gpt-realtime",
        api_key="your-api-key",
        params={
            "output_modalities": ["audio", "text"],
            "audio": {
                "input": {"turn_detection": {"type": "server_vad"}},
                "output": {"voice": "alloy"}
            }
        }
    )
    
    # Create agent
    agent = BidirectionalAgent(model=model)
    await agent.start()
    
    # Audio setup (simplified)
    audio = pyaudio.PyAudio()
    microphone = audio.open(channels=1, format=pyaudio.paInt16, 
                           input=True, rate=24000, frames_per_buffer=1024)
    speaker = audio.open(channels=1, format=pyaudio.paInt16, 
                        output=True, rate=24000, frames_per_buffer=1024)
    
    # Audio processing loop
    async def process_audio():
        async for event in agent.receive():
            if "audioOutput" in event:
                speaker.write(event["audioOutput"]["audioData"])
    
    async def send_audio():
        while True:
            audio_data = microphone.read(1024)
            await agent.send({
                "audioData": audio_data,
                "format": "pcm",
                "sampleRate": 24000,
                "channels": 1
            })
    
    # Run concurrently
    await asyncio.gather(process_audio(), send_audio())

asyncio.run(voice_chat())
```

## Configuration Options

### Model Parameters

```python
model = OpenAIRealtimeBidirectionalModel(
    model_id="gpt-realtime",  # OpenAI Realtime model
    api_key="your-api-key",
    base_url=None,  # Optional custom base URL
    organization=None,  # Optional organization ID
    project=None,  # Optional project ID
    params={
        # Output modalities
        "output_modalities": ["audio", "text"],  # or ["text"] for text-only
        
        # Audio configuration
        "audio": {
            "input": {
                "turn_detection": {"type": "server_vad"}  # or None for manual
            },
            "output": {
                "voice": "alloy"  # alloy, echo, fable, onyx, nova, shimmer
            }
        },
        
        # System instructions
        "instructions": "You are a helpful assistant.",
        
        # Function calling
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            }
        ]
    }
)
```

### Voice Options

Available voices for audio output:
- `alloy`: Balanced, neutral voice
- `echo`: Clear, expressive voice  
- `fable`: Warm, engaging voice
- `onyx`: Deep, authoritative voice
- `nova`: Bright, energetic voice
- `shimmer`: Soft, gentle voice

### Turn Detection

- `"server_vad"`: Automatic voice activity detection (recommended)
- `None`: Manual turn detection (requires explicit `input_audio_buffer.commit`)

## Function Calling

The OpenAI Realtime model supports function calling during conversations:

```python
from strands_tools import calculator

# Add tools to agent
agent = BidirectionalAgent(
    model=model,
    tools=[calculator],  # Strands tools
    system_prompt="You can help with calculations."
)

# Tools are automatically called when needed
await agent.send("What's 15 * 23?")
```

## Event Types

The model emits these provider-agnostic events:

- `audioOutput`: Audio data from the model
- `textOutput`: Text content from the model  
- `toolUse`: Function call requests
- `interruptionDetected`: User interruption detected
- `BidirectionalConnectionStart`: Connection established
- `BidirectionalConnectionEnd`: Connection closed

## Audio Format

- **Input**: 24kHz, 16-bit PCM, mono
- **Output**: 24kHz, 16-bit PCM, mono
- **Encoding**: Base64 for transmission

## Error Handling

```python
try:
    await agent.start()
    # ... conversation logic
except Exception as e:
    logger.error(f"Conversation error: {e}")
finally:
    await agent.end()  # Always cleanup
```

## Limitations

1. **Image Support**: Limited compared to Chat API
2. **Session Duration**: Maximum 30 minutes per session
3. **Audio Quality**: Dependent on network conditions
4. **Rate Limits**: Subject to OpenAI API rate limits

## Examples

See the `examples/bidirectional_streaming/` directory for complete examples:

- `openai_realtime_example.py`: Text and audio conversation examples
- `test_openai_realtime.py`: Unit tests and verification

## Troubleshooting

### Common Issues

1. **Audio not working**: Ensure PyAudio is installed and microphone/speakers are available
2. **Connection errors**: Check API key and network connectivity
3. **Import errors**: Install required dependencies with `pip install strands-agents[openai,bidirectional-streaming]`

### Debug Logging

Enable debug logging to see detailed event flow:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## API Reference

### OpenAIRealtimeBidirectionalModel

Main model class for OpenAI Realtime API integration.

**Parameters:**
- `model_id` (str): Model identifier (default: "gpt-realtime")
- `api_key` (str, optional): OpenAI API key
- `base_url` (str, optional): Custom API base URL
- `organization` (str, optional): Organization ID
- `project` (str, optional): Project ID
- `**config`: Additional configuration parameters

**Methods:**
- `create_bidirectional_connection()`: Create a new streaming session

### OpenAIRealtimeSession

Session class managing the connection to OpenAI Realtime API.

**Methods:**
- `send_audio_content(audio_input)`: Send audio data
- `send_text_content(text)`: Send text message
- `send_image_content(image_input)`: Send image data
- `send_interrupt()`: Interrupt current generation
- `send_tool_result(tool_use_id, result)`: Send function call result
- `receive_events()`: Async generator for receiving events
- `close()`: Close the connection

## Contributing

When contributing to the OpenAI Realtime integration:

1. Follow the existing patterns from other model providers
2. Add comprehensive type hints
3. Include proper error handling
4. Add tests for new functionality
5. Update documentation

## License

This module is part of the Strands Agents SDK and follows the same Apache 2.0 license.