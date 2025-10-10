#!/usr/bin/env python3
"""OpenAI Realtime API bidirectional streaming example.

This example demonstrates how to use the OpenAI Realtime API with the Strands
bidirectional streaming module for real-time voice conversations.

Requirements:
- pip install strands-agents[openai,bidirectional-streaming]
- OPENAI_API_KEY environment variable
- Microphone and speakers for audio interaction

Usage:
    python examples/bidirectional_streaming/openai_realtime_example.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the src directory to Python path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    print("PyAudio not available. Install with: pip install pyaudio")
    print("Also ensure OpenAI SDK >= 1.107.0: pip install openai>=1.107.0")
    AUDIO_AVAILABLE = False

from strands.experimental.bidirectional_streaming.agent.agent import BidirectionalAgent
from strands.experimental.bidirectional_streaming.models.openai_realtime import OpenAIRealtimeBidirectionalModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def text_conversation_example():
    """Example of text-based conversation with OpenAI Realtime API."""
    print("OpenAI Realtime API - Text Conversation Example")
    print("=" * 50)
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set OPENAI_API_KEY environment variable")
        return
    
    # Initialize model with text output
    model = OpenAIRealtimeBidirectionalModel(
        model_id="gpt-realtime",
        api_key=api_key,
        params={
            "output_modalities": ["text"],
            "instructions": "You are a helpful assistant. Keep responses concise and friendly."
        }
    )
    
    # Create agent
    agent = BidirectionalAgent(
        model=model,
        system_prompt="You are a helpful assistant. Keep responses concise and friendly."
    )
    
    try:
        # Start the conversation
        await agent.start()
        print("Agent started. Type 'quit' to exit.\n")
        
        while True:
            # Get user input
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            # Send message to agent
            await agent.send(user_input)
            
            # Collect and display response
            response_text = ""
            async for event in agent.receive():
                if "textOutput" in event:
                    text_chunk = event["textOutput"]["text"]
                    response_text += text_chunk
                    print(text_chunk, end="", flush=True)
                elif "BidirectionalConnectionEnd" in event:
                    break
                # Stop after getting a complete response (you might want to adjust this logic)
                elif response_text and response_text.endswith(('.', '!', '?')):
                    # Simple heuristic to detect end of response
                    await asyncio.sleep(0.1)  # Brief pause to collect any remaining text
                    break
            
            print("\n")  # New line after response
    
    except KeyboardInterrupt:
        print("\nConversation interrupted by user")
    except Exception as e:
        logger.error("Error during conversation: %s", e)
    finally:
        await agent.end()
        print("Conversation ended.")


async def audio_conversation_example():
    """Example of audio-based conversation with OpenAI Realtime API."""
    if not AUDIO_AVAILABLE:
        print("Audio not available - skipping audio example")
        return
    
    print("OpenAI Realtime API - Audio Conversation Example")
    print("=" * 50)
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set OPENAI_API_KEY environment variable")
        return
    
    # Initialize model with audio output
    model = OpenAIRealtimeBidirectionalModel(
        model_id="gpt-realtime",
        api_key=api_key,
        params={
            "output_modalities": ["audio", "text"],
            "audio": {
                "input": {
                    "turn_detection": {"type": "server_vad"}  # Voice activity detection
                },
                "output": {
                    "voice": "alloy"  # Choose voice: alloy, echo, fable, onyx, nova, shimmer
                }
            },
            "instructions": "You are a helpful voice assistant. Keep responses brief and conversational."
        }
    )
    
    # Create agent
    agent = BidirectionalAgent(
        model=model,
        system_prompt="You are a helpful voice assistant. Keep responses brief and conversational."
    )
    
    # Audio setup
    audio = pyaudio.PyAudio()
    
    # Microphone setup
    microphone = audio.open(
        channels=1,
        format=pyaudio.paInt16,
        frames_per_buffer=1024,
        input=True,
        rate=24000,  # OpenAI Realtime expects 24kHz
    )
    
    # Speaker setup
    speaker = audio.open(
        channels=1,
        format=pyaudio.paInt16,
        output=True,
        rate=24000,
        frames_per_buffer=1024,
    )
    
    try:
        # Start the conversation
        await agent.start()
        print("Voice agent started. Speak into your microphone!")
        print("Press Ctrl+C to exit.\n")
        
        # Create tasks for audio input/output
        async def record_audio():
            """Record audio from microphone and send to agent."""
            while True:
                try:
                    audio_data = microphone.read(1024, exception_on_overflow=False)
                    audio_event = {
                        "audioData": audio_data,
                        "format": "pcm",
                        "sampleRate": 24000,
                        "channels": 1
                    }
                    await agent.send(audio_event)
                    await asyncio.sleep(0.01)
                except Exception as e:
                    logger.error("Error recording audio: %s", e)
                    break
        
        async def play_audio():
            """Play audio output from agent."""
            async for event in agent.receive():
                if "audioOutput" in event:
                    audio_data = event["audioOutput"]["audioData"]
                    speaker.write(audio_data)
                elif "textOutput" in event:
                    # Also print text for debugging
                    text = event["textOutput"]["text"]
                    if text.strip():
                        print(f"Assistant: {text}")
                elif "BidirectionalConnectionEnd" in event:
                    break
        
        # Run audio input and output concurrently
        await asyncio.gather(
            record_audio(),
            play_audio(),
            return_exceptions=True
        )
    
    except KeyboardInterrupt:
        print("\nVoice conversation interrupted by user")
    except Exception as e:
        logger.error("Error during voice conversation: %s", e)
    finally:
        # Cleanup
        microphone.close()
        speaker.close()
        audio.terminate()
        await agent.end()
        print("Voice conversation ended.")


async def main():
    """Main function to run examples."""
    print("OpenAI Realtime API Examples")
    print("=" * 30)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    print("Choose an example:")
    print("1. Text conversation")
    print("2. Audio conversation (requires microphone/speakers)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        await text_conversation_example()
    elif choice == "2":
        await audio_conversation_example()
    else:
        print("Invalid choice. Running text conversation example...")
        await text_conversation_example()


if __name__ == "__main__":
    asyncio.run(main())