"""Test suite for bidirectional streaming with real-time audio interaction.

Tests the complete bidirectional streaming system including audio input/output,
interruption handling, and concurrent tool execution using Nova Sonic.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
import time

import pyaudio
from strands_tools import calculator

from strands.experimental.bidirectional_streaming.agent.agent import BidirectionalAgent
from strands.experimental.bidirectional_streaming.models.novasonic import NovaSonicBidirectionalModel
from strands.experimental.bidirectional_streaming.models.gemini_live import GeminiLiveBidirectionalModel

# Configure logging - debug only for Gemini Live, info for everything else
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
gemini_logger = logging.getLogger('strands.experimental.bidirectional_streaming.models.gemini_live')
gemini_logger.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


async def play(context):
    """Play audio output with responsive interruption support."""
    audio = pyaudio.PyAudio()
    speaker = audio.open(
        channels=1,
        format=pyaudio.paInt16,
        output=True,
        rate=24000,
        frames_per_buffer=1024,
    )

    try:
        while context["active"]:
            try:
                # Check for interruption first
                if context.get("interrupted", False):
                    # Clear entire audio queue immediately
                    while not context["audio_out"].empty():
                        try:
                            context["audio_out"].get_nowait()
                        except asyncio.QueueEmpty:
                            break

                    context["interrupted"] = False
                    await asyncio.sleep(0.05)
                    continue

                # Get next audio data
                audio_data = await asyncio.wait_for(context["audio_out"].get(), timeout=0.1)

                if audio_data and context["active"]:
                    chunk_size = 1024
                    for i in range(0, len(audio_data), chunk_size):
                        # Check for interruption before each chunk
                        if context.get("interrupted", False) or not context["active"]:
                            break

                        end = min(i + chunk_size, len(audio_data))
                        chunk = audio_data[i:end]
                        speaker.write(chunk)
                        await asyncio.sleep(0.001)

            except asyncio.TimeoutError:
                continue  # No audio available
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                break

    except asyncio.CancelledError:
        pass
    finally:
        speaker.close()
        audio.terminate()


async def record(context):
    """Record audio input from microphone."""
    audio = pyaudio.PyAudio()
    
    # List all available audio devices
    print("Available audio devices:")
    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:  # Only show input devices
            print(f"  Device {i}: {device_info['name']} (inputs: {device_info['maxInputChannels']})")
    
    # Get default input device info
    default_device = audio.get_default_input_device_info()
    print(f"\nUsing default input device: {default_device['name']} (Device {default_device['index']})")
    
    microphone = audio.open(
        channels=1,
        format=pyaudio.paInt16,
        frames_per_buffer=1024,
        input=True,
        rate=16000,
        input_device_index=6,  # Use default, or specify a device index
    )

    try:
        while context["active"]:
            try:
                audio_bytes = microphone.read(1024, exception_on_overflow=False)
                context["audio_in"].put_nowait(audio_bytes)
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                break
    except asyncio.CancelledError:
        pass
    finally:
        microphone.close()
        audio.terminate()


async def receive(agent, context):
    """Receive and process events from agent."""
    try:
        async for event in agent.receive():
            # Handle audio output
            if "audioOutput" in event:
                if not context.get("interrupted", False):
                    context["audio_out"].put_nowait(event["audioOutput"]["audioData"])

            # Handle interruption events
            elif "interruptionDetected" in event:
                context["interrupted"] = True
            elif "interrupted" in event:
                context["interrupted"] = True

            # Handle text output with interruption detection
            elif "textOutput" in event:
                text_content = event["textOutput"].get("content", "")
                role = event["textOutput"].get("role", "unknown")

                # Check for text-based interruption patterns
                if '{ "interrupted" : true }' in text_content:
                    context["interrupted"] = True
                elif "interrupted" in text_content.lower():
                    context["interrupted"] = True

                # Log text output
                if role.upper() == "USER":
                    print(f"User: {text_content}")
                elif role.upper() == "ASSISTANT":
                    print(f"Assistant: {text_content}")
            
            # Handle turn complete events (if we add them back)
            elif "turnComplete" in event:
                logger.debug("Turn complete event received - model ready for next input")
                # Reset interrupted state since the turn is complete
                context["interrupted"] = False

    except asyncio.CancelledError:
        pass


async def send(agent, context):
    """Send audio input to agent."""
    try:
        while time.time() - context["start_time"] < context["duration"]:
            try:
                audio_bytes = context["audio_in"].get_nowait()
                audio_event = {"audioData": audio_bytes, "format": "pcm", "sampleRate": 16000, "channels": 1}
                await agent.send(audio_event)
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.01)  # Restored to working timing
            except asyncio.CancelledError:
                break

        context["active"] = False
    except asyncio.CancelledError:
        pass


async def main(duration=180):
    """Main function for bidirectional streaming test."""
    print("Starting bidirectional streaming test...")
    print("Audio optimizations: 1024-byte buffers, balanced smooth playback + responsive interruption")

    # Initialize model and agent
    # Get API key from environment variable
    import os
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    
    if api_key:
        # Use Gemini Live with proper configuration
        logger.info("Initializing Gemini Live model with API key")
        
        # Use a model ID that works with v1alpha API
        model = GeminiLiveBidirectionalModel(
            model_id="gemini-2.5-flash-native-audio-preview-09-2025",  # Add models/ prefix
            api_key=api_key,
            params={
                "response_modalities": ["AUDIO"]
            }
        )
        logger.info("Gemini Live model initialized successfully")
        print("Using Gemini Live model")
    else:
        # Fallback to Nova Sonic
        logger.info("No Gemini API key found, using Nova Sonic")
        model = NovaSonicBidirectionalModel(region="us-east-1")
        print("Using Nova Sonic model (no Gemini API key found)")
    agent = BidirectionalAgent(model=model, tools=[calculator], system_prompt="You are a helpful assistant.")

    await agent.start()

    # Create shared context for all tasks
    context = {
        "active": True,
        "audio_in": asyncio.Queue(),
        "audio_out": asyncio.Queue(),
        "connection": agent._session,
        "duration": duration,
        "start_time": time.time(),
        "interrupted": False,
    }

    print("Speak into microphone. Press Ctrl+C to exit.")

    try:
        # Run all tasks concurrently
        await asyncio.gather(
            play(context), record(context), receive(agent, context), send(agent, context), return_exceptions=True
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except asyncio.CancelledError:
        print("\nTest cancelled")
    finally:
        print("Cleaning up...")
        context["active"] = False
        await agent.end()


if __name__ == "__main__":
    asyncio.run(main())
