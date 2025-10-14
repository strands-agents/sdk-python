"""Test suite for bidirectional streaming with real-time audio interaction.

Tests the complete bidirectional streaming system including audio input/output,
interruption handling, and concurrent tool execution using Nova Sonic.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
import os
import time

import pyaudio
from strands_tools import calculator

from strands.experimental.bidirectional_streaming.agent.agent import BidirectionalAgent
from strands.experimental.bidirectional_streaming.models.novasonic import NovaSonicBidirectionalModel


def test_direct_tools():
    """Test direct tool calling."""
    print("Testing direct tool calling...")

    # Check AWS credentials
    if not all([os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY")]):
        print("AWS credentials not set - skipping test")
        return

    try:
        model = NovaSonicBidirectionalModel()
        agent = BidirectionalAgent(model=model, tools=[calculator])

        # Test calculator
        result = agent.tool.calculator(expression="2 * 3")
        content = result.get("content", [{}])[0].get("text", "")
        print(f"Result: {content}")
        print("Test completed")

    except Exception as e:
        print(f"Test failed: {e}")


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
    microphone = audio.open(
        channels=1,
        format=pyaudio.paInt16,
        frames_per_buffer=1024,
        input=True,
        rate=16000,
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
    model = NovaSonicBidirectionalModel(region="us-east-1")
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
    # Test direct tool calling first
    test_direct_tools()

    asyncio.run(main())
