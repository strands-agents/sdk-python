"""Test suite for bidirectional streaming with real-time audio interaction.

Tests the complete bidirectional streaming system including audio input/output,
interruption handling, and concurrent tool execution using Nova Sonic.
"""

import asyncio
import base64
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
import os
import time

import pyaudio
from strands_tools import calculator

from strands.experimental.bidirectional_streaming.agent.agent import BidiAgent
from strands.experimental.bidirectional_streaming.models.novasonic import BidiNovaSonicModel


def test_direct_tools():
    """Test direct tool calling."""
    print("Testing direct tool calling...")

    # Check AWS credentials
    if not all([os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY")]):
        print("AWS credentials not set - skipping test")
        return

    try:
        model = BidiNovaSonicModel()
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
            event_type = event.get("type", "unknown")
            
            # Handle audio stream events (bidi_audio_stream)
            if event_type == "bidi_audio_stream":
                if not context.get("interrupted", False):
                    # Decode base64 audio string to bytes for playback
                    audio_b64 = event["audio"]
                    audio_data = base64.b64decode(audio_b64)
                    context["audio_out"].put_nowait(audio_data)

            # Handle interruption events (bidi_interruption)
            elif event_type == "bidi_interruption":
                context["interrupted"] = True

            # Handle transcript events (bidi_transcript_stream)
            elif event_type == "bidi_transcript_stream":
                text_content = event.get("text", "")
                role = event.get("role", "unknown")
                
                # Log transcript output
                if role == "user":
                    print(f"User: {text_content}")
                elif role == "assistant":
                    print(f"Assistant: {text_content}")
            
            # Handle response complete events (bidi_response_complete)
            elif event_type == "bidi_response_complete":
                # Reset interrupted state since the turn is complete
                context["interrupted"] = False
            
            # Handle tool use events (tool_use_stream)
            elif event_type == "tool_use_stream":
                tool_use = event.get("current_tool_use", {})
                tool_name = tool_use.get("name", "unknown")
                tool_input = tool_use.get("input", {})
                print(f"🔧 Tool called: {tool_name} with input: {tool_input}")
            
            # Handle tool result events (tool_result)
            elif event_type == "tool_result":
                tool_result = event.get("tool_result", {})
                tool_name = tool_result.get("name", "unknown")
                result_content = tool_result.get("content", [])
                result_text = ""
                for block in result_content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        result_text = block.get("text", "")
                        break
                print(f"✅ Tool result from {tool_name}: {result_text}")

    except asyncio.CancelledError:
        pass


async def send(agent, context):
    """Send audio input to agent."""
    try:
        while time.time() - context["start_time"] < context["duration"]:
            try:
                audio_bytes = context["audio_in"].get_nowait()
                # Create audio event using TypedEvent
                from strands.experimental.bidirectional_streaming.types.events import BidiAudioInputEvent
                
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                audio_event = BidiAudioInputEvent(
                    audio=audio_b64,
                    format="pcm",
                    sample_rate=16000,
                    channels=1
                )
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
    model = BidiNovaSonicModel(region="us-east-1")
    agent = BidiAgent(model=model, tools=[calculator], system_prompt="You are a helpful assistant.")

    await agent.start()

    # Create shared context for all tasks
    context = {
        "active": True,
        "audio_in": asyncio.Queue(),
        "audio_out": asyncio.Queue(),
        "connection": agent._agent_loop,
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
        await agent.stop()


if __name__ == "__main__":
    # Test direct tool calling first
    test_direct_tools()

    asyncio.run(main())
