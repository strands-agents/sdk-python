#!/usr/bin/env python3
"""Test OpenAI Realtime API speech-to-speech interaction."""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pyaudio
from strands_tools import calculator

from strands.experimental.bidirectional_streaming.agent.agent import BidirectionalAgent
from strands.experimental.bidirectional_streaming.models.openai import OpenAIRealtimeBidirectionalModel


def test_direct_tool_calling():
    """Test direct tool calling functionality."""
    print("Testing direct tool calling...")
    
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("OPENAI_API_KEY not set - skipping test")
            return
        
        model = OpenAIRealtimeBidirectionalModel(model="gpt-4o-realtime-preview", api_key=api_key)
        agent = BidirectionalAgent(model=model, tools=[calculator])
        
        # Test calculator
        result = agent.tool.calculator(expression="2 * 3")
        content = result.get("content", [{}])[0].get("text", "")
        print(f"Result: {content}")
        print("Test completed")
        
    except Exception as e:
        print(f"Test failed: {e}")


async def play(context):
    """Handle audio playback with interruption support."""
    audio = pyaudio.PyAudio()
    
    try:
        speaker = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,  # OpenAI Realtime uses 24kHz
            output=True,
            frames_per_buffer=1024,
        )
        
        while context["active"]:
            try:
                # Check for interruption
                if context.get("interrupted", False):
                    # Clear audio queue on interruption
                    while not context["audio_out"].empty():
                        try:
                            context["audio_out"].get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    
                    context["interrupted"] = False
                    await asyncio.sleep(0.05)
                    continue
                
                # Get audio data with timeout
                try:
                    audio_data = await asyncio.wait_for(context["audio_out"].get(), timeout=0.1)
                    
                    if audio_data and context["active"]:
                        # Play in chunks to allow interruption
                        chunk_size = 1024
                        for i in range(0, len(audio_data), chunk_size):
                            if context.get("interrupted", False) or not context["active"]:
                                break
                            
                            chunk = audio_data[i:i + chunk_size]
                            speaker.write(chunk)
                            await asyncio.sleep(0.001)  # Brief pause for responsiveness
                
                except asyncio.TimeoutError:
                    continue
                    
            except asyncio.CancelledError:
                break
        
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Audio playback error: {e}")
    finally:
        try:
            speaker.close()
        except Exception:
            pass
        audio.terminate()


async def record(context):
    """Handle microphone recording."""
    audio = pyaudio.PyAudio()
    
    try:
        microphone = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,  # Match OpenAI's expected input rate
            input=True,
            frames_per_buffer=1024,
        )
        
        while context["active"]:
            try:
                audio_bytes = microphone.read(1024, exception_on_overflow=False)
                await context["audio_in"].put(audio_bytes)
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                break
        
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Microphone recording error: {e}")
    finally:
        try:
            microphone.close()
        except Exception:
            pass
        audio.terminate()


async def receive(agent, context):
    """Handle events from the agent."""
    try:
        async for event in agent.receive():
            if not context["active"]:
                break
            
            # Handle audio output
            if "audioOutput" in event:
                audio_data = event["audioOutput"]["audioData"]
                
                if not context.get("interrupted", False):
                    await context["audio_out"].put(audio_data)
            
            # Handle text output (transcripts)
            elif "textOutput" in event:
                text_output = event["textOutput"]
                role = text_output.get("role", "assistant")
                text = text_output.get("text", "").strip()
                
                if text:
                    if role == "user":
                        print(f"User: {text}")
                    elif role == "assistant":
                        print(f"Assistant: {text}")
            
            # Handle interruption detection
            elif "interruptionDetected" in event:
                context["interrupted"] = True
            
            # Handle connection events
            elif "BidirectionalConnectionStart" in event:
                pass  # Silent connection start
            elif "BidirectionalConnectionEnd" in event:
                context["active"] = False
                break
    
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Receive handler error: {e}")
    finally:
        pass


async def send(agent, context):
    """Send audio from microphone to agent."""
    try:
        while context["active"]:
            try:
                audio_bytes = await asyncio.wait_for(context["audio_in"].get(), timeout=0.1)
                
                # Create audio event in expected format
                audio_event = {
                    "audioData": audio_bytes,
                    "format": "pcm",
                    "sampleRate": 24000,
                    "channels": 1
                }
                
                await agent.send(audio_event)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
    
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Send handler error: {e}")
    finally:
        pass


async def main():
    """Main test function for OpenAI voice chat."""
    print("Starting OpenAI Realtime API test...")
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY environment variable not set")
        return False
    
    # Check audio system
    try:
        audio = pyaudio.PyAudio()
        audio.terminate()
    except Exception as e:
        print(f"Audio system error: {e}")
        return False
    
    # Create OpenAI model
    model = OpenAIRealtimeBidirectionalModel(
        model="gpt-4o-realtime-preview",
        api_key=api_key,
        session={
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "silence_duration_ms": 700
                    }
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "voice": "alloy"
                }
            }
        }
    )
    
    # Create agent
    agent = BidirectionalAgent(
        model=model,
        tools=[calculator],
        system_prompt=(
            "You are a helpful voice assistant. Keep your responses brief and natural. "
            "Say hello when you first connect."
        )
    )
    
    # Start the session
    await agent.start()
    
    # Create shared context
    context = {
        "active": True,
        "audio_in": asyncio.Queue(),
        "audio_out": asyncio.Queue(),
        "interrupted": False,
        "start_time": time.time()
    }
    
    print("Speak into your microphone. Press Ctrl+C to stop.")
    
    try:
        # Run all tasks concurrently
        await asyncio.gather(
            play(context),
            record(context),
            receive(agent, context),
            send(agent, context),
            return_exceptions=True
        )
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except asyncio.CancelledError:
        print("\nTest cancelled")
    except Exception as e:
        print(f"\nError during voice chat: {e}")
    finally:
        print("Cleaning up...")
        context["active"] = False
        
        try:
            await agent.end()
        except Exception as e:
            print(f"Cleanup error: {e}")
    
    return True


if __name__ == "__main__":
    # Test direct tool calling first
    print("OpenAI Realtime API Test Suite")
    test_direct_tool_calling()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()