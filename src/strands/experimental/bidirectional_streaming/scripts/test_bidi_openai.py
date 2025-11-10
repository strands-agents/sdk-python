#!/usr/bin/env python3
"""Test OpenAI Realtime API speech-to-speech interaction."""

import asyncio
import base64
import os
import sys
import time
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pyaudio
from strands_tools import calculator

from strands.experimental.bidirectional_streaming.agent.agent import BidirectionalAgent
from strands.experimental.bidirectional_streaming.models.openai import BidiOpenAIRealtimeModel


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
        except:
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
        except:
            pass
        audio.terminate()


async def receive(agent, context):
    """Handle events from the agent."""
    try:
        async for event in agent.receive():
            if not context["active"]:
                break
            
            # Get event type
            event_type = event.get("type", "unknown")
            
            # Handle audio stream events (bidirectional_audio_stream)
            if event_type == "bidirectional_audio_stream":
                # Decode base64 audio string to bytes for playback
                audio_b64 = event["audio"]
                audio_data = base64.b64decode(audio_b64)
                
                if not context.get("interrupted", False):
                    await context["audio_out"].put(audio_data)
            
            # Handle transcript events (bidirectional_transcript_stream)
            elif event_type == "bidirectional_transcript_stream":
                source = event.get("source", "assistant")
                text = event.get("text", "").strip()
                
                if text:
                    if source == "user":
                        print(f"🎤 User: {text}")
                    elif source == "assistant":
                        print(f"🔊 Assistant: {text}")
            
            # Handle interruption events (bidirectional_interruption)
            elif event_type == "bidirectional_interruption":
                context["interrupted"] = True
                print("⚠️  Interruption detected")
            
            # Handle session start events (bidirectional_session_start)
            elif event_type == "bidirectional_session_start":
                print(f"✓ Session started: {event.get('model', 'unknown')}")
            
            # Handle session end events (bidirectional_session_end)
            elif event_type == "bidirectional_session_end":
                print(f"✓ Session ended: {event.get('reason', 'unknown')}")
                context["active"] = False
                break
            
            # Handle turn complete events (bidirectional_turn_complete)
            elif event_type == "bidirectional_turn_complete":
                # Reset interrupted state since the turn is complete
                context["interrupted"] = False
    
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
                
                # Create audio event using TypedEvent
                # Encode audio bytes to base64 string for JSON serializability
                from strands.experimental.bidirectional_streaming.types.events import BidiAudioInputEvent
                
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                audio_event = BidiAudioInputEvent(
                    audio=audio_b64,
                    format="pcm",
                    sample_rate=24000,
                    channels=1
                )
                
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
    model = BidiOpenAIRealtimeModel(
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
        system_prompt="You are a helpful voice assistant. Keep your responses brief and natural. Say hello when you first connect."
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
            await agent.stop()
        except Exception as e:
            print(f"Cleanup error: {e}")
        
        return True


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()