"""Test suite for Gemini Live bidirectional streaming with camera support.

Tests the Gemini Live API with real-time audio and video interaction including:
- Audio input/output streaming
- Camera frame capture and transmission
- Interruption handling
- Concurrent tool execution
- Transcript events

Requirements:
- pip install opencv-python pillow pyaudio google-genai
- Camera access permissions
- GOOGLE_AI_API_KEY environment variable
"""

import asyncio
import base64
import io
import logging
import os
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
import time

try:
    import cv2
    import PIL.Image
    CAMERA_AVAILABLE = True
except ImportError as e:
    print(f"Camera dependencies not available: {e}")
    print("Install with: pip install opencv-python pillow")
    CAMERA_AVAILABLE = False

import pyaudio
from strands_tools import calculator

from strands.experimental.bidirectional_streaming.agent.agent import BidiAgent
from strands.experimental.bidirectional_streaming.models.gemini_live import BidiGeminiLiveModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
                print("⚠️  Interruption detected")

            # Handle transcript events (bidi_transcript_stream)
            elif event_type == "bidi_transcript_stream":
                transcript_text = event.get("text", "")
                transcript_role = event.get("role", "unknown")
                is_final = event.get("is_final", False)
                
                # Print transcripts with special formatting
                if transcript_role == "user":
                    print(f"🎤 User: {transcript_text}")
                elif transcript_role == "assistant":
                    print(f"🔊 Assistant: {transcript_text}")
            
            # Handle response complete events (bidi_response_complete)
            elif event_type == "bidi_response_complete":
                # Reset interrupted state since the response is complete
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
                # Extract text from content blocks
                result_text = ""
                for block in result_content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        result_text = block.get("text", "")
                        break
                print(f"✅ Tool result from {tool_name}: {result_text}")

    except asyncio.CancelledError:
        pass


def _get_frame(cap):
    """Capture and process a frame from camera."""
    if not CAMERA_AVAILABLE:
        return None
        
    # Read the frame
    ret, frame = cap.read()
    # Check if the frame was read successfully
    if not ret:
        return None
    # Convert BGR to RGB color space
    # OpenCV captures in BGR but PIL expects RGB format
    # This prevents the blue tint in the video feed
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = PIL.Image.fromarray(frame_rgb)
    img.thumbnail([1024, 1024])

    image_io = io.BytesIO()
    img.save(image_io, format="jpeg")
    image_io.seek(0)

    mime_type = "image/jpeg"
    image_bytes = image_io.read()
    return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}


async def get_frames(context):
    """Capture frames from camera and send to agent."""
    if not CAMERA_AVAILABLE:
        print("Camera not available - skipping video capture")
        return
        
    # This takes about a second, and will block the whole program
    # causing the audio pipeline to overflow if you don't to_thread it.
    cap = await asyncio.to_thread(cv2.VideoCapture, 0)  # 0 represents the default camera
    
    print("Camera initialized. Starting video capture...")

    try:
        while context["active"] and time.time() - context["start_time"] < context["duration"]:
            frame = await asyncio.to_thread(_get_frame, cap)
            if frame is None:
                break

            # Send frame to agent as image input
            try:
                from strands.experimental.bidirectional_streaming.types.events import BidiImageInputEvent
                
                image_event = BidiImageInputEvent(
                    image=frame["data"],  # Already base64 encoded
                    mime_type=frame["mime_type"]
                )
                await context["agent"].send(image_event)
                print("📸 Frame sent to model")
            except Exception as e:
                logger.error(f"Error sending frame: {e}")

            # Wait 1 second between frames (1 FPS)
            await asyncio.sleep(1.0)

    except asyncio.CancelledError:
        pass
    finally:
        # Release the VideoCapture object
        cap.release()


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
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                break

        context["active"] = False
    except asyncio.CancelledError:
        pass


async def main(duration=180):
    """Main function for Gemini Live bidirectional streaming test with camera support."""
    print("Starting Gemini Live bidirectional streaming test with camera...")
    print("Audio optimizations: 1024-byte buffers, balanced smooth playback + responsive interruption")
    print("Video: Camera frames sent at 1 FPS to model")

    # Get API key from environment variable
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    
    if not api_key:
        print("ERROR: GOOGLE_AI_API_KEY environment variable not set")
        print("Please set it with: export GOOGLE_AI_API_KEY=your_api_key")
        return
    
    # Initialize Gemini Live model with proper configuration
    logger.info("Initializing Gemini Live model with API key")
    
    model = BidiGeminiLiveModel(
        model_id="gemini-2.5-flash-native-audio-preview-09-2025",
        api_key=api_key,
        live_config={
            "response_modalities": ["AUDIO"],
            "output_audio_transcription": {},  # Enable output transcription
            "input_audio_transcription": {}    # Enable input transcription
        }
    )
    logger.info("Gemini Live model initialized successfully")
    print("Using Gemini Live model")
    
    agent = BidiAgent(
        model=model, 
        tools=[calculator], 
        system_prompt="You are a helpful assistant."
    )

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
        "agent": agent,  # Add agent reference for camera task
    }

    print("Speak into microphone and show things to camera. Press Ctrl+C to exit.")

    try:
        # Run all tasks concurrently including camera
        await asyncio.gather(
            play(context), 
            record(context), 
            receive(agent, context), 
            send(agent, context),
            get_frames(context),  # Add camera task
            return_exceptions=True
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
    asyncio.run(main())