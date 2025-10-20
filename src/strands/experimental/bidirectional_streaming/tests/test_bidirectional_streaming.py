"""Test suite for bidirectional streaming with real-time audio and video interaction.

Tests the complete bidirectional streaming system including audio input/output,
image/video input from camera, interruption handling, and concurrent tool execution.

Supports multiple providers:
- Gemini Live (GOOGLE_AI_API_KEY)
- Nova Sonic (AWS credentials)
- OpenAI Realtime (OPENAI_API_KEY)

Requirements:
- pip install opencv-python pillow pyaudio
- Camera access permissions
- Provider API keys (see above)

Usage:
    python test_bidirectional_streaming.py [--provider gemini|nova|openai] [--duration 180] [--camera] [--debug]
"""

import argparse
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

from strands import tool
from strands.experimental.bidirectional_streaming.agent.agent import BidirectionalAgent
from strands.experimental.bidirectional_streaming.models.gemini_live import GeminiLiveBidirectionalModel
from strands.experimental.bidirectional_streaming.models.novasonic import NovaSonicBidirectionalModel
from strands.experimental.bidirectional_streaming.models.openai import OpenAIRealtimeBidirectionalModel
from strands.experimental.bidirectional_streaming.types.bidirectional_streaming import (
    AudioInputEvent,
    ImageInputEvent,
)

# Logger will be configured in main() based on --debug flag
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
        if device_info["maxInputChannels"] > 0:  # Only show input devices
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
        # input_device_index=6,  # Use default, or specify a device index
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
    """Receive and process events from agent - now returns dicts like normal agent."""
    try:
        async for event in agent.receive():
            # All events are now dicts (converted via as_dict())
            # Check for specific event types by their discriminator keys
            
            if event.get("session_start"):
                logger.info(f"Session started: {event.get('session_id')} with model {event.get('model')}")
                
            elif event.get("turn_start"):
                logger.debug(f"Turn started: {event.get('turn_id')}")
                
            elif event.get("audio_stream"):
                if not context.get("interrupted", False):
                    context["audio_out"].put_nowait(event.get("audio"))
                    
            elif event.get("transcript_stream"):
                # Print transcripts with special formatting
                source = event.get("source")
                text = event.get("text")
                if source == "user":
                    print(f"🎤 User: {text}")
                elif source == "assistant":
                    print(f"🔊 Assistant: {text}")
                    
            elif event.get("current_tool_use"):
                # ToolUseStreamEvent from core
                current = event.get("current_tool_use", {})
                tool_name = current.get("name", "unknown")
                tool_id = current.get("toolUseId", "unknown")
                logger.info(f"Tool use requested: {tool_name} (id: {tool_id})")
                print(f"Using tools: {tool_name}  (id: {tool_id})")
            elif event.get("interruption"):
                logger.debug(f"Interruption detected: {event.get('reason')}")
                context["interrupted"] = True
                
            elif event.get("turn_complete"):
                logger.debug(f"Turn complete: {event.get('turn_id')} (reason: {event.get('stop_reason')})")
                context["interrupted"] = False
                
            elif event.get("type") == "multimodal_usage":
                logger.debug(f"Usage: {event.get('inputTokens')} in, {event.get('outputTokens')} out")
                
            elif event.get("session_end"):
                logger.info(f"Session ended: {event.get('reason')}")
                context["active"] = False
                
            # Fallback: Handle legacy TypedDict events for backward compatibility
            elif "audioOutput" in event:
                if not context.get("interrupted", False):
                    context["audio_out"].put_nowait(event["audioOutput"]["audioData"])
            
            elif "interruptionDetected" in event or "interrupted" in event:
                context["interrupted"] = True
            
            elif "textOutput" in event:
                text_content = event["textOutput"].get("text", "")
                role = event["textOutput"].get("role", "unknown")
                if role.upper() == "USER":
                    print(f"User: {text_content}")
                elif role.upper() == "ASSISTANT":
                    print(f"Assistant: {text_content}")
            
            elif "transcript" in event:
                transcript_text = event["transcript"].get("text", "")
                transcript_role = event["transcript"].get("role", "unknown")
                if transcript_role.upper() == "USER":
                    print(f"🎤 User (transcript): {transcript_text}")
                elif transcript_role.upper() == "ASSISTANT":
                    print(f"🔊 Assistant (transcript): {transcript_text}")

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
    # Fix: Convert BGR to RGB color space
    # OpenCV captures in BGR but PIL expects RGB format
    # This prevents the blue tint in the video feed
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
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

            # Send frame to agent as image input using new TypedEvent API
            try:
                image_event = ImageInputEvent(
                    image=frame["data"],
                    mime_type=frame["mime_type"],
                    encoding="base64"
                )
                await context["agent"].send(image_event)
                print("📸 Frame sent to model")
            except Exception as e:
                logger.error(f"Error sending frame: {e}")

            # Wait 1 second between frames (1 FPS like the example)
            await asyncio.sleep(1.0)

    except asyncio.CancelledError:
        pass
    finally:
        # Release the VideoCapture object
        cap.release()


async def send(agent, context):
    """Send audio input to agent using new TypedEvent API."""
    try:
        while time.time() - context["start_time"] < context["duration"]:
            try:
                audio_bytes = context["audio_in"].get_nowait()
                audio_event = AudioInputEvent(
                    audio=audio_bytes,
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


def create_model(provider: str):
    """Create a model instance based on provider selection."""
    if provider == "gemini":
        api_key = os.getenv("GOOGLE_AI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_AI_API_KEY environment variable not set")

        logger.info("Initializing Gemini Live model")
        model = GeminiLiveBidirectionalModel(
            model_id="gemini-2.5-flash-native-audio-preview-09-2025",
            api_key=api_key,
            params={
                "response_modalities": ["AUDIO"],
                "output_audio_transcription": {},  # Enable output transcription
                "input_audio_transcription": {},  # Enable input transcription
            },
        )
        print("✓ Using Gemini Live model")
        return model

    elif provider == "nova":
        logger.info("Initializing Nova Sonic model")
        model = NovaSonicBidirectionalModel(region="us-east-1")
        print("✓ Using Nova Sonic model")
        return model

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        logger.info("Initializing OpenAI Realtime model")
        model = OpenAIRealtimeBidirectionalModel(
            model_id="gpt-4o-realtime-preview-2024-12-17",
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
        print("✓ Using OpenAI Realtime model")
        return model

    else:
        raise ValueError(f"Unknown provider: {provider}. Choose from: gemini, nova, openai")


def configure_logging(debug: bool = False):
    """Configure logging based on debug flag."""

    # Console handler - only errors and critical
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    if debug:
        # Debug mode: All logs to file, only errors to console
        log_filename = f"bidirectional_streaming_debug_{int(time.time())}.log"
        
        
        # File handler - captures everything at DEBUG level
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    
        root_logger.addHandler(file_handler)
        print(f"🐛 Debug logging enabled - writing to {log_filename}")
        print(f"   All logs → {log_filename}")
        print(f"   Errors only → terminal\n")
        
        # Enable debug logging for all bidirectional streaming modules
        for module_name in [
            "strands.experimental.bidirectional_streaming.models.gemini_live",
            "strands.experimental.bidirectional_streaming.models.novasonic",
            "strands.experimental.bidirectional_streaming.models.openai",
            "strands.experimental.bidirectional_streaming.agent.agent",
            "strands.experimental.bidirectional_streaming.event_loop.bidirectional_event_loop",
        ]:
            logging.getLogger(module_name).setLevel(logging.DEBUG)
    else:
        # Normal mode: Log to console with INFO level
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )


async def main(provider: str = "nova", duration: int = 180, enable_camera: bool = False, debug: bool = False):
    """Main function for bidirectional streaming test with camera support."""
    # Configure logging first
    configure_logging(debug)
    
    print(f"\n{'='*60}")
    print("Bidirectional Streaming Test")
    print(f"{'='*60}")
    print(f"Provider: {provider.upper()}")
    print(f"Duration: {duration}s")
    print(f"Camera: {'Enabled' if enable_camera and CAMERA_AVAILABLE else 'Disabled'}")
    if debug:
        print(f"Debug: Enabled (logging to file)")
    print(f"{'='*60}\n")

    # Initialize model and agent
    try:
        model = create_model(provider)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return

    # Define a simple calculator tool
    @tool
    def calculator(operation: str, a: float, b: float) -> float:
        """Perform basic math operations.
        
        Args:
            operation: The operation to perform (add, subtract, multiply, divide)
            a: First number
            b: Second number
            
        Returns:
            Result of the operation
        """
        print(f"🧮 Calculator: {operation} {a} {b}")
        if operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            return a / b if b != 0 else float('inf')
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
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
        "agent": agent,  # Add agent reference for camera task
    }

    print("🎤 Speak into microphone")
    if enable_camera and CAMERA_AVAILABLE:
        print("📸 Show things to camera")
    print("⌨️  Press Ctrl+C to exit\n")

    try:
        # Build task list
        tasks = [play(context), record(context), receive(agent, context), send(agent, context)]

        # Add camera task if enabled
        if enable_camera and CAMERA_AVAILABLE:
            tasks.append(get_frames(context))

        # Run all tasks concurrently
        await asyncio.gather(*tasks, return_exceptions=True)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except asyncio.CancelledError:
        print("\n\n⚠️  Test cancelled")
    finally:
        print("🧹 Cleaning up...")
        context["active"] = False
        await agent.end()
        print("✓ Done\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test bidirectional streaming with multiple providers")
    parser.add_argument(
        "--provider",
        type=str,
        default="nova",
        choices=["gemini", "nova", "openai"],
        help="Model provider to use (default: nova)",
    )
    parser.add_argument("--duration", type=int, default=180, help="Test duration in seconds (default: 180)")
    parser.add_argument("--camera", action="store_true", help="Enable camera/video input (disabled by default)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging to file")

    args = parser.parse_args()

    asyncio.run(main(
        provider=args.provider,
        duration=args.duration,
        enable_camera=args.camera,
        debug=args.debug
    ))
