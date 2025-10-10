"""Test script for OpenAI Realtime API bidirectional streaming.

Simple test to verify the OpenAI Realtime model provider works correctly
with text input/output and basic conversation flow.

Requirements:
- pip install openai>=1.107.0
- OPENAI_API_KEY environment variable
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from strands.experimental.bidirectional_streaming.agent.agent import BidirectionalAgent
from strands.experimental.bidirectional_streaming.models.openai_realtime import OpenAIRealtimeBidirectionalModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_openai_realtime_text():
    """Test OpenAI Realtime API with text input/output."""
    print("Testing OpenAI Realtime API with text...")
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Initialize model and agent
    model = OpenAIRealtimeBidirectionalModel(
        model_id="gpt-realtime",
        api_key=api_key,
        params={
            "output_modalities": ["text"],  # Text only for this test
            "instructions": "You are a helpful assistant. Keep responses brief and friendly."
        }
    )
    
    agent = BidirectionalAgent(
        model=model,
        system_prompt="You are a helpful assistant. Keep responses brief and friendly."
    )
    
    try:
        # Start the agent
        await agent.start()
        print("Agent started successfully")
        
        # Send a text message
        test_message = "Hello! Can you tell me a short joke?"
        print(f"Sending: {test_message}")
        await agent.send(test_message)
        
        # Receive and print responses
        response_count = 0
        async for event in agent.receive():
            if "textOutput" in event:
                text_content = event["textOutput"]["text"]
                role = event["textOutput"]["role"]
                print(f"{role.capitalize()}: {text_content}")
                response_count += 1
                
                # Stop after receiving some responses
                if response_count >= 10:  # Adjust as needed
                    break
            
            elif "BidirectionalConnectionStart" in event:
                print("Connection started")
            
            elif "BidirectionalConnectionEnd" in event:
                print("Connection ended")
                break
        
        print("Test completed successfully!")
        
    except Exception as e:
        logger.error("Error during test: %s", e)
        raise
    finally:
        # Clean up
        await agent.end()
        print("Agent stopped")


async def test_openai_realtime_audio():
    """Test OpenAI Realtime API with audio input/output (requires audio hardware)."""
    print("Testing OpenAI Realtime API with audio...")
    
    try:
        import pyaudio
    except ImportError:
        print("PyAudio not available - skipping audio test")
        print("Install with: pip install pyaudio")
        return
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Initialize model and agent
    model = OpenAIRealtimeBidirectionalModel(
        model_id="gpt-realtime",
        api_key=api_key,
        params={
            "output_modalities": ["audio", "text"],
            "audio": {
                "input": {"turn_detection": {"type": "server_vad"}},
                "output": {"voice": "alloy"}
            },
            "instructions": "You are a helpful assistant. Keep responses brief."
        }
    )
    
    agent = BidirectionalAgent(
        model=model,
        system_prompt="You are a helpful assistant. Keep responses brief."
    )
    
    try:
        # Start the agent
        await agent.start()
        print("Agent started successfully")
        print("Audio test would require microphone and speaker setup")
        print("For now, just testing connection...")
        
        # Test with text input to verify connection
        await agent.send("Hello, this is a test message.")
        
        # Receive a few events to verify it's working
        event_count = 0
        async for event in agent.receive():
            if "textOutput" in event:
                print(f"Received text: {event['textOutput']['text']}")
                event_count += 1
                if event_count >= 5:
                    break
            elif "audioOutput" in event:
                print(f"Received audio: {len(event['audioOutput']['audioData'])} bytes")
                event_count += 1
                if event_count >= 5:
                    break
            elif "BidirectionalConnectionEnd" in event:
                break
        
        print("Audio test connection verified!")
        
    except Exception as e:
        logger.error("Error during audio test: %s", e)
        raise
    finally:
        # Clean up
        await agent.end()
        print("Agent stopped")


async def main():
    """Run OpenAI Realtime API tests."""
    print("OpenAI Realtime API Test Suite")
    print("=" * 40)
    
    # Test text functionality
    await test_openai_realtime_text()
    
    print("\n" + "=" * 40)
    
    # Test audio functionality (basic connection test)
    await test_openai_realtime_audio()
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    asyncio.run(main())