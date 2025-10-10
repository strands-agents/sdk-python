#!/usr/bin/env python3
"""Validation script for OpenAI Realtime model provider.

This script validates that the OpenAI Realtime model provider can be imported
and initialized correctly without requiring an actual API key or connection.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

def test_imports():
    """Test that all required imports work correctly."""
    print("Testing imports...")
    
    try:
        # Import directly to avoid other model dependencies
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        
        from openai_realtime import (
            OpenAIRealtimeBidirectionalModel,
            OpenAIRealtimeSession
        )
        print("✓ OpenAI Realtime model imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    try:
        from openai import AsyncOpenAI
        import openai
        
        # Check version
        version = openai.__version__
        major, minor, patch = map(int, version.split('.'))
        min_version = (1, 107, 0)
        current_version = (major, minor, patch)
        
        if current_version >= min_version:
            print(f"✓ OpenAI SDK import successful (version {version})")
        else:
            print(f"✗ OpenAI SDK version {version} is too old")
            print("  Realtime API requires >= 1.107.0")
            print("  Install with: pip install openai>=1.107.0")
            return False
            
    except ImportError as e:
        print(f"✗ OpenAI SDK not available: {e}")
        print("  Install with: pip install openai>=1.107.0")
        return False
    
    return True


def test_model_initialization():
    """Test that the model can be initialized without API key."""
    print("\nTesting model initialization...")
    
    try:
        from openai_realtime import (
            OpenAIRealtimeBidirectionalModel
        )
        
        # Test initialization without API key (should work for validation)
        model = OpenAIRealtimeBidirectionalModel(
            model_id="gpt-realtime",
            api_key="test-key",  # Dummy key for validation
            params={
                "output_modalities": ["text"],
                "instructions": "Test instructions"
            }
        )
        
        print("✓ Model initialization successful")
        print(f"  Model ID: {model.model_id}")
        print(f"  Config keys: {list(model.config.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model initialization error: {e}")
        return False


def test_type_annotations():
    """Test that type annotations are working correctly."""
    print("\nTesting type annotations...")
    
    try:
        from openai_realtime import (
            OpenAIRealtimeSession
        )
        from bidirectional_model import (
            BidirectionalModelSession
        )
        
        # Check that OpenAIRealtimeSession is a subclass of BidirectionalModelSession
        if issubclass(OpenAIRealtimeSession, BidirectionalModelSession):
            print("✓ Type hierarchy correct")
        else:
            print("✗ Type hierarchy incorrect")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Type annotation error: {e}")
        return False


def test_event_types():
    """Test that event type definitions are working."""
    print("\nTesting event types...")
    
    try:
        from strands.experimental.bidirectional_streaming.types.bidirectional_streaming import (
            AudioInputEvent,
            AudioOutputEvent,
            TextOutputEvent,
            InterruptionDetectedEvent,
            BidirectionalConnectionStartEvent,
            BidirectionalConnectionEndEvent
        )
        
        print("✓ Event type imports successful")
        
        # Test creating sample events
        audio_input: AudioInputEvent = {
            "audioData": b"test",
            "format": "pcm",
            "sampleRate": 24000,
            "channels": 1
        }
        
        audio_output: AudioOutputEvent = {
            "audioData": b"test",
            "format": "pcm", 
            "sampleRate": 24000,
            "channels": 1,
            "encoding": "base64"
        }
        
        print("✓ Event type creation successful")
        return True
        
    except Exception as e:
        print(f"✗ Event type error: {e}")
        return False


def main():
    """Run all validation tests."""
    print("OpenAI Realtime Model Provider Validation")
    print("=" * 45)
    
    tests = [
        test_imports,
        test_model_initialization,
        test_type_annotations,
        test_event_types
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 45)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All validation tests passed!")
        print("\nThe OpenAI Realtime model provider is ready to use.")
        print("Set OPENAI_API_KEY environment variable to test with real API.")
        return True
    else:
        print("✗ Some validation tests failed.")
        print("Please check the errors above and install missing dependencies.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)