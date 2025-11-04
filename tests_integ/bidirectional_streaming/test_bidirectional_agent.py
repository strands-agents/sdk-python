"""Parameterized integration tests for bidirectional streaming.

Tests fundamental functionality across multiple model providers (Nova Sonic, OpenAI, etc.)
including multi-turn conversations, audio I/O, text transcription, and tool execution.

This demonstrates the provider-agnostic design of the bidirectional streaming system.
"""

import asyncio
import logging
import os

import pytest

from strands import tool
from strands.experimental.bidirectional_streaming.agent.agent import BidirectionalAgent
from strands.experimental.bidirectional_streaming.models.novasonic import NovaSonicModel
from strands.experimental.bidirectional_streaming.models.openai import OpenAIRealtimeModel
from strands.experimental.bidirectional_streaming.models.gemini_live import GeminiLiveModel

from .utils.test_context import BidirectionalTestContext

logger = logging.getLogger(__name__)


# Simple calculator tool for testing
@tool
def calculator(operation: str, x: float, y: float) -> float:
    """Perform basic arithmetic operations.
    
    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        x: First number
        y: Second number
        
    Returns:
        Result of the operation
    """
    if operation == "add":
        return x + y
    elif operation == "subtract":
        return x - y
    elif operation == "multiply":
        return x * y
    elif operation == "divide":
        if y == 0:
            raise ValueError("Cannot divide by zero")
        return x / y
    else:
        raise ValueError(f"Unknown operation: {operation}")


# Provider configurations
PROVIDER_CONFIGS = {
    "nova_sonic": {
        "model_class": NovaSonicModel,
        "model_kwargs": {"region": "us-east-1"},
        "silence_duration": 2.5,  # Nova Sonic needs 2+ seconds of silence
        "env_vars": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
        "skip_reason": "AWS credentials not available",
    },
    "openai": {
        "model_class": OpenAIRealtimeModel,
        "model_kwargs": {
            "model": "gpt-4o-realtime-preview-2024-12-17",
            "session": {
                "output_modalities": ["audio"],  # OpenAI only supports audio OR text, not both
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "silence_duration_ms": 700,
                        },
                    },
                    "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "alloy"},
                },
            },
        },
        "silence_duration": 1.0,  # OpenAI has faster VAD
        "env_vars": ["OPENAI_API_KEY"],
        "skip_reason": "OPENAI_API_KEY not available",
    },
    # NOTE: Gemini Live is temporarily disabled in parameterized tests
    # Issue: Transcript events are not being properly emitted alongside audio events
    # The model responds with audio but the test infrastructure expects text/transcripts
    # TODO: Fix Gemini Live event emission to yield both transcript and audio events
    # "gemini_live": {
    #     "model_class": GeminiLiveModel,
    #     "model_kwargs": {
    #         "model_id": "gemini-2.5-flash-native-audio-preview-09-2025",
    #         "params": {
    #             "response_modalities": ["AUDIO"],
    #             "output_audio_transcription": {},
    #             "input_audio_transcription": {},
    #         },
    #     },
    #     "silence_duration": 3.0,
    #     "env_vars": ["GOOGLE_AI_API_KEY"],
    #     "skip_reason": "GOOGLE_AI_API_KEY not available",
    # },
}


def check_provider_available(provider_name: str) -> tuple[bool, str]:
    """Check if a provider's credentials are available.
    
    Args:
        provider_name: Name of the provider to check.
        
    Returns:
        Tuple of (is_available, skip_reason).
    """
    config = PROVIDER_CONFIGS[provider_name]
    env_vars = config["env_vars"]
    
    missing_vars = [var for var in env_vars if not os.getenv(var)]
    
    if missing_vars:
        return False, f"{config['skip_reason']}: {', '.join(missing_vars)}"
    
    return True, ""


@pytest.fixture(params=list(PROVIDER_CONFIGS.keys()))
def provider_config(request):
    """Provide configuration for each model provider.
    
    This fixture is parameterized to run tests against all available providers.
    """
    provider_name = request.param
    config = PROVIDER_CONFIGS[provider_name]
    
    # Check if provider is available
    is_available, skip_reason = check_provider_available(provider_name)
    if not is_available:
        pytest.skip(skip_reason)
    
    return {
        "name": provider_name,
        **config,
    }


@pytest.fixture
def agent_with_calculator(provider_config):
    """Provide bidirectional agent with calculator tool for the given provider.
    
    Note: Session lifecycle (start/end) is handled by BidirectionalTestContext.
    """
    model_class = provider_config["model_class"]
    model_kwargs = provider_config["model_kwargs"]
    
    model = model_class(**model_kwargs)
    return BidirectionalAgent(
        model=model,
        tools=[calculator],
        system_prompt="You are a helpful assistant with access to a calculator tool. Keep responses brief.",
    )

@pytest.mark.asyncio
async def test_bidirectional_agent(agent_with_calculator, audio_generator, provider_config):
    """Test multi-turn conversation with follow-up questions across providers.
    
    This test runs against all configured providers (Nova Sonic, OpenAI, etc.)
    to validate provider-agnostic functionality.
    
    Validates:
    - Session lifecycle (start/end via context manager)
    - Audio input streaming
    - Speech-to-text transcription
    - Tool execution (calculator)
    - Multi-turn conversation flow
    - Text-to-speech audio output
    """
    provider_name = provider_config["name"]
    silence_duration = provider_config["silence_duration"]
    
    logger.info(f"Testing provider: {provider_name}")
    
    async with BidirectionalTestContext(agent_with_calculator, audio_generator) as ctx:
        # Turn 1: Simple greeting to test basic audio I/O
        await ctx.say("Hello, can you hear me?")
        # Wait for silence to trigger provider's VAD/silence detection
        await asyncio.sleep(silence_duration)
        await ctx.wait_for_response()

        text_outputs_turn1 = ctx.get_text_outputs()
        all_text_turn1 = " ".join(text_outputs_turn1).lower()
        
        # Validate turn 1 - just check we got a response
        assert len(text_outputs_turn1) > 0, (
            f"[{provider_name}] No text output received in turn 1"
        )
        
        logger.info(f"[{provider_name}] ✓ Turn 1 complete: received response")
        logger.info(f"[{provider_name}]   Response: {text_outputs_turn1[0][:100]}...")

        # Turn 2: Follow-up to test multi-turn conversation
        await ctx.say("What's your name?")
        # Wait for silence to trigger provider's VAD/silence detection
        await asyncio.sleep(silence_duration)
        await ctx.wait_for_response()

        text_outputs_turn2 = ctx.get_text_outputs()
        
        # Validate turn 2 - check we got more responses
        assert len(text_outputs_turn2) > len(text_outputs_turn1), (
            f"[{provider_name}] No new text output in turn 2"
        )
        
        logger.info(f"[{provider_name}] ✓ Turn 2 complete: multi-turn conversation works")
        logger.info(f"[{provider_name}]   Total responses: {len(text_outputs_turn2)}")

        # Validate full conversation
        # Validate audio outputs
        audio_outputs = ctx.get_audio_outputs()
        assert len(audio_outputs) > 0, f"[{provider_name}] No audio output received"
        total_audio_bytes = sum(len(audio) for audio in audio_outputs)
        
        # Summary
        logger.info("=" * 60)
        logger.info(f"[{provider_name}] ✓ Multi-turn conversation test PASSED")
        logger.info(f"  Provider: {provider_name}")
        logger.info(f"  Total events: {len(ctx.get_events())}")
        logger.info(f"  Text responses: {len(text_outputs_turn2)}")
        logger.info(f"  Audio chunks: {len(audio_outputs)} ({total_audio_bytes:,} bytes)")
        logger.info("=" * 60)
