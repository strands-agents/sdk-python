"""Basic integration tests for Nova Sonic bidirectional streaming.

Tests fundamental functionality including multi-turn conversations, audio I/O,
text transcription, and tool execution using the new context manager approach.
"""

import logging

import pytest
from strands_tools import calculator

from strands.experimental.bidirectional_streaming.agent.agent import BidirectionalAgent
from strands.experimental.bidirectional_streaming.models.novasonic import NovaSonicBidirectionalModel

from .utils.test_context import BidirectionalTestContext

logger = logging.getLogger(__name__)


@pytest.fixture
def agent_with_calculator():
    """Provide bidirectional agent with calculator tool.
    
    Note: Session lifecycle (start/end) is handled by BidirectionalTestContext.
    """
    model = NovaSonicBidirectionalModel(region="us-east-1")
    return BidirectionalAgent(
        model=model,
        tools=[calculator],
        system_prompt="You are a helpful assistant with access to a calculator tool.",
    )

@pytest.mark.asyncio
async def test_bidirectional_agent(agent_with_calculator, audio_generator):
    """Test multi-turn conversation with follow-up questions.
    
    Validates:
    - Session lifecycle (start/end via context manager)
    - Audio input streaming
    - Speech-to-text transcription
    - Tool execution (calculator)
    - Multi-turn conversation flow
    - Text-to-speech audio output
    """
    async with BidirectionalTestContext(agent_with_calculator, audio_generator) as ctx:
        # Turn 1: Initial question
        await ctx.say("What is five plus three?")
        await ctx.wait_for_response()

        text_outputs_turn1 = ctx.get_text_outputs()
        all_text_turn1 = " ".join(text_outputs_turn1).lower()
        
        # Validate turn 1
        assert "8" in all_text_turn1 or "eight" in all_text_turn1, (
            f"Answer '8' not found in turn 1: {text_outputs_turn1}"
        )
        logger.info(f"✓ Turn 1 complete: {len(ctx.get_events())} events")

        # Turn 2: Follow-up question
        await ctx.say("Now multiply that by two")
        await ctx.wait_for_response()

        text_outputs_turn2 = ctx.get_text_outputs()
        all_text_turn2 = " ".join(text_outputs_turn2).lower()
        
        # Validate turn 2
        assert "16" in all_text_turn2 or "sixteen" in all_text_turn2, (
            f"Answer '16' not found in turn 2: {text_outputs_turn2}"
        )
        logger.info(f"✓ Turn 2 complete: {len(ctx.get_events())} total events")

        # Validate full conversation
        assert len(text_outputs_turn2) > len(text_outputs_turn1), "No new text outputs in turn 2"
        
        # Validate audio outputs
        audio_outputs = ctx.get_audio_outputs()
        assert len(audio_outputs) > 0, "No audio output received"
        total_audio_bytes = sum(len(audio) for audio in audio_outputs)
        logger.info(f"✓ Audio output: {len(audio_outputs)} chunks, {total_audio_bytes} bytes")
        
        # Summary
        logger.info("=" * 60)
        logger.info("✓ Multi-turn conversation test passed")
        logger.info(f"  Total events: {len(ctx.get_events())}")
        logger.info(f"  Text outputs: {len(text_outputs_turn2)}")
        logger.info(f"  Audio chunks: {len(audio_outputs)}")
        logger.info("=" * 60)
