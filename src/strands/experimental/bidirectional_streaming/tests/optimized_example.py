"""Example using the OptimizedAudioAdapter - clean and simple."""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from strands.experimental.bidirectional_streaming.agent.clean_agent import CleanBidirectionalAgent
from strands.experimental.bidirectional_streaming.agent.agent import BidirectionalAgent

from strands.experimental.bidirectional_streaming.models.novasonic import NovaSonicBidirectionalModel
from strands.experimental.bidirectional_streaming.adapters.optimized_audio_adapter import OptimizedAudioAdapter
from strands_tools import calculator


async def main():
    """Test the optimized audio adapter."""
    # Nova Sonic model
    model = NovaSonicBidirectionalModel()
    
    # Clean agent with tools
    agent = BidirectionalAgent(model=model, tools=[calculator])
    
    # Optimized audio adapter
    adapter = OptimizedAudioAdapter(agent)
    
    # Simple chat using context manager for automatic cleanup
    await agent.run(sender=adapter.create_output(), receiver=adapter.create_input())


if __name__ == "__main__":
    asyncio.run(main())