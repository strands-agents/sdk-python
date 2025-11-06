"""Test BidirectionalAgent with simple developer experience."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from strands.experimental.bidirectional_streaming.agent.agent import BidirectionalAgent
from strands.experimental.bidirectional_streaming.models.novasonic import NovaSonicBidirectionalModel
from strands_tools import calculator


async def main():
    """Test the BidirectionalAgent API."""

    
    # Nova Sonic model
    model = NovaSonicBidirectionalModel()

    async with BidirectionalAgent(model=model, tools=[calculator]) as agent:
        print("New BidirectionalAgent Experience")
        print("Try asking: 'What is 25 times 8?' or 'Calculate the square root of 144'")
        await agent.connect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Conversation ended by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
