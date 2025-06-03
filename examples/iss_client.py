"""Example client for the ISS agent.

This shows how to interact with the ISS agent once it's running.
First run: uv run examples/iss_agent.py
Then run: uv run examples/iss_client.py
"""

import time
from strands.protocols import A2AProtocolClient
import asyncio

# The URL where your ISS agent is running
AGENT_URL = "http://localhost:8000"

async def test_agent():
    """Test the ISS agent with better error handling."""
    
    # Check if server is running first
    print("Checking if agent server is running...")
    
    async with A2AProtocolClient(AGENT_URL) as client:
        try:
            # Try to fetch agent card with shorter timeout first
            print("Fetching agent card...")
            agent_card = await client.fetch_agent_card()
            print(f"\n✅ Connected to agent!")
            print(f"Agent: {agent_card.name}")
            print(f"Description: {agent_card.description}")
            print(f"Available skills: {[skill.name for skill in agent_card.skills]}")
            
        except Exception as e:
            print(f"❌ Failed to connect to agent at {AGENT_URL}")
            print(f"Error: {e}")
            print("\nMake sure the agent server is running:")
            print("  uv run examples/iss_agent.py")
            return
        
        # Now send your ISS question with longer timeout
        print("\n" + "="*50)
        print("Sending ISS question to agent...")
        print("This may take a while as the agent needs to:")
        print("- Look up real-time ISS position")
        print("- Calculate distances to multiple cities")
        print("- Perform complex calculations")
        print("="*50 + "\n")
        
        try:
            # Use longer timeout for complex calculation
            response = await client.send_task_and_wait(
                message="Who is the closest to the ISS? People in: "
                        "Portland, Vancouver, Seattle, or New York? "
                        "First, lookup realtime information about the position of the ISS. "
                        "Give me the altitude of the ISS, and the distance and vector from the closest city to the ISS. "
                        "After you give me the answer, explain your reasoning and show me any code you used",
                timeout=120.0  # 2 minutes for complex calculation
            )
            
            print("🚀 ISS Agent Response:")
            print("="*50)
            print(response)
            
        except TimeoutError as e:
            print(f"⏱️  Request timed out: {e}")
            print("The agent may be taking longer than expected.")
            print("Try again or increase the timeout.")
            
        except Exception as e:
            print(f"❌ Error during request: {e}")

# Run the async client
if __name__ == "__main__":
    asyncio.run(test_agent()) 