"""ISS Location Agent Server

This agent can answer questions about the International Space Station's location
and calculate distances to various cities.

Run with: uv run examples/iss.py
Then test with: uv run examples/iss_client.py
"""

from strands import Agent
from strands_tools import http_request, python_repl
from strands.protocols import A2AProtocolServer

# Create the ISS agent with tools for web requests and calculations
agent = Agent(
    tools=[http_request, python_repl],
    system_prompt="You are a helpful assistant that can answer questions about the International Space Station's location and calculate distances to various cities.",
    name="ISS Location Agent",
    description="An intelligent agent that tracks the International Space Station's real-time position and calculates distances to cities worldwide. Provides accurate geospatial analysis and space-related information.",
    # Uncomment to use a specific model:
    # model="us.amazon.nova-premier-v1:0",
    # model="us.anthropic.claude-sonnet-4-20250514-v1:0",
)

# Configure the A2A server
server_config = A2AProtocolServer(
    port=8000,
    host="0.0.0.0",
    version="1.2.3"
)

print(f"Starting ISS Location Agent...")
print(f"Model: {agent.model.config}")

# Serve the agent - it's now ready to handle requests!
server = agent.serve(server_config)

print("\n" + "="*50)
print("ISS Agent is now running!")
print(f"- Agent card: http://localhost:8000/.well-known/agent.json")
print(f"- Send requests to: http://localhost:8000/")
print("- Test with: uv run examples/iss_client.py")
print("="*50)

# Keep the server running
try:
    import time
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n\nShutting down ISS agent...")
    server.stop()
