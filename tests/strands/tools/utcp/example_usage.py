"""Example usage of UTCP integration with Strands SDK.

This example demonstrates how to use the UTCP integration to connect to UTCP providers
and use their tools within the Strands agent framework.
"""

import asyncio
import json
from pathlib import Path

from .utcp_client import UTCPClient


async def example_utcp_integration():
    """Example showing how to use UTCP integration with Strands."""

    # Example providers.json configuration
    providers_config = [
        {"name": "example_api", "provider_type": "http", "url": "https://api.example.com/utcp", "http_method": "GET"},
        {"name": "local_tools", "provider_type": "text", "file_path": "/path/to/local/tools.json"},
    ]

    # Create a temporary providers.json file for this example
    providers_file = Path("/tmp/providers.json")
    with open(providers_file, "w") as f:
        json.dump(providers_config, f, indent=2)

    # UTCP client configuration
    utcp_config = {
        "providers_file_path": str(providers_file),
        "load_variables_from": [{"type": "dotenv", "env_file_path": ".env"}],
    }

    # Use UTCP client with async context manager
    async with UTCPClient(utcp_config) as utcp_client:
        print("UTCP client initialized successfully")

        # List all available tools
        tools_list = utcp_client.list_tools_sync()
        print(f"Found {len(tools_list.items)} UTCP tools:")

        for tool in tools_list.items:
            print(f"  - {tool.tool_name}: {tool.tool_spec['description']}")

        # Search for specific tools
        weather_tools = await utcp_client.search_tools("weather", max_results=5)
        print(f"\nFound {len(weather_tools)} weather-related tools:")

        for tool in weather_tools:
            print(f"  - {tool.tool_name}: {tool.tool_spec['description']}")

        # Example of calling a tool (if available)
        if tools_list.items:
            example_tool = tools_list.items[0]
            print(f"\nCalling example tool: {example_tool.tool_name}")

            try:
                # Create a mock tool use request
                tool_use = {
                    "toolUseId": "example-123",
                    "input": {},  # Add appropriate input based on tool schema
                }

                # Call the tool asynchronously
                result = await utcp_client.call_tool_async(
                    tool_use_id=tool_use["toolUseId"], tool_name=example_tool.tool_name, arguments=tool_use["input"]
                )

                print(f"Tool result: {result}")

            except Exception as e:
                print(f"Error calling tool: {e}")


def example_utcp_with_strands_agent():
    """Example showing how to integrate UTCP tools with a Strands agent."""

    # This would be used within a Strands agent context
    # The UTCP tools can be registered with the agent's tool registry

    print("""
    Example integration with Strands Agent:
    
    from strands.tools.utcp import UTCPClient
    from strands.agents import Agent
    
    async def create_agent_with_utcp_tools():
        # Initialize UTCP client
        utcp_config = {
            "providers_file_path": "./providers.json"
        }
        
        async with UTCPClient(utcp_config) as utcp_client:
            # Get UTCP tools
            utcp_tools = utcp_client.list_tools_sync()
            
            # Create agent with UTCP tools
            agent = Agent(
                name="UTCP-enabled Agent",
                tools=utcp_tools.items  # UTCPAgentTool instances
            )
            
            # Use the agent normally
            response = await agent.run("Get the weather for London")
            return response
    """)


if __name__ == "__main__":
    print("UTCP Integration Example")
    print("=" * 40)

    # Run the async example
    try:
        asyncio.run(example_utcp_integration())
    except Exception as e:
        print(f"Example failed (expected if no real UTCP providers): {e}")

    # Show the Strands integration example
    example_utcp_with_strands_agent()
