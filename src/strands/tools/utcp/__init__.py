"""Universal Tool Calling Protocol (UTCP) integration.

This package provides integration with the Universal Tool Calling Protocol (UTCP), allowing agents to use tools
provided by UTCP servers and providers. UTCP offers a simple and scalable support for
multiple transport protocols.

Usage:
    ```python
    from strands.agent import Agent
    from strands.tools.utcp import UTCPClient

    # Create UTCP client
    config = {"providers_file_path": "/path/to/providers.json"}
    async with UTCPClient(config) as utcp_client:
        # Use UTCP tools directly with Strands agent (same pattern as MCP)
        agent = Agent(tools=utcp_client.list_tools_sync())
        result = agent("Use the weather tool to get current conditions")
    ```

- Docs: https://github.com/universal-tool-calling-protocol/python-utcp
"""

from .utcp_agent_tool import UTCPAgentTool
from .utcp_client import UTCPClient

__all__ = ["UTCPAgentTool", "UTCPClient"]
