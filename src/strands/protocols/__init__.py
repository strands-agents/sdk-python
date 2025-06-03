"""Protocol implementations for Strands agents.

This module provides protocol server implementations for exposing agents
through various communication protocols.
"""

from .a2a import (
    A2AProtocolServer, 
    A2AProtocolClient,
    A2ARemoteTool,
    create_agent_tool,
    create_agent_tools_from_skills,
    fetch_agent_card_sync,
    send_a2a_request_sync
)

# Protocol registry for dynamic server creation
PROTOCOL_REGISTRY = {
    "a2a": A2AProtocolServer,
    # Future protocols can be added here:
    # "mcp": MCPProtocolServer,
    # "graphql": GraphQLProtocolServer,
    # "grpc": GRPCProtocolServer,
}

__all__ = [
    # A2A Protocol
    "A2AProtocolServer",
    "A2AProtocolClient", 
    "A2ARemoteTool",
    "create_agent_tool",
    "create_agent_tools_from_skills",
    "fetch_agent_card_sync",
    "send_a2a_request_sync",
    
    # Registry
    "PROTOCOL_REGISTRY"
] 