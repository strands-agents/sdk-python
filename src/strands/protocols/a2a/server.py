"""A2A Protocol Server implementation.

This module implements the A2A (Agent-to-Agent) protocol server following
the ProtocolServer abstraction pattern.
"""

import json
import logging
import threading
from typing import Any, Dict, Optional, TypedDict, Union
from typing_extensions import Unpack, override

import uvicorn
from fastapi import FastAPI, Request, Response
from a2a.types import AgentCard, AgentSkill

from ...types.protocols import ProtocolServer

logger = logging.getLogger(__name__)


class A2AProtocolServer(ProtocolServer):
    """A2A protocol server implementation with auto-discovery."""
    
    class A2AConfig(TypedDict, total=False):
        """Configuration options for A2A servers.
        
        Attributes:
            host: Host to bind the server to. Defaults to "0.0.0.0".
            port: Port to bind the server to. Defaults to 8000.
            version: A2A protocol version. Defaults to "1.0.0".
            enable_auth: Whether to enable authentication. Defaults to False.
            auth_token: Bearer token for authentication if enabled.
            max_concurrent_tasks: Maximum concurrent tasks. Defaults to 1.
            tls_cert: Path to TLS certificate file.
            tls_key: Path to TLS private key file.
            cors_origins: List of allowed CORS origins. Defaults to ["*"].
        """
        
        host: str
        port: int  
        version: str
        enable_auth: bool
        auth_token: Optional[str]
        max_concurrent_tasks: int
        tls_cert: Optional[str]
        tls_key: Optional[str]
        cors_origins: list[str]
    
    def __init__(
        self,
        **server_config: Unpack[A2AConfig]
    ) -> None:
        """Initialize A2A server instance.
        
        Args:
            **server_config: Configuration options for the A2A server.
        """
        self.config = A2AProtocolServer.A2AConfig(
            host="0.0.0.0",
            port=8000,
            version="1.0.0",
            enable_auth=False,
            max_concurrent_tasks=1,
            cors_origins=["*"]
        )
        self.update_config(**server_config)
        
        logger.debug("config=<%s> | initializing", self.config)
        
        self.app = FastAPI()
        self.agent: Optional[Any] = None
        self.agent_card: Optional[AgentCard] = None
        self._is_running = False
        self._server_thread: Optional[threading.Thread] = None
        self._uvicorn_server: Optional[uvicorn.Server] = None
        
    @override
    def update_config(self, **server_config: Unpack[A2AConfig]) -> None:  # type: ignore[override]
        """Update the A2A server configuration.
        
        Args:
            **server_config: Configuration overrides.
        """
        self.config.update(server_config)
        
    @override
    def get_config(self) -> A2AConfig:
        """Get the A2A server configuration.
        
        Returns:
            The A2A server configuration.
        """
        return self.config
    
    @property
    @override
    def protocol_name(self) -> str:
        """Return the protocol name."""
        return "a2a"
    
    @property
    @override
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._is_running
        
    @override
    def get_endpoint(self) -> str:
        """Get the server's endpoint URL.
        
        Returns:
            The URL where the A2A server is accessible.
        """
        host = self.config.get("host", "0.0.0.0")
        port = self.config.get("port", 8000)
        
        # Convert 0.0.0.0 to localhost for easier access
        if host == "0.0.0.0":
            host = "localhost"
            
        protocol = "https" if self.config.get("tls_cert") else "http"
        return f"{protocol}://{host}:{port}"
    
    def _create_agent_card(self, agent: Any) -> AgentCard:
        """Create an A2A agent card from the agent's tools.
        
        Args:
            agent: The Strands agent instance.
            
        Returns:
            An A2A AgentCard with auto-discovered capabilities.
        """
        # Get all tool configurations
        tool_configs = agent.tool_registry.get_all_tools_config()
        
        # Generate skills from tools
        skills = []
        for tool_name, tool_spec in tool_configs.items():
            # Generate intelligent tags from description
            description = tool_spec.get("description", "")
            tags = self._generate_tags(description)
            
            # Generate examples from input schema
            examples = self._generate_examples(tool_spec)
            
            skill = {
                "id": tool_name,  # Use tool name as skill ID
                "name": tool_spec.get("name", tool_name),
                "description": description,
                "tags": tags,
                "examples": examples,
                "inputMode": "text/plain",
                "outputMode": "text/plain"
            }
            skills.append(skill)
        
        # Create agent card
        name = getattr(agent, "name", agent.__class__.__name__)
        description = getattr(agent, "description", f"A Strands agent with {len(skills)} tools")
        
        return AgentCard(
            name=name,
            description=description,
            url=self.get_endpoint(),
            version=self.config.get("version", "1.0.0"),
            skills=skills,
            capabilities={
                "streaming": False,
                "pushNotifications": False,
                "stateTransitionHistory": True
            },
            authentication={
                "schemes": ["Bearer"] if self.config.get("enable_auth") else []
            },
            defaultInputModes=["text", "text/plain"],
            defaultOutputModes=["text", "text/plain"]
        )
    
    def _generate_tags(self, description: str) -> list[str]:
        """Generate intelligent tags from a description."""
        tags = []
        desc_lower = description.lower()
        
        # Category mappings
        tag_mappings = {
            "calculation": ["calculat", "math", "comput", "sum", "average"],
            "search": ["search", "find", "lookup", "query", "fetch"],
            "file": ["file", "read", "write", "save", "load"],
            "web": ["http", "url", "web", "api", "request"],
            "data": ["data", "process", "transform", "analyz", "parse"],
            "communication": ["email", "message", "notif", "send", "chat"],
            "time": ["time", "date", "schedule", "calendar", "timezone"]
        }
        
        for tag, keywords in tag_mappings.items():
            if any(keyword in desc_lower for keyword in keywords):
                tags.append(tag)
                
        # Default tag if none found
        if not tags:
            tags.append("general")
            
        return tags
    
    def _generate_examples(self, tool_spec: dict) -> list[str]:
        """Generate example usage from tool input schema."""
        examples = []
        
        # Get custom examples if provided
        if "examples" in tool_spec:
            return tool_spec["examples"]
            
        # Otherwise generate from schema
        input_schema = tool_spec.get("input_schema", {})
        properties = input_schema.get("properties", {})
        
        if properties:
            # Create a simple example with required fields
            required = input_schema.get("required", [])
            example_parts = []
            
            for prop, schema in properties.items():
                if prop in required:
                    prop_type = schema.get("type", "string")
                    if prop_type == "string":
                        example_parts.append(f"{prop}: 'example'")
                    elif prop_type == "number":
                        example_parts.append(f"{prop}: 123")
                    elif prop_type == "boolean":
                        example_parts.append(f"{prop}: true")
                        
            if example_parts:
                examples.append(f"Use with {', '.join(example_parts)}")
                
        # Add a generic example based on description
        if not examples and "description" in tool_spec:
            examples.append(f"Help me {tool_spec['description'].lower()}")
            
        return examples
    
    def _setup_routes(self) -> None:
        """Set up the FastAPI routes for A2A."""
        
        @self.app.get("/.well-known/agent.json")
        async def get_agent_card() -> Dict[str, Any]:
            """Serve the agent card."""
            if not self.agent_card:
                return {"error": "Agent card not initialized"}
            return self.agent_card.dict()
        
        @self.app.post("/")
        async def handle_task(request: Request) -> Dict[str, Any]:
            """Handle A2A task requests."""
            # Check authentication if enabled
            if self.config.get("enable_auth"):
                auth_header = request.headers.get("Authorization")
                expected_token = f"Bearer {self.config.get('auth_token')}"
                
                if auth_header != expected_token:
                    return {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32001,
                            "message": "Unauthorized"
                        }
                    }
            
            # Parse JSON-RPC request
            try:
                body = await request.json()
                method = body.get("method")
                params = body.get("params", {})
                request_id = body.get("id")
                
                # Extract the user message
                user_message = params.get("message", params.get("prompt", ""))
                
                # Call the agent
                result = self.agent(user_message)
                
                # Format response
                response_text = str(result)
                
                return {
                    "jsonrpc": "2.0",
                    "result": {
                        "content": response_text,
                        "status": "completed"
                    },
                    "id": request_id
                }
                
            except Exception as e:
                logger.error(f"Error processing A2A request: {e}")
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": str(e)
                    },
                    "id": body.get("id") if "body" in locals() else None
                }
    
    @override
    def start(self, agent: Any) -> None:
        """Start the A2A server for the given agent.
        
        Args:
            agent: The agent to expose via A2A.
        """
        if self._is_running:
            logger.warning("A2A server already running")
            return
            
        self.agent = agent
        self.agent_card = self._create_agent_card(agent)
        self._setup_routes()
        
        # Configure uvicorn
        config = uvicorn.Config(
            app=self.app,
            host=self.config.get("host", "0.0.0.0"),
            port=self.config.get("port", 8000),
            log_level="info"
        )
        
        # Add TLS if configured
        if self.config.get("tls_cert") and self.config.get("tls_key"):
            config.ssl_certfile = self.config["tls_cert"]
            config.ssl_keyfile = self.config["tls_key"]
        
        self._uvicorn_server = uvicorn.Server(config)
        
        # Start in background thread
        def run_server():
            try:
                self._is_running = True
                logger.info(f"Starting A2A server on {self.get_endpoint()}")
                logger.info(f"Agent Card: {self.get_endpoint()}/.well-known/agent.json")
                
                # Print discovered skills
                logger.info(f"Discovered {len(self.agent_card.skills)} skills:")
                for skill in self.agent_card.skills:
                    skill_info = skill if isinstance(skill, dict) else {
                        "name": getattr(skill, "name", "Unknown"),
                        "description": getattr(skill, "description", ""),
                        "tags": getattr(skill, "tags", [])
                    }
                    logger.info(f"  • {skill_info['name']}: {skill_info['description']}")
                    if skill_info.get("tags"):
                        logger.info(f"    Tags: {', '.join(skill_info['tags'])}")
                        
                self._uvicorn_server.run()
            finally:
                self._is_running = False
        
        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        
        # Give server time to start
        import time
        time.sleep(0.5)
    
    @override  
    def stop(self) -> None:
        """Stop the A2A server."""
        if not self._is_running:
            return
            
        logger.info("Stopping A2A server")
        
        if self._uvicorn_server:
            self._uvicorn_server.should_exit = True
            
        if self._server_thread:
            self._server_thread.join(timeout=5)
            
        self._is_running = False
        self.agent = None
        self.agent_card = None 