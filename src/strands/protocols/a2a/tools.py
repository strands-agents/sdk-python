"""A2A Remote Tool implementation.

This module provides tools that wrap remote A2A agents, enabling the
"Agent as Tool" pattern for seamless agent composition.
"""

import asyncio
import logging
from typing import Any, Dict, Optional, List, Union

from .client import A2AProtocolClient
from ...types.tools import AgentTool, ToolSpec, ToolUse, ToolResult, ToolResultContent

logger = logging.getLogger(__name__)


class A2ARemoteTool(AgentTool):
    """Wraps a remote A2A agent as a Strands tool."""
    
    def __init__(
        self, 
        agent_url: str, 
        skill_id: Optional[str] = None,
        name: Optional[str] = None,
        auth_token: Optional[str] = None
    ):
        """Initialize A2A remote tool.
        
        Args:
            agent_url: URL of the remote A2A agent
            skill_id: Optional specific skill to expose as tool
            name: Optional custom name for the tool
            auth_token: Optional authentication token
        """
        super().__init__()
        self.agent_url = agent_url
        self.skill_id = skill_id
        self._custom_name = name
        self.client = A2AProtocolClient(agent_url, auth_token)
        self.agent_card = None
        self._discovered = False
        self._tool_spec: Optional[ToolSpec] = None
        
    def _ensure_discovered(self):
        """Ensure agent capabilities are discovered."""
        if not self._discovered:
            # Use sync discovery for simplicity
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                self.agent_card = loop.run_until_complete(self.client.fetch_agent_card())
                self._discovered = True
                # Generate and cache tool spec
                self._tool_spec = self._generate_tool_spec()
            finally:
                loop.close()
    
    @property
    def tool_name(self) -> str:
        """The unique name of the tool used for identification and invocation."""
        if self._custom_name:
            return self._custom_name
            
        self._ensure_discovered()
        
        if self.skill_id and self.agent_card:
            # Find the specific skill
            for skill in self.agent_card.skills:
                skill_dict = skill if isinstance(skill, dict) else skill.dict()
                if skill_dict.get("id") == self.skill_id:
                    return f"remote_{skill_dict.get('name', self.skill_id).lower().replace(' ', '_')}"
        
        # Use agent name
        if self.agent_card:
            return f"remote_{self.agent_card.name.lower().replace(' ', '_')}"
            
        return "remote_agent"
    
    @property
    def tool_spec(self) -> ToolSpec:
        """Tool specification that describes its functionality and parameters."""
        self._ensure_discovered()
        if self._tool_spec:
            return self._tool_spec
        # Generate on demand if needed
        return self._generate_tool_spec()
    
    @property
    def tool_type(self) -> str:
        """The type of the tool implementation."""
        return "a2a_remote"
    
    @property
    def supports_hot_reload(self) -> bool:
        """Remote tools don't support hot reload."""
        return False
    
    def _generate_tool_spec(self) -> ToolSpec:
        """Generate tool spec from A2A agent card."""
        if not self.agent_card:
            raise RuntimeError("Failed to discover remote agent")
        
        # If targeting a specific skill
        if self.skill_id:
            for skill in self.agent_card.skills:
                skill_dict = skill if isinstance(skill, dict) else skill.dict()
                if skill_dict.get("id") == self.skill_id:
                    return self._skill_to_tool_spec(skill_dict)
            raise ValueError(f"Skill {self.skill_id} not found in agent")
        
        # Otherwise, create a general tool spec for the entire agent
        skill_names = []
        all_examples = []
        all_tags = set()
        
        for skill in self.agent_card.skills:
            skill_dict = skill if isinstance(skill, dict) else skill.dict()
            skill_names.append(skill_dict.get("name", skill_dict.get("id", "unknown")))
            all_examples.extend(skill_dict.get("examples", []))
            all_tags.update(skill_dict.get("tags", []))
        
        # Build inputSchema in the correct format
        json_schema = {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message or task to send to the remote agent"
                },
                "skill": {
                    "type": "string",
                    "description": f"Optional specific skill to use",
                    "enum": [s.get("id") if isinstance(s, dict) else s.id for s in self.agent_card.skills]
                }
            },
            "required": ["message"]
        }
        
        return ToolSpec(
            name=self.tool_name,
            description=f"{self.agent_card.name}: {self.agent_card.description}. Available skills: {', '.join(skill_names)}",
            inputSchema={"json": json_schema}
        )
    
    def _skill_to_tool_spec(self, skill: Dict[str, Any]) -> ToolSpec:
        """Convert an A2A skill to a Strands tool spec."""
        # Build inputSchema in the correct format
        json_schema = {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": f"Input for {skill.get('name', 'the skill')}"
                }
            },
            "required": ["message"]
        }
        
        return ToolSpec(
            name=self.tool_name,
            description=skill.get("description", f"Use the {skill.get('name', 'remote')} skill"),
            inputSchema={"json": json_schema}
        )
    
    def get_tool_spec(self) -> Dict[str, Any]:
        """Legacy method - returns the tool spec as a dict."""
        return dict(self.tool_spec)
    
    async def _execute_async(self, **kwargs) -> str:
        """Execute the remote agent call asynchronously."""
        message = kwargs.get("message", "")
        skill = kwargs.get("skill", self.skill_id)
        
        if not message:
            raise ValueError("Message is required")
            
        try:
            result = await self.client.send_task_and_wait(
                message=message,
                skill_id=skill
            )
            return result
        except Exception as e:
            logger.error(f"Error calling remote agent {self.agent_url}: {e}")
            raise
    
    def invoke(self, tool: ToolUse, *args: Any, **kwargs: dict[str, Any]) -> ToolResult:
        """Execute the tool's functionality with the given tool use request.
        
        Args:
            tool: The tool use request containing tool ID and parameters.
            *args: Positional arguments to pass to the tool.
            **kwargs: Keyword arguments to pass to the tool.
            
        Returns:
            The result of the tool execution.
        """
        # Extract parameters from tool use
        tool_input = tool.get("input", {})
        message = tool_input.get("message", "")
        skill = tool_input.get("skill", self.skill_id)
        
        # Run async method in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_text = loop.run_until_complete(
                self._execute_async(message=message, skill=skill)
            )
            
            # Return properly formatted ToolResult
            return ToolResult(
                toolUseId=tool["toolUseId"],
                content=[ToolResultContent(text=result_text)],
                status="success"
            )
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return ToolResult(
                toolUseId=tool["toolUseId"],
                content=[ToolResultContent(text=f"Error: {str(e)}")],
                status="error"
            )
        finally:
            loop.close()
    
    def __call__(self, **kwargs) -> str:
        """Execute remote agent call (legacy interface).
        
        Args:
            message: Message to send to the remote agent
            skill: Optional specific skill to use
            **kwargs: Additional arguments (ignored)
            
        Returns:
            The agent's response as a string
        """
        # Create a synthetic tool use for the legacy interface
        tool_use = ToolUse(
            toolUseId=f"legacy_{id(self)}",
            name=self.tool_name,
            input=kwargs
        )
        
        result = self.invoke(tool_use)
        
        # Extract text from result
        if result["status"] == "success" and result["content"]:
            for content in result["content"]:
                if "text" in content:
                    return content["text"]
        
        return "No response"


def create_agent_tool(
    agent_url: str, 
    name: Optional[str] = None,
    skill_id: Optional[str] = None,
    auth_token: Optional[str] = None
) -> A2ARemoteTool:
    """Factory to create a tool from a remote A2A agent.
    
    Args:
        agent_url: URL of the remote A2A agent
        name: Optional custom name for the tool
        skill_id: Optional specific skill to expose
        auth_token: Optional authentication token
        
    Returns:
        A2ARemoteTool instance ready to be registered
        
    Example:
        ```python
        # Add a remote agent as a tool
        agent.register_tool(create_agent_tool("http://research-agent:8000"))
        
        # Or with a custom name
        agent.register_tool(create_agent_tool(
            "http://research-agent:8000", 
            name="research"
        ))
        ```
    """
    tool = A2ARemoteTool(agent_url, skill_id, name, auth_token)
    
    # Pre-discover to validate the agent exists
    tool._ensure_discovered()
    
    return tool


def create_agent_tools_from_skills(
    agent_url: str,
    skills: Optional[List[str]] = None,
    auth_token: Optional[str] = None
) -> List[A2ARemoteTool]:
    """Create multiple tools from an agent's skills.
    
    Args:
        agent_url: URL of the remote A2A agent
        skills: List of skill IDs to create tools for (None = all skills)
        auth_token: Optional authentication token
        
    Returns:
        List of A2ARemoteTool instances, one per skill
        
    Example:
        ```python
        # Register specific skills as separate tools
        tools = create_agent_tools_from_skills(
            "http://utility-agent:8000",
            skills=["calculate", "translate"]
        )
        for tool in tools:
            agent.register_tool(tool)
        ```
    """
    # First, discover the agent
    temp_tool = A2ARemoteTool(agent_url, auth_token=auth_token)
    temp_tool._ensure_discovered()
    
    if not temp_tool.agent_card:
        raise RuntimeError(f"Failed to discover agent at {agent_url}")
    
    # Get all available skill IDs
    available_skills = []
    for skill in temp_tool.agent_card.skills:
        skill_dict = skill if isinstance(skill, dict) else skill.dict()
        available_skills.append(skill_dict.get("id"))
    
    # Filter to requested skills
    if skills:
        skill_ids = [s for s in skills if s in available_skills]
        if not skill_ids:
            raise ValueError(f"No requested skills found. Available: {available_skills}")
    else:
        skill_ids = available_skills
    
    # Create a tool for each skill
    tools = []
    for skill_id in skill_ids:
        tool = A2ARemoteTool(
            agent_url, 
            skill_id=skill_id,
            auth_token=auth_token
        )
        tool.agent_card = temp_tool.agent_card  # Reuse discovered card
        tool._discovered = True
        tools.append(tool)
    
    return tools 