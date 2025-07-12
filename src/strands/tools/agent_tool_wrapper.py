"""Agent tool wrapper that enables using Agent objects as tools."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..agent.agent import Agent

from ..types.tools import AgentTool, ToolGenerator, ToolResult, ToolResultContent, ToolSpec, ToolUse


class AgentToolWrapper(AgentTool):
    """Wrapper that makes an Agent usable as a tool.

    This class enables the agents-as-tools pattern by wrapping an Agent
    and implementing the AgentTool interface. The wrapped agent can then
    be used as a tool by other agents.
    """

    def __init__(self, agent: "Agent"):
        """Initialize the agent tool wrapper.

        Args:
            agent: The Agent instance to wrap
        """
        super().__init__()
        self._agent = agent
        self._validate_agent()
        self._name = agent.name
        self._description = agent.description or ""

    def _validate_agent(self) -> None:
        # Check if agent has the required attributes and they are properly set
        if (
            not hasattr(self._agent, "name")
            or not hasattr(self._agent, "description")
            or not self._agent.name
            or self._agent.name == "Strands Agents"  # Default agent name
            or not self._agent.description
        ):
            raise ValueError(
                "Agent must have both 'name' and 'description' parameters "
                "to be used as a tool. 'name' must not be default agent name: 'Strands Agents'. "
                "Initialize the Agent with: "
                "Agent(name='tool_name', description='tool_description', ...)"
            )

    @property
    def tool_name(self) -> str:
        """The unique name of the tool used for identification and invocation."""
        return self._name

    @property
    def tool_spec(self) -> ToolSpec:
        """Tool specification that describes its functionality and parameters."""
        return ToolSpec(
            name=self._name,
            description=self._description,
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The query or task to send to the sub-agent"}
                },
                "required": ["query"],
            },
        )

    @property
    def tool_type(self) -> str:
        """The type of the tool implementation."""
        return "agent"

    async def stream(self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any) -> ToolGenerator:
        """Stream tool events by delegating to the wrapped agent.

        Args:
            tool_use: The tool use request containing tool ID and parameters
            invocation_state: Context for the tool invocation, including agent state
            **kwargs: Additional keyword arguments for future extensibility

        Yields:
            Tool events with the last being the tool result
        """
        try:
            # Extract the query from tool input
            query = tool_use["input"].get("query", "")

            # Invoke the sub-agent
            result = await self._agent.invoke_async(query)

            # Convert agent response to tool result format
            tool_result = ToolResult(
                toolUseId=tool_use["toolUseId"], status="success", content=[ToolResultContent(text=str(result))]
            )

            yield tool_result

        except Exception as e:
            # Return error result
            tool_result = ToolResult(
                toolUseId=tool_use["toolUseId"],
                status="error",
                content=[ToolResultContent(text=f"Error executing '{self._name}': {str(e)}")],
            )
            yield tool_result

    def get_display_properties(self) -> dict[str, str]:
        """Get properties to display in UI representations of this tool.

        Returns:
            Dictionary of property names and their string values
        """
        return {
            "Name": self.tool_name,
            "Type": self.tool_type,
            "Description": self._description,
        }
