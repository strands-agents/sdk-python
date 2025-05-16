"""Tool-related type definitions for the SDK.

These types are modeled after the Bedrock API.

- Bedrock docs: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_Types_Amazon_Bedrock_Runtime.html
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from typing_extensions import TypedDict

from .media import DocumentContent, ImageContent

if TYPE_CHECKING:
    from .content import Messages
    from .models import Model

JSONSchema = dict
"""Type alias for JSON Schema dictionaries."""


class ToolSpec(TypedDict):
    """Specification for a tool that can be used by an agent.

    Attributes:
        description: A human-readable description of what the tool does.
        inputSchema: JSON Schema defining the expected input parameters.
        name: The unique name of the tool.
    """

    description: str
    inputSchema: JSONSchema
    name: str


class Tool(TypedDict):
    """A tool that can be provided to a model.

    This type wraps a tool specification for inclusion in a model request.

    Attributes:
        toolSpec: The specification of the tool.
    """

    toolSpec: ToolSpec


class ToolUse(TypedDict):
    """A request from the model to use a specific tool with the provided input.

    Attributes:
        input: The input parameters for the tool.
            Can be any JSON-serializable type.
        name: The name of the tool to invoke.
        toolUseId: A unique identifier for this specific tool use request.
    """

    input: Any
    name: str
    toolUseId: str


class ToolResultContent(TypedDict, total=False):
    """Content returned by a tool execution.

    Attributes:
        document: Document content returned by the tool.
        image: Image content returned by the tool.
        json: JSON-serializable data returned by the tool.
        text: Text content returned by the tool.
    """

    document: DocumentContent
    image: ImageContent
    json: Any
    text: str


ToolResultStatus = Literal["success", "error"]
"""Status of a tool execution result."""


class ToolResult(TypedDict):
    """Result of a tool execution.

    Attributes:
        content: List of result content returned by the tool.
        status: The status of the tool execution ("success" or "error").
        toolUseId: The unique identifier of the tool use request that produced this result.
    """

    content: List[ToolResultContent]
    status: ToolResultStatus
    toolUseId: str


class ToolChoiceAuto(TypedDict):
    """Configuration for automatic tool selection.

    This represents the configuration for automatic tool selection, where the model decides whether and which tool to
    use based on the context.
    """

    pass


class ToolChoiceAny(TypedDict):
    """Configuration indicating that the model must request at least one tool."""

    pass


class ToolChoiceTool(TypedDict):
    """Configuration for forcing the use of a specific tool.

    Attributes:
        name: The name of the tool that the model must use.
    """

    name: str


ToolChoice = Union[
    Dict[Literal["auto"], ToolChoiceAuto],
    Dict[Literal["any"], ToolChoiceAny],
    Dict[Literal["tool"], ToolChoiceTool],
]
"""
Configuration for how the model should choose tools.

- "auto": The model decides whether to use tools based on the context
- "any": The model must use at least one tool (any tool)
- "tool": The model must use the specified tool
"""


class ToolConfig(TypedDict):
    """Configuration for tools in a model request.

    Attributes:
        tools: List of tools available to the model.
        toolChoice: Configuration for how the model should choose tools.
    """

    tools: List[Tool]
    toolChoice: ToolChoice


class AgentTool(ABC):
    """Abstract base class for all SDK tools.

    This class defines the interface that all tool implementations must follow. Each tool must provide its name,
    specification, and implement an invoke method that executes the tool's functionality.
    """

    _is_dynamic: bool

    def __init__(self) -> None:
        """Initialize the base agent tool with default dynamic state."""
        self._is_dynamic = False

    @property
    @abstractmethod
    # pragma: no cover
    def tool_name(self) -> str:
        """The unique name of the tool used for identification and invocation."""
        pass

    @property
    @abstractmethod
    # pragma: no cover
    def tool_spec(self) -> ToolSpec:
        """Tool specification that describes its functionality and parameters."""
        pass

    @property
    @abstractmethod
    # pragma: no cover
    def tool_type(self) -> str:
        """The type of the tool implementation (e.g., 'python', 'javascript', 'lambda').

        Used for categorization and appropriate handling.
        """
        pass

    @property
    def supports_hot_reload(self) -> bool:
        """Whether the tool supports automatic reloading when modified.

        Returns:
            False by default.
        """
        return False

    @abstractmethod
    # pragma: no cover
    def invoke(self, tool: ToolUse, *args: Any, **kwargs: dict[str, Any]) -> ToolResult:
        """Execute the tool's functionality with the given tool use request.

        Args:
            tool: The tool use request containing tool ID and parameters.
            *args: Positional arguments to pass to the tool.
            **kwargs: Keyword arguments to pass to the tool.

        Returns:
            The result of the tool execution.
        """
        pass

    @property
    def is_dynamic(self) -> bool:
        """Whether the tool was dynamically loaded during runtime.

        Dynamic tools may have different lifecycle management.

        Returns:
            True if loaded dynamically, False otherwise.
        """
        return self._is_dynamic

    def mark_dynamic(self) -> None:
        """Mark this tool as dynamically loaded."""
        self._is_dynamic = True

    def get_display_properties(self) -> dict[str, str]:
        """Get properties to display in UI representations of this tool.

        Subclasses can extend this to include additional properties.

        Returns:
            Dictionary of property names and their string values.
        """
        return {
            "Name": self.tool_name,
            "Type": self.tool_type,
        }


class ToolHandler(ABC):
    """Abstract base class for handling tool execution within the agent framework."""

    @abstractmethod
    # pragma: no cover
    def preprocess(
        self,
        tool: ToolUse,
        tool_config: ToolConfig,
        **kwargs: Any,
    ) -> Optional[ToolResult]:
        """Preprocess a tool use request before execution.

        Args:
            tool: The tool use request to preprocess.
            tool_config: The tool configuration for the current session.
            **kwargs: Additional context-specific arguments.

        Returns:
            A preprocessed tool result object.
        """
        ...

    @abstractmethod
    # pragma: no cover
    def process(
        self,
        tool: ToolUse,
        *,
        messages: "Messages",
        model: "Model",
        system_prompt: Optional[str],
        tool_config: ToolConfig,
        callback_handler: Any,
        **kwargs: Any,
    ) -> ToolResult:
        """Process a tool use request and execute the tool.

        Args:
            tool: The tool use request to process.
            messages: The current conversation history.
            model: The model being used for the conversation.
            system_prompt: The system prompt for the conversation.
            tool_config: The tool configuration for the current session.
            callback_handler: Callback for processing events as they happen.
            **kwargs: Additional context-specific arguments.

        Returns:
            The result of the tool execution.
        """
        ...
