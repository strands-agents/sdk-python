"""Bidirectional Agent for real-time streaming conversations.

Provides real-time audio and text interaction through persistent streaming sessions.
Unlike traditional request-response patterns, this agent maintains long-running
conversations where users can interrupt, provide additional input, and receive
continuous responses including audio output.

Key capabilities:
- Persistent conversation sessions with concurrent processing
- Real-time audio input/output streaming
- Mid-conversation interruption and tool execution
- Event-driven communication with model providers
"""

import asyncio
import json
import logging
from typing import Any, AsyncIterable, Mapping, Optional, Union, TYPE_CHECKING

from .... import _identifier
from ....hooks import HookProvider, HookRegistry
from ....telemetry.metrics import EventLoopMetrics
from ....tools.caller import ToolCaller
from ....tools.executors import ConcurrentToolExecutor
from ....tools.executors._executor import ToolExecutor
from ....tools.registry import ToolRegistry
from ....tools.watcher import ToolWatcher
from ....types.content import Message, Messages
from ....types.tools import ToolResult, ToolUse
from ....types.traces import AttributeValue
from ..event_loop.bidirectional_event_loop import start_bidirectional_connection, stop_bidirectional_connection
from ..models.bidirectional_model import BidirectionalModel
from ..types.bidirectional_streaming import AudioInputEvent, BidirectionalStreamEvent
from ..models.novasonic import NovaSonicBidirectionalModel

if TYPE_CHECKING:
    from ..event_loop.bidirectional_event_loop import BidirectionalEventLoop


logger = logging.getLogger(__name__)

_DEFAULT_AGENT_NAME = "Strands Agents"
_DEFAULT_AGENT_ID = "default"


class BidirectionalAgent:
    """Agent for bidirectional streaming conversations.

    Enables real-time audio and text interaction with AI models through persistent
    sessions. Supports concurrent tool execution and interruption handling.
    """

    def __init__(
        self,
        model: Union[BidirectionalModel, str, None] = None,
        tools: Optional[list[Union[str, dict[str, str], Any]]] = None,
        system_prompt: Optional[str] = None,
        messages: Optional[Messages] = None,
        record_direct_tool_call: bool = True,
        load_tools_from_directory: bool = False,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        tool_executor: Optional[ToolExecutor] = None,
        hooks: Optional[list[HookProvider]] = None,
        trace_attributes: Optional[Mapping[str, AttributeValue]] = None,
        description: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize bidirectional agent with flexible model support and extensible configuration.

        Args:
            model: BidirectionalModel instance, string model_id, or None for default detection.
            tools: Optional list of tools with flexible format support.
            system_prompt: Optional system prompt for conversations.
            messages: Optional conversation history to initialize with.
            record_direct_tool_call: Whether to record direct tool calls in message history.
            load_tools_from_directory: Whether to load and automatically reload tools in the `./tools/` directory.
            agent_id: Optional ID for the agent, useful for session management and multi-agent scenarios.
            name: Name of the Agent.
            tool_executor: Definition of tool execution strategy (e.g., sequential, concurrent, etc.).
            hooks: Hooks to be added to the agent hook registry.
            trace_attributes: Custom trace attributes to apply to the agent's trace span.
            description: Description of what the Agent does.
            **kwargs: Additional configuration for future extensibility.

        Raises:
            ValueError: If model configuration is invalid.
            TypeError: If model type is unsupported.
        """

        self.model = (
            NovaSonicBidirectionalModel()
            if not model
            else NovaSonicBidirectionalModel(model_id=model)
            if isinstance(model, str)
            else model
        )
        self.system_prompt = system_prompt
        self.messages = messages or []

        # Agent identification
        self.agent_id = _identifier.validate(agent_id or _DEFAULT_AGENT_ID, _identifier.Identifier.AGENT)
        self.name = name or _DEFAULT_AGENT_NAME
        self.description = description

        # Tool execution configuration
        self.record_direct_tool_call = record_direct_tool_call
        self.load_tools_from_directory = load_tools_from_directory

        # Process trace attributes to ensure they're of compatible types
        self.trace_attributes: dict[str, AttributeValue] = {}
        if trace_attributes:
            for k, v in trace_attributes.items():
                if isinstance(v, (str, int, float, bool)) or (
                    isinstance(v, list) and all(isinstance(x, (str, int, float, bool)) for x in v)
                ):
                    self.trace_attributes[k] = v

        # Initialize tool registry
        self.tool_registry = ToolRegistry()

        if tools is not None:
            self.tool_registry.process_tools(tools)

        self.tool_registry.initialize_tools(self.load_tools_from_directory)

        # Initialize tool watcher if directory loading is enabled
        if self.load_tools_from_directory:
            self.tool_watcher = ToolWatcher(tool_registry=self.tool_registry)

        # Initialize tool executor
        self.tool_executor = tool_executor or ConcurrentToolExecutor()

        # Initialize hooks system
        self.hooks = HookRegistry()
        if hooks:
            for hook in hooks:
                self.hooks.add_hook(hook)

        # Initialize other components
        self.event_loop_metrics = EventLoopMetrics()
        self.tool_caller = ToolCaller(self)

        # Session management
        self._session = None
        self._output_queue = asyncio.Queue()

        # Store extensibility kwargs for future use
        self._config_kwargs = kwargs

    @property
    def tool(self) -> ToolCaller:
        """Call tool as a function.

        Returns:
            ToolCaller for method-style tool execution.

        Example:
            ```
            agent = BidirectionalAgent(model=model, tools=[calculator])
            agent.tool.calculator(expression="2+2")
            ```
        """
        return self.tool_caller

    @property
    def tool_names(self) -> list[str]:
        """Get a list of all registered tool names.

        Returns:
            Names of all tools available to this agent.
        """
        all_tools = self.tool_registry.get_all_tools_config()
        return list(all_tools.keys())

    def _record_tool_execution(
        self,
        tool: ToolUse,
        tool_result: ToolResult,
        user_message_override: Optional[str],
    ) -> None:
        """Record a tool execution in the message history.

        Creates a sequence of messages that represent the tool execution:

        1. A user message describing the tool call
        2. An assistant message with the tool use
        3. A user message with the tool result
        4. An assistant message acknowledging the tool call

        Args:
            tool: The tool call information.
            tool_result: The result returned by the tool.
            user_message_override: Optional custom message to include.
        """
        # Filter tool input parameters to only include those defined in tool spec
        filtered_input = self._filter_tool_parameters_for_recording(tool["name"], tool["input"])

        # Create user message describing the tool call
        input_parameters = json.dumps(filtered_input, default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>")

        user_msg_content = [
            {"text": (f"agent.tool.{tool['name']} direct tool call.\nInput parameters: {input_parameters}\n")}
        ]

        # Add override message if provided
        if user_message_override:
            user_msg_content.insert(0, {"text": f"{user_message_override}\n"})

        # Create filtered tool use for message history
        filtered_tool: ToolUse = {
            "toolUseId": tool["toolUseId"],
            "name": tool["name"],
            "input": filtered_input,
        }

        # Create the message sequence
        user_msg: Message = {
            "role": "user",
            "content": user_msg_content,
        }
        tool_use_msg: Message = {
            "role": "assistant",
            "content": [{"toolUse": filtered_tool}],
        }
        tool_result_msg: Message = {
            "role": "user",
            "content": [{"toolResult": tool_result}],
        }
        assistant_msg: Message = {
            "role": "assistant",
            "content": [{"text": f"agent.tool.{tool['name']} was called."}],
        }

        # Add to message history
        self.messages.append(user_msg)
        self.messages.append(tool_use_msg)
        self.messages.append(tool_result_msg)
        self.messages.append(assistant_msg)

        logger.debug("Direct tool call recorded in message history: %s", tool["name"])

    def _filter_tool_parameters_for_recording(self, tool_name: str, input_params: dict[str, Any]) -> dict[str, Any]:
        """Filter input parameters to only include those defined in the tool specification.

        Args:
            tool_name: Name of the tool to get specification for
            input_params: Original input parameters

        Returns:
            Filtered parameters containing only those defined in tool spec
        """
        all_tools_config = self.tool_registry.get_all_tools_config()
        tool_spec = all_tools_config.get(tool_name)

        if not tool_spec or "inputSchema" not in tool_spec:
            return input_params.copy()

        properties = tool_spec["inputSchema"]["json"]["properties"]
        return {k: v for k, v in input_params.items() if k in properties}

    async def start(self) -> None:
        """Start a persistent bidirectional conversation session.

        Initializes the streaming session and starts background tasks for processing
        model events, tool execution, and session management.

        Raises:
            ValueError: If conversation already active.
            ConnectionError: If session creation fails.
        """
        if self._session and self._session.active:
            raise ValueError("Conversation already active. Call end() first.")

        logger.debug("Conversation start - initializing session")

        self._session = await start_bidirectional_connection(self)
        logger.debug("Conversation ready")

    async def send(self, input_data: Union[str, AudioInputEvent]) -> None:
        """Send input to the model (text or audio).

        Unified method for sending both text and audio input to the model during
        an active conversation session. User input is automatically added to
        conversation history for complete message tracking.

        Args:
            input_data: Either a string for text input or AudioInputEvent for audio input.

        Raises:
            ValueError: If no active session or invalid input type.
        """
        self._validate_active_session()

        if isinstance(input_data, str):
            # Add user text message to history
            user_message: Message = {"role": "user", "content": [{"text": input_data}]}

            self.messages.append(user_message)

            logger.debug("Text sent: %d characters", len(input_data))
            await self._session.model_session.send_text_content(input_data)
        elif isinstance(input_data, dict) and "audioData" in input_data:
            # Handle audio input
            await self._session.model_session.send_audio_content(input_data)
        else:
            raise ValueError(
                "Input must be either a string (text) or AudioInputEvent "
                "(dict with audioData, format, sampleRate, channels)"
            )

    async def receive(self) -> AsyncIterable[BidirectionalStreamEvent]:
        """Receive events from the model including audio, text, and tool calls.

        Yields model output events processed by background tasks including audio output,
        text responses, tool calls, and session updates.

        Yields:
            BidirectionalStreamEvent: Events from the model session.
        """
        while self._session and self._session.active:
            try:
                event = await asyncio.wait_for(self._output_queue.get(), timeout=0.1)
                yield event
            except asyncio.TimeoutError:
                continue

    async def interrupt(self) -> None:
        """Interrupt the current model generation and clear audio buffers.

        Sends interruption signal to stop generation immediately and clears
        pending audio output for responsive conversation flow.

        Raises:
            ValueError: If no active session.
        """
        self._validate_active_session()
        await self._session.model_session.send_interrupt()

    async def end(self) -> None:
        """End the conversation session and cleanup all resources.

        Terminates the streaming session, cancels background tasks, and
        closes the connection to the model provider.
        """
        if self._session:
            await stop_bidirectional_connection(self._session)
            self._session = None

    def _validate_active_session(self) -> None:
        """Validate that an active session exists.

        Raises:
            ValueError: If no active session.
        """
        if not self._session or not self._session.active:
            raise ValueError("No active conversation. Call start() first.")
