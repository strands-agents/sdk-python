"""Bidirectional Agent for real-time streaming conversations.

Provides real-time audio and text interaction through persistent streaming connections.
Unlike traditional request-response patterns, this agent maintains long-running
conversations where users can interrupt, provide additional input, and receive
continuous responses including audio output.

Key capabilities:
- Persistent conversation connections with concurrent processing
- Real-time audio input/output streaming
- Automatic interruption detection and tool execution
- Event-driven communication with model providers
"""

import asyncio
import json
import logging
from typing import Any, AsyncIterable, Mapping, Optional, Union, Callable

from .... import _identifier
from ....telemetry.metrics import EventLoopMetrics
from ....tools.caller import ToolCaller
from ....tools.executors import ConcurrentToolExecutor
from ....tools.executors._executor import ToolExecutor
from ....tools.registry import ToolRegistry
from ....tools.watcher import ToolWatcher
from ....types.content import Message, Messages
from ....types.tools import ToolResult, ToolUse
from ....types.traces import AttributeValue
from ..adapters.audio_adapter import AudioAdapter
from ..event_loop.bidirectional_event_loop import BidirectionalAgentLoop
from ..models.bidirectional_model import BidirectionalModel
from ..models.novasonic import NovaSonicBidirectionalModel
from ..types.bidirectional_streaming import AudioInputEvent, BidirectionalStreamEvent, ImageInputEvent

logger = logging.getLogger(__name__)

_DEFAULT_AGENT_NAME = "Strands Agents"
_DEFAULT_AGENT_ID = "default"
# Type alias for cleaner send() method signature
BidirectionalInput = str | AudioInputEvent | ImageInputEvent


class BidirectionalAgent:
    """Agent for bidirectional streaming conversations.

    Enables real-time audio and text interaction with AI models through persistent
    connections. Supports concurrent tool execution and interruption handling.
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
        description: Optional[str] = None,
        adapters: Optional[list[Any]] = None,
        **kwargs: Any,
    ):
        """Initialize bidirectional agent.

        Args:
            model: BidirectionalModel instance, string model_id, or None for default detection.
            tools: Optional list of tools with flexible format support.
            system_prompt: Optional system prompt for conversations.
            messages: Optional conversation history to initialize with.
            record_direct_tool_call: Whether to record direct tool calls in message history.
            load_tools_from_directory: Whether to load and automatically reload tools in the `./tools/` directory.
            agent_id: Optional ID for the agent, useful for connection management and multi-agent scenarios.
            name: Name of the Agent.
            tool_executor: Definition of tool execution strategy (e.g., sequential, concurrent, etc.).
            description: Description of what the Agent does.
            adapters: Optional list of adapter instances (e.g., AudioAdapter) for hardware abstraction.
                     If None, automatically creates default AudioAdapter for basic audio functionality.
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

        # Initialize other components
        self.event_loop_metrics = EventLoopMetrics()
        self.tool_caller = ToolCaller(self)

        # connection management
        self._agentloop: Optional["BidirectionalAgentLoop"] = None
        self._output_queue = asyncio.Queue()

        # Initialize adapters - auto-create AudioAdapter as default
        if adapters is None:
            # Create default AudioAdapter for basic audio functionality
            default_audio_adapter = AudioAdapter(audio_config={"input_sample_rate": 16000})
            self.adapters = [default_audio_adapter]
        else:
            self.adapters = adapters

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
        """Start a persistent bidirectional conversation connection.

        Initializes the streaming connection and starts background tasks for processing
        model events, tool execution, and connection management.

        Raises:
            ValueError: If conversation already active.
            ConnectionError: If connection creation fails.
        """
        if self._agentloop and self._agentloop.active:
            raise ValueError("Conversation already active. Call end() first.")

        logger.debug("Conversation start - initializing connection")

        # Create model session and event loop directly
        model_session = await self.model.create_bidirectional_connection(
            system_prompt=self.system_prompt, tools=self.tool_registry.get_all_tool_specs(), messages=self.messages
        )

        self._agentloop = BidirectionalAgentLoop(model_session=model_session, agent=self)
        await self._agentloop.start()

        logger.debug("Conversation ready")

    async def send(self, input_data: BidirectionalInput) -> None:
        """Send input to the model (text or audio).

        Unified method for sending both text and audio input to the model during
        an active conversation connection. User input is automatically added to
        conversation history for complete message tracking.

        Args:
            input_data: String for text, AudioInputEvent for audio, or ImageInputEvent for images.

        Raises:
            ValueError: If no active connection or invalid input type.
        """
        self._validate_active_connection()

        if isinstance(input_data, str):
            # Add user text message to history
            user_message: Message = {"role": "user", "content": [{"text": input_data}]}

            self.messages.append(user_message)

            logger.debug("Text sent: %d characters", len(input_data))
            await self._agentloop.model_session.send_text_content(input_data)
        elif isinstance(input_data, dict) and "audioData" in input_data:
            # Handle audio input
            await self._agentloop.model_session.send_audio_content(input_data)
        elif isinstance(input_data, dict) and "imageData" in input_data:
            # Handle image input (ImageInputEvent)
            await self._agentloop.model_session.send_image_content(input_data)
        else:
            raise ValueError(
                "Input must be either a string (text), AudioInputEvent "
                "(dict with audioData, format, sampleRate, channels), or ImageInputEvent "
                "(dict with imageData, mimeType, encoding)"
            )

    async def receive(self) -> AsyncIterable[BidirectionalStreamEvent]:
        """Receive events from the model including audio, text, and tool calls.

        Yields model output events processed by background tasks including audio output,
        text responses, tool calls, and connection updates.

        Yields:
            BidirectionalStreamEvent: Events from the model session.
        """
        while self.active:
            try:
                event = await self._output_queue.get()
                yield event
            except asyncio.TimeoutError:
                continue

    async def end(self) -> None:
        """End the conversation connection and cleanup all resources.

        Terminates the streaming connection, cancels background tasks, and
        closes the connection to the model provider.
        """
        if self._agentloop:
            await self._agentloop.stop()
            self._agentloop = None

    async def __aenter__(self) -> "BidirectionalAgent":
        """Async context manager entry point.

        Automatically starts the bidirectional connection when entering the context.

        Returns:
            Self for use in the context.

        Raises:
            ValueError: If connection is already active.
            ConnectionError: If connection creation fails.
        """
        logger.debug("Entering async context manager - starting connection")
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit point.

        Automatically ends the connection and cleans up resources including adapters
        when exiting the context, regardless of whether an exception occurred.

        Args:
            exc_type: Exception type if an exception occurred, None otherwise.
            exc_val: Exception value if an exception occurred, None otherwise.
            exc_tb: Exception traceback if an exception occurred, None otherwise.
        """
        try:
            logger.debug("Exiting async context manager - cleaning up adapters and connection")

            # Cleanup adapters first
            for adapter in self.adapters:
                if hasattr(adapter, "_cleanup_audio"):
                    try:
                        adapter._cleanup_audio()
                        logger.debug(f"Cleaned up adapter: {type(adapter).__name__}")
                    except Exception as adapter_error:
                        logger.warning(f"Error cleaning up adapter: {adapter_error}")

            # Then cleanup agent connection
            await self.end()

        except Exception as cleanup_error:
            if exc_type is None:
                # No original exception, re-raise cleanup error
                logger.error("Error during context manager cleanup: %s", cleanup_error)
                raise
            else:
                # Original exception exists, log cleanup error but don't suppress original
                logger.error(
                    "Error during context manager cleanup (suppressed due to original exception): %s", cleanup_error
                )

    @property
    def active(self) -> bool:
        """Check if the agent connection is currently active.

        Returns:
            True if connection is active and ready for communication, False otherwise.
        """
        return self._agentloop is not None and self._agentloop.active

    async def connect(self) -> None:
        """Connect the agent using configured adapters for bidirectional communication.

        Automatically uses configured adapters to establish bidirectional communication
        with the model. If no adapters are provided in constructor, uses default AudioAdapter.

        Example:
            ```python
            # Simple - uses default AudioAdapter
            agent = BidirectionalAgent(model=model, tools=[calculator])
            await agent.connect()

            # Custom adapter
            adapter = AudioAdapter(audio_config={"input_sample_rate": 24000})
            agent = BidirectionalAgent(model=model, tools=[calculator], adapters=[adapter])
            await agent.connect()
            ```

        Raises:
            Exception: Any exception from the transport layer.
        """
        # Use first adapter (always available due to default initialization)
        adapter = self.adapters[0]
        sender = adapter.create_output()
        receiver = adapter.create_input()

        if self.active:
            # Use existing connection
            await self._run(sender, receiver)
        else:
            # Use async context manager for automatic lifecycle management
            async with self:
                await self._run(sender, receiver)

    async def _run(
        self,
        sender: Callable[[Any], Any],
        receiver: Callable[[], Any],
    ) -> None:
        """Internal method to run send/receive loops with an active connection.

        Args:
            sender: Async callable that sends events to the client.
            receiver: Async callable that receives events from the client.
        """

        async def receive_from_agent():
            """Receive events from agent and send to client."""
            try:
                async for event in self.receive():
                    await sender(event)
            except Exception as e:
                logger.debug(f"Receive from agent stopped: {e}")
                raise

        async def send_to_agent():
            """Receive events from client and send to agent."""
            try:
                while self.active:
                    event = await receiver()
                    await self.send(event)
            except Exception as e:
                logger.debug(f"Send to agent stopped: {e}")
                raise

        # Run both loops concurrently
        await asyncio.gather(receive_from_agent(), send_to_agent(), return_exceptions=True)

    def _validate_active_connection(self) -> None:
        """Validate that an active connection exists.

        Raises:
            ValueError: If no active connection.
        """
        if not self.active:
            raise ValueError("No active conversation. Call start() first or use async context manager.")
