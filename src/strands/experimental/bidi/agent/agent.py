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
from typing import Any, AsyncIterable

from .... import _identifier
from ....tools.caller import _ToolCaller
from ....tools.executors import ConcurrentToolExecutor
from ....tools.executors._executor import ToolExecutor
from ....tools.registry import ToolRegistry
from ....tools.watcher import ToolWatcher
from ....types.content import Message, Messages
from ....types.tools import ToolResult, ToolUse, AgentTool

from .loop import _BidiAgentLoop
from ..models.bidi_model import BidiModel
from ..models.novasonic import BidiNovaSonicModel
from ..types.agent import BidiAgentInput
from ..types.events import BidiAudioInputEvent, BidiImageInputEvent, BidiTextInputEvent, BidiInputEvent, BidiOutputEvent
from ..types.io import BidiInput, BidiOutput
from ...tools import ToolProvider

logger = logging.getLogger(__name__)

_DEFAULT_AGENT_NAME = "Strands Agents"
_DEFAULT_AGENT_ID = "default"


class BidiAgent:
    """Agent for bidirectional streaming conversations.

    Enables real-time audio and text interaction with AI models through persistent
    connections. Supports concurrent tool execution and interruption handling.
    """

    def __init__(
        self,
        model: BidiModel| str | None = None,
        tools: list[str| AgentTool| ToolProvider]| None = None,
        system_prompt: str | None = None,
        messages: Messages | None = None,
        record_direct_tool_call: bool = True,
        load_tools_from_directory: bool = False,
        agent_id: str | None = None,
        name: str | None = None,
        tool_executor: ToolExecutor | None = None,
        description: str | None = None,
        **kwargs: Any,
    ):
        """Initialize bidirectional agent.

        Args:
            model: BidiModel instance, string model_id, or None for default detection.
            tools: Optional list of tools with flexible format support.
            system_prompt: Optional system prompt for conversations.
            messages: Optional conversation history to initialize with.
            record_direct_tool_call: Whether to record direct tool calls in message history.
            load_tools_from_directory: Whether to load and automatically reload tools in the `./tools/` directory.
            agent_id: Optional ID for the agent, useful for connection management and multi-agent scenarios.
            name: Name of the Agent.
            tool_executor: Definition of tool execution strategy (e.g., sequential, concurrent, etc.).
            description: Description of what the Agent does.
            **kwargs: Additional configuration for future extensibility.

        Raises:
            ValueError: If model configuration is invalid.
            TypeError: If model type is unsupported.
        """
        self.model = (
            BidiNovaSonicModel()
            if not model
            else BidiNovaSonicModel(model_id=model)
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
        self._tool_caller = _ToolCaller(self)

        self._current_adapters = []  # Track adapters for cleanup

        self._loop = _BidiAgentLoop(self)

    @property
    def tool(self) -> _ToolCaller:
        """Call tool as a function.

        Returns:
            ToolCaller for method-style tool execution.

        Example:
            ```
            agent = BidiAgent(model=model, tools=[calculator])
            agent.tool.calculator(expression="2+2")
            ```
        """
        return self._tool_caller

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
        user_message_override: str | None,
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
        """
        logger.debug("starting agent")

        await self._loop.start()

    async def send(self, input_data: BidiAgentInput) -> None:
        """Send input to the model (text, audio, image, or event dict).
        
        Unified method for sending text, audio, and image input to the model during
        an active conversation session. Accepts TypedEvent instances or plain dicts
        (e.g., from WebSocket clients) which are automatically reconstructed.
        
        Args:
            input_data: Can be:
                - str: Text message from user
                - BidiAudioInputEvent: Audio data with format/sample rate
                - BidiImageInputEvent: Image data with MIME type
                - dict: Event dictionary (will be reconstructed to TypedEvent)
            
        Raises:
            ValueError: If no active session or invalid input type.
            
        Example:
            await agent.send("Hello")
            await agent.send(BidiAudioInputEvent(audio="base64...", format="pcm", ...))
            await agent.send({"type": "bidirectional_text_input", "text": "Hello", "role": "user"})
        """
        self._validate_active_connection()

        # Handle string input
        if isinstance(input_data, str):
            # Add user text message to history
            user_message: Message = {"role": "user", "content": [{"text": input_data}]}

            self.messages.append(user_message)

            logger.debug("Text sent: %d characters", len(input_data))
            # Create BidiTextInputEvent for send()
            text_event = BidiTextInputEvent(text=input_data, role="user")
            await self.model.send(text_event)
            return
        
        # Handle BidiInputEvent instances
        # Check this before dict since TypedEvent inherits from dict
        if isinstance(input_data, BidiInputEvent):
            await self.model.send(input_data)
            return
        
        # Handle plain dict - reconstruct TypedEvent for WebSocket integration
        if isinstance(input_data, dict) and "type" in input_data:
            event_type = input_data["type"]
            if event_type == "bidi_text_input":
                input_event = BidiTextInputEvent(text=input_data["text"], role=input_data["role"])
            elif event_type == "bidi_audio_input":
                input_event = BidiAudioInputEvent(
                    audio=input_data["audio"],
                    format=input_data["format"],
                    sample_rate=input_data["sample_rate"],
                    channels=input_data["channels"]
                )
            elif event_type == "bidi_image_input":
                input_event = BidiImageInputEvent(
                    image=input_data["image"],
                    mime_type=input_data["mime_type"]
                )
            else:
                raise ValueError(f"Unknown event type: {event_type}")
            
            # Send the reconstructed TypedEvent
            await self.model.send(input_event)
            return
        
        # If we get here, input type is invalid
        raise ValueError(
            f"Input must be a string, BidiInputEvent (BidiTextInputEvent/BidiAudioInputEvent/BidiImageInputEvent), or event dict with 'type' field, got: {type(input_data)}"
        )

    async def receive(self) -> AsyncIterable[BidiOutputEvent]:
        """Receive events from the model including audio, text, and tool calls.

        Yields model output events processed by background tasks including audio output,
        text responses, tool calls, and connection updates.

        Yields:
            Model and tool call events.
        """
        async for event in self._loop.receive():
            yield event

    async def stop(self) -> None:
        """End the conversation connection and cleanup all resources.

        Terminates the streaming connection, cancels background tasks, and
        closes the connection to the model provider.
        """
        await self._loop.stop()

    async def __aenter__(self) -> "BidiAgent":
        """Async context manager entry point.

        Automatically starts the bidirectional connection when entering the context.

        Returns:
            Self for use in the context.
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
            
            # Cleanup adapters if any are currently active
            for adapter in self._current_adapters:
                if hasattr(adapter, "cleanup"):
                    try:
                        adapter.stop()
                        logger.debug(f"Cleaned up adapter: {type(adapter).__name__}")
                    except Exception as adapter_error:
                        logger.warning(f"Error cleaning up adapter: {adapter_error}")
            
            # Clear current adapters
            self._current_adapters = []
            
            # Cleanup agent connection
            await self.stop()

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
        """True if agent loop started, False otherwise."""
        return self._loop.active

    async def run(self, inputs: list[BidiInput], outputs: list[BidiOutput]) -> None:
        """Run the agent using provided IO channels for bidirectional communication.

        Args:
            inputs: Input callables to read data from a source
            outputs: Output callables to receive events from the agent
                
        Example:
            ```python
            audio_io = BidiAudioIO(audio_config={"input_sample_rate": 16000})
            text_io = BidiTextIO()
            agent = BidiAgent(model=model, tools=[calculator])
            await agent.run(inputs=[audio_io.input()], outputs=[audio_io.output(), text_io.output()])
            ```
        """
        async def run_inputs():
            while self.active:
                for input_ in inputs:
                    event = await input_()
                    await self.send(event)

                    # TODO: Need to make tool result send in Nova provider atomic. Audio input events end up interleaving
                    # and leading to failures. Adding a sleep here as a temporary solution.
                    await asyncio.sleep(0.001)

        async def run_outputs():
            async for event in self.receive():
                for output in outputs:
                    await output(event)

        for input_ in inputs:
            if hasattr(input_, "start"):
                await input_.start()

        for output in outputs:
            if hasattr(output, "start"):
                await output.start()

        # Start agent after all IO is ready
        await self.start()
        try:
            await asyncio.gather(run_inputs(), run_outputs(), return_exceptions=True)

        finally:
            await self.stop()

            for input_ in inputs:
                if hasattr(input_, "stop"):
                    await input_.stop()

            for output in outputs:
                if hasattr(output, "stop"):
                    await output.stop()

    def _validate_active_connection(self) -> None:
        """Validate that an active connection exists.

        Raises:
            ValueError: If no active connection.
        """
        if not self.active:
            raise ValueError("No active conversation. Call start() first or use async context manager.")
