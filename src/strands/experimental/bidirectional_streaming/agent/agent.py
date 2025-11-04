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
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterable, Callable, Mapping, Optional

from .... import _identifier
from ....hooks import HookProvider, HookRegistry
from ....telemetry.metrics import EventLoopMetrics
from ....tools.executors import ConcurrentToolExecutor
from ....tools.executors._executor import ToolExecutor
from ....tools.registry import ToolRegistry
from ....tools.watcher import ToolWatcher
from ....types.content import Message, Messages
from ....types.tools import ToolResult, ToolUse
from ....types.traces import AttributeValue
from ..event_loop.bidirectional_event_loop import start_bidirectional_connection, stop_bidirectional_connection
from ..models.bidirectional_model import BidirectionalModel
from ..types.bidirectional_streaming import AudioInputEvent, ImageInputEvent, OutputEvent

logger = logging.getLogger(__name__)

_DEFAULT_AGENT_NAME = "Strands Agents"
_DEFAULT_AGENT_ID = "default"


class BidirectionalAgent:
    """Agent for bidirectional streaming conversations.

    Enables real-time audio and text interaction with AI models through persistent
    sessions. Supports concurrent tool execution and interruption handling.
    """

    class ToolCaller:
        """Call tool as a function for bidirectional agent."""

        def __init__(self, agent: "BidirectionalAgent") -> None:
            """Initialize tool caller with agent reference."""
            # WARNING: Do not add any other member variables or methods as this could result in a name conflict with
            #          agent tools and thus break their execution.
            self._agent = agent

        def __getattr__(self, name: str) -> Callable[..., Any]:
            """Call tool as a function.

            This method enables the method-style interface (e.g., `agent.tool.tool_name(param="value")`).
            It matches underscore-separated names to hyphenated tool names (e.g., 'some_thing' matches 'some-thing').

            Args:
                name: The name of the attribute (tool) being accessed.

            Returns:
                A function that when called will execute the named tool.

            Raises:
                AttributeError: If no tool with the given name exists or if multiple tools match the given name.
            """

            def caller(
                user_message_override: Optional[str] = None,
                record_direct_tool_call: Optional[bool] = None,
                **kwargs: Any,
            ) -> Any:
                """Call a tool directly by name.

                Args:
                    user_message_override: Optional custom message to record instead of default
                    record_direct_tool_call: Whether to record direct tool calls in message history. 
                        For bidirectional agents, this is always True to maintain conversation history.
                    **kwargs: Keyword arguments to pass to the tool.

                Returns:
                    The result returned by the tool.

                Raises:
                    AttributeError: If the tool doesn't exist.
                """
                normalized_name = self._find_normalized_tool_name(name)

                # Create unique tool ID and set up the tool request
                tool_id = f"tooluse_{name}_{random.randint(100000000, 999999999)}"
                tool_use: ToolUse = {
                    "toolUseId": tool_id,
                    "name": normalized_name,
                    "input": kwargs.copy(),
                }
                tool_results: list[ToolResult] = []
                invocation_state = kwargs

                async def acall() -> ToolResult:
                    async for event in ToolExecutor._stream(self._agent, tool_use, tool_results, invocation_state):
                        _ = event

                    return tool_results[0]

                def tcall() -> ToolResult:
                    return asyncio.run(acall())

                with ThreadPoolExecutor() as executor:
                    future = executor.submit(tcall)
                    tool_result = future.result()

                # Always record direct tool calls for bidirectional agents to maintain conversation history
                # Use agent's record_direct_tool_call setting if not overridden
                if record_direct_tool_call is not None:
                    should_record_direct_tool_call = record_direct_tool_call
                else:
                    should_record_direct_tool_call = self._agent.record_direct_tool_call

                if should_record_direct_tool_call:
                    # Create a record of this tool execution in the message history
                    self._agent._record_tool_execution(tool_use, tool_result, user_message_override)

                return tool_result

            return caller

        def _find_normalized_tool_name(self, name: str) -> str:
            """Lookup the tool represented by name, replacing characters with underscores as necessary."""
            tool_registry = self._agent.tool_registry.registry

            if tool_registry.get(name, None):
                return name

            # If the desired name contains underscores, it might be a placeholder for characters that can't be
            # represented as python identifiers but are valid as tool names, such as dashes. In that case, find
            # all tools that can be represented with the normalized name
            if "_" in name:
                filtered_tools = [
                    tool_name for (tool_name, tool) in tool_registry.items() if tool_name.replace("-", "_") == name
                ]

                # The registry itself defends against similar names, so we can just take the first match
                if filtered_tools:
                    return filtered_tools[0]

            raise AttributeError(f"Tool '{name}' not found")

    def __init__(
        self,
        model: BidirectionalModel,
        tools: list | None = None,
        system_prompt: str | None = None,
        messages: Messages | None = None,
        record_direct_tool_call: bool = True,
        load_tools_from_directory: bool = False,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        tool_executor: Optional[ToolExecutor] = None,
        hooks: Optional[list[HookProvider]] = None,
        trace_attributes: Optional[Mapping[str, AttributeValue]] = None,
        description: Optional[str] = None,
    ):
        """Initialize bidirectional agent with required model and optional configuration.

        Args:
            model: BidirectionalModel instance supporting streaming sessions.
            tools: Optional list of tools available to the model.
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
        """
        self.model = model
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
        self.tool_caller = BidirectionalAgent.ToolCaller(self)

        # Session management
        self._session = None
        self._output_queue = asyncio.Queue()

    @property
    def tool(self) -> ToolCaller:
        """Call tool as a function.

        Returns:
            Tool caller through which user can invoke tool as a function.

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
    
    async def send(self, input_data: str | AudioInputEvent | ImageInputEvent | dict) -> None:
        """Send input to the model (text, audio, image, or event dict).
        
        Unified method for sending text, audio, and image input to the model during
        an active conversation session. Accepts TypedEvent instances or plain dicts
        (e.g., from WebSocket clients) which are automatically reconstructed.
        
        Args:
            input_data: Can be:
                - str: Text message from user
                - AudioInputEvent: Audio data with format/sample rate
                - ImageInputEvent: Image data with MIME type
                - dict: Event dictionary (will be reconstructed to TypedEvent)
            
        Raises:
            ValueError: If no active session or invalid input type.
            
        Example:
            await agent.send("Hello")
            await agent.send(AudioInputEvent(audio="base64...", format="pcm", ...))
            await agent.send({"type": "bidirectional_text_input", "text": "Hello", "role": "user"})
        """
        self._validate_active_session()

        # Handle string input
        if isinstance(input_data, str):
            # Add user text message to history
            self.messages.append({"role": "user", "content": input_data})
            logger.debug("Text sent: %d characters", len(input_data))
            from ..types.bidirectional_streaming import TextInputEvent
            text_event = TextInputEvent(text=input_data, role="user")
            await self._session.model.send(text_event)
            return
        
        # Handle dict - reconstruct TypedEvent for WebSocket integration
        if isinstance(input_data, dict) and "type" in input_data:
            from ..types.bidirectional_streaming import TextInputEvent
            event_type = input_data["type"]
            if event_type == "bidirectional_text_input":
                input_data = TextInputEvent(text=input_data["text"], role=input_data["role"])
            elif event_type == "bidirectional_audio_input":
                input_data = AudioInputEvent(
                    audio=input_data["audio"],
                    format=input_data["format"],
                    sample_rate=input_data["sample_rate"],
                    channels=input_data["channels"]
                )
            elif event_type == "bidirectional_image_input":
                input_data = ImageInputEvent(
                    image=input_data["image"],
                    mime_type=input_data["mime_type"]
                )
            else:
                raise ValueError(f"Unknown event type: {event_type}")
        
        # Handle TypedEvent instances
        if isinstance(input_data, (AudioInputEvent, ImageInputEvent, TextInputEvent)):
            await self._session.model.send(input_data)
        else:
            raise ValueError(
                f"Input must be a string, TypedEvent, or event dict, got: {type(input_data)}"
            )

    async def receive(self) -> AsyncIterable[dict[str, Any]]:
        """Receive events from the model including audio, text, and tool calls.

        Yields model output events processed by background tasks including audio output,
        text responses, tool calls, and session updates.

        Yields:
            dict: Event dictionaries from the model session. Each event is a TypedEvent
                converted to a dictionary for consistency with the standard Agent API.
        """
        while self._session and self._session.active:
            try:
                event = await asyncio.wait_for(self._output_queue.get(), timeout=0.1)
                # Convert TypedEvent to dict for consistency with Agent.stream_async
                if hasattr(event, 'as_dict'):
                    yield event.as_dict()
                else:
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
        # Interruption is now handled internally by models through audio/event processing
        # No explicit interrupt method needed in unified interface
        logger.debug("Interrupt requested - handled by model's audio processing")

    async def end(self) -> None:
        """End the conversation session and cleanup all resources.

        Terminates the streaming session, cancels background tasks, and
        closes the connection to the model provider.
        """
        if self._session:
            await stop_bidirectional_connection(self._session)
            self._session = None

    async def run(
        self,
        *,
        sender: Callable[[Any], Any],
        receiver: Callable[[], Any],
    ) -> None:
        """Run the agent with send/receive loop management.

        Starts the session, pipes events between the agent and transport layer,
        and handles cleanup on disconnection.

        Args:
            sender: Async callable that sends events to the client (e.g., websocket.send_json).
            receiver: Async callable that receives events from the client (e.g., websocket.receive_json).

        Example:
            ```python
            # With WebSocket
            agent = BidirectionalAgent(model=model, tools=[calculator])
            await agent.run(sender=websocket.send_json, receiver=websocket.receive_json)

            # With custom transport
            async def custom_send(event):
                # Custom send logic
                pass

            async def custom_receive():
                # Custom receive logic
                return event

            await agent.run(sender=custom_send, receiver=custom_receive)
            ```

        Raises:
            Exception: Any exception from the transport layer (e.g., WebSocketDisconnect).
        """
        await self.start()

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
                while self._session and self._session.active:
                    event = await receiver()
                    await self.send(event)
            except Exception as e:
                logger.debug(f"Send to agent stopped: {e}")
                raise

        try:
            # Run both loops concurrently
            await asyncio.gather(
                receive_from_agent(),
                send_to_agent(),
                return_exceptions=True
            )
        finally:
            try:
                await self.end()
            except Exception as e:
                logger.debug(f"Error during cleanup: {e}")

    def _validate_active_session(self) -> None:
        """Validate that an active session exists.

        Raises:
            ValueError: If no active session.
        """
        if not self._session or not self._session.active:
            raise ValueError("No active conversation. Call start() first.")
