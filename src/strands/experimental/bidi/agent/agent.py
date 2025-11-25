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
import logging
from typing import Any, AsyncGenerator

from .... import _identifier
from ....agent.state import AgentState
from ....hooks import HookProvider, HookRegistry
from ....interrupt import _InterruptState
from ....tools._caller import _ToolCaller
from ....tools.executors import ConcurrentToolExecutor
from ....tools.executors._executor import ToolExecutor
from ....tools.registry import ToolRegistry
from ....tools.watcher import ToolWatcher
from ....types.content import Message, Messages
from ....types.tools import AgentTool
from ...hooks.events import BidiAgentInitializedEvent, BidiMessageAddedEvent
from ...tools import ToolProvider
from .._async import stop_all
from ..models.bidi_model import BidiModel
from ..models.novasonic import BidiNovaSonicModel
from ..types.agent import BidiAgentInput
from ..types.events import BidiAudioInputEvent, BidiImageInputEvent, BidiInputEvent, BidiOutputEvent, BidiTextInputEvent
from ..types.io import BidiInput, BidiOutput
from .loop import _BidiAgentLoop

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
        model: BidiModel | str | None = None,
        tools: list[str | AgentTool | ToolProvider] | None = None,
        system_prompt: str | None = None,
        messages: Messages | None = None,
        record_direct_tool_call: bool = True,
        load_tools_from_directory: bool = False,
        agent_id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        hooks: list[HookProvider] | None = None,
        state: AgentState | dict | None = None,
        tool_executor: ToolExecutor | None = None,
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
            description: Description of what the Agent does.
            hooks: Optional list of hook providers to register for lifecycle events.
            state: Stateful information for the agent. Can be either an AgentState object, or a json serializable dict.
            tool_executor: Definition of tool execution strategy (e.g., sequential, concurrent, etc.).
            **kwargs: Additional configuration for future extensibility.

        Raises:
            ValueError: If model configuration is invalid or state is invalid type.
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

        # Initialize agent state management
        if state is not None:
            if isinstance(state, dict):
                self.state = AgentState(state)
            elif isinstance(state, AgentState):
                self.state = state
            else:
                raise ValueError("state must be an AgentState object or a dict")
        else:
            self.state = AgentState()

        # Initialize other components
        self.tool_caller = _ToolCaller(self)

        # Initialize tool executor
        self.tool_executor = tool_executor or ConcurrentToolExecutor()

        # Initialize hooks registry
        self.hooks = HookRegistry()
        if hooks:
            for hook in hooks:
                self.hooks.add_hook(hook)

        self._loop = _BidiAgentLoop(self)

        # Emit initialization event
        self.hooks.invoke_callbacks(BidiAgentInitializedEvent(agent=self))

        # TODO: Determine if full support is required
        self._interrupt_state = _InterruptState()

        self._started = False

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
        return self.tool_caller

    @property
    def tool_names(self) -> list[str]:
        """Get a list of all registered tool names.

        Returns:
            Names of all tools available to this agent.
        """
        all_tools = self.tool_registry.get_all_tools_config()
        return list(all_tools.keys())



    async def _append_message(self, message: Message) -> None:
        """Appends a message to the agent's list of messages and invokes the callbacks for the MessageAddedEvent."""
        self.messages.append(message)
        await self.hooks.invoke_callbacks_async(BidiMessageAddedEvent(agent=self, message=message))

    async def start(self, invocation_state: dict[str, Any] | None = None) -> None:
        """Start a persistent bidirectional conversation connection.

        Initializes the streaming connection and starts background tasks for processing
        model events, tool execution, and connection management.

        Args:
            invocation_state: Optional context to pass to tools during execution.
                This allows passing custom data (user_id, session_id, database connections, etc.)
                that tools can access via their invocation_state parameter.

        Raises:
            RuntimeError:
                If agent already started.

        Example:
            ```python
            await agent.start(invocation_state={
                "user_id": "user_123",
                "session_id": "session_456",
                "database": db_connection,
            })
            ```
        """
        if self._started:
            raise RuntimeError("agent already started | call stop before starting again")

        logger.debug("agent starting")
        await self._loop.start(invocation_state)
        self._started = True

    async def send(self, input_data: BidiAgentInput | dict[str, Any]) -> None:
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
            RuntimeError: If start has not been called.
            ValueError: If invalid input type.

        Example:
            await agent.send("Hello")
            await agent.send(BidiAudioInputEvent(audio="base64...", format="pcm", ...))
            await agent.send({"type": "bidirectional_text_input", "text": "Hello", "role": "user"})
        """
        if not self._started:
            raise RuntimeError("agent not started | call start before sending")

        # Handle string input
        if isinstance(input_data, str):
            # Add user text message to history
            user_message: Message = {"role": "user", "content": [{"text": input_data}]}

            self.messages.append(user_message)
            await self.hooks.invoke_callbacks_async(BidiMessageAddedEvent(agent=self, message=user_message))

            logger.debug("text_length=<%d> | text sent to model", len(input_data))
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
            input_event: BidiInputEvent
            if event_type == "bidi_text_input":
                input_event = BidiTextInputEvent(text=input_data["text"], role=input_data["role"])
            elif event_type == "bidi_audio_input":
                input_event = BidiAudioInputEvent(
                    audio=input_data["audio"],
                    format=input_data["format"],
                    sample_rate=input_data["sample_rate"],
                    channels=input_data["channels"],
                )
            elif event_type == "bidi_image_input":
                input_event = BidiImageInputEvent(image=input_data["image"], mime_type=input_data["mime_type"])
            else:
                raise ValueError(f"Unknown event type: {event_type}")

            # Send the reconstructed TypedEvent
            await self.model.send(input_event)
            return

        # If we get here, input type is invalid
        raise ValueError(
            f"Input must be a string, BidiInputEvent "
            f"(BidiTextInputEvent/BidiAudioInputEvent/BidiImageInputEvent), "
            f"or event dict with 'type' field, got: {type(input_data)}"
        )

    async def receive(self) -> AsyncGenerator[BidiOutputEvent, None]:
        """Receive events from the model including audio, text, and tool calls.

        Yields:
            Model output events processed by background tasks including audio output,
            text responses, tool calls, and connection updates.

        Raises:
            RuntimeError: If start has not been called.
        """
        if not self._started:
            raise RuntimeError("agent not started | call start before receiving")

        async for event in self._loop.receive():
            yield event

    async def stop(self) -> None:
        """End the conversation connection and cleanup all resources.

        Terminates the streaming connection, cancels background tasks, and
        closes the connection to the model provider.
        """
        self._started = False
        await self._loop.stop()

    async def __aenter__(self, invocation_state: dict[str, Any] | None = None) -> "BidiAgent":
        """Async context manager entry point.

        Automatically starts the bidirectional connection when entering the context.

        Args:
            invocation_state: Optional context to pass to tools during execution.
                This allows passing custom data (user_id, session_id, database connections, etc.)
                that tools can access via their invocation_state parameter.

        Returns:
            Self for use in the context.
        """
        logger.debug("context_manager=<enter> | starting agent")
        await self.start(invocation_state)
        return self

    async def __aexit__(self, *_: Any) -> None:
        """Async context manager exit point.

        Automatically ends the connection and cleans up resources including
        when exiting the context, regardless of whether an exception occurred.
        """
        logger.debug("context_manager=<exit> | stopping agent")
        await self.stop()

    async def run(
        self, inputs: list[BidiInput], outputs: list[BidiOutput], invocation_state: dict[str, Any] | None = None
    ) -> None:
        """Run the agent using provided IO channels for bidirectional communication.

        Args:
            inputs: Input callables to read data from a source
            outputs: Output callables to receive events from the agent
            invocation_state: Optional context to pass to tools during execution.
                This allows passing custom data (user_id, session_id, database connections, etc.)
                that tools can access via their invocation_state parameter.

        Example:
            ```python
            audio_io = BidiAudioIO(input_rate=16000)
            text_io = BidiTextIO()
            agent = BidiAgent(model=model, tools=[calculator])
            await agent.run(
                inputs=[audio_io.input()],
                outputs=[audio_io.output(), text_io.output()],
                invocation_state={"user_id": "user_123"}
            )
            ```
        """

        async def run_inputs() -> None:
            async def task(input_: BidiInput) -> None:
                while True:
                    event = await input_()
                    await self.send(event)

            tasks = [task(input_) for input_ in inputs]
            await asyncio.gather(*tasks)

        async def run_outputs() -> None:
            async for event in self.receive():
                tasks = [output(event) for output in outputs]
                await asyncio.gather(*tasks)

        try:
            # Only start if not already started (e.g., when used with async context manager)
            if not self._started:
                await self.start(invocation_state)

            start_inputs = [input_.start for input_ in inputs if hasattr(input_, "start")]
            start_outputs = [output.start for output in outputs if hasattr(output, "start")]
            for start in [*start_inputs, *start_outputs]:
                await start()

            async with asyncio.TaskGroup() as task_group:
                task_group.create_task(run_inputs())
                task_group.create_task(run_outputs())

        finally:
            stop_inputs = [input_.stop for input_ in inputs if hasattr(input_, "stop")]
            stop_outputs = [output.stop for output in outputs if hasattr(output, "stop")]

            await stop_all(*stop_inputs, *stop_outputs, self.stop)
