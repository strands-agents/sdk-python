"""Agent Interface.

This module implements the core Agent class that serves as the primary entry point for interacting with foundation
models and tools in the SDK.

The Agent interface supports two complementary interaction patterns:

1. Natural language for conversation: `agent("Analyze this data")`
2. Method-style for direct tool access: `agent.tool.tool_name(param1="value")`
"""

import logging
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Mapping,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

from opentelemetry import trace as trace_api
from pydantic import BaseModel

from .. import _identifier
from .._async import run_async
from ..event_loop.event_loop import event_loop_cycle
from ..tools._tool_helpers import generate_missing_tool_result_content

if TYPE_CHECKING:
    from ..experimental.tools import ToolProvider
from ..handlers.callback_handler import PrintingCallbackHandler, null_callback_handler
from ..hooks import (
    AfterInvocationEvent,
    AgentInitializedEvent,
    BeforeInvocationEvent,
    HookProvider,
    HookRegistry,
    MessageAddedEvent,
)
from ..interrupt import _InterruptState
from ..models.bedrock import BedrockModel
from ..models.model import Model
from ..session.session_manager import SessionManager
from ..telemetry.metrics import EventLoopMetrics
from ..telemetry.tracer import get_tracer, serialize
from ..tools._caller import _ToolCaller
from ..tools.executors import ConcurrentToolExecutor
from ..tools.executors._executor import ToolExecutor
from ..tools.registry import ToolRegistry
from ..tools.structured_output._structured_output_context import StructuredOutputContext
from ..tools.watcher import ToolWatcher
from ..types._events import AgentResultEvent, EventLoopStopEvent, InitEventLoopEvent, ModelStreamChunkEvent, TypedEvent
from ..types.agent import AgentInput
from ..types.content import ContentBlock, Message, Messages, SystemContentBlock
from ..types.exceptions import ContextWindowOverflowException
from ..types.traces import AttributeValue
from .agent_result import AgentResult
from .conversation_manager import (
    ConversationManager,
    SlidingWindowConversationManager,
)
from .state import AgentState

logger = logging.getLogger(__name__)

# TypeVar for generic structured output
T = TypeVar("T", bound=BaseModel)


# Sentinel class and object to distinguish between explicit None and default parameter value
class _DefaultCallbackHandlerSentinel:
    """Sentinel class to distinguish between explicit None and default parameter value."""

    pass


_DEFAULT_CALLBACK_HANDLER = _DefaultCallbackHandlerSentinel()
_DEFAULT_AGENT_NAME = "Strands Agents"
_DEFAULT_AGENT_ID = "default"


class Agent:
    """Core Agent interface.

    An agent orchestrates the following workflow:

    1. Receives user input
    2. Processes the input using a language model
    3. Decides whether to use tools to gather information or perform actions
    4. Executes those tools and receives results
    5. Continues reasoning with the new information
    6. Produces a final response
    """

    # For backwards compatibility
    ToolCaller = _ToolCaller

    def __init__(
        self,
        model: Union[Model, str, None] = None,
        messages: Optional[Messages] = None,
        tools: Optional[list[Union[str, dict[str, str], "ToolProvider", Any]]] = None,
        system_prompt: Optional[str | list[SystemContentBlock]] = None,
        structured_output_model: Optional[Type[BaseModel]] = None,
        callback_handler: Optional[
            Union[Callable[..., Any], _DefaultCallbackHandlerSentinel]
        ] = _DEFAULT_CALLBACK_HANDLER,
        conversation_manager: Optional[ConversationManager] = None,
        record_direct_tool_call: bool = True,
        load_tools_from_directory: bool = False,
        trace_attributes: Optional[Mapping[str, AttributeValue]] = None,
        *,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        state: Optional[Union[AgentState, dict]] = None,
        hooks: Optional[list[HookProvider]] = None,
        session_manager: Optional[SessionManager] = None,
        tool_executor: Optional[ToolExecutor] = None,
    ):
        """Initialize the Agent with the specified configuration.

        Args:
            model: Provider for running inference or a string representing the model-id for Bedrock to use.
                Defaults to strands.models.BedrockModel if None.
            messages: List of initial messages to pre-load into the conversation.
                Defaults to an empty list if None.
            tools: List of tools to make available to the agent.
                Can be specified as:

                - String tool names (e.g., "retrieve")
                - File paths (e.g., "/path/to/tool.py")
                - Imported Python modules (e.g., from strands_tools import current_time)
                - Dictionaries with name/path keys (e.g., {"name": "tool_name", "path": "/path/to/tool.py"})
                - ToolProvider instances for managed tool collections
                - Functions decorated with `@strands.tool` decorator.

                If provided, only these tools will be available. If None, all tools will be available.
            system_prompt: System prompt to guide model behavior.
                Can be a string or a list of SystemContentBlock objects for advanced features like caching.
                If None, the model will behave according to its default settings.
            structured_output_model: Pydantic model type(s) for structured output.
                When specified, all agent calls will attempt to return structured output of this type.
                This can be overridden on the agent invocation.
                Defaults to None (no structured output).
            callback_handler: Callback for processing events as they happen during agent execution.
                If not provided (using the default), a new PrintingCallbackHandler instance is created.
                If explicitly set to None, null_callback_handler is used.
            conversation_manager: Manager for conversation history and context window.
                Defaults to strands.agent.conversation_manager.SlidingWindowConversationManager if None.
            record_direct_tool_call: Whether to record direct tool calls in message history.
                Defaults to True.
            load_tools_from_directory: Whether to load and automatically reload tools in the `./tools/` directory.
                Defaults to False.
            trace_attributes: Custom trace attributes to apply to the agent's trace span.
            agent_id: Optional ID for the agent, useful for session management and multi-agent scenarios.
                Defaults to "default".
            name: name of the Agent
                Defaults to "Strands Agents".
            description: description of what the Agent does
                Defaults to None.
            state: stateful information for the agent. Can be either an AgentState object, or a json serializable dict.
                Defaults to an empty AgentState object.
            hooks: hooks to be added to the agent hook registry
                Defaults to None.
            session_manager: Manager for handling agent sessions including conversation history and state.
                If provided, enables session-based persistence and state management.
            tool_executor: Definition of tool execution strategy (e.g., sequential, concurrent, etc.).

        Raises:
            ValueError: If agent id contains path separators.
        """
        self.model = BedrockModel() if not model else BedrockModel(model_id=model) if isinstance(model, str) else model
        self.messages = messages if messages is not None else []
        # initializing self._system_prompt for backwards compatibility
        self._system_prompt, self._system_prompt_content = self._initialize_system_prompt(system_prompt)
        self._default_structured_output_model = structured_output_model
        self.agent_id = _identifier.validate(agent_id or _DEFAULT_AGENT_ID, _identifier.Identifier.AGENT)
        self.name = name or _DEFAULT_AGENT_NAME
        self.description = description

        # If not provided, create a new PrintingCallbackHandler instance
        # If explicitly set to None, use null_callback_handler
        # Otherwise use the passed callback_handler
        self.callback_handler: Union[Callable[..., Any], PrintingCallbackHandler]
        if isinstance(callback_handler, _DefaultCallbackHandlerSentinel):
            self.callback_handler = PrintingCallbackHandler()
        elif callback_handler is None:
            self.callback_handler = null_callback_handler
        else:
            self.callback_handler = callback_handler

        self.conversation_manager = conversation_manager if conversation_manager else SlidingWindowConversationManager()

        # Process trace attributes to ensure they're of compatible types
        self.trace_attributes: dict[str, AttributeValue] = {}
        if trace_attributes:
            for k, v in trace_attributes.items():
                if isinstance(v, (str, int, float, bool)) or (
                    isinstance(v, list) and all(isinstance(x, (str, int, float, bool)) for x in v)
                ):
                    self.trace_attributes[k] = v

        self.record_direct_tool_call = record_direct_tool_call
        self.load_tools_from_directory = load_tools_from_directory

        self.tool_registry = ToolRegistry()

        # Process tool list if provided
        if tools is not None:
            self.tool_registry.process_tools(tools)

        # Initialize tools and configuration
        self.tool_registry.initialize_tools(self.load_tools_from_directory)
        if load_tools_from_directory:
            self.tool_watcher = ToolWatcher(tool_registry=self.tool_registry)

        self.event_loop_metrics = EventLoopMetrics()

        # Initialize tracer instance (no-op if not configured)
        self.tracer = get_tracer()
        self.trace_span: Optional[trace_api.Span] = None

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

        self.tool_caller = _ToolCaller(self)

        self.hooks = HookRegistry()

        self._interrupt_state = _InterruptState()

        # Initialize session management functionality
        self._session_manager = session_manager
        if self._session_manager:
            self.hooks.add_hook(self._session_manager)

        # Allow conversation_managers to subscribe to hooks
        self.hooks.add_hook(self.conversation_manager)

        self.tool_executor = tool_executor or ConcurrentToolExecutor()

        if hooks:
            for hook in hooks:
                self.hooks.add_hook(hook)
        self.hooks.invoke_callbacks(AgentInitializedEvent(agent=self))

    @property
    def system_prompt(self) -> str | None:
        """Get the system prompt as a string for backwards compatibility.

        Returns the system prompt as a concatenated string when it contains text content,
        or None if no text content is present. This maintains backwards compatibility
        with existing code that expects system_prompt to be a string.

        Returns:
            The system prompt as a string, or None if no text content exists.
        """
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str | list[SystemContentBlock] | None) -> None:
        """Set the system prompt and update internal content representation.

        Accepts either a string or list of SystemContentBlock objects.
        When set, both the backwards-compatible string representation and the internal
        content block representation are updated to maintain consistency.

        Args:
            value: System prompt as string, list of SystemContentBlock objects, or None.
                  - str: Simple text prompt (most common use case)
                  - list[SystemContentBlock]: Content blocks with features like caching
                  - None: Clear the system prompt
        """
        self._system_prompt, self._system_prompt_content = self._initialize_system_prompt(value)

    @property
    def tool(self) -> _ToolCaller:
        """Call tool as a function.

        Returns:
            Tool caller through which user can invoke tool as a function.

        Example:
            ```
            agent = Agent(tools=[calculator])
            agent.tool.calculator(...)
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

    def __call__(
        self,
        prompt: AgentInput = None,
        *,
        invocation_state: dict[str, Any] | None = None,
        structured_output_model: Type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Process a natural language prompt through the agent's event loop.

        This method implements the conversational interface with multiple input patterns:
        - String input: `agent("hello!")`
        - ContentBlock list: `agent([{"text": "hello"}, {"image": {...}}])`
        - Message list: `agent([{"role": "user", "content": [{"text": "hello"}]}])`
        - No input: `agent()` - uses existing conversation history

        Args:
            prompt: User input in various formats:
                - str: Simple text input
                - list[ContentBlock]: Multi-modal content blocks
                - list[Message]: Complete messages with roles
                - None: Use existing conversation history
            invocation_state: Additional parameters to pass through the event loop.
            structured_output_model: Pydantic model type(s) for structured output (overrides agent default).
            **kwargs: Additional parameters to pass through the event loop.[Deprecating]

        Returns:
            Result object containing:

                - stop_reason: Why the event loop stopped (e.g., "end_turn", "max_tokens")
                - message: The final message from the model
                - metrics: Performance metrics from the event loop
                - state: The final state of the event loop
                - structured_output: Parsed structured output when structured_output_model was specified
        """
        return run_async(
            lambda: self.invoke_async(
                prompt, invocation_state=invocation_state, structured_output_model=structured_output_model, **kwargs
            )
        )

    async def invoke_async(
        self,
        prompt: AgentInput = None,
        *,
        invocation_state: dict[str, Any] | None = None,
        structured_output_model: Type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Process a natural language prompt through the agent's event loop.

        This method implements the conversational interface with multiple input patterns:
        - String input: Simple text input
        - ContentBlock list: Multi-modal content blocks
        - Message list: Complete messages with roles
        - No input: Use existing conversation history

        Args:
            prompt: User input in various formats:
                - str: Simple text input
                - list[ContentBlock]: Multi-modal content blocks
                - list[Message]: Complete messages with roles
                - None: Use existing conversation history
            invocation_state: Additional parameters to pass through the event loop.
            structured_output_model: Pydantic model type(s) for structured output (overrides agent default).
            **kwargs: Additional parameters to pass through the event loop.[Deprecating]

        Returns:
            Result: object containing:

                - stop_reason: Why the event loop stopped (e.g., "end_turn", "max_tokens")
                - message: The final message from the model
                - metrics: Performance metrics from the event loop
                - state: The final state of the event loop
        """
        events = self.stream_async(
            prompt, invocation_state=invocation_state, structured_output_model=structured_output_model, **kwargs
        )
        async for event in events:
            _ = event

        return cast(AgentResult, event["result"])

    def structured_output(self, output_model: Type[T], prompt: AgentInput = None) -> T:
        """This method allows you to get structured output from the agent.

        If you pass in a prompt, it will be used temporarily without adding it to the conversation history.
        If you don't pass in a prompt, it will use only the existing conversation history to respond.

        For smaller models, you may want to use the optional prompt to add additional instructions to explicitly
        instruct the model to output the structured data.

        Args:
            output_model: The output model (a JSON schema written as a Pydantic BaseModel)
                that the agent will use when responding.
            prompt: The prompt to use for the agent in various formats:
                - str: Simple text input
                - list[ContentBlock]: Multi-modal content blocks
                - list[Message]: Complete messages with roles
                - None: Use existing conversation history

        Raises:
            ValueError: If no conversation history or prompt is provided.
        """
        warnings.warn(
            "Agent.structured_output method is deprecated."
            " You should pass in `structured_output_model` directly into the agent invocation."
            " see: https://strandsagents.com/latest/documentation/docs/user-guide/concepts/agents/structured-output/",
            category=DeprecationWarning,
            stacklevel=2,
        )

        return run_async(lambda: self.structured_output_async(output_model, prompt))

    async def structured_output_async(self, output_model: Type[T], prompt: AgentInput = None) -> T:
        """This method allows you to get structured output from the agent.

        If you pass in a prompt, it will be used temporarily without adding it to the conversation history.
        If you don't pass in a prompt, it will use only the existing conversation history to respond.

        For smaller models, you may want to use the optional prompt to add additional instructions to explicitly
        instruct the model to output the structured data.

        Args:
            output_model: The output model (a JSON schema written as a Pydantic BaseModel)
                that the agent will use when responding.
            prompt: The prompt to use for the agent (will not be added to conversation history).

        Raises:
            ValueError: If no conversation history or prompt is provided.
        -
        """
        if self._interrupt_state.activated:
            raise RuntimeError("cannot call structured output during interrupt")

        warnings.warn(
            "Agent.structured_output_async method is deprecated."
            " You should pass in `structured_output_model` directly into the agent invocation."
            " see: https://strandsagents.com/latest/documentation/docs/user-guide/concepts/agents/structured-output/",
            category=DeprecationWarning,
            stacklevel=2,
        )
        await self.hooks.invoke_callbacks_async(BeforeInvocationEvent(agent=self))
        with self.tracer.tracer.start_as_current_span(
            "execute_structured_output", kind=trace_api.SpanKind.CLIENT
        ) as structured_output_span:
            try:
                if not self.messages and not prompt:
                    raise ValueError("No conversation history or prompt provided")

                temp_messages: Messages = self.messages + await self._convert_prompt_to_messages(prompt)

                structured_output_span.set_attributes(
                    {
                        "gen_ai.system": "strands-agents",
                        "gen_ai.agent.name": self.name,
                        "gen_ai.agent.id": self.agent_id,
                        "gen_ai.operation.name": "execute_structured_output",
                    }
                )
                if self.system_prompt:
                    structured_output_span.add_event(
                        "gen_ai.system.message",
                        attributes={"role": "system", "content": serialize([{"text": self.system_prompt}])},
                    )
                for message in temp_messages:
                    structured_output_span.add_event(
                        f"gen_ai.{message['role']}.message",
                        attributes={"role": message["role"], "content": serialize(message["content"])},
                    )
                events = self.model.structured_output(output_model, temp_messages, system_prompt=self.system_prompt)
                async for event in events:
                    if isinstance(event, TypedEvent):
                        event.prepare(invocation_state={})
                        if event.is_callback_event:
                            self.callback_handler(**event.as_dict())

                structured_output_span.add_event(
                    "gen_ai.choice", attributes={"message": serialize(event["output"].model_dump())}
                )
                return event["output"]

            finally:
                await self.hooks.invoke_callbacks_async(AfterInvocationEvent(agent=self))

    def cleanup(self) -> None:
        """Clean up resources used by the agent.

        This method cleans up all tool providers that require explicit cleanup,
        such as MCP clients. It should be called when the agent is no longer needed
        to ensure proper resource cleanup.

        Note: This method uses a "belt and braces" approach with automatic cleanup
        through finalizers as a fallback, but explicit cleanup is recommended.
        """
        self.tool_registry.cleanup()

    def __del__(self) -> None:
        """Clean up resources when agent is garbage collected."""
        # __del__ is called even when an exception is thrown in the constructor,
        # so there is no guarantee tool_registry was set..
        if hasattr(self, "tool_registry"):
            self.tool_registry.cleanup()

    async def stream_async(
        self,
        prompt: AgentInput = None,
        *,
        invocation_state: dict[str, Any] | None = None,
        structured_output_model: Type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Process a natural language prompt and yield events as an async iterator.

        This method provides an asynchronous interface for streaming agent events with multiple input patterns:
        - String input: Simple text input
        - ContentBlock list: Multi-modal content blocks
        - Message list: Complete messages with roles
        - No input: Use existing conversation history

        Args:
            prompt: User input in various formats:
                - str: Simple text input
                - list[ContentBlock]: Multi-modal content blocks
                - list[Message]: Complete messages with roles
                - None: Use existing conversation history
            invocation_state: Additional parameters to pass through the event loop.
            structured_output_model: Pydantic model type(s) for structured output (overrides agent default).
            **kwargs: Additional parameters to pass to the event loop.[Deprecating]

        Yields:
            An async iterator that yields events. Each event is a dictionary containing
                information about the current state of processing, such as:

                - data: Text content being generated
                - complete: Whether this is the final chunk
                - current_tool_use: Information about tools being executed
                - And other event data provided by the callback handler

        Raises:
            Exception: Any exceptions from the agent invocation will be propagated to the caller.

        Example:
            ```python
            async for event in agent.stream_async("Analyze this data"):
                if "data" in event:
                    yield event["data"]
            ```
        """
        self._interrupt_state.resume(prompt)

        self.event_loop_metrics.reset_usage_metrics()

        merged_state = {}
        if kwargs:
            warnings.warn("`**kwargs` parameter is deprecating, use `invocation_state` instead.", stacklevel=2)
            merged_state.update(kwargs)
            if invocation_state is not None:
                merged_state["invocation_state"] = invocation_state
        else:
            if invocation_state is not None:
                merged_state = invocation_state

        callback_handler = self.callback_handler
        if kwargs:
            callback_handler = kwargs.get("callback_handler", self.callback_handler)

        # Process input and get message to add (if any)
        messages = await self._convert_prompt_to_messages(prompt)

        self.trace_span = self._start_agent_trace_span(messages)

        with trace_api.use_span(self.trace_span):
            try:
                events = self._run_loop(messages, merged_state, structured_output_model)

                async for event in events:
                    event.prepare(invocation_state=merged_state)

                    if event.is_callback_event:
                        as_dict = event.as_dict()
                        callback_handler(**as_dict)
                        yield as_dict

                result = AgentResult(*event["stop"])
                callback_handler(result=result)
                yield AgentResultEvent(result=result).as_dict()

                self._end_agent_trace_span(response=result)

            except Exception as e:
                self._end_agent_trace_span(error=e)
                raise

    async def _run_loop(
        self,
        messages: Messages,
        invocation_state: dict[str, Any],
        structured_output_model: Type[BaseModel] | None = None,
    ) -> AsyncGenerator[TypedEvent, None]:
        """Execute the agent's event loop with the given message and parameters.

        Args:
            messages: The input messages to add to the conversation.
            invocation_state: Additional parameters to pass to the event loop.
            structured_output_model: Optional Pydantic model type for structured output.

        Yields:
            Events from the event loop cycle.
        """
        await self.hooks.invoke_callbacks_async(BeforeInvocationEvent(agent=self))

        agent_result: AgentResult | None = None
        try:
            yield InitEventLoopEvent()

            await self._append_messages(*messages)

            structured_output_context = StructuredOutputContext(
                structured_output_model or self._default_structured_output_model
            )

            # Execute the event loop cycle with retry logic for context limits
            events = self._execute_event_loop_cycle(invocation_state, structured_output_context)
            async for event in events:
                # Signal from the model provider that the message sent by the user should be redacted,
                # likely due to a guardrail.
                if (
                    isinstance(event, ModelStreamChunkEvent)
                    and event.chunk
                    and event.chunk.get("redactContent")
                    and event.chunk["redactContent"].get("redactUserContentMessage")
                ):
                    self.messages[-1]["content"] = self._redact_user_content(
                        self.messages[-1]["content"], str(event.chunk["redactContent"]["redactUserContentMessage"])
                    )
                    if self._session_manager:
                        self._session_manager.redact_latest_message(self.messages[-1], self)
                yield event

            # Capture the result from the final event if available
            if isinstance(event, EventLoopStopEvent):
                agent_result = AgentResult(*event["stop"])

        finally:
            self.conversation_manager.apply_management(self)
            await self.hooks.invoke_callbacks_async(AfterInvocationEvent(agent=self, result=agent_result))

    async def _execute_event_loop_cycle(
        self, invocation_state: dict[str, Any], structured_output_context: StructuredOutputContext | None = None
    ) -> AsyncGenerator[TypedEvent, None]:
        """Execute the event loop cycle with retry logic for context window limits.

        This internal method handles the execution of the event loop cycle and implements
        retry logic for handling context window overflow exceptions by reducing the
        conversation context and retrying.

        Args:
            invocation_state: Additional parameters to pass to the event loop.
            structured_output_context: Optional structured output context for this invocation.

        Yields:
            Events of the loop cycle.
        """
        # Add `Agent` to invocation_state to keep backwards-compatibility
        invocation_state["agent"] = self

        if structured_output_context:
            structured_output_context.register_tool(self.tool_registry)

        try:
            events = event_loop_cycle(
                agent=self,
                invocation_state=invocation_state,
                structured_output_context=structured_output_context,
            )
            async for event in events:
                yield event

        except ContextWindowOverflowException as e:
            # Try reducing the context size and retrying
            self.conversation_manager.reduce_context(self, e=e)

            # Sync agent after reduce_context to keep conversation_manager_state up to date in the session
            if self._session_manager:
                self._session_manager.sync_agent(self)

            events = self._execute_event_loop_cycle(invocation_state, structured_output_context)
            async for event in events:
                yield event

        finally:
            if structured_output_context:
                structured_output_context.cleanup(self.tool_registry)

    async def _convert_prompt_to_messages(self, prompt: AgentInput) -> Messages:
        if self._interrupt_state.activated:
            return []

        messages: Messages | None = None
        if prompt is not None:
            # Check if the latest message is toolUse
            if len(self.messages) > 0 and any("toolUse" in content for content in self.messages[-1]["content"]):
                # Add toolResult message after to have a valid conversation
                logger.info(
                    "Agents latest message is toolUse, appending a toolResult message to have valid conversation."
                )
                tool_use_ids = [
                    content["toolUse"]["toolUseId"] for content in self.messages[-1]["content"] if "toolUse" in content
                ]
                await self._append_messages(
                    {
                        "role": "user",
                        "content": generate_missing_tool_result_content(tool_use_ids),
                    }
                )
            if isinstance(prompt, str):
                # String input - convert to user message
                messages = [{"role": "user", "content": [{"text": prompt}]}]
            elif isinstance(prompt, list):
                if len(prompt) == 0:
                    # Empty list
                    messages = []
                # Check if all item in input list are dictionaries
                elif all(isinstance(item, dict) for item in prompt):
                    # Check if all items are messages
                    if all(all(key in item for key in Message.__annotations__.keys()) for item in prompt):
                        # Messages input - add all messages to conversation
                        messages = cast(Messages, prompt)

                    # Check if all items are content blocks
                    elif all(any(key in ContentBlock.__annotations__.keys() for key in item) for item in prompt):
                        # Treat as List[ContentBlock] input - convert to user message
                        # This allows invalid structures to be passed through to the model
                        messages = [{"role": "user", "content": cast(list[ContentBlock], prompt)}]
        else:
            messages = []
        if messages is None:
            raise ValueError("Input prompt must be of type: `str | list[Contentblock] | Messages | None`.")
        return messages

    def _start_agent_trace_span(self, messages: Messages) -> trace_api.Span:
        """Starts a trace span for the agent.

        Args:
            messages: The input messages.
        """
        model_id = self.model.config.get("model_id") if hasattr(self.model, "config") else None
        return self.tracer.start_agent_span(
            messages=messages,
            agent_name=self.name,
            model_id=model_id,
            tools=self.tool_names,
            system_prompt=self.system_prompt,
            custom_trace_attributes=self.trace_attributes,
            tools_config=self.tool_registry.get_all_tools_config(),
        )

    def _end_agent_trace_span(
        self,
        response: Optional[AgentResult] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """Ends a trace span for the agent.

        Args:
            span: The span to end.
            response: Response to record as a trace attribute.
            error: Error to record as a trace attribute.
        """
        if self.trace_span:
            trace_attributes: dict[str, Any] = {
                "span": self.trace_span,
            }

            if response:
                trace_attributes["response"] = response
            if error:
                trace_attributes["error"] = error

            self.tracer.end_agent_span(**trace_attributes)

    def _initialize_system_prompt(
        self, system_prompt: str | list[SystemContentBlock] | None
    ) -> tuple[str | None, list[SystemContentBlock] | None]:
        """Initialize system prompt fields from constructor input.

        Maintains backwards compatibility by keeping system_prompt as str when string input
        provided, avoiding breaking existing consumers.

        Maps system_prompt input to both string and content block representations:
        - If string: system_prompt=string, _system_prompt_content=[{text: string}]
        - If list with text elements: system_prompt=concatenated_text, _system_prompt_content=list
        - If list without text elements: system_prompt=None, _system_prompt_content=list
        - If None: system_prompt=None, _system_prompt_content=None
        """
        if isinstance(system_prompt, str):
            return system_prompt, [{"text": system_prompt}]
        elif isinstance(system_prompt, list):
            # Concatenate all text elements for backwards compatibility, None if no text found
            text_parts = [block["text"] for block in system_prompt if "text" in block]
            system_prompt_str = "\n".join(text_parts) if text_parts else None
            return system_prompt_str, system_prompt
        else:
            return None, None

    async def _append_messages(self, *messages: Message) -> None:
        """Appends messages to history and invoke the callbacks for the MessageAddedEvent."""
        for message in messages:
            self.messages.append(message)
            await self.hooks.invoke_callbacks_async(MessageAddedEvent(agent=self, message=message))

    def _redact_user_content(self, content: list[ContentBlock], redact_message: str) -> list[ContentBlock]:
        """Redact user content preserving toolResult blocks.

        Args:
            content: content blocks to be redacted
            redact_message: redact message to be replaced

        Returns:
            Redacted content, as follows:
            - if the message contains at least a toolResult block,
                all toolResult blocks(s) are kept, redacting only the result content;
            - otherwise, the entire content of the message is replaced
                with a single text block with the redact message.
        """
        redacted_content = []
        for block in content:
            if "toolResult" in block:
                block["toolResult"]["content"] = [{"text": redact_message}]
                redacted_content.append(block)

        if not redacted_content:
            # Text content is added only if no toolResult blocks were found
            redacted_content = [{"text": redact_message}]

        return redacted_content
