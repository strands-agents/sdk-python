"""Agent Interface.

This module implements the core Agent class that serves as the primary entry point for interacting with foundation
models and tools in the SDK.

The Agent interface supports two complementary interaction patterns:

1. Natural language for conversation: `agent("Analyze this data")`
2. Method-style for direct tool access: `agent.tool.tool_name(param1="value")`
"""

import asyncio
import json
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from typing import Any, AsyncIterator, Callable, Dict, List, Mapping, Optional, Union
from uuid import uuid4

from opentelemetry import trace

from ..event_loop.event_loop import event_loop_cycle
from ..handlers.callback_handler import CompositeCallbackHandler, PrintingCallbackHandler, null_callback_handler
from ..handlers.tool_handler import AgentToolHandler
from ..models.bedrock import BedrockModel
from ..telemetry.metrics import EventLoopMetrics
from ..telemetry.tracer import get_tracer
from ..tools.registry import ToolRegistry
from ..tools.thread_pool_executor import ThreadPoolExecutorWrapper
from ..tools.watcher import ToolWatcher
from ..types.content import ContentBlock, Message, Messages
from ..types.exceptions import ContextWindowOverflowException
from ..types.models import Model
from ..types.tools import ToolConfig
from ..types.traces import AttributeValue
from .agent_result import AgentResult
from .conversation_manager import (
    ConversationManager,
    SlidingWindowConversationManager,
)

logger = logging.getLogger(__name__)


# Sentinel class and object to distinguish between explicit None and default parameter value
class _DefaultCallbackHandlerSentinel:
    """Sentinel class to distinguish between explicit None and default parameter value."""

    pass


_DEFAULT_CALLBACK_HANDLER = _DefaultCallbackHandlerSentinel()


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

    class ToolCaller:
        """Call tool as a function."""

        def __init__(self, agent: "Agent") -> None:
            """Initialize instance.

            Args:
                agent: Agent reference that will accept tool results.
            """
            # WARNING: Do not add any other member variables or methods as this could result in a name conflict with
            #          agent tools and thus break their execution.
            self._agent = agent

        def __getattr__(self, name: str) -> Callable[..., Any]:
            """Call tool as a function.

            This method enables the method-style interface (e.g., `agent.tool.tool_name(param="value")`).

            Args:
                name: The name of the attribute (tool) being accessed.

            Returns:
                A function that when called will execute the named tool.

            Raises:
                AttributeError: If no tool with the given name exists.
            """

            def caller(**kwargs: Any) -> Any:
                """Call a tool directly by name.

                Args:
                    **kwargs: Keyword arguments to pass to the tool.

                        - user_message_override: Custom message to record instead of default
                        - tool_execution_handler: Custom handler for tool execution
                        - event_loop_metrics: Custom metrics collector
                        - messages: Custom message history to use
                        - tool_config: Custom tool configuration
                        - callback_handler: Custom callback handler
                        - record_direct_tool_call: Whether to record this call in history

                Returns:
                    The result returned by the tool.

                Raises:
                    AttributeError: If the tool doesn't exist.
                """
                if name not in self._agent.tool_registry.registry:
                    raise AttributeError(f"Tool '{name}' not found")

                # Create unique tool ID and set up the tool request
                tool_id = f"tooluse_{name}_{random.randint(100000000, 999999999)}"
                tool_use = {
                    "toolUseId": tool_id,
                    "name": name,
                    "input": kwargs.copy(),
                }

                # Extract tool execution parameters
                user_message_override = kwargs.get("user_message_override", None)
                tool_execution_handler = kwargs.get("tool_execution_handler", self._agent.thread_pool_wrapper)
                event_loop_metrics = kwargs.get("event_loop_metrics", self._agent.event_loop_metrics)
                messages = kwargs.get("messages", self._agent.messages)
                tool_config = kwargs.get("tool_config", self._agent.tool_config)
                callback_handler = kwargs.get("callback_handler", self._agent.callback_handler)
                record_direct_tool_call = kwargs.get("record_direct_tool_call", self._agent.record_direct_tool_call)

                # Process tool call
                handler_kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if k
                    not in [
                        "tool_execution_handler",
                        "event_loop_metrics",
                        "messages",
                        "tool_config",
                        "callback_handler",
                        "tool_handler",
                        "system_prompt",
                        "model",
                        "model_id",
                        "user_message_override",
                        "agent",
                        "record_direct_tool_call",
                    ]
                }

                # Execute the tool
                tool_result = self._agent.tool_handler.process(
                    tool=tool_use,
                    model=self._agent.model,
                    system_prompt=self._agent.system_prompt,
                    messages=messages,
                    tool_config=tool_config,
                    callback_handler=callback_handler,
                    tool_execution_handler=tool_execution_handler,
                    event_loop_metrics=event_loop_metrics,
                    agent=self._agent,
                    **handler_kwargs,
                )

                if record_direct_tool_call:
                    # Create a record of this tool execution in the message history
                    self._agent._record_tool_execution(tool_use, tool_result, user_message_override, messages)

                # Apply window management
                self._agent.conversation_manager.apply_management(self._agent)

                return tool_result

            return caller

    def __init__(
        self,
        model: Union[Model, str, None] = None,
        messages: Optional[Messages] = None,
        tools: Optional[List[Union[str, Dict[str, str], Any]]] = None,
        system_prompt: Optional[str] = None,
        callback_handler: Optional[
            Union[Callable[..., Any], _DefaultCallbackHandlerSentinel]
        ] = _DEFAULT_CALLBACK_HANDLER,
        conversation_manager: Optional[ConversationManager] = None,
        max_parallel_tools: int = os.cpu_count() or 1,
        record_direct_tool_call: bool = True,
        load_tools_from_directory: bool = True,
        trace_attributes: Optional[Mapping[str, AttributeValue]] = None,
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
                - Functions decorated with `@strands.tool` decorator.

                If provided, only these tools will be available. If None, all tools will be available.
            system_prompt: System prompt to guide model behavior.
                If None, the model will behave according to its default settings.
            callback_handler: Callback for processing events as they happen during agent execution.
                If not provided (using the default), a new PrintingCallbackHandler instance is created.
                If explicitly set to None, null_callback_handler is used.
            conversation_manager: Manager for conversation history and context window.
                Defaults to strands.agent.conversation_manager.SlidingWindowConversationManager if None.
            max_parallel_tools: Maximum number of tools to run in parallel when the model returns multiple tool calls.
                Defaults to os.cpu_count() or 1.
            record_direct_tool_call: Whether to record direct tool calls in message history.
                Defaults to True.
            load_tools_from_directory: Whether to load and automatically reload tools in the `./tools/` directory.
                Defaults to True.
            trace_attributes: Custom trace attributes to apply to the agent's trace span.

        Raises:
            ValueError: If max_parallel_tools is less than 1.
        """
        self.model = BedrockModel() if not model else BedrockModel(model_id=model) if isinstance(model, str) else model
        self.messages = messages if messages is not None else []

        self.system_prompt = system_prompt

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
        self.trace_attributes: Dict[str, AttributeValue] = {}
        if trace_attributes:
            for k, v in trace_attributes.items():
                if isinstance(v, (str, int, float, bool)) or (
                    isinstance(v, list) and all(isinstance(x, (str, int, float, bool)) for x in v)
                ):
                    self.trace_attributes[k] = v

        # If max_parallel_tools is 1, we execute tools sequentially
        self.thread_pool = None
        self.thread_pool_wrapper = None
        if max_parallel_tools > 1:
            self.thread_pool = ThreadPoolExecutor(max_workers=max_parallel_tools)
            self.thread_pool_wrapper = ThreadPoolExecutorWrapper(self.thread_pool)
        elif max_parallel_tools < 1:
            raise ValueError("max_parallel_tools must be greater than 0")

        self.record_direct_tool_call = record_direct_tool_call
        self.load_tools_from_directory = load_tools_from_directory

        self.tool_registry = ToolRegistry()
        self.tool_handler = AgentToolHandler(tool_registry=self.tool_registry)

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
        self.trace_span: Optional[trace.Span] = None

        self.tool_caller = Agent.ToolCaller(self)

    @property
    def tool(self) -> ToolCaller:
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
    def tool_names(self) -> List[str]:
        """Get a list of all registered tool names.

        Returns:
            Names of all tools available to this agent.
        """
        all_tools = self.tool_registry.get_all_tools_config()
        return list(all_tools.keys())

    @property
    def tool_config(self) -> ToolConfig:
        """Get the tool configuration for this agent.

        Returns:
            The complete tool configuration.
        """
        return self.tool_registry.initialize_tool_config()

    def __del__(self) -> None:
        """Clean up resources when Agent is garbage collected.

        Ensures proper shutdown of the thread pool executor if one exists.
        """
        if self.thread_pool_wrapper and hasattr(self.thread_pool_wrapper, "shutdown"):
            self.thread_pool_wrapper.shutdown(wait=False)
            logger.debug("thread pool executor shutdown complete")

    def __call__(self, prompt: str, **kwargs: Any) -> AgentResult:
        """Process a natural language prompt through the agent's event loop.

        This method implements the conversational interface (e.g., `agent("hello!")`). It adds the user's prompt to
        the conversation history, processes it through the model, executes any tool calls, and returns the final result.

        Args:
            prompt: The natural language prompt from the user.
            **kwargs: Additional parameters to pass to the event loop.

        Returns:
            Result object containing:

                - stop_reason: Why the event loop stopped (e.g., "end_turn", "max_tokens")
                - message: The final message from the model
                - metrics: Performance metrics from the event loop
                - state: The final state of the event loop
        """
        self._start_agent_trace_span(prompt)

        try:
            # Run the event loop and get the result
            result = self._run_loop(prompt, kwargs)

            self._end_agent_trace_span(response=result)

            return result
        except Exception as e:
            self._end_agent_trace_span(error=e)

            # Re-raise the exception to preserve original behavior
            raise

    async def stream_async(self, prompt: str, **kwargs: Any) -> AsyncIterator[Any]:
        """Process a natural language prompt and yield events as an async iterator.

        This method provides an asynchronous interface for streaming agent events, allowing
        consumers to process stream events programmatically through an async iterator pattern
        rather than callback functions. This is particularly useful for web servers and other
        async environments.

        Args:
            prompt: The natural language prompt from the user.
            **kwargs: Additional parameters to pass to the event loop.

        Returns:
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
        self._start_agent_trace_span(prompt)

        _stop_event = uuid4()

        queue = asyncio.Queue[Any]()
        loop = asyncio.get_event_loop()

        def enqueue(an_item: Any) -> None:
            nonlocal queue
            nonlocal loop
            loop.call_soon_threadsafe(queue.put_nowait, an_item)

        def queuing_callback_handler(**handler_kwargs: Any) -> None:
            enqueue(handler_kwargs.copy())

        def target_callback() -> None:
            nonlocal kwargs

            try:
                result = self._run_loop(prompt, kwargs, supplementary_callback_handler=queuing_callback_handler)
                self._end_agent_trace_span(response=result)
            except Exception as e:
                self._end_agent_trace_span(error=e)
                enqueue(e)
            finally:
                enqueue(_stop_event)

        thread = Thread(target=target_callback, daemon=True)
        thread.start()

        try:
            while True:
                item = await queue.get()
                if item == _stop_event:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            thread.join()

    def _run_loop(
        self, prompt: str, kwargs: Dict[str, Any], supplementary_callback_handler: Optional[Callable[..., Any]] = None
    ) -> AgentResult:
        """Execute the agent's event loop with the given prompt and parameters."""
        try:
            # If the call had a callback_handler passed in, then for this event_loop
            # cycle we call both handlers as the callback_handler
            invocation_callback_handler = (
                CompositeCallbackHandler(self.callback_handler, supplementary_callback_handler)
                if supplementary_callback_handler is not None
                else self.callback_handler
            )

            # Extract key parameters
            invocation_callback_handler(init_event_loop=True, **kwargs)

            # Set up the user message with optional knowledge base retrieval
            message_content: List[ContentBlock] = [{"text": prompt}]
            new_message: Message = {"role": "user", "content": message_content}
            self.messages.append(new_message)

            # Execute the event loop cycle with retry logic for context limits
            return self._execute_event_loop_cycle(invocation_callback_handler, kwargs)

        finally:
            self.conversation_manager.apply_management(self)

    def _execute_event_loop_cycle(self, callback_handler: Callable[..., Any], kwargs: Dict[str, Any]) -> AgentResult:
        """Execute the event loop cycle with retry logic for context window limits.

        This internal method handles the execution of the event loop cycle and implements
        retry logic for handling context window overflow exceptions by reducing the
        conversation context and retrying.

        Returns:
            The result of the event loop cycle.
        """
        # Extract parameters with fallbacks to instance values
        system_prompt = kwargs.pop("system_prompt", self.system_prompt)
        model = kwargs.pop("model", self.model)
        tool_execution_handler = kwargs.pop("tool_execution_handler", self.thread_pool_wrapper)
        event_loop_metrics = kwargs.pop("event_loop_metrics", self.event_loop_metrics)
        callback_handler_override = kwargs.pop("callback_handler", callback_handler)
        tool_handler = kwargs.pop("tool_handler", self.tool_handler)
        messages = kwargs.pop("messages", self.messages)
        tool_config = kwargs.pop("tool_config", self.tool_config)
        kwargs.pop("agent", None)  # Remove agent to avoid conflicts

        try:
            # Execute the main event loop cycle
            stop_reason, message, metrics, state = event_loop_cycle(
                model=model,
                system_prompt=system_prompt,
                messages=messages,  # will be modified by event_loop_cycle
                tool_config=tool_config,
                callback_handler=callback_handler_override,
                tool_handler=tool_handler,
                tool_execution_handler=tool_execution_handler,
                event_loop_metrics=event_loop_metrics,
                agent=self,
                event_loop_parent_span=self.trace_span,
                **kwargs,
            )

            return AgentResult(stop_reason, message, metrics, state)

        except ContextWindowOverflowException as e:
            # Try reducing the context size and retrying

            self.conversation_manager.reduce_context(self, e=e)
            return self._execute_event_loop_cycle(callback_handler_override, kwargs)

    def _record_tool_execution(
        self,
        tool: Dict[str, Any],
        tool_result: Dict[str, Any],
        user_message_override: Optional[str],
        messages: List[Dict[str, Any]],
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
            messages: The message history to append to.
        """
        # Create user message describing the tool call
        user_msg_content = [
            {"text": (f"agent.tool.{tool['name']} direct tool call.\nInput parameters: {json.dumps(tool['input'])}\n")}
        ]

        # Add override message if provided
        if user_message_override:
            user_msg_content.insert(0, {"text": f"{user_message_override}\n"})

        # Create the message sequence
        user_msg = {
            "role": "user",
            "content": user_msg_content,
        }
        tool_use_msg = {
            "role": "assistant",
            "content": [{"toolUse": tool}],
        }
        tool_result_msg = {
            "role": "user",
            "content": [{"toolResult": tool_result}],
        }
        assistant_msg = {
            "role": "assistant",
            "content": [{"text": f"agent.{tool['name']} was called"}],
        }

        # Add to message history
        messages.append(user_msg)
        messages.append(tool_use_msg)
        messages.append(tool_result_msg)
        messages.append(assistant_msg)

    def _start_agent_trace_span(self, prompt: str) -> None:
        """Starts a trace span for the agent.

        Args:
            prompt: The natural language prompt from the user.
        """
        model_id = self.model.config.get("model_id") if hasattr(self.model, "config") else None

        self.trace_span = self.tracer.start_agent_span(
            prompt=prompt,
            model_id=model_id,
            tools=self.tool_names,
            system_prompt=self.system_prompt,
            custom_trace_attributes=self.trace_attributes,
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
            trace_attributes: Dict[str, Any] = {
                "span": self.trace_span,
            }

            if response:
                trace_attributes["response"] = response
            if error:
                trace_attributes["error"] = error

            self.tracer.end_agent_span(**trace_attributes)
