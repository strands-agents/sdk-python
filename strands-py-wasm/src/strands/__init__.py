"""Strands Agents SDK Python surface.

Wire types live in :mod:`strands.types` (a re-export of the machine-generated
:mod:`strands._generated` package). Records are dataclasses; variant arms are
nested ``VariantCase`` subclasses (e.g. ``StreamEvent.TextDelta``) that pass
``isinstance`` and ``match`` natively.

This module overrides a handful of generated config dataclasses to fold in
ergonomic transforms (seconds → nanoseconds, ``**extras`` → JSON-encoded
``additional_config``, dict → list-of-pairs). Variant-arm wrapping is handled
at the boundary that consumes the value — Agent for its own slots,
SessionManager for storage, McpClientConfig for transport, etc.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import typing
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import asdict, is_dataclass
from typing import Any, Protocol, TypeVar, get_type_hints, runtime_checkable

from strands import _marshalling, types
from strands._runtime import _AgentRuntime

# First-class types users construct when wiring up an agent. Anything reached
# only through return values or pattern matching (StreamEvent, Interrupt,
# StopReason, Metrics, Usage, ContentBlock arms, etc.) lives in
# :mod:`strands.types` and is imported from there when needed.
from strands.types import (  # noqa: F401
    AgentSkills,
    ContextOffloader,
    CustomStorage,
    FileStorage,
    S3Storage,
    SlidingWindowConversationManager,
    SummarizingConversationManager,
)


class StrandsError(Exception):
    """Base class for all SDK-raised errors."""


class _ModelErrorBase(StrandsError):
    """Base for errors surfaced by a model provider."""


class ContextWindowOverflowError(_ModelErrorBase):
    """Input exceeded the model's context window and no recovery was possible."""


class MaxTokensError(_ModelErrorBase):
    """Model stopped generating because it hit the max-tokens budget."""

    def __init__(self, message: str, partial_message: types.Message | None = None) -> None:
        super().__init__(message)
        self.partial_message = partial_message


class ModelThrottledError(_ModelErrorBase):
    """Model provider throttled the request. Hook into retry to recover."""


class ProviderTokenCountError(_ModelErrorBase):
    """Provider-native token counting failed; base heuristic should run instead."""


class ToolValidationError(StrandsError):
    """A tool failed validation at registration or invocation time."""


class JsonValidationError(StrandsError):
    """A value could not be serialized to JSON."""


class StructuredOutputError(StrandsError):
    """Model refused to use the structured-output tool even after being forced."""


class ConcurrentInvocationError(StrandsError):
    """Agent is already processing an invocation; concurrent calls are not allowed."""


class SessionError(StrandsError):
    """Session storage read/write failed."""


# Bare payload records the user can pass as content. The SDK wraps each one
# in the matching ``ContentBlock`` arm before sending it on the wire.
_ContentPayload = (
    types.TextBlock
    | types.JsonBlock
    | types.ToolUseBlock
    | types.ToolResultBlock
    | types.ReasoningBlock
    | types.CachePointBlock
    | types.ImageBlock
    | types.VideoBlock
    | types.DocumentBlock
    | types.CitationsBlock
    | types.InterruptResponseBlock
)
ContentInput = str | _ContentPayload | types.ContentBlock
PromptInput = str | list[ContentInput]


class Message(types.Message):
    def __init__(
        self,
        *,
        role: types.Role,
        content: Iterable[ContentInput],
        metadata: types.MessageMetadata | None = None,
    ) -> None:
        super().__init__(
            role=role,
            content=[_marshalling.as_content_block(c) for c in content],
            metadata=metadata,
        )

    @classmethod
    def user(cls, *content: ContentInput, metadata: types.MessageMetadata | None = None) -> Message:
        return cls(role=types.Role.USER, content=content, metadata=metadata)

    @classmethod
    def assistant(cls, *content: ContentInput, metadata: types.MessageMetadata | None = None) -> Message:
        return cls(role=types.Role.ASSISTANT, content=content, metadata=metadata)


class BedrockModel(types.BedrockModel):
    def __init__(
        self,
        model_id: str = "us.anthropic.claude-opus-4-7-v1:0",
        *,
        region: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        session_token: str | None = None,
        **extras: Any,
    ) -> None:
        # The wasm bundle links the AWS SDK browser build, which has no credential
        # chain. Resolve via botocore so users get the same behavior they'd get
        # from any other Python AWS app (env vars, ~/.aws, SSO, IMDS, etc.).
        if access_key_id is None and secret_access_key is None:
            try:
                import botocore.session

                creds = botocore.session.Session().get_credentials()
                if creds is not None:
                    frozen = creds.get_frozen_credentials()
                    access_key_id = frozen.access_key
                    secret_access_key = frozen.secret_key
                    session_token = frozen.token
            except ImportError:
                pass

        super().__init__(
            model_id=model_id,
            region=region,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
            additional_config=_marshalling.extras_to_json(extras),
        )


class AnthropicModel(types.AnthropicModel):
    def __init__(self, model_id: str | None = None, *, api_key: str | None = None, **extras: Any) -> None:
        super().__init__(model_id=model_id, api_key=api_key, additional_config=_marshalling.extras_to_json(extras))


class OpenaiModel(types.OpenaiModel):
    def __init__(self, model_id: str | None = None, *, api_key: str | None = None, **extras: Any) -> None:
        super().__init__(model_id=model_id, api_key=api_key, additional_config=_marshalling.extras_to_json(extras))


class GoogleModel(types.GoogleModel):
    def __init__(self, model_id: str | None = None, *, api_key: str | None = None, **extras: Any) -> None:
        super().__init__(model_id=model_id, api_key=api_key, additional_config=_marshalling.extras_to_json(extras))


class CustomModel(types.CustomModel):
    def __init__(
        self,
        provider_id: str,
        *,
        model_id: str | None = None,
        stateful: bool = False,
        **extras: Any,
    ) -> None:
        super().__init__(
            provider_id=provider_id,
            model_id=model_id,
            additional_config=_marshalling.extras_to_json(extras),
            stateful=stateful,
        )


def agent_node(*, id: str, agent_config: Any, description: str | None, timeout: int | None) -> types.AgentNode:
    """Build an ``AgentNode`` with a JSON-encoded nested agent config.

    ``agent_config`` may be a string (passed through), a dataclass, or any
    object with a ``__dict__``; serialization happens here so call sites
    don't import :mod:`json`.
    """
    encoded = (
        agent_config if isinstance(agent_config, str) else json.dumps(agent_config, default=_marshalling.json_default)
    )
    return types.AgentNode(id=id, description=description, timeout=timeout, agent_config=encoded)


def multi_agent_node(*, id: str, orchestrator: Any, description: str | None) -> types.MultiAgentNode:
    """Build a ``MultiAgentNode`` with a JSON-encoded nested orchestrator."""
    encoded = (
        orchestrator if isinstance(orchestrator, str) else json.dumps(orchestrator, default=_marshalling.json_default)
    )
    return types.MultiAgentNode(id=id, description=description, orchestrator=encoded)


class StdioTransport(types.StdioTransport):
    """``StdioTransport`` with ``dict[str, str]`` env shorthand."""

    def __init__(
        self,
        *,
        command: str,
        args: list[str],
        env: dict[str, str],
        cwd: str | None,
    ) -> None:
        super().__init__(
            command=command,
            args=args,
            env=[types.EnvVar(key=k, value=v) for k, v in env.items()],
            cwd=cwd,
        )


class HttpTransport(types.HttpTransport):
    """``HttpTransport`` with ``dict[str, str]`` headers shorthand."""

    def __init__(self, *, url: str, headers: dict[str, str]) -> None:
        super().__init__(
            url=url,
            headers=[types.HttpHeader(name=k, value=v) for k, v in headers.items()],
        )


class SseTransport(types.SseTransport):
    """``SseTransport`` with ``dict[str, str]`` headers shorthand."""

    def __init__(self, *, url: str, headers: dict[str, str]) -> None:
        super().__init__(
            url=url,
            headers=[types.HttpHeader(name=k, value=v) for k, v in headers.items()],
        )


class InterruptResponse(types.RespondArgs):
    """Reply to a paused agent via ``response-stream.respond``."""

    def __init__(self, *, interrupt_id: str, response: Any) -> None:
        payload = response if isinstance(response, str) else json.dumps(response)
        super().__init__(interrupt_id=interrupt_id, response=payload)


class DecoratedTool:
    """A Python function exposed to the agent as a tool.

    Build one with the ``@tool`` decorator, then pass it to
    :class:`Agent` via ``tools=[...]``. The agent will call the function
    when the model invokes the tool by name.

    Callbacks must be synchronous. Async functions are not yet supported.

    Example::

        @tool
        def get_weather(city: str) -> str:
            \"\"\"Return the current weather for a city.\"\"\"
            return f"It is 72F and sunny in {city}."

        agent = Agent(model=BedrockModel(...), tools=[get_weather])
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        func: Callable[..., Any],
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.func = func

    def to_spec(self) -> types.ToolSpec:
        return types.ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=json.dumps(self.input_schema),
        )

    def invoke(self, raw_input: str) -> list[types.ToolResultContent]:
        """Run the tool with a JSON object of keyword arguments."""
        kwargs = json.loads(raw_input) if raw_input else {}
        return _normalize_tool_result(self.func(**kwargs))


def _normalize_tool_result(result: Any) -> list[types.ToolResultContent]:
    if isinstance(result, str):
        return [types.ToolResultContent.Text(types.TextBlock(text=result))]
    if isinstance(result, types.TextBlock):
        return [types.ToolResultContent.Text(result)]
    if isinstance(result, types.JsonBlock):
        return [types.ToolResultContent.Json(result)]
    if isinstance(result, dict):
        return [types.ToolResultContent.Json(types.JsonBlock(json=json.dumps(result)))]
    if is_dataclass(result) and not isinstance(result, type):
        return [types.ToolResultContent.Json(types.JsonBlock(json=json.dumps(asdict(result))))]
    if isinstance(result, list):
        return result
    return [types.ToolResultContent.Text(types.TextBlock(text=str(result)))]


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Decorator that turns a Python function into a :class:`DecoratedTool`.

    Only synchronous functions are supported at this time.
    """

    def wrap(f: Callable[..., Any]) -> DecoratedTool:
        hints = get_type_hints(f)
        sig = inspect.signature(f)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param_name, param in sig.parameters.items():
            properties[param_name] = _marshalling.py_type_to_schema(hints.get(param_name, str))
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
        schema: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return DecoratedTool(
            name=name or f.__name__,
            description=description or (f.__doc__ or "").strip() or f.__name__,
            input_schema=schema,
            func=f,
        )

    return wrap(func) if func is not None else wrap


# String shorthand picks a tool by name; otherwise pass a tagged ToolChoice arm.
_ToolChoiceInput = str | types.ToolChoice | None


class Agent:
    """Strands agent. Construct once; call :meth:`invoke` or :meth:`stream_async`."""

    def __init__(
        self,
        *,
        model: types.ModelInput | None = None,
        messages: list[types.Message] | None = None,
        system_prompt: PromptInput | None = None,
        tools: list[DecoratedTool] | None = None,
        agent_tools: list[types.AgentAsToolConfig] | None = None,
        vended_tools: list[types.VendedToolInput] | None = None,
        vended_plugins: list[types.VendedPluginInput] | None = None,
        mcp_clients: list[types.McpClientConfig] | None = None,
        name: str | None = None,
        id: str | None = None,
        description: str | None = None,
        tool_executor: types.ToolExecutorStrategy | None = None,
        display_output: bool | None = None,
        trace_attributes: list[types.TraceAttribute] | None = None,
        trace_context: types.TraceContext | None = None,
        session: types.SessionManager | None = None,
        conversation_manager: types.ConversationManagerInput | None = None,
        retry: types.RetryConfig | None = None,
        structured_output_schema: str | None = None,
        app_state: dict[str, Any] | None = None,
        model_state: dict[str, Any] | None = None,
    ) -> None:
        self._tools: list[DecoratedTool] = list(tools or [])
        identity = None
        if name is not None or id is not None or description is not None:
            identity = types.AgentIdentity(name=name, id=id, description=description)

        wrapped_vended_tools = (
            [_marshalling.wrap(v, _marshalling.VENDED_TOOL_ARM_BY_TYPE) for v in vended_tools] if vended_tools else None
        )
        wrapped_vended_plugins = (
            [_marshalling.wrap(p, _marshalling.VENDED_PLUGIN_ARM_BY_TYPE) for p in vended_plugins]
            if vended_plugins
            else None
        )

        self._config = types.AgentConfig(
            model=_marshalling.wrap(model, _marshalling.MODEL_ARM_BY_TYPE),
            model_params=None,
            messages=messages,
            system_prompt=(_marshalling.coerce_prompt(system_prompt) if system_prompt is not None else None),
            tools=[t.to_spec() for t in self._tools] or None,
            agent_tools=agent_tools,
            vended_tools=wrapped_vended_tools,
            vended_plugins=wrapped_vended_plugins,
            mcp_clients=mcp_clients,
            identity=identity,
            tool_executor=tool_executor,
            display_output=display_output,
            trace_attributes=trace_attributes,
            trace_context=trace_context,
            session=session,
            conversation_manager=_marshalling.wrap(conversation_manager, _marshalling.CM_ARM_BY_TYPE),
            retry=retry,
            structured_output_schema=structured_output_schema,
            app_state=json.dumps(app_state) if app_state else None,
            model_state=json.dumps(model_state) if model_state else None,
        )
        self._runtime = _AgentRuntime(self)
        self._runtime.init()

    @property
    def config(self) -> types.AgentConfig:
        return self._config

    def _lookup_tool(self, name: str) -> DecoratedTool:
        for t in self._tools:
            if getattr(t, "name", None) == name:
                return t
        raise KeyError(f"no tool registered under name {name!r}")

    def _build_invoke_args(
        self,
        prompt: PromptInput,
        tools: list[DecoratedTool] | None,
        tool_choice: _ToolChoiceInput,
        structured_output_schema: str | None,
    ) -> types.InvokeArgs:
        extra_tools = [t.to_spec() for t in (tools or [])] or None
        return types.InvokeArgs(
            input=_marshalling.coerce_prompt(prompt),
            tools=extra_tools,
            tool_choice=_marshalling.coerce_tool_choice(tool_choice),
            structured_output_schema=structured_output_schema,
        )

    async def stream_async(
        self,
        prompt: PromptInput,
        *,
        tools: list[DecoratedTool] | None = None,
        tool_choice: _ToolChoiceInput = None,
        structured_output_schema: str | None = None,
    ) -> AsyncIterator[types.StreamEvent]:
        """Yield :class:`StreamEvent` arms as the agent runs."""
        args = self._build_invoke_args(prompt, tools, tool_choice, structured_output_schema)
        stream = await self._runtime.generate(args)
        async for event in stream:
            yield event

    async def invoke_async(
        self,
        prompt: PromptInput,
        *,
        tools: list[DecoratedTool] | None = None,
        tool_choice: _ToolChoiceInput = None,
        structured_output_schema: str | None = None,
    ) -> AgentResult:
        """Run the agent to completion and return an :class:`AgentResult`."""
        accumulator = _AgentResultAccumulator()
        async for event in self.stream_async(
            prompt,
            tools=tools,
            tool_choice=tool_choice,
            structured_output_schema=structured_output_schema,
        ):
            accumulator.consume(event)
        return accumulator.finalize(self)

    def invoke(
        self,
        prompt: PromptInput,
        *,
        tools: list[DecoratedTool] | None = None,
        tool_choice: _ToolChoiceInput = None,
        structured_output_schema: str | None = None,
    ) -> AgentResult:
        """Synchronous wrapper around :meth:`invoke_async`.

        Raises :class:`RuntimeError` if called from a running event loop. Use
        :meth:`invoke_async` directly in Jupyter or async frameworks.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "Agent.invoke() cannot run inside an existing event loop. Use 'await agent.invoke_async(...)' instead."
            )
        return asyncio.run(
            self.invoke_async(
                prompt,
                tools=tools,
                tool_choice=tool_choice,
                structured_output_schema=structured_output_schema,
            )
        )

    def cancel(self) -> None:
        """Cancel the in-flight invocation. Fire-and-forget."""
        self._runtime.cancel()

    async def respond(self, interrupt_id: str, response: Any) -> None:
        payload = response if isinstance(response, str) else json.dumps(response)
        await self._runtime.respond(types.RespondArgs(interrupt_id=interrupt_id, response=payload))

    async def get_messages(self) -> list[types.Message]:
        return await self._runtime.get_messages()

    async def set_messages(self, messages: list[types.Message]) -> None:
        await self._runtime.set_messages(messages)

    def get_app_state(self) -> dict[str, Any]:
        return self._runtime.get_app_state()

    def set_app_state(self, state: dict[str, Any]) -> None:
        self._runtime.set_app_state(state)

    async def get_model_state(self) -> dict[str, Any]:
        return await self._runtime.get_model_state()

    async def set_model_state(self, state: dict[str, Any]) -> None:
        await self._runtime.set_model_state(state)


class _AgentResultAccumulator:
    """Folds the stream of events into the fields of an :class:`AgentResult`."""

    def __init__(self) -> None:
        self._stop: types.StopEvent | None = None
        self._last_message: types.Message | None = None
        self._interrupts: list[types.Interrupt] = []

    def consume(self, event: types.StreamEvent) -> None:
        if isinstance(event, types.StreamEvent.MessageAdded):
            self._last_message = event.value.message
        elif isinstance(event, types.StreamEvent.ModelMessage):
            self._last_message = event.value.message
        elif isinstance(event, types.StreamEvent.Stop):
            self._stop = event.value
        elif isinstance(event, types.StreamEvent.AgentResult):
            self._stop = event.value.stop
        elif isinstance(event, types.StreamEvent.Interrupt):
            self._interrupts.append(event.value)

    def finalize(self, agent: Agent) -> AgentResult:
        stop = self._stop
        last = self._last_message
        if last is None:
            last = types.Message(role=types.Role.ASSISTANT, content=[], metadata=None)
        return AgentResult(
            stop_reason=stop.reason if stop is not None else types.StopReason.END_TURN,
            last_message=last,
            usage=stop.usage if stop is not None else None,
            metrics=None,
            structured_output=(json.loads(stop.structured_output) if stop and stop.structured_output else None),
            interrupts=self._interrupts or None,
        )


_HookEventT = TypeVar("_HookEventT", bound=types.StreamEvent)
_HookCallback = Callable[[types.StreamEvent], Any]


@runtime_checkable
class HookProvider(Protocol):
    """Bundle of related hook registrations."""

    def register_hooks(self, registry: HookRegistry) -> None: ...


class HookRegistry:
    """Register callbacks keyed by ``StreamEvent`` arm class.

    Each arm of the wire ``stream-event`` variant is a distinct Python class
    (``StreamEvent.TextDelta``, ``StreamEvent.Stop``, ...). Subscribers match
    by exact class.

    Callbacks for arms whose name begins with ``After`` dispatch in reverse
    registration order, mirroring teardown semantics. Everything else
    dispatches FIFO.
    """

    def __init__(self) -> None:
        self._callbacks: dict[type, list[_HookCallback]] = {}

    def add_callback(
        self,
        event_type: type[_HookEventT],
        callback: Callable[[_HookEventT], Any],
    ) -> Callable[[], None]:
        entries = self._callbacks.setdefault(event_type, [])
        entry = typing.cast(_HookCallback, callback)
        entries.append(entry)

        def _remove() -> None:
            try:
                self._callbacks[event_type].remove(entry)
            except (KeyError, ValueError):
                pass

        return _remove

    def add_hook(self, provider: HookProvider) -> None:
        provider.register_hooks(self)

    def dispatch(self, event: Any) -> None:
        callbacks = self._callbacks_for(event)
        if any(inspect.iscoroutinefunction(cb) for cb in callbacks):
            raise RuntimeError(f"event={type(event).__name__} | use dispatch_async for async callbacks")
        for cb in callbacks:
            cb(event)

    async def dispatch_async(self, event: Any) -> None:
        for cb in self._callbacks_for(event):
            result = cb(event)
            if inspect.iscoroutine(result):
                await typing.cast(Awaitable[Any], result)

    def _callbacks_for(self, event: Any) -> list[_HookCallback]:
        entries = self._callbacks.get(type(event), [])
        return list(reversed(entries)) if type(event).__name__.startswith("After") else list(entries)


class AgentResult:
    """Terminal result of an agent invocation."""

    def __init__(
        self,
        *,
        stop_reason: types.StopReason,
        last_message: types.Message,
        invocation_state: dict[str, Any] | None = None,
        traces: list[types.AgentTrace] | None = None,
        metrics: types.AgentMetrics | None = None,
        usage: types.Usage | None = None,
        structured_output: Any = None,
        interrupts: list[types.Interrupt] | None = None,
    ) -> None:
        self.stop_reason = stop_reason
        self.last_message = last_message
        self.invocation_state = invocation_state if invocation_state is not None else {}
        self.traces = traces
        self.metrics = metrics
        self.usage = usage
        self.structured_output = structured_output
        self.interrupts = interrupts

    @property
    def context_size(self) -> int | None:
        return self.metrics.latest_context_size if self.metrics else None

    @property
    def projected_context_size(self) -> int | None:
        return self.metrics.projected_context_size if self.metrics else None

    def __str__(self) -> str:
        """Concatenate text from TextBlock and ReasoningBlock content."""
        chunks: list[str] = []
        for block in self.last_message.content:
            tag = getattr(block, "tag", None)
            payload = getattr(block, "payload", None)
            if tag == "text" and payload is not None:
                chunks.append(payload.text)
            elif tag == "reasoning" and payload is not None and payload.text:
                chunks.append(payload.text)
        return "\n".join(chunks)
