"""Wasmtime-backed runtime adapter for :class:`strands.Agent`.

Implementation detail of the public SDK surface in ``strands/__init__.py``.
This module owns:

* the process-wide ``Engine`` + ``Component`` (one wasm load per process)
* the per-Agent ``Store`` + ``Linker`` + ``Instance`` lifecycle
* host-import callbacks (``tool-provider.call-tool``, ``host-log``)
* the host-side ``tool-event-stream`` resource the wasm component drains
* the async :class:`EventStream` wrapper that uses ``call_async`` to yield
  back to the asyncio event loop while the guest awaits I/O

External SDK code talks to ``_AgentRuntime`` and ``EventStream``; everything
else (WASI setup, marshaling, resource bookkeeping) is private.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wasmtime import Config, Engine, Store, WasiConfig
from wasmtime.component import (
    Component,
    Linker,
    ResourceAny,
    ResourceHost,
    ResourceType,
    Variant,
)

from . import _generated as _t

if TYPE_CHECKING:
    from . import Agent


_GUEST_LOGGER = logging.getLogger("strands.guest")
logger = logging.getLogger(__name__)


_singleton_lock = threading.RLock()
_engine: Engine | None = None
_component: Component | None = None


def _wasm_path() -> Path:
    env = os.environ.get("STRANDS_AGENT_WASM")
    if env:
        return Path(env)

    bundled = Path(__file__).resolve().parent / "strands-agent.wasm"
    if bundled.exists():
        return bundled

    sdk_root = Path(__file__).resolve().parents[3]
    dev = sdk_root / "strands-wasm" / "dist" / "strands-agent.wasm"
    if dev.exists():
        return dev

    raise FileNotFoundError(
        "Could not locate strands-agent.wasm. Set STRANDS_AGENT_WASM, install the "
        "bundled wheel, or build strands-wasm/dist/strands-agent.wasm."
    )


def _get_engine() -> Engine:
    global _engine
    with _singleton_lock:
        if _engine is None:
            cfg = Config()
            cfg.wasm_component_model = True
            cfg.wasm_component_model_async = True
            cfg.async_allow_sync = True
            _engine = Engine(cfg)
        return _engine


def _get_component() -> Component:
    global _component
    with _singleton_lock:
        if _component is None:
            _component = Component.from_file(_get_engine(), str(_wasm_path()))
        return _component


_TOOL_STREAM_RESOURCE_TYPE_TAG = 0x1
_tool_stream_resource_type: ResourceType | None = None


class _HostToolEventStream:
    """Host-side tool-event-stream backing a single ``call-tool`` invocation.

    Holds a queue of WIT ``tool-stream-event`` Variants. ``read()`` returns
    them one at a time, then ``None`` after the terminal ``complete`` /
    ``error`` event has been delivered.
    """

    def __init__(self) -> None:
        self._events: deque[Variant] = deque()
        self._closed = False

    def push(self, event: Variant) -> None:
        self._events.append(event)

    def close(self) -> None:
        self._closed = True

    def read(self) -> Variant | None:
        if self._events:
            return self._events.popleft()
        return None


class _ToolStreamRegistry:
    """Per-runtime registry for live host-side tool-event-stream reps.

    Scoped to a single :class:`_AgentRuntime` so streams are bounded by the
    runtime's lifetime: when the runtime is GC'd, its registry goes with it
    and any stragglers (e.g. from an aborted invocation) are released.
    """

    def __init__(self) -> None:
        self._reps: dict[int, _HostToolEventStream] = {}
        self._next_rep = 1
        self._lock = threading.Lock()

    def register(self, stream: _HostToolEventStream) -> int:
        with self._lock:
            rep = self._next_rep
            self._next_rep += 1
            self._reps[rep] = stream
            return rep

    def lookup(self, rep: int) -> _HostToolEventStream:
        with self._lock:
            return self._reps[rep]

    def drop(self, rep: int) -> None:
        with self._lock:
            self._reps.pop(rep, None)


def _ensure_tool_stream_type() -> ResourceType:
    # ResourceType identity must be process-stable so wasmtime-py recognizes
    # the same WIT resource across runtimes.
    global _tool_stream_resource_type
    if _tool_stream_resource_type is None:
        _tool_stream_resource_type = ResourceType.host(_TOOL_STREAM_RESOURCE_TYPE_TAG)
    return _tool_stream_resource_type


def _make_tool_call_handler(agent: Agent, registry: _ToolStreamRegistry):
    def call_tool(store: Any, args: Any) -> ResourceHost:
        name = getattr(args, "name", "")
        raw_input = getattr(args, "input", "")
        stream = _HostToolEventStream()
        try:
            tool = agent._lookup_tool(name)
            content_list = tool.invoke(raw_input)
        except Exception as exc:  # noqa: BLE001  surface any tool exception as a tool-error
            logger.exception("tool %r raised; surfacing as tool-error to guest", name)
            stream.push(_t.ToolStreamEvent.error(_t.ToolError.execution_failed(str(exc))))
            stream.close()
        else:
            # content_list items are already ToolResultContent wire variants.
            stream.push(_t.ToolStreamEvent.complete(content_list))
            stream.close()

        # Register, then hand ownership to the guest. On failure, drop the rep
        # ourselves since the guest never received it and won't drop it.
        rep = registry.register(stream)
        try:
            return ResourceHost.own(rep, _TOOL_STREAM_RESOURCE_TYPE_TAG)
        except BaseException:
            registry.drop(rep)
            raise

    return call_tool


def _host_log(_store: Any, entry: Any) -> None:
    level = getattr(entry, "level", "info")
    if not isinstance(level, str):
        level = str(level)
    message = getattr(entry, "message", "")
    context_raw = getattr(entry, "context", None)
    extra = {}
    if context_raw:
        try:
            extra = {"context": json.loads(context_raw)}
        except Exception:
            extra = {"context": context_raw}
    py_level = {
        "trace": logging.DEBUG,
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARNING,
        "error": logging.ERROR,
    }.get(level.lower(), logging.INFO)
    _GUEST_LOGGER.log(py_level, message, extra={"strands": extra} if extra else None)


def _make_tool_event_stream_read(registry: _ToolStreamRegistry):
    def _tool_event_stream_read(store: Any, handle: ResourceAny) -> Variant | None:
        host = handle.to_host(store)
        rep = host.rep
        stream = registry.lookup(rep)
        return stream.read()
    return _tool_event_stream_read


def _trap(name: str):
    def _f(*_a, **_k):
        raise RuntimeError(f"host import not implemented: {name}")
    return _f


_MODEL_EVENT_STREAM_TYPE_TAG = 0x2


def _register_imports(linker: Linker, agent: Agent, registry: _ToolStreamRegistry) -> None:
    tool_stream_type = _ensure_tool_stream_type()
    # model-provider's host-side stream type. Only reached on custom providers.
    model_event_stream_type = ResourceType.host(_MODEL_EVENT_STREAM_TYPE_TAG)

    with linker.root() as root:
        with root.add_instance("strands:agent/host-log@0.1.0") as ns:
            ns.add_func("log", _host_log)

        with root.add_instance("strands:agent/tools@0.1.0") as ns:
            ns.add_resource("tool-event-stream", tool_stream_type, lambda _store, rep: registry.drop(rep))
            ns.add_func("[method]tool-event-stream.read", _make_tool_event_stream_read(registry))

        with root.add_instance("strands:agent/tool-provider@0.1.0") as ns:
            ns.add_func("call-tool", _make_tool_call_handler(agent, registry))

        # Stubs for the imports the basic Agent.invoke flow never reaches.
        with root.add_instance("strands:agent/model-provider@0.1.0") as ns:
            ns.add_resource("model-event-stream", model_event_stream_type, lambda _s, _r: None)
            ns.add_func("[method]model-event-stream.read", _trap("model-event-stream.read"))
            ns.add_func("start-stream", _trap("model-provider.start-stream"))
            ns.add_func("count-tokens", _trap("model-provider.count-tokens"))

        with root.add_instance("strands:agent/snapshot-storage@0.1.0") as ns:
            for fname in ("save-snapshot", "load-snapshot", "list-snapshot-ids",
                          "delete-session", "load-manifest", "save-manifest"):
                ns.add_func(fname, _trap(f"snapshot-storage.{fname}"))

        with root.add_instance("strands:agent/snapshot-trigger-handler@0.1.0") as ns:
            ns.add_func("should-snapshot", _trap("snapshot-trigger-handler.should-snapshot"))

        with root.add_instance("strands:agent/edge-handler-registry@0.1.0") as ns:
            ns.add_func("evaluate", _trap("edge-handler-registry.evaluate"))

        with root.add_instance("strands:agent/elicitation-handler@0.1.0") as ns:
            ns.add_func("elicit", _trap("elicitation-handler.elicit"))


# --- Store + Linker -----------------------------------------------------

def _make_store_and_linker(
    agent: Agent, registry: _ToolStreamRegistry
) -> tuple[Store, Linker]:
    engine = _get_engine()
    store = Store(engine)
    wasi = WasiConfig()
    wasi.inherit_stdout()
    wasi.inherit_stderr()
    wasi.inherit_env()
    store.set_wasi(wasi)
    store.set_wasi_http()

    linker = Linker(engine)
    linker.allow_shadowing = True
    linker.add_wasip2_async()
    linker.add_wasi_http_async()
    _register_imports(linker, agent, registry)
    return store, linker


class EventStream:
    """Async iterator over guest-emitted :class:`StreamEvent` values.

    Each ``__anext__`` uses ``call_async`` to drive the WASM guest's
    ``event-stream.read`` export. This yields back to the asyncio event loop
    while the guest awaits network I/O (model calls), enabling true
    cooperative concurrency without thread pools.
    """

    def __init__(self, runtime: _AgentRuntime, handle: ResourceAny) -> None:
        self._runtime = runtime
        self._handle: ResourceAny | None = handle
        self._closed = False

    def __aiter__(self) -> EventStream:
        return self

    async def __anext__(self) -> Any:
        if self._closed or self._handle is None:
            raise StopAsyncIteration
        raw = await self._runtime.event_stream_read(self._handle)
        if raw is None:
            self._closed = True
            handle = self._handle
            self._handle = None
            handle.drop(self._runtime._store)
            raise StopAsyncIteration
        return _t.StreamEvent.lift(raw)


class _AgentRuntime:
    """Lazy wrapper around the wasm Agent resource for one ``strands.Agent``.

    Uses ``call_async`` for all WASM function invocations so that I/O-bound
    operations (HTTP calls to model providers) yield back to the asyncio
    event loop via a socket-pair notification bridge.
    """

    def __init__(self, agent: Agent) -> None:
        self._agent = agent
        self._tool_streams = _ToolStreamRegistry()
        self._store, self._linker = _make_store_and_linker(agent, self._tool_streams)
        self._instance = self._linker.instantiate(self._store, _get_component())
        self._funcs = _ApiFuncs(self._store, self._instance)
        self._handle: ResourceAny | None = None
        self._current_response: ResourceAny | None = None

    async def async_init(self) -> None:
        if self._handle is not None:
            return
        self._handle = await self._funcs.constructor.call_async(self._store, self._agent._config)
        self._funcs.constructor.post_return(self._store)

    async def generate(self, args: _t.InvokeArgs) -> EventStream:
        response_handle: ResourceAny = await self._funcs.generate.call_async(
            self._store, self._handle, args
        )
        self._funcs.generate.post_return(self._store)
        self._current_response = response_handle
        stream_handle: ResourceAny = await self._funcs.events.call_async(
            self._store, response_handle
        )
        self._funcs.events.post_return(self._store)
        return EventStream(self, stream_handle)

    def cancel(self) -> None:
        handle = self._current_response
        if handle is None:
            return
        self._funcs.cancel(self._store, handle)
        self._funcs.cancel.post_return(self._store)

    async def respond(self, args: _t.RespondArgs) -> None:
        handle = self._current_response
        if handle is None:
            from . import StrandsError

            raise StrandsError("respond() called with no in-flight invocation")
        res = await self._funcs.respond.call_async(self._store, handle, args)
        self._funcs.respond.post_return(self._store)
        _raise_on_err(res)

    async def get_messages(self) -> list[Any]:
        raw = await self._funcs.get_messages.call_async(self._store, self._handle)
        self._funcs.get_messages.post_return(self._store)
        return raw

    async def set_messages(self, messages: Iterable[Any]) -> None:
        wit = list(messages)
        res = await self._funcs.set_messages.call_async(self._store, self._handle, wit)
        self._funcs.set_messages.post_return(self._store)
        _raise_on_err(res)

    async def get_app_state(self) -> dict[str, Any]:
        raw = await self._funcs.get_app_state.call_async(self._store, self._handle)
        self._funcs.get_app_state.post_return(self._store)
        return json.loads(raw) if raw else {}

    async def set_app_state(self, state: dict[str, Any]) -> None:
        payload = json.dumps(state)
        res = await self._funcs.set_app_state.call_async(self._store, self._handle, payload)
        self._funcs.set_app_state.post_return(self._store)
        _raise_on_err(res)

    async def get_model_state(self) -> dict[str, Any]:
        raw = await self._funcs.get_model_state.call_async(self._store, self._handle)
        self._funcs.get_model_state.post_return(self._store)
        return json.loads(raw) if raw else {}

    async def set_model_state(self, state: dict[str, Any]) -> None:
        payload = json.dumps(state)
        res = await self._funcs.set_model_state.call_async(self._store, self._handle, payload)
        self._funcs.set_model_state.post_return(self._store)
        _raise_on_err(res)

    async def event_stream_read(self, handle: ResourceAny) -> Variant | None:
        raw = await self._funcs.event_stream_read.call_async(self._store, handle)
        self._funcs.event_stream_read.post_return(self._store)
        return raw


def _raise_on_err(res: Any) -> None:
    if isinstance(res, Variant) and res.tag == "err":
        from . import StrandsError

        # res.payload is a wire AgentError Variant; surface it raw for now.
        raise StrandsError(f"agent call failed: {res.payload!r}")


class _ApiFuncs:
    """Caches every exported function from ``strands:agent/api@0.1.0``."""

    def __init__(self, store: Store, instance: Any) -> None:
        api = instance.get_export_index(store, "strands:agent/api@0.1.0")
        if api is None:
            raise RuntimeError("component is missing strands:agent/api@0.1.0 export")

        def f(name: str):
            fn = instance.get_func(store, instance.get_export_index(store, name, api))
            if fn is None:
                raise RuntimeError(f"missing api export: {name}")
            return fn

        self.constructor = f("[constructor]agent")
        self.generate = f("[method]agent.generate")
        self.get_messages = f("[method]agent.get-messages")
        self.set_messages = f("[method]agent.set-messages")
        self.get_app_state = f("[method]agent.get-app-state")
        self.set_app_state = f("[method]agent.set-app-state")
        self.get_model_state = f("[method]agent.get-model-state")
        self.set_model_state = f("[method]agent.set-model-state")
        self.events = f("[method]response-stream.events")
        self.respond = f("[method]response-stream.respond")
        self.cancel = f("[method]response-stream.cancel")
        self.event_stream_read = f("[method]event-stream.read")
