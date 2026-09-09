import asyncio
import logging
import unittest.mock
import warnings

import pytest
import pytest_asyncio

from strands import tool
from strands.experimental.bidi import BidiAgent
from strands.experimental.bidi.agent.loop import (
    _MAX_TOOL_RESULT_RECOVERY_BYTES,
    _format_tool_result_recovery,
    _ReaderError,
)
from strands.experimental.bidi.models import BidiModel, BidiModelTimeoutError
from strands.experimental.bidi.types.events import (
    BidiAudioStreamEvent,
    BidiConnectionCloseEvent,
    BidiConnectionRestartEvent,
    BidiConnectionWarningEvent,
    BidiResponseCompleteEvent,
    BidiResponseStartEvent,
    BidiTextInputEvent,
    BidiTranscriptStreamEvent,
    BidiUsageEvent,
)
from strands.experimental.hooks.events import BidiBeforeConnectionRestartEvent
from strands.types._events import ToolResultEvent, ToolResultMessageEvent, ToolUseStreamEvent
from strands.types.tools import ToolResult, ToolUse


@pytest.fixture
def time_tool():
    @tool(name="time_tool")
    async def func():
        return "12:00"

    return func


@pytest.fixture
def agent(time_tool):
    model = unittest.mock.AsyncMock(spec=BidiModel)
    model.restart = unittest.mock.AsyncMock()
    return BidiAgent(model=model, tools=[time_tool])


@pytest_asyncio.fixture
async def loop(agent):
    return agent._loop


@pytest.mark.asyncio
async def test_bidi_agent_loop_receive_restart_connection(loop, agent, agenerator):
    timeout_error = BidiModelTimeoutError("test timeout", test_restart_config=1)
    text_event = BidiTextInputEvent(text="test after restart")

    agent.model.receive = unittest.mock.Mock(side_effect=[timeout_error, agenerator([text_event])])

    await loop.start()

    tru_events = []
    async for event in loop.receive():
        tru_events.append(event)
        if len(tru_events) >= 2:
            break

    exp_events = [
        BidiConnectionRestartEvent(reason="timeout", timeout_error=timeout_error),
        text_event,
    ]
    assert tru_events == exp_events

    # The reactive path restarts through the provider method and forwards the timeout config.
    assert agent.model.start.call_count == 1
    agent.model.restart.assert_called_once_with(
        agent.system_prompt,
        agent.tool_registry.get_all_tool_specs(),
        agent.messages,
        test_restart_config=1,
    )


@pytest.mark.asyncio
async def test_reactive_restart_failure_yields_event_before_raising(loop, agent, agenerator):
    """A failed reactive restart still notifies the caller before surfacing the failure."""
    timeout_error = BidiModelTimeoutError("test timeout")
    restart_error = RuntimeError("restart failed")
    agent.model.connection_config = {}
    agent.model.receive = unittest.mock.Mock(side_effect=timeout_error)
    agent.model.restart.side_effect = restart_error

    await loop.start()
    consumer = loop.receive()

    event = await consumer.__anext__()
    assert event == BidiConnectionRestartEvent(reason="timeout", timeout_error=timeout_error)
    with pytest.raises(RuntimeError, match="restart failed"):
        await consumer.__anext__()

    await loop.stop()


@pytest.mark.asyncio
async def test_bidi_agent_loop_auto_reconnect_default_on(loop, agent, agenerator):
    """Auto reconnect is the default: a timeout triggers reconnect without any opt-in."""
    # AsyncMock(spec=BidiModel) generates connection_config as a Mock; force the realistic
    # "provider declared nothing" case so the loop falls back to its default (reconnect).
    agent.model.connection_config = {}
    timeout_error = BidiModelTimeoutError("test timeout")
    text_event = BidiTextInputEvent(text="after restart")
    agent.model.receive = unittest.mock.Mock(side_effect=[timeout_error, agenerator([text_event])])

    await loop.start()

    received = []
    async for event in loop.receive():
        received.append(event)
        if len(received) >= 2:
            break

    agent.model.restart.assert_called_once()


@pytest.mark.asyncio
async def test_bidi_agent_loop_auto_reconnect_opt_out_surfaces_timeout(loop, agent, agenerator):
    """A provider opting out with auto_reconnect=False surfaces the timeout instead of reconnecting."""
    agent.model.connection_config = {"auto_reconnect": False}
    timeout_error = BidiModelTimeoutError("test timeout")
    agent.model.receive = unittest.mock.Mock(side_effect=[timeout_error, agenerator([])])

    await loop.start()

    with pytest.raises(BidiModelTimeoutError):
        async for _ in loop.receive():
            pass

    agent.model.restart.assert_not_called()


@pytest.mark.asyncio
async def test_bidi_agent_loop_proactive_reconnect_before_deadline(loop, agent, agenerator):
    """A declared limit arms the timer, which emits a warning and reconnects proactively."""
    agent.model.connection_config = {"restart_after_s": 5}
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))

    # Drive timing without wall time: the first cycle's sleeps return immediately; the re-armed
    # cycle after the swap parks, so exactly one proactive reconnect fires.
    sleep_count = 0

    async def fake_sleep(_seconds):
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count > 2:
            await asyncio.Event().wait()
        await asyncio.sleep(0)

    loop._reconnect_timer._sleep = fake_sleep

    await loop.start()

    # The proactive timer emits the warning and scheduled restart on the bounded event stream.
    warning = await loop._event_queue.get()
    assert warning == BidiConnectionWarningEvent(time_left_s=5)

    restart = await loop._event_queue.get()
    assert restart == BidiConnectionRestartEvent(reason="scheduled", turn_interrupted=False)

    agent.model.restart.assert_called()

    await loop.stop()


@pytest.mark.asyncio
async def test_scheduled_restart_event_emitted_before_model_restart(loop, agent, agenerator):
    """The scheduled restart event precedes provider restart and new-connection output."""
    agent.model.connection_config = {}
    output = BidiTextInputEvent(text="new-connection output")
    agent.model.receive = unittest.mock.Mock(side_effect=[agenerator([]), agenerator([output])])
    order = []

    await loop.start()
    loop._reconnect_timer.cancel()

    original_put = loop._event_queue.put

    async def recording_put(event):
        if isinstance(event, BidiConnectionRestartEvent):
            order.append("event")
        await original_put(event)

    agent.model.restart.side_effect = lambda *_args, **_kwargs: order.append("restart")

    with unittest.mock.patch.object(loop._event_queue, "put", side_effect=recording_put):
        await loop._on_reconnect_deadline()

    assert order == ["event", "restart"]
    restart = await loop._event_queue.get()
    assert restart == BidiConnectionRestartEvent(reason="scheduled", turn_interrupted=False)
    assert await asyncio.wait_for(loop._event_queue.get(), timeout=2.0) is output

    await loop.stop()


@pytest.mark.asyncio
async def test_no_proactive_timer_when_restart_after_not_positive(loop, agent, agenerator):
    """A non-positive restart_after_s must not arm a zero-deadline hot reconnect loop."""
    agent.model.connection_config = {"restart_after_s": 0}
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))

    await loop.start()

    assert loop._reconnect_timer._task is None  # proactive disabled; reactive path remains

    await loop.stop()


@pytest.mark.asyncio
async def test_bidi_agent_loop_no_timer_without_declared_limit(loop, agent, agenerator):
    """A provider that declares no limit arms no proactive timer; reconnect stays reactive-only."""
    agent.model.connection_config = {}
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))

    await loop.start()

    assert loop._reconnect_timer._task is None

    await loop.stop()


@pytest.mark.asyncio
async def test_bidi_agent_loop_no_timer_when_auto_reconnect_disabled(loop, agent, agenerator):
    """auto_reconnect=False is the only opt-out: no proactive timer arms."""
    agent.model.connection_config = {"restart_after_s": 420, "auto_reconnect": False}
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))

    await loop.start()

    assert loop._reconnect_timer._task is None

    await loop.stop()


class _NonRestartableModel(BidiModel):
    """A provider without an optimized restart implementation."""

    def __init__(self):
        self.config = {}
        self.connection_config = {}
        self.started: list = []
        self.stopped = 0

    async def start(self, system_prompt=None, tools=None, messages=None, **kwargs):
        self.started.append(system_prompt)

    async def stop(self):
        self.stopped += 1

    def receive(self): ...

    async def send(self, content): ...


@pytest.mark.asyncio
async def test_restart_falls_back_to_stop_start_when_provider_is_not_restartable():
    """A non-restartable provider is restarted through stop() and start()."""
    model = _NonRestartableModel()
    agent = BidiAgent(model=model, system_prompt="hi")

    await agent._loop._restart_model({})

    assert model.stopped == 1
    assert model.started == ["hi"]  # start() called once with the agent's system prompt


class _StreamModel(BidiModel):
    """Reader blocks on a live 'stream' and raises when stop() closes it, like Nova/awscrt.

    The reader is terminated by the stream closing (an OSError), not by a force-cancel, so
    a reconnect must fence that error instead of forwarding it to the consumer.
    """

    def __init__(self):
        self.config = {}
        self.connection_config = {}
        self.restart_calls = 0
        self._closed = asyncio.Event()
        self._inbox: asyncio.Queue = asyncio.Queue()

    async def start(self, system_prompt=None, tools=None, messages=None, **kwargs):
        self._closed = asyncio.Event()
        self._inbox = asyncio.Queue()

    async def stop(self):
        self._closed.set()

    async def restart(self, system_prompt=None, tools=None, messages=None, **kwargs):
        self.restart_calls += 1
        await self.stop()
        await self.start(system_prompt, tools, messages, **kwargs)

    async def send(self, content):
        return None

    async def emit(self, event):
        await self._inbox.put(event)

    async def receive(self):
        closed, inbox = self._closed, self._inbox
        while True:
            getter = asyncio.ensure_future(inbox.get())
            waiter = asyncio.ensure_future(closed.wait())
            done, pending = await asyncio.wait({getter, waiter}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            if waiter in done:
                getter.cancel()
                raise OSError("stream closed")
            yield getter.result()


@pytest.mark.asyncio
async def test_reconnect_fences_superseded_reader_stream_close_error():
    """Reconnect closes the old stream (reader raises); that error must not leak to the consumer."""
    model = _StreamModel()
    agent = BidiAgent(model=model, system_prompt="hi")
    loop = agent._loop

    await loop.start()

    first = BidiTextInputEvent(text="first")
    await model.emit(first)
    assert await loop._event_queue.get() is first

    # Proactive-style restart closes the old stream, so the old
    # reader raises OSError. It is superseded, so that error must be dropped, not queued.
    await loop._restart_connection(None, loop._generation)
    assert model.restart_calls == 1

    second = BidiTextInputEvent(text="second")
    await model.emit(second)
    # The new connection's event arrives; a leaked OSError would have surfaced here instead.
    assert await loop._event_queue.get() is second

    await loop.stop()


@pytest.mark.asyncio
async def test_stale_reader_event_does_not_corrupt_state_across_reconnect():
    """A reader suspended in the full-queue put across a swap must not record its stale event.

    Guards the generation re-check after the queue put: without it, a cumulative-usage provider
    double-counts the old connection's running total onto the already-folded baseline.
    """
    model = _StreamModel()
    model.usage_is_cumulative = True  # like Nova: usage events report a running total
    agent = BidiAgent(model=model, system_prompt="hi")
    loop = agent._loop

    await loop.start()
    loop._reconnect_timer.cancel()

    await model.emit(BidiUsageEvent(input_tokens=60, output_tokens=40, total_tokens=100))
    await model.emit(BidiUsageEvent(input_tokens=90, output_tokens=60, total_tokens=150))
    for _ in range(30):
        await asyncio.sleep(0)
    # usage1 recorded; the reader is now suspended inside put(usage2) on the full queue.
    assert loop._accumulated_total_tokens == 100

    swap = asyncio.create_task(loop._restart_connection(None, loop._generation))
    for _ in range(30):
        await asyncio.sleep(0)
    await loop._event_queue.get()  # drain, unblocking the old reader's put(usage2)
    await swap
    for _ in range(30):
        await asyncio.sleep(0)

    # The stale usage2 must not be recorded onto the new connection (no cumulative double count).
    assert loop._accumulated_total_tokens == 100

    await loop.stop()


async def _feed_after_drain(loop, event):
    """Put ``event`` once the queue has drained (so a maxsize-1 put does not block)."""
    while loop._event_queue.qsize() > 0:
        await asyncio.sleep(0)
    await loop._event_queue.put(event)


@pytest.mark.asyncio
async def test_stale_reader_error_is_dropped_not_raised(loop, agent, agenerator):
    """A generic error from a superseded reader must be dropped, not surfaced into the new connection.

    Without the generation tag, a stale error re-raised by receive() kills the healthy, just-swapped
    session.
    """
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    await loop.start()
    loop._reconnect_timer.cancel()

    # An error raised on a superseded (older) generation.
    await loop._event_queue.put(_ReaderError(loop._generation - 1, OSError("stale connection error")))

    sentinel = BidiTextInputEvent(text="after stale error")
    feed = asyncio.create_task(_feed_after_drain(loop, sentinel))
    # receive() must drop the stale error and go on to the next event, not raise it.
    result = await asyncio.wait_for(loop.receive().__anext__(), timeout=2.0)
    assert result is sentinel
    await feed

    await loop.stop()


@pytest.mark.asyncio
async def test_current_reader_error_is_surfaced(loop, agent, agenerator):
    """A genuine error from the current reader must still surface to the consumer."""
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    await loop.start()
    loop._reconnect_timer.cancel()

    await loop._event_queue.put(_ReaderError(loop._generation, OSError("live connection error")))

    with pytest.raises(OSError, match="live connection error"):
        await asyncio.wait_for(loop.receive().__anext__(), timeout=2.0)

    await loop.stop()


@pytest.mark.asyncio
async def test_stale_reactive_timeout_dropped_after_proactive_swap(loop, agent, agenerator):
    """A timeout raised on an old generation, dequeued after a proactive swap, must not reconnect again."""
    agent.model.connection_config = {}
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    await loop.start()
    loop._reconnect_timer.cancel()

    stale_generation = loop._generation
    await loop._restart_connection(None, loop._generation)  # a proactive swap advances the generation
    restarts = agent.model.restart.call_count

    # A timeout tagged with the pre-swap generation is now stale; receive() must drop it.
    await loop._event_queue.put(_ReaderError(stale_generation, BidiModelTimeoutError("stale timeout")))

    sentinel = BidiTextInputEvent(text="after stale timeout")
    feed = asyncio.create_task(_feed_after_drain(loop, sentinel))
    result = await asyncio.wait_for(loop.receive().__anext__(), timeout=2.0)
    assert result is sentinel
    await feed
    assert agent.model.restart.call_count == restarts  # no second restart from the stale timeout

    await loop.stop()


@pytest.mark.asyncio
async def test_reactive_timeout_during_scheduled_restart_emits_no_duplicate(loop, agent, agenerator):
    """A timeout cannot emit another restart event after a scheduled restart is accepted."""
    agent.model.connection_config = {}
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    hook_started = asyncio.Event()
    release_hook = asyncio.Event()

    async def block_restart(_event):
        hook_started.set()
        await release_hook.wait()

    agent.hooks.add_callback(BidiBeforeConnectionRestartEvent, block_restart)

    await loop.start()
    loop._reconnect_timer.cancel()
    generation = loop._generation

    deadline = asyncio.create_task(loop._on_reconnect_deadline())
    await hook_started.wait()

    consumer = loop.receive()
    scheduled = await consumer.__anext__()
    assert scheduled == BidiConnectionRestartEvent(reason="scheduled")

    await loop._event_queue.put(_ReaderError(generation, BidiModelTimeoutError("duplicate timeout")))
    next_event = asyncio.create_task(consumer.__anext__())
    await asyncio.sleep(0)
    assert not next_event.done()

    sentinel = BidiTextInputEvent(text="after duplicate timeout")
    await loop._event_queue.put(sentinel)
    assert await asyncio.wait_for(next_event, timeout=2.0) is sentinel

    release_hook.set()
    await deadline
    agent.model.restart.assert_called_once()

    await loop.stop()


@pytest.mark.asyncio
async def test_connection_events_share_bounded_event_queue(loop, agent, agenerator):
    """Connection events preserve FIFO order and backpressure on the size-one event queue."""
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    await loop.start()
    loop._reconnect_timer.cancel()

    data = BidiTextInputEvent(text="new-connection output")
    await loop._event_queue.put(data)
    warning_put = asyncio.create_task(loop._on_reconnect_warning(10))
    await asyncio.sleep(0)
    assert not warning_put.done()

    first = await asyncio.wait_for(loop.receive().__anext__(), timeout=2.0)
    assert first is data
    await warning_put

    second = await asyncio.wait_for(loop.receive().__anext__(), timeout=2.0)
    assert second == BidiConnectionWarningEvent(time_left_s=10)
    assert loop._event_queue.maxsize == 1

    await loop.stop()


@pytest.mark.asyncio
async def test_connection_event_delivered_while_consumer_idle(loop, agent, agenerator):
    """A connection event emitted while the queue is empty wakes receive()."""
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    await loop.start()
    loop._reconnect_timer.cancel()

    consumer = loop.receive()

    async def emit():
        await asyncio.sleep(0)
        await loop._on_reconnect_warning(10)

    asyncio.create_task(emit())
    first = await asyncio.wait_for(consumer.__anext__(), timeout=2.0)
    assert first == BidiConnectionWarningEvent(time_left_s=10)

    await loop.stop()


@pytest.mark.asyncio
async def test_tool_result_completed_during_reconnect_is_recovered(agenerator):
    """A tool released during reconnect is recovered only on the replacement connection."""
    order = []
    release_tool = asyncio.Event()

    @tool
    async def slow_tool():
        await release_tool.wait()
        return "result"

    model = unittest.mock.AsyncMock(spec=BidiModel)
    model.restart = unittest.mock.AsyncMock(side_effect=lambda *a, **k: order.append("restart"))
    model.connection_config = {}
    model.send.side_effect = lambda event: order.append("recover" if isinstance(event, BidiTextInputEvent) else "send")
    model.receive = unittest.mock.Mock(return_value=agenerator([]))

    agent = BidiAgent(model=model, tools=[slow_tool], system_prompt="hi")
    loop = agent._loop
    await loop.start()
    loop._reconnect_timer.cancel()

    async def drain():
        while True:
            await loop._event_queue.get()

    drain_task = asyncio.create_task(drain())
    tool_use = {"toolUseId": "t1", "name": "slow_tool", "input": {}}
    tool_use_key = await loop._register_tool_use(tool_use, loop._generation)
    assert tool_use_key is not None
    tool_task = asyncio.create_task(loop._run_tool(tool_use, tool_use_key))
    for _ in range(10):
        await asyncio.sleep(0)

    async def before_restart_hook(event):
        # Release the tool mid-reconnect: the gate is closed but the generation not yet bumped.
        release_tool.set()
        for _ in range(50):
            await asyncio.sleep(0)

    agent.hooks.add_callback(BidiBeforeConnectionRestartEvent, before_restart_hook)

    await loop._restart_connection(None, loop._generation)
    await asyncio.wait_for(tool_task, timeout=2)
    drain_task.cancel()

    assert "send" not in order
    assert order.count("recover") == 1
    assert order.index("restart") < order.index("recover")


@pytest.mark.asyncio
async def test_tool_completion_during_stop_cleans_continuity_state(agenerator):
    """Stopping while a completed tool is sending cancels delivery and removes its entry."""
    release_tool = asyncio.Event()
    send_started = asyncio.Event()

    @tool
    async def slow_tool():
        await release_tool.wait()
        return "result"

    model = unittest.mock.AsyncMock(spec=BidiModel)
    model.connection_config = {}
    model.receive = unittest.mock.Mock(return_value=agenerator([]))

    async def stalled_send(_event):
        send_started.set()
        await asyncio.Event().wait()

    model.send.side_effect = stalled_send
    agent = BidiAgent(model=model, tools=[slow_tool])
    loop = agent._loop
    await loop.start()
    loop._reconnect_timer.cancel()

    async def drain():
        while True:
            await loop._event_queue.get()

    drain_task = asyncio.create_task(drain())
    tool_use: ToolUse = {"toolUseId": "t1", "name": "slow_tool", "input": {}}
    tool_use_key = await loop._register_tool_use(tool_use, loop._generation)
    assert tool_use_key is not None
    tool_task = loop._task_pool.create(loop._run_tool(tool_use, tool_use_key))

    release_tool.set()
    await asyncio.wait_for(send_started.wait(), timeout=0.5)
    await asyncio.wait_for(loop.stop(), timeout=0.5)

    drain_task.cancel()
    await asyncio.gather(drain_task, return_exceptions=True)
    assert tool_task.done()
    assert loop._running_tools == {}


@pytest.mark.asyncio
async def test_tool_result_rechecks_send_gate_after_connection_lock(loop, agent, agenerator):
    """Result delivery does not hold the connection lock while a closed send gate blocks it."""
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    await loop.start()
    loop._reconnect_timer.cancel()

    tool_use: ToolUse = {"toolUseId": "t1", "name": "time_tool", "input": {}}
    tool_use_key = await loop._register_tool_use(tool_use, loop._generation)
    assert tool_use_key is not None
    tool_result: ToolResult = {"toolUseId": "t1", "status": "success", "content": [{"text": "12:00"}]}

    await loop._connection_lock.acquire()
    delivery_task = asyncio.create_task(loop._deliver_tool_result(tool_use_key, ToolResultEvent(tool_result)))
    try:
        await asyncio.sleep(0)
        loop._send_gate.clear()
        loop._connection_lock.release()
        await asyncio.sleep(0)

        await asyncio.wait_for(loop._connection_lock.acquire(), timeout=0.5)
        loop._connection_lock.release()

        loop._send_gate.set()
        await asyncio.wait_for(delivery_task, timeout=2)
        agent.model.send.assert_awaited_once()
    finally:
        loop._send_gate.set()
        await asyncio.gather(delivery_task, return_exceptions=True)
        await loop.stop()


@pytest.mark.asyncio
async def test_send_rechecks_gate_after_connection_lock(loop, agent, agenerator):
    """Ordinary input waits for the replacement connection when the gate closes before lock acquisition."""
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    await loop.start()
    loop._reconnect_timer.cancel()

    await loop._connection_lock.acquire()
    send_task = asyncio.create_task(loop.send(BidiTextInputEvent(text="hello")))
    try:
        await asyncio.sleep(0)
        loop._send_gate.clear()
        loop._connection_lock.release()
        await asyncio.sleep(0)

        agent.model.send.assert_not_awaited()

        loop._send_gate.set()
        await asyncio.wait_for(send_task, timeout=2)
        agent.model.send.assert_awaited_once()
        assert len(agent.messages) == 1
    finally:
        loop._send_gate.set()
        await asyncio.gather(send_task, return_exceptions=True)
        await loop.stop()


@pytest.mark.asyncio
async def test_send_timeout_does_not_block_reconnect(loop, agent, agenerator, monkeypatch):
    """A stalled ordinary send releases the connection lock so reconnect can proceed."""
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    await loop.start()
    loop._reconnect_timer.cancel()
    monkeypatch.setattr("strands.experimental.bidi.agent.loop._MODEL_SEND_TIMEOUT_S", 0.01)

    send_started = asyncio.Event()

    async def stalled_send(_event):
        send_started.set()
        await asyncio.Event().wait()

    agent.model.send.side_effect = stalled_send
    send_task = asyncio.create_task(loop.send(BidiTextInputEvent(text="hello")))
    await send_started.wait()
    restart_task = asyncio.create_task(loop._restart_connection(None, loop._generation))

    with pytest.raises(TimeoutError):
        await send_task
    assert await asyncio.wait_for(restart_task, timeout=1) is True

    await loop.stop()


@pytest.mark.asyncio
async def test_send_waiting_for_reconnect_exits_when_loop_stops(loop, agent, agenerator):
    """Stopping the loop wakes an input blocked behind the reconnect gate."""
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    await loop.start()
    loop._reconnect_timer.cancel()
    loop._send_gate.clear()

    send_task = asyncio.create_task(loop.send(BidiTextInputEvent(text="hello")))
    await asyncio.sleep(0)
    await loop.stop()

    with pytest.raises(RuntimeError, match="loop stopped before event could be sent"):
        await asyncio.wait_for(send_task, timeout=0.5)


@pytest.mark.asyncio
async def test_tool_result_send_timeout_does_not_block_reconnect(loop, agent, agenerator, monkeypatch, caplog):
    """A stalled tool-result send releases reconnect and retries on the replacement connection."""
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    await loop.start()
    loop._reconnect_timer.cancel()
    monkeypatch.setattr("strands.experimental.bidi.agent.loop._MODEL_SEND_TIMEOUT_S", 0.01)

    tool_use: ToolUse = {"toolUseId": "t1", "name": "time_tool", "input": {}}
    tool_use_key = await loop._register_tool_use(tool_use, loop._generation)
    assert tool_use_key is not None
    tool_result: ToolResult = {"toolUseId": "t1", "status": "success", "content": [{"text": "12:00"}]}

    send_started = asyncio.Event()
    retry_complete = asyncio.Event()
    tru_sent_events = []

    async def send(event):
        tru_sent_events.append(event)
        if len(tru_sent_events) == 1:
            send_started.set()
            await asyncio.Event().wait()
        retry_complete.set()

    agent.model.send.side_effect = send
    with caplog.at_level(logging.WARNING, logger="strands.experimental.bidi.agent.loop"):
        delivery_task = asyncio.create_task(loop._deliver_tool_result(tool_use_key, ToolResultEvent(tool_result)))
        await send_started.wait()
        restart_task = asyncio.create_task(loop._restart_connection(None, loop._generation))
        tru_retained, tru_restarted = await asyncio.wait_for(asyncio.gather(delivery_task, restart_task), timeout=1)
        await asyncio.wait_for(retry_complete.wait(), timeout=0.5)

    assert "mode=<native>, timeout_s=<0.01> | tool result delivery timed out" in caplog.text
    assert tru_retained is True
    assert tru_restarted is True
    assert len(tru_sent_events) == 2
    assert isinstance(tru_sent_events[0], ToolResultEvent)
    assert isinstance(tru_sent_events[1], BidiTextInputEvent)
    assert set(loop._running_tools) == {tool_use_key}

    await loop._bind_recovery_response(loop._generation, "recovery")
    await loop._clear_recovered_tool_result("recovery")
    assert loop._running_tools == {}

    await loop.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["success", "error"])
async def test_semantic_recovery_matches_late_exact_reissue(loop, agent, agenerator, status):
    """An exact reissue before the recovery response completes reuses the retained result."""
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    await loop.start()
    loop._reconnect_timer.cancel()

    original: ToolUse = {"toolUseId": "old", "name": "time_tool", "input": {}}
    original_key = await loop._register_tool_use(original, loop._generation)
    assert original_key is not None
    result = ToolResultEvent(ToolResult(toolUseId="old", status=status, content=[{"text": "12:00"}]))

    loop._generation += 1
    assert await loop._deliver_tool_result(original_key, result) is True
    assert set(loop._running_tools) == {original_key}

    reissue: ToolUse = {"toolUseId": "new", "name": "time_tool", "input": {}}
    assert await loop._register_tool_use(reissue, loop._generation) is None

    for _ in range(10):
        await asyncio.sleep(0)
        if original_key not in loop._running_tools:
            break

    tru_sent_events = [call.args[0] for call in agent.model.send.await_args_list]
    assert len(tru_sent_events) == 2
    assert isinstance(tru_sent_events[0], BidiTextInputEvent)
    assert f'"status": "{status}"' in tru_sent_events[0].text
    assert isinstance(tru_sent_events[1], ToolResultEvent)
    assert tru_sent_events[1].tool_result["toolUseId"] == "new"
    assert tru_sent_events[1].tool_result["status"] == status
    assert loop._running_tools == {}
    assert list(loop._pending_recovery_responses) == []
    assert list(loop._bound_recovery_responses) == []

    await loop.stop()


@pytest.mark.asyncio
async def test_response_complete_expires_unclaimed_semantic_recovery(loop, agent, agenerator):
    """A recovered result expires when its prompted response completes without a reissue."""
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    await loop.start()
    loop._reconnect_timer.cancel()

    original: ToolUse = {"toolUseId": "old", "name": "time_tool", "input": {}}
    original_key = await loop._register_tool_use(original, loop._generation)
    assert original_key is not None
    result = ToolResultEvent(ToolResult(toolUseId="old", status="success", content=[{"text": "12:00"}]))

    loop._generation += 1
    assert await loop._deliver_tool_result(original_key, result) is True
    assert set(loop._running_tools) == {original_key}

    response = [
        BidiResponseStartEvent(response_id="recovery"),
        BidiResponseCompleteEvent(response_id="recovery", stop_reason="complete"),
    ]
    agent.model.receive = unittest.mock.Mock(return_value=agenerator(response))

    async def drain():
        while True:
            await loop._event_queue.get()

    drain_task = asyncio.create_task(drain())
    try:
        await loop._run_model(loop._generation)
    finally:
        drain_task.cancel()

    assert loop._running_tools == {}
    later: ToolUse = {"toolUseId": "later", "name": "time_tool", "input": {}}
    assert await loop._register_tool_use(later, loop._generation) == (loop._generation, "later")

    await loop.stop()


@pytest.mark.asyncio
async def test_unrelated_response_complete_does_not_expire_semantic_recovery(loop, agent, agenerator):
    """A response already in flight cannot consume an unbound recovery result."""
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    await loop.start()
    loop._reconnect_timer.cancel()

    original: ToolUse = {"toolUseId": "old", "name": "time_tool", "input": {}}
    original_key = await loop._register_tool_use(original, loop._generation)
    assert original_key is not None
    result = ToolResultEvent(ToolResult(toolUseId="old", status="success", content=[{"text": "12:00"}]))

    loop._generation += 1
    assert await loop._deliver_tool_result(original_key, result) is True

    await loop._clear_recovered_tool_result("unrelated")

    assert set(loop._running_tools) == {original_key}
    assert list(loop._pending_recovery_responses) == [original_key]
    assert list(loop._bound_recovery_responses) == []

    await loop.stop()


@pytest.mark.asyncio
async def test_recovered_results_expire_with_their_own_responses(loop, agent, agenerator):
    """One swap redelivers multiple results and retains each until its response completes."""
    agent.model.receive = unittest.mock.Mock(side_effect=lambda: agenerator([]))
    await loop.start()
    loop._reconnect_timer.cancel()

    first: ToolUse = {"toolUseId": "first", "name": "first_tool", "input": {}}
    second: ToolUse = {"toolUseId": "second", "name": "second_tool", "input": {}}
    first_key = await loop._register_tool_use(first, loop._generation)
    second_key = await loop._register_tool_use(second, loop._generation)
    assert first_key is not None
    assert second_key is not None

    first_result = ToolResultEvent(ToolResult(toolUseId="first", status="success", content=[{"text": "first result"}]))
    second_result = ToolResultEvent(
        ToolResult(toolUseId="second", status="success", content=[{"text": "second result"}])
    )
    async with loop._tool_lock:
        loop._running_tools[first_key].result_event = first_result
        loop._running_tools[second_key].result_event = second_result

    assert await loop._restart_connection(None, loop._generation) is True
    for _ in range(20):
        await asyncio.sleep(0)
        if agent.model.send.await_count == 2:
            break

    tru_sent_events = [call.args[0] for call in agent.model.send.await_args_list]
    assert len(tru_sent_events) == 2
    assert all(isinstance(event, BidiTextInputEvent) for event in tru_sent_events)
    assert set(loop._running_tools) == {first_key, second_key}

    await loop._bind_recovery_response(loop._generation, "shared-response")
    await loop._bind_recovery_response(loop._generation, "shared-response")
    await loop._clear_recovered_tool_result("shared-response")

    assert first_key not in loop._running_tools
    assert second_key in loop._running_tools

    await loop._clear_recovered_tool_result("shared-response")
    assert loop._running_tools == {}

    await loop.stop()


@pytest.mark.asyncio
async def test_tool_result_recovery_timeout_is_delivered_to_next_exact_reissue(
    loop, agent, agenerator, monkeypatch, caplog
):
    """A timed-out recovery result remains available to the next exact provider reissue."""
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    await loop.start()
    loop._reconnect_timer.cancel()
    monkeypatch.setattr("strands.experimental.bidi.agent.loop._MODEL_SEND_TIMEOUT_S", 0.01)

    tool_use: ToolUse = {"toolUseId": "old", "name": "time_tool", "input": {}}
    tool_use_key = await loop._register_tool_use(tool_use, loop._generation)
    assert tool_use_key is not None
    loop._generation += 1

    tru_sent_events = []
    redelivery_complete = asyncio.Event()

    async def send(event):
        tru_sent_events.append(event)
        if len(tru_sent_events) == 1:
            await asyncio.Event().wait()
        redelivery_complete.set()

    agent.model.send.side_effect = send

    async def drain():
        while True:
            await loop._event_queue.get()

    drain_task = asyncio.create_task(drain())
    try:
        with caplog.at_level(logging.WARNING, logger="strands.experimental.bidi.agent.loop"):
            await loop._run_tool(tool_use, tool_use_key)

        assert "mode=<recovery>, timeout_s=<0.01> | tool result delivery timed out" in caplog.text
        assert tool_use_key in loop._running_tools

        loop._generation += 1
        reissue: ToolUse = {"toolUseId": "new", "name": "time_tool", "input": {}}
        assert await loop._register_tool_use(reissue, loop._generation) is None
        await asyncio.wait_for(redelivery_complete.wait(), timeout=0.5)
    finally:
        drain_task.cancel()

    assert len(tru_sent_events) == 2
    assert isinstance(tru_sent_events[0], BidiTextInputEvent)
    assert isinstance(tru_sent_events[1], ToolResultEvent)
    assert tru_sent_events[1].tool_result["toolUseId"] == "new"
    assert loop._running_tools == {}

    await loop.stop()


@pytest.mark.asyncio
async def test_completed_reissue_delivery_does_not_block_model_reader(loop, agent, agenerator):
    """A retained-result send runs independently of subsequent model output."""
    reissue: ToolUse = {"toolUseId": "new", "name": "time_tool", "input": {}}
    reissue_event = ToolUseStreamEvent(current_tool_use=reissue, delta="")
    audio_event = BidiAudioStreamEvent(audio="dGVzdA==", format="pcm", sample_rate=24000, channels=1)

    async def replacement_events():
        yield reissue_event
        yield audio_event

    agent.model.receive = unittest.mock.Mock(side_effect=[agenerator([]), replacement_events()])
    await loop.start()
    loop._reconnect_timer.cancel()

    original: ToolUse = {"toolUseId": "old", "name": "time_tool", "input": {}}
    original_key = await loop._register_tool_use(original, loop._generation)
    assert original_key is not None
    result = ToolResultEvent(ToolResult(toolUseId="old", status="success", content=[{"text": "12:00"}]))
    async with loop._tool_lock:
        loop._running_tools[original_key].result_event = result

    send_started = asyncio.Event()
    release_send = asyncio.Event()

    async def delayed_send(_event):
        send_started.set()
        await release_send.wait()

    agent.model.send.side_effect = delayed_send

    try:
        await loop._restart_connection(None, loop._generation)
        await asyncio.wait_for(send_started.wait(), timeout=0.5)
        tru_event = await asyncio.wait_for(loop.receive().__anext__(), timeout=0.5)
        assert tru_event is audio_event
    finally:
        release_send.set()
        await loop.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("reconnected", [False, True])
async def test_background_tool_result_delivery_error_is_retained(loop, agent, agenerator, caplog, reconnected):
    """A retained-result send failure remains recoverable without ending the session."""
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    await loop.start()
    loop._reconnect_timer.cancel()

    tool_use: ToolUse = {"toolUseId": "old", "name": "time_tool", "input": {}}
    tool_use_key = await loop._register_tool_use(tool_use, loop._generation)
    assert tool_use_key is not None
    result = ToolResultEvent(ToolResult(toolUseId="old", status="success", content=[{"text": "12:00"}]))
    if reconnected:
        loop._generation += 1
    agent.model.send.side_effect = OSError("delivery failed")

    with caplog.at_level(logging.WARNING, logger="strands.experimental.bidi.agent.loop"):
        await loop._run_tool_result_delivery(tool_use_key, result)

    assert tool_use_key in loop._running_tools
    assert loop._running_tools[tool_use_key].reissue_after_generation == loop._generation
    assert loop._started is True
    assert loop._event_queue.empty()
    exp_mode = "recovery" if reconnected else "native"
    assert f"mode=<{exp_mode}>, error=<delivery failed> | tool result delivery failed" in caplog.text

    await loop.stop()


@pytest.mark.asyncio
async def test_prepare_tool_result_recovery_prunes_only_unclaimed_results_after_recovery_window(
    loop, agent, agenerator
):
    """Expired completed results are removed without dropping active or claimed calls."""
    agent.model.receive = unittest.mock.Mock(side_effect=lambda: agenerator([]))
    await loop.start()
    loop._reconnect_timer.cancel()

    completed: ToolUse = {"toolUseId": "completed", "name": "time_tool", "input": {}}
    active: ToolUse = {"toolUseId": "active", "name": "time_tool", "input": {"active": True}}
    claimed: ToolUse = {"toolUseId": "claimed", "name": "time_tool", "input": {"claimed": True}}
    completed_key = await loop._register_tool_use(completed, loop._generation)
    active_key = await loop._register_tool_use(active, loop._generation)
    claimed_key = await loop._register_tool_use(claimed, loop._generation)
    assert completed_key is not None
    assert active_key is not None
    assert claimed_key is not None

    result = ToolResultEvent(ToolResult(toolUseId="completed", status="success", content=[{"text": "12:00"}]))
    async with loop._tool_lock:
        loop._running_tools[completed_key].result_event = result
        loop._running_tools[claimed_key].result_event = result

    loop._generation += 1
    tru_retained_results = await loop._prepare_tool_result_recovery()
    assert {tool_use_key for tool_use_key, _ in tru_retained_results} == {completed_key, claimed_key}
    assert set(loop._running_tools) == {completed_key, active_key, claimed_key}

    async with loop._tool_lock:
        loop._running_tools[claimed_key].replacement_key = (loop._generation, "claimed-new")

    loop._generation += 1
    assert await loop._prepare_tool_result_recovery() == []
    assert set(loop._running_tools) == {active_key, claimed_key}

    await loop.stop()


@pytest.mark.asyncio
async def test_stale_reactive_restart_ignored_after_proactive_swap(agent, agenerator):
    """A stale timeout restart (raised for an old generation) must not tear down the new connection."""
    agent.model.connection_config = {}
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))

    loop = agent._loop
    await loop.start()
    loop._reconnect_timer.cancel()

    stale_generation = loop._generation
    await loop._restart_connection(None, loop._generation)  # a proactive swap advances the generation
    assert loop._generation == stale_generation + 1
    restarts = agent.model.restart.call_count

    await loop._restart_connection(BidiModelTimeoutError("stale"), stale_generation)
    assert agent.model.restart.call_count == restarts  # stale trigger ignored

    await loop.stop()


@pytest.mark.asyncio
async def test_deadline_callback_does_not_reconnect_after_stop(agent, agenerator):
    """A proactive deadline callback in flight during stop() must not reconnect the model."""
    agent.model.connection_config = {"restart_after_s": 415}
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))

    loop = agent._loop
    await loop.start()
    loop._reconnect_timer.cancel()

    loop._response_active = True  # mid-turn: the callback waits for the boundary
    loop._update_turn_state()

    deadline_task = asyncio.create_task(loop._on_reconnect_deadline())
    for _ in range(10):
        await asyncio.sleep(0)

    await loop.stop()  # stop() releases the boundary wait; the callback no-ops on _started
    await asyncio.wait_for(deadline_task, timeout=2)

    agent.model.restart.assert_not_called()
    assert loop._event_queue.empty()


@pytest.mark.asyncio
async def test_deadline_callback_does_not_restart_after_stop_while_queue_full(agent, agenerator):
    """A restart blocked on event backpressure must not restart the model after stop()."""
    agent.model.connection_config = {}
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))

    loop = agent._loop
    await loop.start()
    loop._reconnect_timer.cancel()

    queued = BidiTextInputEvent(text="queued")
    await loop._event_queue.put(queued)
    deadline_task = asyncio.create_task(loop._on_reconnect_deadline())
    for _ in range(10):
        await asyncio.sleep(0)
        if loop._reconnecting:
            break
    assert loop._reconnecting

    await loop.stop()
    assert loop._event_queue.get_nowait() is queued
    await asyncio.wait_for(deadline_task, timeout=2)

    agent.model.restart.assert_not_called()
    assert loop._event_queue.get_nowait() == BidiConnectionRestartEvent(reason="scheduled", turn_interrupted=False)


@pytest.mark.asyncio
async def test_send_user_text_marks_turn_awaiting_response(loop, agent, agenerator):
    """A user text turn owes a reply, so it holds the turn boundary like a finished audio turn."""
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    await loop.start()

    await loop.send(BidiTextInputEvent(text="hello", role="user"))
    assert loop._awaiting_response is True
    assert not loop._turn_complete.is_set()  # a proactive reconnect would now wait

    await loop.stop()


@pytest.mark.asyncio
async def test_send_assistant_text_does_not_mark_awaiting_response(loop, agent, agenerator):
    """Injected assistant context is not an owed user turn and must not hold the boundary."""
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    await loop.start()

    await loop.send(BidiTextInputEvent(text="injected context", role="assistant"))
    assert loop._awaiting_response is False
    assert loop._turn_complete.is_set()

    await loop.stop()


@pytest.mark.asyncio
async def test_user_transcript_marks_turn_awaiting_response_before_final(loop, agent, agenerator):
    """A non-final user transcript owes a reply, so a proactive reconnect holds even when the
    provider never flags is_final on user speech (the Gemini case). History is not appended yet."""
    partial = BidiTranscriptStreamEvent(
        delta={"text": "what's the"}, text="what's the", role="user", is_final=False, current_transcript="what's the"
    )
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([partial]))

    await loop.start()
    for _ in range(10):
        await asyncio.sleep(0)

    assert loop._awaiting_response is True
    assert not loop._turn_complete.is_set()  # a proactive reconnect would now wait for the reply
    assert agent.messages == []  # non-final transcript is not committed to history

    await loop.stop()


@pytest.mark.asyncio
async def test_assistant_transcript_does_not_mark_awaiting_response(loop, agent, agenerator):
    """A model (assistant) transcript is output, not an owed user turn, so it must not hold."""
    partial = BidiTranscriptStreamEvent(
        delta={"text": "hi there"}, text="hi there", role="assistant", is_final=False, current_transcript="hi there"
    )
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([partial]))

    await loop.start()
    for _ in range(10):
        await asyncio.sleep(0)

    assert loop._awaiting_response is False

    await loop.stop()


@pytest.mark.asyncio
async def test_response_complete_clears_awaiting_response(loop, agent, agenerator):
    """A completed reply clears the awaited-response latch, so a user transcript that lagged into
    the reply does not leave the turn falsely open (which would burn the alignment wait and flag a
    spurious turn_interrupted)."""
    events = [
        BidiResponseStartEvent(response_id="r1"),
        # A lagging user input transcript arrives during the reply and re-latches awaiting.
        BidiTranscriptStreamEvent(
            delta={"text": "earlier question"},
            text="earlier question",
            role="user",
            is_final=False,
            current_transcript="earlier question",
        ),
        BidiResponseCompleteEvent(response_id="r1", stop_reason="complete"),
    ]
    agent.model.receive = unittest.mock.Mock(return_value=agenerator(events))

    await loop.start()
    # Drain through the reply so all three events are applied (the event queue has maxsize=1, so a
    # reader with no consumer would stall after the first event).
    async for event in loop.receive():
        if isinstance(event, BidiResponseCompleteEvent):
            break
    # Let _run_model apply the post-dequeue turn-state update for the complete event.
    for _ in range(10):
        await asyncio.sleep(0)

    assert loop._awaiting_response is False
    assert loop._turn_complete.is_set()  # turn is idle, so a proactive reconnect fires immediately

    await loop.stop()


@pytest.mark.asyncio
async def test_forced_swap_flags_interrupted_turn(agent, agenerator):
    """A swap forced while a turn is owed sets turn_interrupted so the app can re-prompt."""
    agent.model.connection_config = {"restart_after_s": 415}
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    loop = agent._loop
    await loop.start()
    loop._reconnect_timer.cancel()

    loop._response_active = True  # a turn is in progress and will not complete in time
    loop._update_turn_state()

    # Force the turn-alignment wait to time out immediately (no wall-clock wait).
    with unittest.mock.patch("strands.experimental.bidi.agent.loop._MODEL_RESTART_TURN_TIMEOUT_S", 0):
        await loop._on_reconnect_deadline()

    restart = await loop._event_queue.get()
    assert restart == BidiConnectionRestartEvent(reason="scheduled", turn_interrupted=True)

    await loop.stop()


@pytest.mark.asyncio
async def test_proactive_reconnect_waits_for_turn_boundary(loop, agent, agenerator):
    """A proactive reconnect defers until the in-progress turn completes (turn alignment)."""
    # The real timer is cancelled so the deadline is driven manually; the turn state is set
    # directly, and _await_turn_boundary waits up to _MODEL_RESTART_TURN_TIMEOUT_S for the boundary.
    agent.model.connection_config = {"restart_after_s": 60}
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    await loop.start()
    loop._reconnect_timer.cancel()

    # Mid-response: not at a turn boundary.
    loop._response_active = True
    loop._update_turn_state()

    deadline = asyncio.create_task(loop._on_reconnect_deadline())
    for _ in range(10):
        await asyncio.sleep(0)
    assert not agent.model.restart.called  # held: the turn has not finished

    # Turn completes -> boundary reached -> reconnect proceeds.
    loop._response_active = False
    loop._update_turn_state()
    await deadline
    assert agent.model.restart.called

    await loop.stop()


@pytest.mark.asyncio
async def test_bidi_agent_loop_restart_hook_reports_reason(loop, agent, agenerator):
    """The reactive path reports reason='timeout' with the error; proactive reports 'scheduled' with None."""
    from strands.experimental.hooks.events import BidiBeforeConnectionRestartEvent

    before_events = []
    agent.hooks.add_callback(
        BidiBeforeConnectionRestartEvent, lambda event: before_events.append((event.reason, event.timeout_error))
    )
    agent.model.connection_config = {}
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))

    await loop.start()

    timeout_error = BidiModelTimeoutError("boom")
    await loop._restart_connection(timeout_error, loop._generation)
    await loop._restart_connection(None, loop._generation)

    assert before_events[0] == ("timeout", timeout_error)
    assert before_events[1] == ("scheduled", None)

    await loop.stop()


@pytest.mark.asyncio
async def test_bidi_agent_loop_reconnect_is_reentrancy_guarded(loop, agent, agenerator):
    """A second trigger arriving while a reconnect is in flight is a no-op, not a racing duplicate."""
    agent.model.connection_config = {}
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))

    # Block the restart so the first call holds the guard while the second is attempted.
    release = asyncio.Event()
    restart_calls = 0

    async def blocking_restart(*_args, **_kwargs):
        nonlocal restart_calls
        restart_calls += 1
        await release.wait()

    agent.model.restart = blocking_restart

    await loop.start()

    first = asyncio.create_task(loop._restart_connection(None, loop._generation))
    for _ in range(10):
        await asyncio.sleep(0)
        if restart_calls == 1:
            break

    # First restart is now suspended mid-flight, still holding the guard.
    await loop._restart_connection(None, loop._generation)
    assert restart_calls == 1

    release.set()
    await first
    assert restart_calls == 1

    await loop.stop()


@pytest.mark.asyncio
async def test_bidi_agent_loop_proactive_reconnect_completes_when_reconnect_suspends(loop, agent, agenerator):
    """The proactive reconnect runs on the timer's task, so it must not cancel itself mid-flight.

    Guards against the timer cancelling the very task running its deadline callback: with a
    reconnect that actually suspends, a self-cancel would abort the swap and leave the gate closed.
    """
    agent.model.connection_config = {"restart_after_s": 5}
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))

    restart_done = False

    async def suspending_restart(*_args, **_kwargs):
        nonlocal restart_done
        await asyncio.sleep(0)  # genuine suspension after the timer fires its deadline
        restart_done = True

    agent.model.restart = suspending_restart

    # Drive timing without wall time: the first cycle fires immediately, the re-armed cycle parks.
    sleep_count = 0

    async def fake_sleep(_seconds):
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count > 2:
            await asyncio.Event().wait()
        await asyncio.sleep(0)

    loop._reconnect_timer._sleep = fake_sleep

    await loop.start()

    # Drain notification events like a real consumer, so the proactive path is not blocked
    # enqueuing the warning/restart events on the size-1 queue before it reconnects.
    for _ in range(50):
        await asyncio.sleep(0)
        while not loop._event_queue.empty():
            loop._event_queue.get_nowait()
        if restart_done:
            break

    assert restart_done
    assert loop._send_gate.is_set()

    await loop.stop()


@pytest.mark.asyncio
async def test_bidi_agent_loop_cumulative_usage_not_double_counted(loop, agent, agenerator):
    """Cumulative providers replace running counts rather than summing successive totals."""
    from strands.experimental.bidi.types.events import BidiUsageEvent

    agent.model.usage_is_cumulative = True
    events = [
        BidiUsageEvent(input_tokens=100, output_tokens=50, total_tokens=150),
        BidiUsageEvent(input_tokens=250, output_tokens=120, total_tokens=370),
    ]
    agent.model.receive = unittest.mock.Mock(return_value=agenerator(events))

    await loop.start()

    received = []
    async for event in loop.receive():
        received.append(event)
        if len(received) >= 2:
            break

    # Latest cumulative total wins (370), not the sum of the two events (520).
    assert loop._accumulated_input_tokens == 250
    assert loop._accumulated_output_tokens == 120
    assert loop._accumulated_total_tokens == 370

    await loop.stop()


@pytest.mark.asyncio
async def test_bidi_agent_loop_receive_tool_use(loop, agent, agenerator):
    tool_use = {"toolUseId": "t1", "name": "time_tool", "input": {}}
    tool_result = {"toolUseId": "t1", "status": "success", "content": [{"text": "12:00"}]}

    tool_use_event = ToolUseStreamEvent(current_tool_use=tool_use, delta="")
    tool_result_event = ToolResultEvent(tool_result)

    agent.model.receive = unittest.mock.Mock(return_value=agenerator([tool_use_event]))

    await loop.start()

    tru_events = []
    async for event in loop.receive():
        tru_events.append(event)
        if len(tru_events) >= 3:
            break

    exp_events = [
        tool_use_event,
        tool_result_event,
        # The message is assigned a durable tracking_id when appended to history.
        ToolResultMessageEvent(
            {"role": "user", "content": [{"toolResult": tool_result}], "tracking_id": unittest.mock.ANY}
        ),
    ]
    assert tru_events == exp_events

    tru_messages = agent.messages
    exp_messages = [
        {"role": "assistant", "content": [{"toolUse": tool_use}], "tracking_id": unittest.mock.ANY},
        {"role": "user", "content": [{"toolResult": tool_result}], "tracking_id": unittest.mock.ANY},
    ]
    assert tru_messages == exp_messages

    agent.model.send.assert_called_with(tool_result_event)


@pytest.mark.asyncio
async def test_bidi_agent_loop_does_not_execute_tool_after_queue_wait_crosses_reconnect(loop, agent, agenerator):
    """A tool registered before a blocked queue put is dropped if its connection is superseded."""
    tool_use: ToolUse = {"toolUseId": "t1", "name": "time_tool", "input": {}}
    tool_use_event = ToolUseStreamEvent(current_tool_use=tool_use, delta="")
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([tool_use_event]))
    loop._run_tool = unittest.mock.AsyncMock()

    await loop.start()
    loop._reconnect_timer.cancel()
    loop._event_queue.put_nowait(BidiTextInputEvent(text="occupied"))

    for _ in range(10):
        await asyncio.sleep(0)
        if loop._running_tools:
            break
    assert loop._running_tools

    loop._generation += 1
    loop._event_queue.get_nowait()
    assert loop._model_task is not None
    await asyncio.wait_for(loop._model_task, timeout=1)

    loop._run_tool.assert_not_awaited()
    assert loop._running_tools == {}

    await loop.stop()


@pytest.mark.asyncio
async def test_bidi_agent_loop_cancellation_cleans_tool_registered_before_queue_put(loop, agent, agenerator):
    """Cancelling a reader blocked on queue backpressure removes its unscheduled tool call."""
    tool_use: ToolUse = {"toolUseId": "t1", "name": "time_tool", "input": {}}
    tool_use_event = ToolUseStreamEvent(current_tool_use=tool_use, delta="")
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([tool_use_event]))
    loop._run_tool = unittest.mock.AsyncMock()

    await loop.start()
    loop._reconnect_timer.cancel()
    loop._event_queue.put_nowait(BidiTextInputEvent(text="occupied"))

    for _ in range(10):
        await asyncio.sleep(0)
        if loop._running_tools:
            break
    assert loop._running_tools

    assert loop._model_task is not None
    loop._model_task.cancel()
    await asyncio.gather(loop._model_task, return_exceptions=True)

    loop._run_tool.assert_not_awaited()
    assert loop._running_tools == {}

    await loop.stop()


@pytest.mark.asyncio
async def test_bidi_agent_loop_tool_result_recovers_after_reconnect(loop, agent, agenerator):
    """A tool completing on a later generation is delivered as semantic text."""
    tool_use = {"toolUseId": "t1", "name": "time_tool", "input": {}}

    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    await loop.start()

    # A reconnect during tool execution advances the connection generation.
    issuing_generation = loop._generation
    tool_use_key = await loop._register_tool_use(tool_use, issuing_generation)
    assert tool_use_key is not None
    loop._generation += 1

    # Drain the event queue (maxsize=1) so _run_tool's puts do not block.
    async def drain():
        while True:
            await loop._event_queue.get()

    drain_task = asyncio.create_task(drain())
    try:
        await loop._run_tool(tool_use, tool_use_key)
        await asyncio.sleep(0)
    finally:
        drain_task.cancel()

    assert len(agent.messages) == 2
    assert agent.messages[0]["role"] == "assistant"
    assert agent.messages[0]["content"] == [{"toolUse": tool_use}]
    assert agent.messages[1]["content"][0]["toolResult"]["toolUseId"] == "t1"
    agent.model.send.assert_awaited_once()
    tru_recovery_event = agent.model.send.call_args.args[0]
    assert isinstance(tru_recovery_event, BidiTextInputEvent)
    assert '"tool": "time_tool"' in tru_recovery_event.text
    assert "t1" not in tru_recovery_event.text
    assert set(loop._running_tools) == {tool_use_key}

    await loop._bind_recovery_response(loop._generation, "recovery")
    await loop._clear_recovered_tool_result("recovery")
    assert loop._running_tools == {}


def test_format_tool_result_recovery_is_deterministic_and_omits_provider_id():
    tool_use: ToolUse = {
        "toolUseId": "provider-id",
        "name": "weather",
        "input": {"city": "Seattle"},
    }
    tool_result: ToolResult = {
        "toolUseId": "provider-id",
        "status": "success",
        "content": [{"json": {"temperature": 68}}, {"text": "clear"}],
    }

    tru_recovery = _format_tool_result_recovery(tool_use, tool_result)
    exp_recovery = (
        "A tool requested before reconnection has completed. "
        "Use this result to answer the user without invoking the tool again: "
        '{"arguments": {"city": "Seattle"}, "result": [{"json": {"temperature": 68}}, '
        '{"text": "clear"}], "status": "success", "tool": "weather"}'
    )

    assert tru_recovery == exp_recovery
    assert "provider-id" not in tru_recovery


def test_format_tool_result_recovery_omits_unsupported_content():
    tool_use: ToolUse = {"toolUseId": "t1", "name": "image_tool", "input": {}}
    tool_result: ToolResult = {
        "toolUseId": "t1",
        "status": "success",
        "content": [
            {"text": "supported result"},
            {"image": {"format": "png", "source": {"bytes": b"image"}}},
        ],
    }

    tru_recovery = _format_tool_result_recovery(tool_use, tool_result)

    assert "supported result" in tru_recovery
    assert "[image omitted from voice recovery]" in tru_recovery
    assert "bytes" not in tru_recovery


def test_format_tool_result_recovery_rejects_non_serializable_content():
    tool_use: ToolUse = {"toolUseId": "t1", "name": "json_tool", "input": {}}
    tool_result: ToolResult = {
        "toolUseId": "t1",
        "status": "success",
        "content": [{"json": object()}],
    }

    with pytest.raises(ValueError, match="tool recovery content is not JSON serializable"):
        _format_tool_result_recovery(tool_use, tool_result)


def test_format_tool_result_recovery_truncates_oversized_content():
    tool_use: ToolUse = {"toolUseId": "t1", "name": "large_tool", "input": {}}
    tool_result: ToolResult = {
        "toolUseId": "t1",
        "status": "success",
        "content": [{"text": "é" * (_MAX_TOOL_RESULT_RECOVERY_BYTES // 2)}],
    }

    tru_recovery = _format_tool_result_recovery(tool_use, tool_result)

    assert len(tru_recovery.encode("utf-8")) <= _MAX_TOOL_RESULT_RECOVERY_BYTES
    assert "[truncated for voice recovery]" in tru_recovery


@pytest.mark.asyncio
async def test_tool_result_recovery_omits_unsupported_content(loop, agent, agenerator):
    """Unsupported blocks are omitted while the supported recovery turn is delivered."""
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    await loop.start()

    tool_use: ToolUse = {"toolUseId": "old", "name": "image_tool", "input": {}}
    tool_use_key = await loop._register_tool_use(tool_use, loop._generation)
    assert tool_use_key is not None
    loop._generation += 1
    tool_result: ToolResult = {
        "toolUseId": "old",
        "status": "success",
        "content": [{"image": {"format": "png", "source": {"bytes": b"image"}}}],
    }
    agent.tool_executor._stream = unittest.mock.Mock(return_value=agenerator([ToolResultEvent(tool_result)]))

    tru_events = []

    async def drain():
        while True:
            tru_events.append(await loop._event_queue.get())

    drain_task = asyncio.create_task(drain())
    await loop._run_tool(tool_use, tool_use_key)
    await asyncio.sleep(0)
    drain_task.cancel()

    agent.model.send.assert_awaited_once()
    tru_recovery_event = agent.model.send.await_args.args[0]
    assert isinstance(tru_recovery_event, BidiTextInputEvent)
    assert "[image omitted from voice recovery]" in tru_recovery_event.text
    assert loop._started is True
    assert not any(isinstance(event, Exception) for event in tru_events)
    assert len(agent.messages) == 2
    assert set(loop._running_tools) == {tool_use_key}

    await loop.stop()


@pytest.mark.asyncio
async def test_bidi_agent_loop_matches_tool_reissue_after_reconnect(loop, agent, agenerator):
    """One exact replacement-connection call receives the running tool's result."""
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    await loop.start()

    original = {"toolUseId": "old", "name": "time_tool", "input": {}}
    original_key = await loop._register_tool_use(original, loop._generation)
    assert original_key is not None

    loop._generation += 1
    reissue = {"toolUseId": "new", "name": "time_tool", "input": {}}

    assert await loop._register_tool_use(reissue, loop._generation) is None
    assert len(loop._running_tools) == 1

    result = {"toolUseId": "old", "status": "success", "content": [{"text": "12:00"}]}
    await loop._deliver_tool_result(original_key, ToolResultEvent(result))

    tru_delivered_ids = [call.args[0].tool_result["toolUseId"] for call in agent.model.send.await_args_list]
    exp_delivered_ids = ["new"]
    assert tru_delivered_ids == exp_delivered_ids
    assert loop._running_tools == {}

    await loop.stop()


@pytest.mark.asyncio
async def test_bidi_agent_loop_matches_only_one_reissue_per_generation(loop, agent, agenerator):
    """A second identical call on the replacement connection remains an independent execution."""
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    await loop.start()

    original: ToolUse = {"toolUseId": "old", "name": "time_tool", "input": {}}
    original_key = await loop._register_tool_use(original, loop._generation)
    assert original_key is not None

    loop._generation += 1
    first: ToolUse = {"toolUseId": "new-1", "name": "time_tool", "input": {}}
    second: ToolUse = {"toolUseId": "new-2", "name": "time_tool", "input": {}}

    assert await loop._register_tool_use(first, loop._generation) is None
    tru_second_key = await loop._register_tool_use(second, loop._generation)
    exp_second_key = (loop._generation, "new-2")
    assert tru_second_key == exp_second_key
    assert len(loop._running_tools) == 2

    await loop.stop()


@pytest.mark.asyncio
async def test_bidi_agent_loop_does_not_match_different_arguments(loop, agent, agenerator):
    """Calls with different arguments remain independent across a reconnect."""
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    await loop.start()

    original: ToolUse = {"toolUseId": "old", "name": "weather", "input": {"city": "London"}}
    original_key = await loop._register_tool_use(original, loop._generation)
    assert original_key is not None

    loop._generation += 1
    reissue: ToolUse = {"toolUseId": "new", "name": "weather", "input": {"city": "Paris"}}
    tru_reissue_key = await loop._register_tool_use(reissue, loop._generation)
    exp_reissue_key = (loop._generation, "new")
    assert tru_reissue_key == exp_reissue_key
    assert len(loop._running_tools) == 2

    await loop.stop()


@pytest.mark.asyncio
async def test_bidi_agent_loop_matches_running_tool_across_multiple_generations(loop, agent, agenerator):
    """A tool still running after multiple reconnects is not executed again."""
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    await loop.start()

    original: ToolUse = {"toolUseId": "old", "name": "time_tool", "input": {}}
    original_key = await loop._register_tool_use(original, loop._generation)
    assert original_key is not None

    loop._generation += 2
    reissue: ToolUse = {"toolUseId": "new", "name": "time_tool", "input": {}}
    tru_reissue_key = await loop._register_tool_use(reissue, loop._generation)
    assert tru_reissue_key is None
    assert len(loop._running_tools) == 1
    assert loop._running_tools[original_key].replacement_key == (loop._generation, "new")

    await loop.stop()


@pytest.mark.asyncio
async def test_bidi_agent_loop_updates_running_tool_reissue_on_each_generation(loop, agent, agenerator):
    """Each reconnect may replace a running call's obsolete provider-issued ID."""
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    await loop.start()

    original: ToolUse = {"toolUseId": "old", "name": "time_tool", "input": {}}
    original_key = await loop._register_tool_use(original, loop._generation)
    assert original_key is not None

    loop._generation += 1
    first_reissue: ToolUse = {"toolUseId": "new-1", "name": "time_tool", "input": {}}
    assert await loop._register_tool_use(first_reissue, loop._generation) is None
    assert loop._running_tools[original_key].replacement_key == (loop._generation, "new-1")

    loop._generation += 1
    second_reissue: ToolUse = {"toolUseId": "new-2", "name": "time_tool", "input": {}}
    assert await loop._register_tool_use(second_reissue, loop._generation) is None
    assert loop._running_tools[original_key].replacement_key == (loop._generation, "new-2")

    result = ToolResultEvent(ToolResult(toolUseId="old", status="success", content=[{"text": "12:00"}]))
    assert await loop._deliver_tool_result(original_key, result) is False

    agent.model.send.assert_awaited_once()
    tru_result_event = agent.model.send.await_args.args[0]
    assert isinstance(tru_result_event, ToolResultEvent)
    assert tru_result_event.tool_result["toolUseId"] == "new-2"
    assert loop._running_tools == {}

    await loop.stop()


@pytest.mark.asyncio
async def test_bidi_agent_loop_completed_tool_does_not_match_after_recovery_window(loop, agent, agenerator):
    """A completed result cannot satisfy a genuinely later identical tool call."""
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    await loop.start()

    original: ToolUse = {"toolUseId": "old", "name": "time_tool", "input": {}}
    original_key = await loop._register_tool_use(original, loop._generation)
    assert original_key is not None
    result = ToolResultEvent(ToolResult(toolUseId="old", status="success", content=[{"text": "12:00"}]))
    async with loop._tool_lock:
        loop._running_tools[original_key].result_event = result

    loop._generation += 2
    later: ToolUse = {"toolUseId": "later", "name": "time_tool", "input": {}}
    tru_later_key = await loop._register_tool_use(later, loop._generation)
    exp_later_key = (loop._generation, "later")

    assert tru_later_key == exp_later_key
    assert len(loop._running_tools) == 2

    await loop.stop()


@pytest.mark.asyncio
async def test_bidi_agent_loop_preserves_distinct_same_connection_tool_calls(loop, agent, agenerator):
    """Provider-issued calls on one connection remain distinct even when their arguments match."""
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    await loop.start()

    first = {"toolUseId": "first", "name": "time_tool", "input": {}}
    second = {"toolUseId": "second", "name": "time_tool", "input": {}}

    tru_first_key = await loop._register_tool_use(first, loop._generation)
    tru_second_key = await loop._register_tool_use(second, loop._generation)
    exp_first_key = (loop._generation, "first")
    exp_second_key = (loop._generation, "second")

    assert tru_first_key == exp_first_key
    assert tru_second_key == exp_second_key
    assert len(loop._running_tools) == 2

    await loop.stop()
    assert loop._running_tools == {}


@pytest.mark.asyncio
async def test_bidi_agent_loop_executes_ambiguous_reissue_independently(loop, agent, agenerator, caplog):
    """Multiple exact older-generation matches are not associated arbitrarily."""
    agent.model.receive = unittest.mock.Mock(return_value=agenerator([]))
    await loop.start()

    first: ToolUse = {"toolUseId": "first", "name": "time_tool", "input": {}}
    second: ToolUse = {"toolUseId": "second", "name": "time_tool", "input": {}}
    assert await loop._register_tool_use(first, loop._generation) is not None
    assert await loop._register_tool_use(second, loop._generation) is not None

    loop._generation += 1
    reissue: ToolUse = {"toolUseId": "reissue", "name": "time_tool", "input": {}}
    with caplog.at_level(logging.WARNING, logger="strands.experimental.bidi.agent.loop"):
        tru_reissue_key = await loop._register_tool_use(reissue, loop._generation)

    exp_reissue_key = (loop._generation, "reissue")
    assert tru_reissue_key == exp_reissue_key
    assert len(loop._running_tools) == 3
    assert (
        "matched_tool_use_ids=<first,second>, match_count=<2> | "
        "ambiguous tool use after reconnect | executing independently"
    ) in caplog.text

    await loop.stop()


@pytest.mark.asyncio
async def test_bidi_agent_loop_request_state_initialized_for_tools(loop, agent, agenerator):
    """Test that request_state is initialized in invocation_state before tool execution.

    This ensures request_state exists for tools that may need it via invocation_state,
    even when invocation_state is not provided by the user.
    """
    tool_use = {"toolUseId": "t2", "name": "time_tool", "input": {}}
    tool_use_event = ToolUseStreamEvent(current_tool_use=tool_use, delta="")

    agent.model.receive = unittest.mock.Mock(return_value=agenerator([tool_use_event]))

    # Start without providing invocation_state
    await loop.start()

    tru_events = []
    async for event in loop.receive():
        tru_events.append(event)
        if len(tru_events) >= 3:
            break

    # Verify tool executed successfully
    tool_result_event = tru_events[1]
    assert isinstance(tool_result_event, ToolResultEvent)
    assert tool_result_event.tool_result["status"] == "success"

    # Verify request_state was initialized in invocation_state
    assert "request_state" in loop._invocation_state
    assert isinstance(loop._invocation_state["request_state"], dict)


@pytest.mark.asyncio
async def test_bidi_agent_loop_stop_event_loop_flag(agent, agenerator):
    """Test that the stop_event_loop flag in request_state gracefully closes the connection.

    This simulates a tool (like strands_tools.stop) setting the flag via invocation_state.
    """
    # Use a tool that modifies invocation_state to set the stop flag
    # We'll mock the tool executor to simulate this behavior
    loop = agent._loop

    tool_use = {"toolUseId": "t3", "name": "time_tool", "input": {}}
    tool_use_event = ToolUseStreamEvent(current_tool_use=tool_use, delta="")

    agent.model.receive = unittest.mock.Mock(return_value=agenerator([tool_use_event]))

    # Start with request_state that already has stop_event_loop=True
    # This simulates a tool having set it during execution
    await loop.start(invocation_state={"request_state": {"stop_event_loop": True}})

    tru_events = []
    async for event in loop.receive():
        tru_events.append(event)

    # Should receive: tool_use_event, tool_result_event, tool_result_message, connection_close
    assert len(tru_events) == 4

    # Verify tool executed successfully
    tool_result_event = tru_events[1]
    assert isinstance(tool_result_event, ToolResultEvent)
    assert tool_result_event.tool_result["status"] == "success"

    # Verify connection close event was emitted
    connection_close_event = tru_events[3]
    assert isinstance(connection_close_event, BidiConnectionCloseEvent)
    assert connection_close_event["reason"] == "user_request"

    # Verify model.send was NOT called (tool result not sent to model)
    agent.model.send.assert_not_called()


@pytest.mark.asyncio
async def test_bidi_agent_loop_stop_conversation_deprecated_but_works(loop, agent, agenerator):
    """Test that stop_conversation tool still works but emits a deprecation warning.

    The stop_conversation tool is deprecated in favor of request_state["stop_event_loop"],
    but should continue to work for backward compatibility via the name-based check.
    """
    from strands.experimental.bidi.tools import stop_conversation

    agent.tool_registry.register_tool(stop_conversation)

    tool_use = {"toolUseId": "t5", "name": "stop_conversation", "input": {}}
    tool_use_event = ToolUseStreamEvent(current_tool_use=tool_use, delta="")

    agent.model.receive = unittest.mock.Mock(return_value=agenerator([tool_use_event]))

    await loop.start()

    tru_events = []
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        async for event in loop.receive():
            tru_events.append(event)

    # Should receive: tool_use_event, tool_result_event, tool_result_message, connection_close
    assert len(tru_events) == 4

    # Verify tool executed successfully
    tool_result_event = tru_events[1]
    assert isinstance(tool_result_event, ToolResultEvent)
    assert tool_result_event.tool_result["status"] == "success"
    assert "Ending conversation" in tool_result_event.tool_result["content"][0]["text"]

    # Verify connection close event was emitted
    connection_close_event = tru_events[3]
    assert isinstance(connection_close_event, BidiConnectionCloseEvent)
    assert connection_close_event["reason"] == "user_request"

    # Verify model.send was NOT called (tool result not sent to model)
    agent.model.send.assert_not_called()

    # Verify deprecation warnings were emitted (from both the tool itself and the loop name check)
    deprecation_warnings = [w for w in caught_warnings if issubclass(w.category, DeprecationWarning)]
    assert len(deprecation_warnings) >= 1
    assert any("stop_conversation" in str(w.message).lower() for w in deprecation_warnings)


@pytest.mark.asyncio
async def test_bidi_agent_loop_request_state_preserved_with_invocation_state(agent, agenerator):
    """Test that existing invocation_state is preserved when request_state is initialized."""

    @tool(name="check_invocation_state")
    async def check_invocation_state(custom_key: str) -> str:
        return f"custom_key: {custom_key}"

    agent.tool_registry.register_tool(check_invocation_state)

    tool_use = {"toolUseId": "t4", "name": "check_invocation_state", "input": {"custom_key": "from_state"}}
    tool_use_event = ToolUseStreamEvent(current_tool_use=tool_use, delta="")

    agent.model.receive = unittest.mock.Mock(return_value=agenerator([tool_use_event]))

    loop = agent._loop
    # Start with custom invocation_state but no request_state
    await loop.start(invocation_state={"custom_data": "preserved"})

    tru_events = []
    async for event in loop.receive():
        tru_events.append(event)
        if len(tru_events) >= 3:
            break

    # Verify tool executed successfully
    tool_result_event = tru_events[1]
    assert isinstance(tool_result_event, ToolResultEvent)
    assert tool_result_event.tool_result["status"] == "success"

    # Verify request_state was added without removing custom_data
    assert "request_state" in loop._invocation_state
    assert loop._invocation_state.get("custom_data") == "preserved"


@pytest.mark.asyncio
async def test_bidi_agent_loop_send_respects_event_role(loop, agent):
    agent.model.start = unittest.mock.AsyncMock()
    agent.model.send = unittest.mock.AsyncMock()
    await loop.start()
    await loop.send(BidiTextInputEvent(text="injected context", role="assistant"))
    assert agent.messages[-1] == {
        "role": "assistant",
        "content": [{"text": "injected context"}],
        "tracking_id": unittest.mock.ANY,
    }
