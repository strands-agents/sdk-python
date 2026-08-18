"""Unit tests for the proactive reconnect timer.

The timer is exercised with an injected fake clock so timing is deterministic and does
not depend on wall time or a running provider.
"""

import asyncio

import pytest

from strands.experimental.bidi.agent._reconnect_timer import _BidiReconnectTimer, resolve_deadline_s


# resolve_deadline_s


def test_resolve_deadline_none_when_no_limit_declared():
    """No declared limit means no proactive timer; reconnect stays reactive-only."""
    assert resolve_deadline_s({}) is None
    assert resolve_deadline_s({"reconnect_margin_s": 10}) is None


def test_resolve_deadline_is_limit_minus_margin():
    """Deadline is the declared connection limit minus the reconnect margin."""
    config = {"max_connection_s": 600.0, "reconnect_margin_s": 60.0}
    assert resolve_deadline_s(config) == 540.0


def test_resolve_deadline_defaults_margin():
    """Reconnect margin defaults to 60s when not declared."""
    assert resolve_deadline_s({"max_connection_s": 480.0}) == 420.0


def test_resolve_deadline_never_negative():
    """A limit smaller than the margin clamps to zero (reconnect asap), never negative."""
    assert resolve_deadline_s({"max_connection_s": 30.0, "reconnect_margin_s": 60.0}) == 0.0


# _BidiReconnectTimer


@pytest.mark.asyncio
async def test_timer_fires_warning_then_deadline():
    """The timer fires the warning at the lead offset, then the deadline."""
    warnings, deadlines = [], []
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    timer = _BidiReconnectTimer(
        on_warning=lambda t: _record(warnings, t),
        on_deadline=lambda: _record(deadlines, None),
        sleep=fake_sleep,
    )

    # deadline 420, warning 30 before it => sleep 390 then 30.
    timer.arm(deadline_s=420.0, warning_lead_s=30.0)

    await timer._task

    assert sleeps == [390.0, 30.0]
    assert warnings == [30.0]  # time_left_s at warning == warning_lead_s
    assert deadlines == [None]


@pytest.mark.asyncio
async def test_timer_cancel_is_safe_when_idle():
    """cancel() before arming does not raise."""
    timer = _BidiReconnectTimer(on_warning=_noop_arg, on_deadline=_noop)
    timer.cancel()  # should not raise


@pytest.mark.asyncio
async def test_timer_rearm_cancels_previous():
    """Re-arming cancels the prior cycle so only one deadline fires."""
    deadlines = []

    started = asyncio.Event()

    async def slow_sleep(seconds):
        started.set()
        await asyncio.sleep(3600)

    timer = _BidiReconnectTimer(
        on_warning=_noop_arg,
        on_deadline=lambda: _record(deadlines, None),
        sleep=slow_sleep,
    )

    timer.arm(deadline_s=420.0, warning_lead_s=30.0)
    await started.wait()
    first_task = timer._task

    timer.arm(deadline_s=420.0, warning_lead_s=30.0)

    await asyncio.sleep(0)
    assert first_task.cancelled() or first_task.done()

    timer.cancel()


# Helpers


async def _record(sink, value):
    sink.append(value)


async def _noop():
    return None


async def _noop_arg(_value):
    return None
