"""Unit tests for the proactive reconnect timer.

The timer is exercised with an injected fake clock so timing is deterministic and does
not depend on wall time or a running provider.
"""

import asyncio

import pytest

from strands.experimental.bidi.agent._reconnect_timer import _BidiReconnectTimer, resolve_deadline_s

# resolve_deadline_s


def test_resolve_deadline_none_when_not_declared():
    """No declared restart_after_s means no proactive timer; reconnect stays reactive-only."""
    assert resolve_deadline_s({}) is None
    assert resolve_deadline_s({"auto_reconnect": True}) is None


def test_resolve_deadline_is_restart_after_s():
    """Deadline is the declared restart_after_s."""
    assert resolve_deadline_s({"restart_after_s": 540}) == 540


def test_resolve_deadline_none_when_not_positive():
    """A non-positive restart_after_s declares no usable deadline; no proactive timer arms."""
    assert resolve_deadline_s({"restart_after_s": 0}) is None
    assert resolve_deadline_s({"restart_after_s": -5}) is None


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
