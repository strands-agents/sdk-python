"""Tests that Tracer's ThreadingInstrumentor side-effect is opt-in.

These tests run in isolated subprocesses because
``opentelemetry.instrumentation.BaseInstrumentor`` is a process-wide singleton
whose ``_is_instrumented_by_opentelemetry`` flag + wrapped
``concurrent.futures.ThreadPoolExecutor.submit`` leak across tests.
"""

import os
import subprocess
import sys
import textwrap


def _run_in_subprocess(script: str, env: dict | None = None) -> tuple[int, str]:
    """Run a short Python snippet in a fresh subprocess.

    Inherits the parent's environment so user config dirs, locale, and
    tempdir (``HOME``, ``LANG``, ``LC_ALL``, ``TMPDIR``, ``PATH``, etc.)
    remain available. The caller-supplied ``env`` dict overrides specific
    keys; values of ``None`` in that dict mean "remove this variable",
    which lets opt-in tests scrub any ambient
    ``OTEL_PYTHON_DISABLED_INSTRUMENTATIONS`` a CI box might have set.

    Returns (exit_code, combined stdout+stderr).
    """
    proc_env = os.environ.copy()
    if env:
        for k, v in env.items():
            if v is None:
                proc_env.pop(k, None)
            else:
                proc_env[k] = v
    try:
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            capture_output=True,
            text=True,
            env=proc_env,
            timeout=60,
        )
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "")
        raise AssertionError(f"subprocess timed out after 60s. Captured output:\n{out}") from e
    return result.returncode, result.stdout + result.stderr


# Robust wrap-detection snippet: only flag WRAPPED if *strands* wrapped submit,
# not if something in the subprocess pre-wrapped it before `before` was captured.
_WRAP_DETECT = """
    from concurrent.futures import ThreadPoolExecutor
    before = ThreadPoolExecutor.submit
    before_wrapped = hasattr(before, "__wrapped__")
    from strands.telemetry.tracer import Tracer
    {construct}
    after = ThreadPoolExecutor.submit
    newly_wrapped = (hasattr(after, "__wrapped__") or (before is not after)) and not before_wrapped
    print("WRAPPED" if newly_wrapped else "UNWRAPPED")
"""


def test_threading_instrumentation_is_off_by_default():
    """Constructing Tracer() must NOT monkey-patch ThreadPoolExecutor.submit by default.

    Users should be able to use strands without having their global
    ``concurrent.futures.ThreadPoolExecutor.submit`` wrapped unless they
    explicitly opt in.
    """
    script = _WRAP_DETECT.format(construct="Tracer()")
    # Scrub ambient STRANDS_INSTRUMENT_THREADING so a dev shell with opt-in
    # already exported doesn't mask a regression.
    code, out = _run_in_subprocess(script, env={"STRANDS_INSTRUMENT_THREADING": None})
    assert code == 0, f"subprocess failed: {out}"
    assert "UNWRAPPED" in out, (
        "Tracer() wrapped ThreadPoolExecutor.submit by default; threading "
        "instrumentation must be opt-in. Output:\n" + out
    )


def test_threading_instrumentation_respects_otel_disabled_env_var():
    """Honor OTEL_PYTHON_DISABLED_INSTRUMENTATIONS=threading even when opt-in is on.

    The strands ``_threading_opt_in`` resolver honors
    ``OTEL_PYTHON_DISABLED_INSTRUMENTATIONS`` directly (precedence rule #1:
    the disable env var always wins over any kwarg or strands env var).
    """
    script = _WRAP_DETECT.format(construct="Tracer()")
    code, out = _run_in_subprocess(
        script,
        env={
            "STRANDS_INSTRUMENT_THREADING": "true",
            "OTEL_PYTHON_DISABLED_INSTRUMENTATIONS": "threading",
        },
    )
    assert code == 0, f"subprocess failed: {out}"
    assert "UNWRAPPED" in out, (
        "Tracer() wrapped ThreadPoolExecutor.submit despite "
        "OTEL_PYTHON_DISABLED_INSTRUMENTATIONS=threading. Output:\n" + out
    )


def test_threading_instrumentation_kwarg_respects_otel_disabled_env_var():
    """OTEL_PYTHON_DISABLED_INSTRUMENTATIONS=threading beats kwarg=True.

    Precedence rule #1 (disable env var) must beat precedence rule #2
    (explicit kwarg). Without this guarantee, a host application that
    set the disable env var to avoid wrapper stacking would still get
    wrapped when a library passed ``instrument_threading=True`` programmatically.
    """
    script = _WRAP_DETECT.format(construct="Tracer(instrument_threading=True)")
    code, out = _run_in_subprocess(
        script,
        env={"OTEL_PYTHON_DISABLED_INSTRUMENTATIONS": "threading"},
    )
    assert code == 0, f"subprocess failed: {out}"
    assert "UNWRAPPED" in out, (
        "Tracer(instrument_threading=True) wrapped ThreadPoolExecutor.submit "
        "despite OTEL_PYTHON_DISABLED_INSTRUMENTATIONS=threading. Output:\n" + out
    )


def test_threading_instrumentation_opt_in_via_env_var():
    """STRANDS_INSTRUMENT_THREADING=true enables the old behavior."""
    script = _WRAP_DETECT.format(construct="Tracer()")
    # Scrub ambient OTEL_PYTHON_DISABLED_INSTRUMENTATIONS in case a CI box
    # or dev shell has set it to ``threading`` globally — that would
    # suppress opt-in and falsely report UNWRAPPED.
    code, out = _run_in_subprocess(
        script,
        env={
            "STRANDS_INSTRUMENT_THREADING": "true",
            "OTEL_PYTHON_DISABLED_INSTRUMENTATIONS": None,
        },
    )
    assert code == 0, f"subprocess failed: {out}"
    assert "WRAPPED" in out, "Explicit opt-in did not wrap ThreadPoolExecutor.submit. Output:\n" + out


def test_threading_instrumentation_opt_in_via_kwarg():
    """Tracer(instrument_threading=True) enables the old behavior programmatically."""
    script = _WRAP_DETECT.format(construct="Tracer(instrument_threading=True)")
    # Scrub ambient OTEL_PYTHON_DISABLED_INSTRUMENTATIONS (see above).
    code, out = _run_in_subprocess(script, env={"OTEL_PYTHON_DISABLED_INSTRUMENTATIONS": None})
    assert code == 0, f"subprocess failed: {out}"
    assert "WRAPPED" in out, "Tracer(instrument_threading=True) did not wrap ThreadPoolExecutor.submit. Output:\n" + out


def test_threading_instrumentation_kwarg_false_overrides_env_var_opt_in():
    """Precedence rule: explicit ``instrument_threading=False`` beats env-var opt-in.

    If a host application programmatically disables threading instrumentation
    but the user (or CI) has ``STRANDS_INSTRUMENT_THREADING=true`` exported
    globally, the kwarg must win — otherwise programmatic opt-out is
    unreliable.
    """
    script = _WRAP_DETECT.format(construct="Tracer(instrument_threading=False)")
    code, out = _run_in_subprocess(
        script,
        env={
            "STRANDS_INSTRUMENT_THREADING": "true",
            "OTEL_PYTHON_DISABLED_INSTRUMENTATIONS": None,
        },
    )
    assert code == 0, f"subprocess failed: {out}"
    assert "UNWRAPPED" in out, (
        "Tracer(instrument_threading=False) wrapped ThreadPoolExecutor.submit "
        "even though the explicit kwarg should beat the env-var opt-in. "
        "Output:\n" + out
    )


def test_threading_instrumentation_idempotent_when_already_instrumented():
    """If another library already instrumented threading, strands must not double-wrap.

    Respects BaseInstrumentor._is_instrumented_by_opentelemetry — strands
    should check before calling instrument(). This exercises OUR guard in
    ``_maybe_instrument_threading``; we verify that depth after strands
    construction equals depth before (i.e. strands added zero wrappers).
    """
    script = """
        from opentelemetry.instrumentation.threading import ThreadingInstrumentor
        ThreadingInstrumentor().instrument()
        from concurrent.futures import ThreadPoolExecutor

        # Count wrap depth after external instrumentation
        depth_before = 0
        fn = ThreadPoolExecutor.submit
        while hasattr(fn, "__wrapped__"):
            depth_before += 1
            fn = fn.__wrapped__

        # Now construct strands Tracer with opt-in on
        from strands.telemetry.tracer import Tracer
        Tracer()

        depth_after = 0
        fn = ThreadPoolExecutor.submit
        while hasattr(fn, "__wrapped__"):
            depth_after += 1
            fn = fn.__wrapped__

        print(f"DEPTH_BEFORE={depth_before} DEPTH_AFTER={depth_after}")
    """
    code, out = _run_in_subprocess(script, env={"STRANDS_INSTRUMENT_THREADING": "true"})
    assert code == 0, f"subprocess failed: {out}"
    # Parse the printed counts. Assert depth_after == depth_before and that
    # external instrumentation actually wrapped (>= 1). Don't hardcode
    # depth==1 — future OTel versions may legitimately stack multiple wrappers.
    import re

    match = re.search(r"DEPTH_BEFORE=(\d+) DEPTH_AFTER=(\d+)", out)
    assert match, f"could not parse wrap depth from output:\n{out}"
    depth_before = int(match.group(1))
    depth_after = int(match.group(2))
    assert depth_before >= 1, (
        f"external ThreadingInstrumentor did not wrap submit "
        f"(depth_before={depth_before}); idempotency guard untestable. Output:\n{out}"
    )
    assert depth_after == depth_before, (
        f"Strands wrapped an already-instrumented ThreadPoolExecutor.submit a "
        f"second time (depth_before={depth_before}, depth_after={depth_after}). "
        f"Output:\n{out}"
    )


def test_threading_instrumentation_idempotency_guard_fires():
    """Verify OUR ``_maybe_instrument_threading`` guard — not OTel's internal one — fires.

    The previous source code read ``is_instrumented_by_opentelemetry`` (no
    underscore prefix), which happens to resolve via the property method and
    also works, but the documented stable attribute is
    ``_is_instrumented_by_opentelemetry``. This test pre-sets the flag and
    then calls ``_maybe_instrument_threading`` directly with mocking so we
    can observe that ``instrument()`` was NOT invoked by strands.
    """
    script = """
        from unittest.mock import patch
        from opentelemetry.instrumentation.threading import ThreadingInstrumentor
        from strands.telemetry.tracer import Tracer

        # BaseInstrumentor is a singleton, but the flag may be stored on the
        # instance — not just the class — after the first .instrument() call.
        # Construct the singleton explicitly first, then patch the attribute
        # ON THE INSTANCE so _maybe_instrument_threading (which calls
        # ThreadingInstrumentor() to get the same singleton) observes True.
        instance = ThreadingInstrumentor()
        with patch.object(instance, "_is_instrumented_by_opentelemetry", True):
            with patch.object(ThreadingInstrumentor, "instrument") as mock_instrument:
                t = Tracer.__new__(Tracer)
                t._maybe_instrument_threading(True)
                called = mock_instrument.called
        print("INSTRUMENT_CALLED" if called else "GUARD_FIRED")
    """
    code, out = _run_in_subprocess(script)
    assert code == 0, f"subprocess failed: {out}"
    assert "GUARD_FIRED" in out, (
        "_maybe_instrument_threading called ThreadingInstrumentor.instrument() "
        "despite _is_instrumented_by_opentelemetry=True — strands' idempotency "
        "guard is not firing. Output:\n" + out
    )


def test_threading_instrumentation_swallows_instrumentor_failures():
    """Failures inside ``ThreadingInstrumentor.instrument()`` must not crash the host.

    Telemetry is opt-in ancillary functionality; a failure must not crash
    the user's application. The pattern elsewhere in this module (see
    ``_end_span``) is: log + continue.

    Also asserts the log entry carries enough detail (``threading`` and the
    exception class name) to be actionable — a silent swallow is worse than
    a crash for debuggability.
    """
    script = """
        import logging
        import sys
        from unittest.mock import patch
        from opentelemetry.instrumentation.threading import ThreadingInstrumentor
        from strands.telemetry.tracer import Tracer

        # Capture log output to stderr at WARNING and above so we can assert
        # on the log message content from the parent test.
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr, force=True)

        def _boom(self, **kwargs):
            raise RuntimeError("simulated instrumentor failure")

        with patch.object(ThreadingInstrumentor, "instrument", _boom):
            t = Tracer.__new__(Tracer)
            try:
                t._maybe_instrument_threading(True)
            except Exception as e:
                print(f"CRASHED: {e!r}")
            else:
                print("SWALLOWED")
    """
    code, out = _run_in_subprocess(script)
    assert code == 0, f"subprocess failed: {out}"
    assert "SWALLOWED" in out, (
        "_maybe_instrument_threading did NOT swallow an instrumentor "
        "exception; telemetry failure would crash the host application. "
        "Output:\n" + out
    )
    # Log must identify the subsystem and the error class so operators can
    # distinguish a threading instrumentation failure from any other warning.
    assert "threading" in out.lower(), (
        "Log output does not mention 'threading'; operators can't tell which subsystem failed. Output:\n" + out
    )
    assert "RuntimeError" in out, (
        "Log output does not include the exception class name "
        "(expected RuntimeError via exc_info traceback). Output:\n" + out
    )


def test_threading_instrumentation_failure_logged_at_error_when_user_requested():
    """When the user explicitly opted in, instrumentor failure logs at ERROR.

    Rationale: the user asked for threading span propagation. Silently
    degrading to WARNING makes the broken feature easy to miss in log
    dashboards whose default filter is ERROR+. WARNING is reserved for
    auto-enabled paths (currently none, but the distinction is kept so a
    future default flip doesn't escalate every log).
    """
    script = """
        import logging
        import sys
        from unittest.mock import patch
        from opentelemetry.instrumentation.threading import ThreadingInstrumentor
        from strands.telemetry.tracer import Tracer

        # Capture at WARNING so both WARNING and ERROR records surface, then
        # we can discriminate by the emitted level prefix.
        logging.basicConfig(
            level=logging.WARNING,
            stream=sys.stderr,
            format="%(levelname)s:%(name)s:%(message)s",
            force=True,
        )

        def _boom(self, **kwargs):
            raise RuntimeError("simulated instrumentor failure")

        with patch.object(ThreadingInstrumentor, "instrument", _boom):
            t = Tracer.__new__(Tracer)
            # Explicit kwarg opt-in → user_requested=True → ERROR
            t._maybe_instrument_threading(True)
        print("DONE")
    """
    code, out = _run_in_subprocess(script)
    assert code == 0, f"subprocess failed: {out}"
    assert "DONE" in out, f"subprocess did not reach DONE: {out}"
    # ERROR level when the user explicitly requested instrumentation.
    assert "ERROR:strands.telemetry.tracer:" in out, (
        "Expected ERROR-level log when user explicitly opted in via kwarg; "
        "WARNING is not loud enough for a broken feature the user asked for. "
        "Output:\n" + out
    )
