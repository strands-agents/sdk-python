"""Tests for _async module."""

import asyncio
import importlib
import sys
from unittest import mock

import pytest

from strands._async import run_async

# greenback's ensure_portal() has compatibility issues with pytest-asyncio on Python < 3.11
GREENBACK_PORTAL_COMPATIBLE = sys.version_info >= (3, 11)


def test_run_async_with_return_value():
    """Test run_async returns correct value."""

    async def async_with_value():
        return 42

    result = run_async(async_with_value)
    assert result == 42


def test_run_async_exception_propagation():
    """Test that exceptions are properly propagated."""

    async def async_with_exception():
        raise ValueError("test exception")

    with pytest.raises(ValueError, match="test exception"):
        run_async(async_with_exception)


class TestRunAsyncGreenbackMocked:
    """Tests for run_async greenback code paths using mocks."""

    @pytest.fixture
    def mock_greenback(self) -> mock.MagicMock:
        """Create a mock greenback module."""
        return mock.MagicMock()

    def test_uses_greenback_when_portal_active(self, mock_greenback: mock.MagicMock) -> None:
        """Test that greenback.await_ is used when a portal is active."""
        mock_greenback.has_portal.return_value = True
        mock_greenback.await_.return_value = "greenback_result"

        with mock.patch("strands._async._GREENBACK_AVAILABLE", True):
            with mock.patch("strands._async.greenback", mock_greenback):
                result = run_async(lambda: asyncio.sleep(0))

        mock_greenback.has_portal.assert_called_once()
        mock_greenback.await_.assert_called_once()
        assert result == "greenback_result"

    def test_falls_back_without_portal(self, mock_greenback: mock.MagicMock) -> None:
        """Test that ThreadPoolExecutor is used when no portal is active."""
        mock_greenback.has_portal.return_value = False

        async def async_fn() -> str:
            return "thread_result"

        with mock.patch("strands._async._GREENBACK_AVAILABLE", True):
            with mock.patch("strands._async.greenback", mock_greenback):
                result = run_async(async_fn)

        mock_greenback.has_portal.assert_called_once()
        mock_greenback.await_.assert_not_called()
        assert result == "thread_result"

    def test_falls_back_without_greenback_installed(self) -> None:
        """Test that ThreadPoolExecutor is used when greenback is not installed."""

        async def async_fn() -> str:
            return "thread_result"

        with mock.patch("strands._async._GREENBACK_AVAILABLE", False):
            result = run_async(async_fn)

        assert result == "thread_result"


class TestRunAsyncGreenbackReal:
    """Integration tests that run only when greenback is installed."""

    @pytest.fixture
    def greenback_module(self):
        """Import greenback or skip test if not available."""
        return pytest.importorskip("greenback")

    @pytest.mark.skipif(
        not GREENBACK_PORTAL_COMPATIBLE, reason="greenback portal incompatible with pytest-asyncio on Python < 3.11"
    )
    @pytest.mark.asyncio
    async def test_run_async_uses_same_event_loop_with_portal(self, greenback_module) -> None:
        """Test that run_async uses the same event loop when portal is active."""
        await greenback_module.ensure_portal()

        outer_loop = asyncio.get_running_loop()
        inner_loop = None

        async def capture_loop() -> str:
            nonlocal inner_loop
            inner_loop = asyncio.get_running_loop()
            return "result"

        # Call run_async from within the async context
        result = run_async(capture_loop)

        assert result == "result"
        assert inner_loop is outer_loop, "Expected run_async to use the same event loop"

    @pytest.mark.skipif(
        not GREENBACK_PORTAL_COMPATIBLE, reason="greenback portal incompatible with pytest-asyncio on Python < 3.11"
    )
    @pytest.mark.asyncio
    async def test_run_async_can_access_outer_task(self, greenback_module) -> None:
        """Test that async code in run_async can access resources from the outer context."""
        await greenback_module.ensure_portal()

        # Create a shared resource in the outer async context
        shared_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        shared_future.set_result("shared_value")

        async def access_shared() -> str:
            # This would fail with ThreadPoolExecutor (different loop)
            return await shared_future

        result = run_async(access_shared)
        assert result == "shared_value"

    def test_run_async_without_portal_uses_separate_loop(self, greenback_module, monkeypatch) -> None:
        """Test that run_async uses a separate loop when greenback is disabled."""
        import strands._async

        # Disable greenback via env var to ensure we test the fallback path
        monkeypatch.setenv("STRANDS_DISABLE_GREENBACK", "1")
        importlib.reload(strands._async)

        try:

            async def run_test() -> bool:
                main_loop_id = id(asyncio.get_running_loop())

                async def get_inner_loop_id() -> int:
                    return id(asyncio.get_running_loop())

                inner_loop_id = strands._async.run_async(get_inner_loop_id)
                # With greenback disabled, should be different loops
                return inner_loop_id != main_loop_id

            result = asyncio.run(run_test())
            assert result, "Expected different loops when greenback is disabled"
        finally:
            monkeypatch.delenv("STRANDS_DISABLE_GREENBACK", raising=False)
            importlib.reload(strands._async)


class TestRunAsyncGreenbackEndToEnd:
    """End-to-end tests demonstrating the greenback use case from the ticket.

    These tests verify that async resources bound to the main event loop:
    - Are INACCESSIBLE without a greenback portal (different loop)
    - Are ACCESSIBLE with a greenback portal (same loop)
    """

    @pytest.fixture
    def greenback_module(self):
        """Import greenback or skip test if not available."""
        return pytest.importorskip("greenback")

    def test_loop_bound_client_fails_without_portal(self, greenback_module, monkeypatch) -> None:
        """Test that a loop-bound client fails when accessed from run_async without greenback.

        This simulates the real-world scenario where an httpx.AsyncClient or database
        connection pool is created on the main loop and cannot be used from a different loop.
        """
        import strands._async

        # Disable greenback via env var to ensure we test the fallback path
        monkeypatch.setenv("STRANDS_DISABLE_GREENBACK", "1")
        importlib.reload(strands._async)

        try:
            # We need to run this in an async context to have a "main loop"
            async def run_test() -> bool:
                main_loop_id = id(asyncio.get_running_loop())

                class LoopBoundClient:
                    """Simulates an async client that requires same-loop usage."""

                    def __init__(self, bound_loop_id: int):
                        self._bound_loop_id = bound_loop_id

                    async def request(self) -> dict:
                        current_loop_id = id(asyncio.get_running_loop())
                        if current_loop_id != self._bound_loop_id:
                            raise RuntimeError(
                                f"Client bound to loop {self._bound_loop_id}, called from loop {current_loop_id}"
                            )
                        return {"status": "ok"}

                client = LoopBoundClient(main_loop_id)

                async def use_client() -> dict:
                    return await client.request()

                try:
                    strands._async.run_async(use_client)
                    return False  # Should have failed
                except RuntimeError:
                    return True  # Expected failure

            # Run with greenback disabled - should fail
            result = asyncio.run(run_test())
            assert result, "Expected loop-bound client to fail when greenback is disabled"
        finally:
            monkeypatch.delenv("STRANDS_DISABLE_GREENBACK", raising=False)
            importlib.reload(strands._async)

    @pytest.mark.skipif(
        not GREENBACK_PORTAL_COMPATIBLE, reason="greenback portal incompatible with pytest-asyncio on Python < 3.11"
    )
    @pytest.mark.asyncio
    async def test_loop_bound_client_succeeds_with_portal(self, greenback_module) -> None:
        """Test that a loop-bound client succeeds when accessed via greenback portal.

        With the portal active, run_async uses greenback.await_() which keeps us on
        the same event loop, allowing access to loop-bound resources.
        """
        await greenback_module.ensure_portal()

        main_loop_id = id(asyncio.get_running_loop())

        class LoopBoundClient:
            """Simulates an async client that requires same-loop usage."""

            def __init__(self, bound_loop_id: int):
                self._bound_loop_id = bound_loop_id

            async def request(self) -> dict:
                current_loop_id = id(asyncio.get_running_loop())
                if current_loop_id != self._bound_loop_id:
                    raise RuntimeError(
                        f"Client bound to loop {self._bound_loop_id}, called from loop {current_loop_id}"
                    )
                return {"status": "ok"}

        client = LoopBoundClient(main_loop_id)

        async def use_client() -> dict:
            return await client.request()

        # With portal - should succeed
        result = run_async(use_client)
        assert result == {"status": "ok"}


class TestRunAsyncGreenbackEnvVar:
    """Tests for STRANDS_DISABLE_GREENBACK environment variable."""

    @pytest.fixture
    def greenback_module(self):
        """Import greenback or skip test if not available."""
        return pytest.importorskip("greenback")

    def test_env_var_disables_greenback(self, greenback_module, monkeypatch) -> None:
        """Test that STRANDS_DISABLE_GREENBACK=1 forces ThreadPoolExecutor fallback.

        Even with greenback installed and a portal active, the env var should
        force the fallback to ThreadPoolExecutor (different event loop).
        """
        import strands._async

        # Set env var and reload module to pick up the change
        monkeypatch.setenv("STRANDS_DISABLE_GREENBACK", "1")
        importlib.reload(strands._async)

        try:
            # Verify the flag was set correctly
            assert strands._async._GREENBACK_AVAILABLE is False

            # Now run a test that would use greenback if enabled
            async def run_test() -> bool:
                await greenback_module.ensure_portal()
                main_loop_id = id(asyncio.get_running_loop())

                async def get_inner_loop_id() -> int:
                    return id(asyncio.get_running_loop())

                inner_loop_id = strands._async.run_async(get_inner_loop_id)
                # With env var set, should be different loops even with portal
                return inner_loop_id != main_loop_id

            result = asyncio.run(run_test())
            assert result, "Expected different loops when STRANDS_DISABLE_GREENBACK=1"
        finally:
            # Restore module state
            monkeypatch.delenv("STRANDS_DISABLE_GREENBACK", raising=False)
            importlib.reload(strands._async)
