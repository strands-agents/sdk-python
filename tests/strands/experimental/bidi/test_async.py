from unittest.mock import AsyncMock, Mock

import pytest

from strands.experimental.bidi._async import start, stop


@pytest.fixture
def mock_startable():
    return Mock(start=AsyncMock(), stop=AsyncMock())


@pytest.mark.asyncio
async def test_start_exception(mock_startable):
    mock_startable.start.side_effect = ValueError("start failed")

    with pytest.raises(ValueError, match=r"start failed"):
        await start(mock_startable.start)(mock_startable)

    mock_startable.stop.assert_called_once()


@pytest.mark.asyncio
async def test_start_success(mock_startable):
    await start(mock_startable.start)(mock_startable)
    mock_startable.stop.assert_not_called()


@pytest.mark.asyncio
async def test_stop_exception():
    func1 = AsyncMock()
    func2 = AsyncMock(side_effect=ValueError("stop 2 failed"))
    func3 = AsyncMock()

    with pytest.raises(ExceptionGroup) as exc_info:  # type: ignore  # noqa: F821
        await stop(func1, func2, func3)

    func1.assert_called_once()
    func2.assert_called_once()
    func3.assert_called_once()

    assert len(exc_info.value.exceptions) == 1
    with pytest.raises(ValueError, match=r"stop 2 failed"):
        raise exc_info.value.exceptions[0]


@pytest.mark.asyncio
async def test_stop_success():
    func1 = AsyncMock()
    func2 = AsyncMock()
    func3 = AsyncMock()

    await stop(func1, func2, func3)

    func1.assert_called_once()
    func2.assert_called_once()
    func3.assert_called_once()
