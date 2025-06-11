import unittest.mock

import botocore
import pytest

import strands
from strands.types.exceptions import ModelThrottledException


@pytest.fixture
def callback_handler():
    return unittest.mock.Mock()


@pytest.fixture
def kwargs():
    return {"request_state": "value"}


@pytest.fixture
def tool_handler():
    return unittest.mock.Mock()


@pytest.fixture
def model():
    return unittest.mock.Mock()


@pytest.fixture
def tool_config():
    return {}


@pytest.fixture
def system_prompt():
    return "prompt."


@pytest.fixture
def sdk_event_loop():
    with unittest.mock.patch.object(strands.event_loop.event_loop, "recurse_event_loop") as mock:
        yield mock


@pytest.fixture
def event_stream_error(request):
    message = request.param
    return botocore.exceptions.EventStreamError({"Error": {"Message": message}}, "mock_operation")


def test_handle_throttling_error(callback_handler, kwargs):
    exception = ModelThrottledException("ThrottlingException | ConverseStream")
    max_attempts = 2
    delay = 0.1
    max_delay = 1

    tru_retries = []
    tru_delays = []
    for attempt in range(max_attempts):
        retry, delay = strands.event_loop.error_handler.handle_throttling_error(
            exception, attempt, max_attempts, delay, max_delay, callback_handler, kwargs
        )

        tru_retries.append(retry)
        tru_delays.append(delay)

    exp_retries = [True, False]
    exp_delays = [0.2, 0.2]

    assert tru_retries == exp_retries and tru_delays == exp_delays

    callback_handler.assert_has_calls(
        [
            unittest.mock.call(event_loop_throttled_delay=0.1, request_state="value"),
            unittest.mock.call(force_stop=True, force_stop_reason=str(exception)),
        ]
    )


def test_handle_throttling_error_does_not_exist(callback_handler, kwargs):
    exception = ModelThrottledException("Other Error")
    attempt = 0
    max_attempts = 1
    delay = 1
    max_delay = 1

    tru_retry, tru_delay = strands.event_loop.error_handler.handle_throttling_error(
        exception, attempt, max_attempts, delay, max_delay, callback_handler, kwargs
    )

    exp_retry = False
    exp_delay = 1

    assert tru_retry == exp_retry and tru_delay == exp_delay

    callback_handler.assert_called_with(force_stop=True, force_stop_reason=str(exception))
