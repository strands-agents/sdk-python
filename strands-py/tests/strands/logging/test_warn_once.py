import logging

import pytest

from strands.logging import warn_once
from strands.logging.warn_once import warn_once as warn_once_fn


@pytest.fixture(autouse=True)
def reset_warned():
    warn_once._warned.clear()
    yield
    warn_once._warned.clear()


@pytest.fixture
def logger():
    return logging.getLogger("strands.logging.test_warn_once")


def test_emits_warning_first_time(logger, caplog):
    with caplog.at_level(logging.WARNING, logger=logger.name):
        warn_once_fn(logger, "first-seen-msg")

    assert sum("first-seen-msg" in record.message for record in caplog.records) == 1


def test_suppresses_repeats_of_same_message(logger, caplog):
    with caplog.at_level(logging.WARNING, logger=logger.name):
        warn_once_fn(logger, "repeated-msg")
        warn_once_fn(logger, "repeated-msg")
        warn_once_fn(logger, "repeated-msg")

    assert sum("repeated-msg" in record.message for record in caplog.records) == 1


def test_dedupes_on_interpolated_message(logger, caplog):
    with caplog.at_level(logging.WARNING, logger=logger.name):
        warn_once_fn(logger, "value=<%s> | ignored", "alpha")
        warn_once_fn(logger, "value=<%s> | ignored", "alpha")
        warn_once_fn(logger, "value=<%s> | ignored", "beta")

    messages = [record.getMessage() for record in caplog.records]
    assert messages == ["value=<alpha> | ignored", "value=<beta> | ignored"]
