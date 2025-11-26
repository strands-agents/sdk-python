import unittest.mock

import pytest

from strands.experimental.bidi.io import BidiTextIO
from strands.experimental.bidi.types.events import BidiInterruptionEvent, BidiTextInputEvent, BidiTranscriptStreamEvent


@pytest.fixture
def stream_reader():
    with unittest.mock.patch("strands.experimental.bidi.io.text.asyncio.StreamReader") as mock:
        yield mock.return_value


@pytest.fixture
def text_io():
    return BidiTextIO()


@pytest.fixture
def text_input(text_io):
    return text_io.input()


@pytest.fixture
def text_output(text_io):
    return text_io.output()


@pytest.mark.asyncio
async def test_bidi_text_io_input(stream_reader, text_input):
    stream_reader.readline = unittest.mock.AsyncMock()
    stream_reader.readline.return_value = b"test value"

    tru_event = await text_input()
    exp_event = BidiTextInputEvent(text="test value", role="user")
    assert tru_event == exp_event


@pytest.mark.parametrize(
    ("event", "exp_print"),
    [
        (BidiInterruptionEvent(reason="user_speech"), "interrupted"),
        (BidiTranscriptStreamEvent(text="test text", delta="", is_final=False, role="user"), "Preview: test text"),
        (BidiTranscriptStreamEvent(text="test text", delta="", is_final=True, role="user"), "test text"),
    ]
)
@pytest.mark.asyncio
async def test_bidi_text_io_output_interrupt(event, exp_print, text_output, capsys):
    await text_output(event)

    tru_print = capsys.readouterr().out.strip()
    assert tru_print == exp_print
