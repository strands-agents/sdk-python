"""Tests for the bidirectional model base class."""

from collections.abc import AsyncIterable
from typing import Any

import pytest
from pydantic import BaseModel

from strands.experimental.bidi.models.model import AudioCapable, AudioConfig, BidiModel
from strands.experimental.bidi.types.events import BidiInputEvent, BidiOutputEvent
from strands.models import Model
from strands.types._events import ToolResultEvent
from strands.types.content import Messages
from strands.types.tools import ToolSpec


class _Output(BaseModel):
    value: str


class _TestBidiModel(BidiModel):
    def __init__(self) -> None:
        self.config = {"model_id": "test-model"}
        self.connection_config = {}
        self.usage_is_cumulative = False

    async def start(
        self,
        system_prompt: str | None = None,
        tools: list[ToolSpec] | None = None,
        messages: Messages | None = None,
        **kwargs: Any,
    ) -> None:
        pass

    async def stop(self) -> None:
        pass

    def receive(self) -> AsyncIterable[BidiOutputEvent]:
        async def events() -> AsyncIterable[BidiOutputEvent]:
            if False:
                yield

        return events()

    async def send(self, content: BidiInputEvent | ToolResultEvent) -> None:
        pass


class _AudioBidiModel(_TestBidiModel):
    @property
    def audio_config(self) -> AudioConfig:
        return {
            "input_rate": 16000,
            "output_rate": 24000,
            "channels": 1,
            "format": "pcm",
        }


def test_model_is_model():
    assert isinstance(_TestBidiModel(), Model)


def test_audio_capable_identifies_audio_models():
    assert isinstance(_AudioBidiModel(), AudioCapable)
    assert not isinstance(_TestBidiModel(), AudioCapable)


def test_update_config():
    model = _TestBidiModel()

    model.update_config(model_id="updated-model")

    assert model.get_config() == {"model_id": "updated-model"}


def test_get_config_returns_copy():
    model = _TestBidiModel()

    config = model.get_config()
    config["model_id"] = "updated-model"

    assert model.config == {"model_id": "test-model"}


@pytest.mark.asyncio
async def test_reconnect_raises_not_implemented():
    model = _TestBidiModel()

    with pytest.raises(NotImplementedError, match="reconnect is not implemented"):
        await model.reconnect()


def test_stream_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="regular streaming"):
        _TestBidiModel().stream([])


def test_structured_output_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="structured output"):
        _TestBidiModel().structured_output(_Output, [])
