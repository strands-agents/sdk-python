"""Handle text input and output from bidi agent."""

import asyncio
import logging
import sys
from typing import TYPE_CHECKING

from ..types.events import BidiInterruptionEvent, BidiOutputEvent, BidiTextInputEvent, BidiTranscriptStreamEvent
from ..types.io import BidiInput, BidiOutput

if TYPE_CHECKING:
    from ..agent.agent import BidiAgent

logger = logging.getLogger(__name__)


class _BidiTextInput(BidiInput):
    """Handle text input from user."""

    def __init__(self) -> None:
        """Setup async stream reader."""
        self._reader = asyncio.StreamReader()

    async def start(self, agent: "BidiAgent") -> None:
        """Connect reader to stdin."""
        loop = asyncio.get_running_loop()
        protocol = asyncio.StreamReaderProtocol(self._reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    async def __call__(self) -> BidiTextInputEvent:
        """Read user input from stdin."""
        text = (await self._reader.readline()).decode().strip()
        return BidiTextInputEvent(text, role="user")


class _BidiTextOutput(BidiOutput):
    """Handle text output from bidi agent."""

    async def __call__(self, event: BidiOutputEvent) -> None:
        """Print text events to stdout."""
        if isinstance(event, BidiInterruptionEvent):
            logger.debug("reason=<%s> | text output interrupted", event["reason"])
            print("interrupted")

        elif isinstance(event, BidiTranscriptStreamEvent):
            text = event["text"]
            is_final = event["is_final"]
            role = event["role"]

            logger.debug(
                "role=<%s>, is_final=<%s>, text_length=<%d> | text transcript received",
                role,
                is_final,
                len(text),
            )

            if not is_final:
                text = f"Preview: {text}"

            print(text)


class BidiTextIO:
    """Handle text input and output from bidi agent."""

    def input(self) -> _BidiTextInput:
        """Return text processing BidiInput."""
        return _BidiTextInput()

    def output(self) -> _BidiTextOutput:
        """Return text processing BidiOutput."""
        return _BidiTextOutput()
