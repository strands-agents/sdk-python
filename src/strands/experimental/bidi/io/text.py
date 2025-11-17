"""Handle text input and output from bidi agent."""

import logging

from ..types.events import BidiInterruptionEvent, BidiOutputEvent, BidiTranscriptStreamEvent
from ..types.io import BidiOutput

logger = logging.getLogger(__name__)


class _BidiTextOutput(BidiOutput):
    """Handle text output from bidi agent."""

    async def __call__(self, event: BidiOutputEvent) -> None:
        """Print text events to stdout."""
        if isinstance(event, BidiInterruptionEvent):
            print("interrupted")

        elif isinstance(event, BidiTranscriptStreamEvent):
            text = event["text"]
            if not event["is_final"]:
                text = f"Preview: {text}"

            print(text)


class BidiTextIO:
    """Handle text input and output from bidi agent."""

    def output(self) -> _BidiTextOutput:
        """Return text processing BidiOutput"""
        return _BidiTextOutput()
