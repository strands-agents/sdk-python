"""Handle text input and output from bidi agent."""

import logging

from ..types.events import BidiInterruptionEvent, BidiOutputEvent, BidiTranscriptStreamEvent
from ..types.io import BidiOutput

logger = logging.getLogger(__name__)


class _BidiTextOutput(BidiOutput):
    """Handle text output from bidi agent."""

    async def start(self) -> None:
        """Start text output."""
        pass

    async def stop(self) -> None:
        """Stop text output."""
        pass

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

    def output(self) -> _BidiTextOutput:
        """Return text processing BidiOutput."""
        return _BidiTextOutput()
