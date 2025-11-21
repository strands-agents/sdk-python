"""Types for custom async constructs."""

from typing import Any, Awaitable, Protocol


class Startable(Protocol):
    """A construct that must first be started before use."""

    def start(self, *args: Any, **kwargs: Any) -> Awaitable[None]:
        """Setup resources and start connections."""
        ...

    def stop(self) -> Awaitable[None]:
        """Tear down resources and stop connections."""
        ...
