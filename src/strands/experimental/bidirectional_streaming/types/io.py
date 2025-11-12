"""BidiIO protocol for bidirectional streaming IO channels.

Defines the standard interface that all bidirectional IO channels must implement
for integration with BidirectionalAgent. This protocol enables clean
separation between the agent's core logic and hardware-specific implementations.
"""

from typing import Protocol


class BidiIO(Protocol):
    """Base protocol for bidirectional IO channels.
    
    Defines the interface that IO channels must implement to work with
    BidirectionalAgent. IO channels handle hardware abstraction (audio, video,
    WebSocket, etc.) while the agent handles model communication and logic.
    """

    async def start(self) -> dict:

        """Setup IO channels for input and output."""
        ...

    async def send(self) -> dict:
        """Read input data from the IO channel source.
        
        Returns:
            dict: Input event data to send to the model.
        """
        ...

    async def receive(self, event: dict) -> None:
        """Process output event from the model through the IO channel.
        
        Args:
            event: Output event from the model to handle.
        """
        ...

    async def stop(self) -> None:
        """Clean up IO channel resources.
        
        Called by the agent during shutdown to ensure proper
        resource cleanup (streams, connections, etc.).
        """
        ...