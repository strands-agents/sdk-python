"""Event logging utility for bidirectional streaming models.

Logs incoming and outgoing events with truncated content for analysis.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def truncate_value(value: Any, max_length: int = 100) -> Any:
    """Recursively truncate string values in nested structures."""
    if isinstance(value, str):
        if len(value) > max_length:
            return value[:max_length] + f"... (truncated, total: {len(value)} chars)"
        return value
    elif isinstance(value, bytes):
        if len(value) > max_length:
            return f"<bytes: {len(value)} bytes, showing first {max_length}>: {value[:max_length]!r}..."
        return f"<bytes: {len(value)} bytes>: {value!r}"
    elif isinstance(value, dict):
        return {k: truncate_value(v, max_length) for k, v in value.items()}
    elif isinstance(value, list):
        return [truncate_value(item, max_length) for item in value]
    return value


class EventLogger:
    """Logger for bidirectional streaming events."""

    def __init__(self, provider_name: str, log_dir: str = "event_logs"):
        """Initialize event logger.

        Args:
            provider_name: Name of the provider (e.g., "gemini", "nova", "openai")
            log_dir: Directory to store log files
        """
        self.provider_name = provider_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create session-specific log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{provider_name}_{timestamp}.jsonl"

        self.logger = logging.getLogger(f"event_logger.{provider_name}")
        self.event_count = {"incoming": 0, "outgoing": 0}

    def log_event(self, direction: str, event_type: str, event_data: Dict[str, Any]) -> None:
        """Log an event to file and console.

        Args:
            direction: "incoming" or "outgoing"
            event_type: Type of event (e.g., "audio", "text", "tool_call")
            event_data: Event data dictionary
        """
        self.event_count[direction] += 1

        # Truncate long strings in event data
        truncated_data = truncate_value(event_data, max_length=100)

        # Create log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "provider": self.provider_name,
            "direction": direction,
            "event_type": event_type,
            "sequence": self.event_count[direction],
            "data": truncated_data,
        }

        # Write to file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry, indent=None) + "\n")

        # Log to console (debug level)
        self.logger.debug(
            f"[{direction.upper()}] {event_type} #{self.event_count[direction]}: "
            f"{json.dumps(truncated_data, indent=2)}"
        )

    def log_incoming(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Log an incoming event from the provider."""
        self.log_event("incoming", event_type, event_data)

    def log_outgoing(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Log an outgoing event to the provider."""
        self.log_event("outgoing", event_type, event_data)

    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            "provider": self.provider_name,
            "log_file": str(self.log_file),
            "incoming_count": self.event_count["incoming"],
            "outgoing_count": self.event_count["outgoing"],
            "total_count": sum(self.event_count.values()),
        }
