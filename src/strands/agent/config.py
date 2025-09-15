"""Agent configuration parser for agent-format.md support."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class AgentConfig:
    """Parser for agent configuration files following agent-format.md specification."""

    def __init__(self, config_source: Union[str, Dict[str, Any]]):
        """Initialize agent configuration.

        Args:
            config_source: Path to JSON config file or config dictionary
        """
        if isinstance(config_source, str):
            self.config = self._load_from_file(config_source)
        elif isinstance(config_source, dict):
            self.config = config_source.copy()
        else:
            raise ValueError("config_source must be a file path string or dictionary")

    def _load_from_file(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file.

        Args:
            file_path: Path to the configuration file

        Returns:
            Parsed configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file contains invalid JSON
        """
        path = Path(file_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(f"Agent config file not found: {file_path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError(f"Config file {file_path} must contain a JSON object")
                return data
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in config file {file_path}: {e.msg}", e.doc, e.pos) from e

    @property
    def tools(self) -> Optional[List[str]]:
        """Get tools configuration."""
        return self.config.get("tools")

    @property
    def model(self) -> Optional[str]:
        """Get model configuration."""
        return self.config.get("model")

    @property
    def system_prompt(self) -> Optional[str]:
        """Get system prompt from 'prompt' field."""
        return self.config.get("prompt")
